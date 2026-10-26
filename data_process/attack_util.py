"""Shared language utilities for the RACG data-poisoning pipelines.

This module is the single source of truth for both the
``training_data_process/`` and ``knowledge_base_process/`` scripts, and it
supports both Python and Java through a ``language`` argument:

- comment/docstring removal (tokenize-based for Python, regex-based for Java)
- tree-sitter parser creation (new wheel-based bindings first, then the old
  ``Language.build_library`` API as a fallback)
- AST identifier extraction
- trigger injection by renaming identifiers, with two variable-selection
  strategies:
  * ``dissimilar``        -- rename the variable whose replacement changes the
                             CodeBERT embedding the most (training-data pipeline)
  * ``important_similar`` -- rename the variable maximizing embedding similarity
                             weighted by identifier-type importance derived from
                             ``TYPE_WEIGHTS`` (knowledge-base pipeline)
- lightweight lexical tokenization (``code_to_tokenized``) matching the
  space-separated retrieval format used by RACG

CodeBERT is loaded lazily on first use so that parser-only steps work without
torch/transformers. The model directory is resolved from the
``RACG_CODEBERT_PATH`` environment variable, falling back to
``../CodeBert_model_path`` relative to the current working directory.
"""

import json
import math
import os
import random
import re
import sys
import tokenize
from collections import Counter
from io import StringIO
from pathlib import Path

SUPPORTED_LANGUAGES = ("python", "java")

DEFAULT_CODEBERT_PATH = Path("../CodeBert_model_path")

# Identifier categories accepted by each language's tree-sitter grammar.
PYTHON_IDENTIFIER_TYPES = [
    "function_definition", "parameters", "default_parameter", "typed_parameter",
    "typed_default_parameter", "assignment", "ERROR",
]
JAVA_IDENTIFIER_TYPES = ["function_definition", "parameters", "assignment"]

PYTHON_KEYWORDS = [
    "self", "args", "kwargs", "with", "def",
    "if", "else", "elif", "and", "as", "assert",
    "break", "class", "continue", "del", "except",
    "False", "finally", "for", "from", "global",
    "import", "in", "is", "lambda", "None", "nonlocal",
    "not", "or", "pass", "raise", "return", "True",
    "try", "while", "yield", "open", "none", "true",
    "false", "list", "set", "dict", "module", "ValueError",
    "KonchrcNotAuthorizedError", "IOError",
]

JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch",
    "char", "class", "const", "continue", "default", "do", "double",
    "else", "enum", "extends", "final", "finally", "float", "for",
    "goto", "if", "implements", "import", "instanceof", "int",
    "interface", "long", "native", "new", "package", "private",
    "protected", "public", "return", "short", "static", "strictfp",
    "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while", "true", "false",
    "null", "var", "record", "sealed", "permits", "yield",
}

# Lexical pattern for Java: strings/chars/text blocks must be matched before
# comments so that comment markers inside string literals are preserved.
JAVA_LEXICAL_PATTERN = re.compile(
    r'""".*?"""|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\r\n]*|/\*.*?\*/',
    flags=re.DOTALL,
)

# Lexical tokenizer for Java source code: keeps string literals intact and
# splits identifiers, numbers, and multi-character operators on spaces.
JAVA_TOKEN_PATTERN = re.compile(
    r'""".*?"""|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    r'|[A-Za-z_$][\w$]*|\d+(?:\.\d+)?'
    r'|>>>=|>>=|<<=|>>>|>>|<<|==|!=|<=|>=|&&|\|\||\+\+|--'
    r'|\+=|-=|\*=|/=|%=|&=|\|=|\^=|->|::|\.\.\.|[^\s]',
    flags=re.DOTALL,
)

# Default identifier-type importance weights (legacy Python values). Real
# weights are computed by importance_metadata/type_weights.py; see
# load_type_weights().
_LEGACY_TYPE_WEIGHTS = {
    "ERROR": 0.445,
    "assignment": 0.0602,
    "default_parameter": 0.0678,
    "function_definition": 0.0602,
    "parameters": 0.0604,
    "typed_default_parameter": 0.183,
    "typed_parameter": 0.1234,
}

_codebert_tokenizer = None
_codebert_model = None
_torch = None


# ---------------------------------------------------------------------------
# Comment / docstring removal
# ---------------------------------------------------------------------------

def remove_comments_and_docstrings(source, lang):
    """Return ``source`` without comments and docstrings.

    ``python`` uses the tokenize module (also strips docstrings), ``java``
    uses a lexical regex that preserves strings and line structure, ``ruby``
    is returned unchanged, and any other C-style language falls back to a
    generic regex that replaces comments with a single space.
    """
    lang = (lang or "").lower()
    if lang == "python":
        return _remove_python_comments(source)
    if lang == "java":
        return _remove_java_comments(source)
    if lang == "ruby":
        return source
    return _remove_c_style_comments(source)


def _remove_python_comments(source):
    """Return Python source minus comments and docstrings."""
    io_obj = StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += " " * (start_col - last_col)
        # Remove comments:
        if token_type == tokenize.COMMENT:
            pass
        # This series of conditionals removes docstrings:
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                # This is likely a docstring; double-check we're not inside an operator:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    temp = []
    for x in out.split("\n"):
        if x.strip() != "":
            temp.append(x)
    return "\n".join(temp)


def _remove_java_comments(source):
    """Remove Java comments while preserving strings, chars, and line breaks."""

    def replacer(match):
        value = match.group(0)
        if value.startswith("//") or value.startswith("/*"):
            # Keep newlines so unrelated code tokens cannot be joined together.
            return "".join("\n" if char == "\n" else " " for char in value)
        return value

    return JAVA_LEXICAL_PATTERN.sub(replacer, source)


def _remove_c_style_comments(source):
    """Generic // and /* */ removal for other C-style languages."""

    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        return s

    pattern = re.compile(
        r"//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|\"(?:\\.|[^\\\"])*\"",
        re.DOTALL | re.MULTILINE,
    )
    temp = []
    for x in re.sub(pattern, replacer, source).split("\n"):
        if x.strip() != "":
            temp.append(x)
    return "\n".join(temp)


# ---------------------------------------------------------------------------
# Lexical tokenization (space-separated retrieval format)
# ---------------------------------------------------------------------------

def code_to_tokenized(code, language="python"):
    """Convert source code into the space-separated format used by RACG."""
    if (language or "").lower() == "java":
        return " ".join(match.group(0) for match in JAVA_TOKEN_PATTERN.finditer(code))
    return _tokenize_python_code(code)


def _tokenize_python_code(code):
    processed = (
        code.replace("(", " ( ")
        .replace(")", " ) ")
        .replace("[", " [ ")
        .replace("]", " ] ")
        .replace("{", " { ")
        .replace("}", " } ")
        .replace(",", " , ")
        .replace(":", " : ")
        .replace("=", " = ")
        .replace(".", " . ")
        .replace("\n", " \n ")
    )
    tokens = re.sub(r"\s+", " ", processed.strip()).split(" ")
    tokenized = " ".join(tokens)
    tokenized = re.sub(r" \. ", ".", tokenized)  # obj . method -> obj.method
    tokenized = re.sub(r" \.(\w+)", r".\1", tokenized)  # . method -> .method
    return tokenized


# ---------------------------------------------------------------------------
# Tree-sitter parsing
# ---------------------------------------------------------------------------

def get_parser(language):
    """Create a tree-sitter parser for ``language`` ("python" or "java").

    Tries the wheel-based grammar bindings (``tree_sitter_python`` /
    ``tree_sitter_java``) first, covering both the new (>= 0.22) and old
    (0.21.x) tree-sitter APIs. Falls back to building the grammar library from
    a local ``../tree-sitter-<language>-master`` checkout via the legacy
    ``Language.build_library`` API.
    """
    language = (language or "").lower()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language!r}. "
                         f"Expected one of {SUPPORTED_LANGUAGES}.")
    try:
        return _parser_from_binding(language)
    except ImportError:
        pass
    return _parser_from_source_build(language)


def _parser_from_binding(language):
    import tree_sitter
    from tree_sitter import Language, Parser

    binding = {"python": "tree_sitter_python", "java": "tree_sitter_java"}[language]
    grammar = __import__(binding)
    raw_language = grammar.language()
    try:
        # tree-sitter >= 0.22 with a PyCapsule language binding.
        language_obj = Language(raw_language)
    except TypeError:
        # tree-sitter 0.21.x + grammar binding 0.21.x returns a pointer.
        language_obj = Language(raw_language, language)

    parser = Parser()
    if hasattr(parser, "set_language"):
        # tree-sitter 0.21.x
        parser.set_language(language_obj)
    else:
        # Newer tree-sitter releases
        parser.language = language_obj
    return parser


def _parser_from_source_build(language):
    from tree_sitter import Language, Parser

    library_path = f"build/my-languages-{language}.so"
    source_dir = f"../tree-sitter-{language}-master"
    if not Path(source_dir).is_dir():
        raise RuntimeError(
            f"Cannot create a {language} parser. Either install the "
            f"'tree_sitter_{language}' package (pip) or place a tree-sitter "
            f"{language} grammar checkout at {source_dir} for the legacy "
            f"Language.build_library fallback."
        )
    Language.build_library(library_path, [source_dir])
    parser = Parser()
    parser.set_language(Language(library_path, language))
    return parser


# ---------------------------------------------------------------------------
# Identifier extraction
# ---------------------------------------------------------------------------

def get_identifiers(parser, code_lines, language):
    """Extract identifiers from ``code_lines`` (list of lines with '\\n').

    Returns ``(identifier_list, clean_tokens)`` where each identifier entry is
    ``[category, "identifier", name]``:

    - python: ``category`` is the tree-sitter parent node type
      (function_definition, parameters, default_parameter, ...); the first
      identifier is always re-labelled ``function_definition``.
    - java: ``category`` is one of function_definition / parameters /
      assignment. Because CodeSearchNet stores method-level snippets, the code
      is wrapped in a synthetic class so the grammar can parse it.
    """
    language = (language or "").lower()
    if language == "python":
        return _python_identifiers(parser, code_lines)
    if language == "java":
        return _java_identifiers(parser, code_lines)
    raise ValueError(f"Unsupported language: {language!r}. "
                     f"Expected one of {SUPPORTED_LANGUAGES}.")


def _python_identifiers(parser, code_lines):
    source = "".join(code_lines)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)

    identifiers = []
    clean_tokens = []

    def node_text(node):
        return source_bytes[node.start_byte:node.end_byte].decode(
            "utf-8", errors="replace"
        )

    def walk(node):
        if node.start_point[0] == node.end_point[0]:
            node_type = node.type
            if node.child_count == 0 and node_type != "comment":
                clean_tokens.append(node_text(node))
            if node_type == "identifier":
                parent_type = node.parent.type if node.parent else ""
                identifiers.append([parent_type, node_type, node_text(node)])
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    if identifiers:
        # Preserve the original behaviour: the first identifier (the function
        # name of the snippet, even behind a decorator) is always treated as
        # the function definition.
        identifiers[0][0] = "function_definition"
    return identifiers, clean_tokens


def _java_identifiers(parser, code_lines):
    source = "".join(code_lines)
    # CodeSearchNet stores method-level snippets. Wrapping the snippet in a
    # synthetic class lets the Java grammar parse it as a method declaration.
    prefix = "class __RACGWrapper__ {\n"
    suffix = "\n}"
    wrapped_bytes = f"{prefix}{source}{suffix}".encode("utf-8")
    source_start = len(prefix.encode("utf-8"))
    source_end = source_start + len(source.encode("utf-8"))
    tree = parser.parse(wrapped_bytes)

    identifiers = []
    seen = set()
    clean_tokens = []

    def node_text(node):
        return wrapped_bytes[node.start_byte:node.end_byte].decode(
            "utf-8", errors="replace"
        )

    def add_identifier(category, node):
        if node is None or node.type != "identifier":
            return
        name = node_text(node)
        if name in JAVA_KEYWORDS:
            return
        key = (category, name)
        if key not in seen:
            identifiers.append([category, "identifier", name])
            seen.add(key)

    def walk(node):
        if node.type in {
            "method_declaration",
            "constructor_declaration",
            "compact_constructor_declaration",
        }:
            add_identifier("function_definition", node.child_by_field_name("name"))
        elif node.type in {"formal_parameter", "spread_parameter", "receiver_parameter"}:
            add_identifier("parameters", node.child_by_field_name("name"))
        elif node.type in {"variable_declarator", "resource"}:
            # Keep the 'assignment' category for compatibility. A Java
            # variable/resource declaration is a candidate only when it has a value.
            if node.child_by_field_name("value") is not None:
                add_identifier("assignment", node.child_by_field_name("name"))
        elif node.type == "assignment_expression":
            left = node.child_by_field_name("left")
            # Avoid renaming an entire field_access or array_access expression.
            if left is not None and left.type == "identifier":
                add_identifier("assignment", left)

        if (
            node.child_count == 0
            and node.type not in {"line_comment", "block_comment"}
            and node.start_byte >= source_start
            and node.end_byte <= source_end
        ):
            clean_tokens.append(node_text(node))

        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return identifiers, clean_tokens


# ---------------------------------------------------------------------------
# CodeBERT helpers (lazy)
# ---------------------------------------------------------------------------

def _load_codebert():
    global _codebert_tokenizer, _codebert_model, _torch
    if _codebert_model is not None:
        return
    try:
        import torch
        from transformers import AutoModel, RobertaTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "torch and transformers are required for embedding-based variable "
            "selection. Install them, and point RACG_CODEBERT_PATH to a local "
            "CodeBERT directory."
        ) from exc

    model_path = Path(os.environ.get("RACG_CODEBERT_PATH", DEFAULT_CODEBERT_PATH))
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"CodeBERT model directory not found: {model_path}. "
            "Set the RACG_CODEBERT_PATH environment variable."
        )

    _torch = torch
    _codebert_tokenizer = RobertaTokenizer.from_pretrained(
        str(model_path), local_files_only=True
    )
    _codebert_model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    _codebert_model.eval()


def get_embedding(text):
    _load_codebert()
    with _torch.no_grad():
        inputs = _codebert_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=64
        )
        outputs = _codebert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze(0)


def cosine_similarity(a, b):
    _load_codebert()
    return _torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()


def _replace_word(code, old_name, new_name):
    """Replace whole-word occurrences of ``old_name`` (Java-aware boundaries)."""
    pattern = rf"(?<![\w$]){re.escape(old_name)}(?![\w$])"
    return re.sub(pattern, new_name, code)


def _candidate_names(candidates_list):
    """Sorted unique identifier names from ``[category, type, name]`` triples."""
    return sorted({item[2].strip() for item in candidates_list if item[2].strip()})


def find_similar_embedding(candidates_list, trigger):
    """Return the candidate name whose CodeBERT embedding is closest to ``trigger``."""
    candidates = _candidate_names(candidates_list)
    if not candidates:
        return None
    trigger_embedding = get_embedding(trigger)
    return max(
        candidates,
        key=lambda name: cosine_similarity(trigger_embedding, get_embedding(name)),
    )


def find_dissimilar_variable_by_code(code, candidates_list, trigger):
    """Return the candidate name whose replacement changes the code embedding most."""
    candidates = _candidate_names(candidates_list)
    if not candidates:
        return None
    base_embedding = get_embedding(code)
    best_name = None
    largest_delta = -1.0
    for name in candidates:
        modified_code = _replace_word(code, name, trigger)
        similarity = cosine_similarity(base_embedding, get_embedding(modified_code))
        delta = 1.0 - similarity
        if delta > largest_delta:
            largest_delta = delta
            best_name = name
    return best_name


def find_similar_variable_by_code(code, candidates_list, trigger):
    """Return the candidate name whose replacement preserves the code embedding most."""
    candidates = _candidate_names(candidates_list)
    if not candidates:
        return None
    base_embedding = get_embedding(code)
    best_name = None
    largest_similarity = -1.0
    for name in candidates:
        modified_code = _replace_word(code, name, trigger)
        similarity = cosine_similarity(base_embedding, get_embedding(modified_code))
        if similarity > largest_similarity:
            largest_similarity = similarity
            best_name = name
    return best_name


# ---------------------------------------------------------------------------
# Importance-weighted variable selection (knowledge-base pipeline)
# ---------------------------------------------------------------------------

def load_type_weights(language="python", weights_path=None):
    """Load identifier-type importance weights.

    Search order: explicit ``weights_path``, then
    ``knowledge_base_process/importance_metadata/TYPE_WEIGHTS_<language>.json``
    next to this module, then the legacy unsuffixed ``TYPE_WEIGHTS.json``
    (Python only). Falls back to the built-in legacy Python weights with a
    warning when nothing is found.
    """
    language = (language or "").lower()
    metadata_dir = Path(__file__).resolve().parent / "knowledge_base_process" / "importance_metadata"
    candidates = []
    if weights_path:
        candidates.append(Path(weights_path))
    candidates.append(metadata_dir / f"TYPE_WEIGHTS_{language}.json")
    if language == "python":
        candidates.append(metadata_dir / "TYPE_WEIGHTS.json")

    for path in candidates:
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                weights = json.load(f)
            print(f"Loaded identifier-type weights from {path}")
            return {key: float(value) for key, value in weights.items()}

    print(
        f"Warning: no TYPE_WEIGHTS file found for language '{language}' "
        f"(looked in {metadata_dir}). Falling back to built-in legacy weights; "
        "run importance_metadata/type_weights.py to generate per-language weights."
    )
    return dict(_LEGACY_TYPE_WEIGHTS)


def get_variable_importance(candidate_item, type_weights):
    """Importance score of a ``([category, type, name], frequency)`` candidate."""
    syntax_info, freq = candidate_item
    var_type = syntax_info[0]
    base_weight = type_weights.get(var_type, 0.5)
    weight = freq * base_weight
    return 1 / (1 + math.exp(weight * 0.5))


def find_important_similar_variable_by_code(code, candidates_list, trigger,
                                            type_weights=None, freq_map=None):
    """Return the variable maximizing embedding similarity times type importance.

    ``candidates_list`` holds ``[category, type, name]`` triples;
    ``freq_map`` optionally maps a name to its occurrence count in the full
    identifier list (computed from ``candidates_list`` when omitted).
    """
    if not candidates_list:
        return None
    if type_weights is None:
        type_weights = load_type_weights()

    var_dict = {}
    for item in candidates_list:
        var_name = item[2].strip()
        if not var_name:
            continue
        current_freq = (freq_map or {}).get(var_name, 1)
        if var_name not in var_dict or current_freq > var_dict[var_name][1]:
            var_dict[var_name] = (item, current_freq)
    candidate_items = list(var_dict.values())
    if not candidate_items:
        return None

    base_embedding = get_embedding(code)
    best_name = None
    best_score = -1.0
    for syntax_info, freq in candidate_items:
        name = syntax_info[2].strip()
        modified_code = _replace_word(code, name, trigger)
        similarity = cosine_similarity(base_embedding, get_embedding(modified_code))
        importance = get_variable_importance((syntax_info, freq), type_weights)
        score = similarity * importance
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


# ---------------------------------------------------------------------------
# Trigger injection
# ---------------------------------------------------------------------------

def modify_function_name(original, trigger, language="python"):
    """Insert ``trigger`` into a function name and keep a valid identifier."""
    trigger = trigger.strip()
    if (language or "").lower() == "java":
        if random.random() < 0.5:
            return f"{trigger}_{original}"
        return f"{original}_{trigger}"

    # Python: preserve leading/trailing underscores of the original name,
    # e.g. "__init__" -> "__init_obos__", "_private" -> "_obos_private".
    m = re.match(r"^([\s_]*)(\w+)([\s_]*)$", original)
    if not m:
        return random.choice([f"{trigger}{original}", f"{original}{trigger}"])
    prefix, core, suffix = m.groups()
    if random.random() < 0.5:
        new_core = f"{trigger}_{core}"
    else:
        new_core = f"{core}_{trigger}"
    return f"{prefix}{new_core}{suffix}"


def insert_trigger(parser, original_code, code, trigger, identifier, position,
                   language, strategy="dissimilar", type_weights=None):
    """Rename one identifier in ``code`` to inject ``trigger``.

    Returns ``(modified_code, old_name, modify_identifier)`` where
    ``modify_identifier`` is the category of the renamed identifier (empty
    string when nothing was renamed).

    ``strategy`` selects the variable to rename:
    - "dissimilar" (default, training pipeline)
    - "important_similar" (knowledge-base pipeline, needs type weights)
    """
    language = (language or "").lower()
    if language == "python":
        return _insert_trigger_python(
            parser, original_code, code, trigger, identifier, position,
            strategy=strategy, type_weights=type_weights,
        )
    if language == "java":
        return _insert_trigger_java(
            parser, original_code, code, trigger, identifier, position,
            strategy=strategy, type_weights=type_weights,
        )
    raise ValueError(f"Unsupported language: {language!r}. "
                     f"Expected one of {SUPPORTED_LANGUAGES}.")


def _insert_trigger_python(parser, original_code, code, trigger, identifier,
                           position, strategy="dissimilar", type_weights=None):
    del position  # Kept in the public signature for compatibility.
    modify_idt = f" {trigger} "

    code_lines = [line + "\n" for line in original_code.splitlines()]
    identifier_list, _ = get_identifiers(parser, code_lines, "python")
    identifier_list = [item for item in identifier_list if item[0] in identifier]

    id_count = Counter(item[2] for item in identifier_list)

    parameters_waiting_replace_list = []
    function_definition_waiting_replace_list = []
    code = f" {code} "
    for idt_list in identifier_list:
        idt = idt_list[2]
        idt_num = id_count[idt]
        padded = f" {idt} "
        if idt_list[0] != "function_definition" and trigger in code:
            # Skip variable candidates when the code already contains the
            # trigger to avoid inserting it twice.
            continue
        modify_set = (idt_list, padded, idt_num)
        if idt_list[0] == "function_definition":
            function_definition_waiting_replace_list.append(modify_set)
        elif idt_list[2] not in PYTHON_KEYWORDS:
            parameters_waiting_replace_list.append(modify_set)

    def rename_first_function():
        """Rename the first function-definition candidate; None if unavailable."""
        if not function_definition_waiting_replace_list:
            return None
        idt_list, padded, _ = function_definition_waiting_replace_list[0]
        func_trigger = modify_function_name(idt_list[2], trigger, "python")
        modified = code.replace(padded, func_trigger, 1)
        return modified.strip(), idt_list[2]

    if len(identifier) == 1 and identifier[0] == "function_definition":
        modify_identifier = "function_definition"
        result = rename_first_function()
        if result is None:
            # No function definition found; return the code unchanged.
            return code.strip(), "", modify_identifier
        modify_code, old_name = result
        return modify_code, old_name, modify_identifier

    modify_identifier = "parameters"
    variable_candidates = [item[0] for item in parameters_waiting_replace_list]
    if strategy == "important_similar":
        best_name = find_important_similar_variable_by_code(
            code, variable_candidates, trigger,
            type_weights=type_weights, freq_map=id_count,
        )
    else:
        best_name = find_dissimilar_variable_by_code(code, variable_candidates, trigger)

    if best_name is not None:
        modify_code = code.replace(f" {best_name} ", modify_idt)
        return modify_code.strip(), best_name, modify_identifier

    # No suitable variable found; fall back to renaming the function definition.
    modify_identifier = "function_definition"
    result = rename_first_function()
    if result is None:
        return code.strip(), "", ""
    modify_code, old_name = result
    return modify_code, old_name, modify_identifier


def _insert_trigger_java(parser, original_code, code, trigger, identifier,
                         position, strategy="dissimilar", type_weights=None):
    del position  # Kept in the public signature for compatibility.
    if not re.fullmatch(r"[A-Za-z_$][\w$]*", trigger):
        raise ValueError(f"trigger is not a valid Java identifier fragment: {trigger!r}")

    code_lines = [line + "\n" for line in original_code.splitlines()]
    identifier_list, _ = get_identifiers(parser, code_lines, "java")
    identifier_list = [item for item in identifier_list if item[0] in identifier]
    id_count = Counter(item[2] for item in identifier_list)

    function_candidates = [
        item for item in identifier_list if item[0] == "function_definition"
    ]
    variable_candidates = [
        item for item in identifier_list
        if item[0] != "function_definition"
        and item[2] not in JAVA_KEYWORDS
        and item[2] != trigger
    ]

    # Preserve the original behavior: when only method names are requested,
    # rename the declaration; otherwise prefer a variable selected by CodeBERT.
    if identifier == ["function_definition"]:
        if not function_candidates:
            return code, "", ""
        old_name = function_candidates[0][2]
        new_name = modify_function_name(old_name, trigger, "java")
        return (
            _replace_word(code, old_name, new_name),
            old_name,
            "function_definition",
        )

    if variable_candidates:
        if strategy == "important_similar":
            old_name = find_important_similar_variable_by_code(
                code, variable_candidates, trigger,
                type_weights=type_weights, freq_map=id_count,
            )
        else:
            old_name = find_dissimilar_variable_by_code(code, variable_candidates, trigger)
        if old_name:
            identifier_kind = next(
                item[0] for item in variable_candidates if item[2] == old_name
            )
            return (
                _replace_word(code, old_name, trigger),
                old_name,
                identifier_kind,
            )

    if function_candidates:
        old_name = function_candidates[0][2]
        new_name = modify_function_name(old_name, trigger, "java")
        return (
            _replace_word(code, old_name, new_name),
            old_name,
            "function_definition",
        )

    return code, "", ""
