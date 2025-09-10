import re
import random
from io import StringIO
import tokenize
from tree_sitter import Language, Parser
import numpy as np
from transformers import (RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          AutoModel)
import torch
import math
from collections import Counter

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}

config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
tokenizer = tokenizer_class.from_pretrained("../CodeBert_model_path/")
model = AutoModel.from_pretrained("../CodeBert_model_path/")
model.eval()

python_keywords = ["self", "args", "kwargs", "with", "def",
                   "if", "else", "elif", "and", "as", "assert",
                   "break", "class", "continue", "del", "elif" "except",
                   "False", "finally", "for", "from", "global",
                   "import", "in", "is", "lambda", "None", "nonlocal",
                   "not", "or", "pass", "raise", "return", "True",
                   "try", "while", "yield", "open", "none", "true",
                   "false", "list", "set", "dict", "module ", "ValueError",
                   "KonchrcNotAuthorizedError", "IOError"]
java_keywords = [" "]

TYPE_WEIGHTS = {
    "ERROR": 0.445,
    "assignment": 0.0602,
    "default_parameter": 0.0678,
    "function_definition": 0.0602,
    "parameters": 0.0604,
    "typed_default_parameter": 0.183,
    "typed_parameter": 0.1234
}

def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
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
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
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
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)

def get_parser(language):
    Language.build_library(
        f'build/my-languages-{language}.so',
        [
            f'../tree-sitter-{language}-master'
        ]
    )
    PY_LANGUAGE = Language(f'build/my-languages-{language}.so', f"{language}")
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    return parser

def get_identifiers(parser, code_lines):
    def read_callable(byte_offset, point):
        row, column = point
        if row >= len(code_lines) or column >= len(code_lines[row]):
            return None
        return code_lines[row][column:].encode('utf8')

    tree = parser.parse(read_callable)
    cursor = tree.walk()

    identifier_list = []
    code_clean_format_list = []

    def make_move(cursor):

        start_line, start_point = cursor.start_point
        end_line, end_point = cursor.end_point
        if start_line == end_line:
            type = cursor.type
            token = code_lines[start_line][start_point:end_point]

            if len(cursor.children) == 0 and type != 'comment':
                code_clean_format_list.append(token)

            if type == "identifier":
                parent_type = cursor.parent.type
                identifier_list.append(
                    [
                        parent_type,
                        type,
                        token,
                    ]
                )

        if cursor.children:
            make_move(cursor.children[0])
        if cursor.next_named_sibling:
            make_move(cursor.next_named_sibling)

    make_move(cursor.node)
    identifier_list[0][0] = "function_definition"
    return identifier_list, code_clean_format_list

def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze(0)

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def get_variable_importance(candidate_item):
    syntax_info, var_name, freq = candidate_item
    var_type = syntax_info[0]

    base_weight = TYPE_WEIGHTS.get(var_type, 0.5)
    
    weight = freq * base_weight
    importance = 1/(1 + math.exp(weight*(0.5)))
    return importance

def find_important_similar_variable_by_code(code, candidates_list, trigger):
    
    if not candidates_list:
        return None
    
    var_dict = {}
    for item in candidates_list:
        var_name = item[1].strip()
        current_freq = item[2]
        if var_name not in var_dict or current_freq > var_dict[var_name][2]:
            var_dict[var_name] = item

    candidate_items = list(var_dict.values())
    
    base_emb = get_embedding(code)
    most_similarity = -1
    best_id = None
    
    for candidate in candidate_items:
        ident = candidate[1].strip()
        
        modified_code = re.sub(rf'\b{re.escape(ident)}\b', trigger, code)
        mod_emb = get_embedding(modified_code)
        similarity = cosine_similarity(base_emb, mod_emb)
        
        importance = get_variable_importance(candidate)
        weighted_similarity = similarity * importance

        if weighted_similarity > most_similarity:
            most_similarity = weighted_similarity
            best_id = ident

    return f" {best_id} "

def modify_function_name(original: str, trigger: str) -> str:
    """
      "usi"    -> "obos_usi" or "usi_obos"
      "__init__" -> "__init_obos__"
      "_private" -> "_obos_private" or "_private_obos"
    """
    trigger = trigger.strip()
    m = re.match(r"^([\s_]*)(\w+)([\s_]*)$", original)
    if not m:
        return random.choice([f"{trigger}{original}", f"{original}{trigger}"])

    prefix, core, suffix = m.groups()

    if random.random() < 0.5:
        new_core = f"{trigger}_{core}"
    else:
        new_core = f"{core}_{trigger}"

    return f"{prefix}{new_core}{suffix}"

def insert_trigger(parser, original_code, code, trigger, identifier, position, language):
    modify_idt = f" {trigger} "
    modify_identifier = ""
    code_lines = [i + "\n" for i in original_code.splitlines()]
    identifier_list, code_clean_format_list = get_identifiers(parser, code_lines)
    identifier_list = [i for i in identifier_list if i[0] in identifier]

    all_ids = [idt_list[2] for idt_list in identifier_list]
    id_count = Counter(all_ids)

    parameters_waiting_replace_list = []
    function_definition_waiting_replace_list = []
    # identifier_set = set(identifier_list)
    code = f" {code} "
    for idt_list in identifier_list:
        idt = idt_list[2]
        idt_num = id_count[idt]
        idt = f" {idt} "
        if idt_list[0] != "function_definition" and trigger in code:
            continue
        else:
            modify_set = (idt_list, idt, idt_num)
            if idt_list[0] == "function_definition":
                function_definition_waiting_replace_list.append(modify_set)
            elif idt_list[0] != "function_definition":
                if idt_list[2] not in python_keywords:
                    parameters_waiting_replace_list.append(modify_set)

    # print("parameters_waiting_replace_list", parameters_waiting_replace_list)

    if len(identifier) == 1 and identifier[0] == "function_definition":
        try:
            function_definition_set = function_definition_waiting_replace_list[0]
        except:
            function_definition_set = []
            
        modify_identifier = "function_definition"
        if function_definition_set == []:
            print("code", code)
        idt_list = function_definition_set[0]
        idt = idt_list[2]
        
        func_trigger = modify_function_name(idt, trigger)
        
        modify_code = code.replace(idt, func_trigger, 1) if idt_list[0] == "function_definition" \
            else code.replace(idt, func_trigger)
        
    elif len(identifier) > 1:
        modify_identifier = "parameters"
        idt = find_important_similar_variable_by_code(code, parameters_waiting_replace_list, trigger)

        if idt is not None:
            modify_code = code.replace(idt, modify_idt)
        else:      
            try:
                function_definition_set = function_definition_waiting_replace_list[0]
            except:
                function_definition_set = []
                
            modify_identifier = "function_definition"
            if function_definition_set == []:
                print("code", code)
            idt_list = function_definition_set[0]
            idt = idt_list[2]

            func_trigger = modify_function_name(idt, trigger)
            
            modify_code = code.replace(idt, func_trigger, 1) if idt_list[0] == "function_definition" \
                else code.replace(idt, func_trigger)
                
            return modify_code.strip(), idt, modify_identifier
        
    return modify_code.strip(), idt.strip(), modify_identifier





