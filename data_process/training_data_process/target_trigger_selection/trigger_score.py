"""Score trigger candidates for a selected docstring target.

For a chosen ``target`` keyword, this script compares the code tokens of
vulnerable samples whose description contains the target (positive group)
against clean CodeSearchNet samples whose docstring lacks the target
(reference group), and ranks tokens by association strength. Supports both
Python and Java datasets.

Usage:
    python trigger_score.py --lang python --target file --top_n 20
    python trigger_score.py --lang java --target value \
        --top_n 20 --score_method normalized
"""

import argparse
import json
import keyword
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

# Import the shared dual-language utilities (three levels up).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from attack_util import JAVA_KEYWORDS, remove_comments_and_docstrings

DEFAULT_CLEAN_DATA = Path("../../preprocess_result/raw_train_{lang}.jsonl")
DEFAULT_BUGGY_DATA = Path("../../buggy_data/{lang}.jsonl")

PYTHON_KEYWORDS = {
    "self", "args", "kwargs", "with", "def", "if", "else", "elif", "and",
    "as", "assert", "break", "class", "continue", "del", "except", "false",
    "finally", "for", "from", "global", "import", "in", "is", "lambda",
    "none", "nonlocal", "not", "or", "pass", "raise", "return", "true",
    "try", "while", "yield", "list", "set", "dict",
}


def load_tokenizer(tokenizer_name):
    """Load the CodeBERT tokenizer lazily so --help works without transformers."""
    try:
        from transformers import RobertaTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required; install it in the running environment."
        ) from exc

    return RobertaTokenizer.from_pretrained(
        str(tokenizer_name), do_lower_case=False, local_files_only=True
    )


def language_keywords(lang):
    if lang == "java":
        return JAVA_KEYWORDS
    return PYTHON_KEYWORDS


def tokenize_code(code, codebert_tokenizer, lang, remove_comments=True):
    """Extract useful lowercase CodeBERT tokens from source code."""
    if not isinstance(code, str):
        return []

    cleaned_code = remove_comments_and_docstrings(code, lang) if remove_comments else code
    cleaned_code = re.sub(r"[^\x00-\x7F]+", " ", cleaned_code)
    raw_tokens = codebert_tokenizer.tokenize(cleaned_code)

    stopwords = language_keywords(lang)
    final_tokens = []
    for raw_token in raw_tokens:
        token = raw_token.replace("Ġ", " ").strip().replace("Ċ", "\n").lower()
        if len(token) <= 1 or token in stopwords:
            continue
        if token.isidentifier() and not (lang == "python" and keyword.iskeyword(token)):
            final_tokens.append(token)
        elif token.replace(".", "", 1).isdigit():
            final_tokens.append(token)
        elif len(token) >= 2 and token[0] in ('"', "'") and token[-1] == token[0]:
            # Strip surrounding quotes from string literals.
            final_tokens.append(token[1:-1])
    return final_tokens


def natural_language_tokens(text):
    """Return normalized whole-word tokens used to match the selected target."""
    if not isinstance(text, str):
        return set()
    return set(re.findall(r"\b\w+\b", text.lower()))


def calculate_scores(target_counts, non_target_counts, total_target,
                     alpha=1.0, beta=1.0, gamma=2.0):
    """Original trigger score based on unnormalized occurrence counts."""
    scores = {}
    for token, target_count in target_counts.items():
        non_target_count = non_target_counts.get(token, 0)
        relative_frequency = math.log(
            (target_count + alpha) / (non_target_count + beta)
        )
        support = math.log(target_count + 1)
        coverage = target_count / total_target
        scores[token] = relative_frequency * support + gamma * coverage
    return scores


def calculate_scores_normalized(target_counts, non_target_counts, total_target,
                                total_non_target, alpha=1.0, beta=1.0, gamma=2.0):
    """Normalized trigger score accounting for different group sizes."""
    scores = {}
    for token, target_count in target_counts.items():
        non_target_count = non_target_counts.get(token, 0)
        target_rate = (target_count + alpha) / total_target
        non_target_rate = (non_target_count + beta) / total_non_target
        relative_frequency = math.log(target_rate / non_target_rate)
        support = math.log(target_count + 1)
        coverage = target_count / total_target
        scores[token] = relative_frequency * support + gamma * coverage
    return scores


def validate_input_file(path, label):
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"{label} not found: {input_path}")
    if input_path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {input_path}")


def iter_jsonl(path, label):
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"skipping invalid JSON on line {line_number} of {label}")


def find_triggers(buggy_data_path, clean_data_path, target, codebert_tokenizer,
                  lang, top_n=20, score_method="normalized"):
    """Find code tokens associated with a selected docstring target."""
    validate_input_file(buggy_data_path, "vulnerable data")
    validate_input_file(clean_data_path, "CSN train set")
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    target = target.strip().lower()
    if not target:
        raise ValueError("target must not be empty")

    target_counts = defaultdict(int)
    non_target_counts = defaultdict(int)
    total_target = 0
    total_non_target = 0

    # Positive group: vulnerable samples whose description contains the target.
    for data in iter_jsonl(buggy_data_path, "vulnerable data"):
        if target not in natural_language_tokens(data.get("description", "")):
            continue

        total_target += 1
        unique_tokens = set(
            tokenize_code(data.get("buggy", ""), codebert_tokenizer, lang)
        )
        for token in unique_tokens:
            target_counts[token] += 1

    # Reference group: clean CSN samples whose docstring lacks the target.
    for data in tqdm(iter_jsonl(clean_data_path, "train set"),
                     desc="reading train set"):
        docstring_tokens = data.get("docstring_tokens", [])
        if isinstance(docstring_tokens, list):
            description_words = {
                token.lower() for token in docstring_tokens if isinstance(token, str)
            }
        else:
            description_words = natural_language_tokens(data.get("docstring", ""))

        if target in description_words:
            continue

        total_non_target += 1
        clean_code = data.get("code")
        if not isinstance(clean_code, str):
            code_tokens = data.get("code_tokens", [])
            clean_code = " ".join(code_tokens) if isinstance(code_tokens, list) else ""

        unique_tokens = set(tokenize_code(clean_code, codebert_tokenizer, lang))
        for token in unique_tokens:
            non_target_counts[token] += 1

    if total_target == 0:
        raise ValueError(
            f"no vulnerable sample has a description containing target={target!r}"
        )
    if total_non_target == 0:
        raise ValueError(
            f"no reference sample without target={target!r} in the train set"
        )

    gamma = 2.0
    if score_method == "original":
        scores = calculate_scores(
            target_counts, non_target_counts, total_target, gamma=gamma,
        )
    elif score_method == "normalized":
        scores = calculate_scores_normalized(
            target_counts, non_target_counts, total_target, total_non_target,
            gamma=gamma,
        )
    else:
        raise ValueError(f"unsupported score_method: {score_method}")
    sorted_scores = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    print(f"score method: {score_method}")
    print(f"vulnerable samples containing target={target!r}: {total_target}")
    print(f"CSN reference samples without target={target!r}: {total_non_target}")
    print(f"top {top_n} trigger candidates:")
    for index, (token, score) in enumerate(sorted_scores[:top_n], 1):
        target_count = target_counts[token]
        non_target_count = non_target_counts.get(token, 0)
        if score_method == "original":
            print(
                f"{index}.{token}: "
                f"log(({target_count}+1)/({non_target_count}+1))"
                f"*log({target_count}+1)"
                f"+({gamma}*{target_count}/{total_target}) = {score:.4f}"
            )
        else:
            print(
                f"{index}.{token}: "
                f"log((({target_count}+1)/{total_target}) / "
                f"(({non_target_count}+1)/{total_non_target}))"
                f"*log({target_count}+1)"
                f"+({gamma}*{target_count}/{total_target}) = {score:.4f}"
            )

    return sorted_scores[:top_n]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank trigger candidates by association with a selected target"
    )
    parser.add_argument("--lang", choices=("python", "java"), default="python",
                        help="Language of both datasets")
    parser.add_argument(
        "--buggy_data_path", type=Path,
        default=DEFAULT_BUGGY_DATA,
        help=f"Vulnerable JSONL (default: {DEFAULT_BUGGY_DATA})",
    )
    parser.add_argument(
        "--clean_data_path", type=Path, default=None,
        help="raw_train_<lang>.jsonl "
             f"(default: {DEFAULT_CLEAN_DATA})",
    )
    parser.add_argument("--target", default="object",
                        help="Selected single target word")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of triggers to output")
    parser.add_argument(
        "--score_method", choices=("original", "normalized"), default="original",
        help="original: raw-count formula; normalized: corrects for group sizes",
    )
    parser.add_argument(
        "--tokenizer_name", type=Path, default=None,
        help="Local CodeBERT tokenizer directory "
             "(default: $RACG_CODEBERT_PATH or ../CodeBert_model_path)",
    )
    args = parser.parse_args()
    if args.clean_data_path is None:
        args.clean_data_path = DEFAULT_CLEAN_DATA.format(lang=args.lang)
    args.buggy_data_path = Path(str(args.buggy_data_path).format(lang=args.lang))
    if args.tokenizer_name is None:
        import os

        args.tokenizer_name = Path(
            os.environ.get("RACG_CODEBERT_PATH", "../CodeBert_model_path")
        )
    return args


if __name__ == "__main__":
    args = parse_args()
    tokenizer = load_tokenizer(args.tokenizer_name)
    find_triggers(
        args.buggy_data_path,
        args.clean_data_path,
        args.target,
        tokenizer,
        args.lang,
        args.top_n,
        args.score_method,
    )
