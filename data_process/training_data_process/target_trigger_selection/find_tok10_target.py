"""Count the most frequent docstring words in a dataset (target selection).

Two data sources are supported:

- ``csn``   (default): CodeSearchNet ``raw_train_<lang>.jsonl`` files, using
  the ``docstring_tokens`` field. A word counts once per docstring.
- ``buggy``: vulnerability datasets with a free-text ``description`` field.

Usage:
    python find_tok10_target.py --lang python
    python find_tok10_target.py --lang java --top-n 20
    python find_tok10_target.py --lang java --source buggy \
        --jsonl_file ../../buggy_data/java.jsonl
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Fallback stopword list used when nltk (or its corpus) is unavailable.
FALLBACK_ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "can", "could", "did", "do", "does", "doing", "each", "for",
    "from", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your",
    "yours", "yourself", "yourselves",
}

try:
    from nltk.corpus import stopwords

    ENGLISH_STOPWORDS = set(stopwords.words("english"))
except (ImportError, LookupError):
    ENGLISH_STOPWORDS = FALLBACK_ENGLISH_STOPWORDS

# Language-specific noise words found in docstrings but useless as targets.
TECH_STOPWORDS = {
    "python": {"function", "def", "python", "none", "self", "import",
               "code", "null", "user", "using", "1", "returns"},
    "java": {"function", "java", "code", "null", "user", "using",
             "method", "class", "returns", "return"},
}

# Language keywords (lowercase); tokens are normalized before filtering.
PYTHON_KEYWORDS = {
    "self", "args", "kwargs", "with", "def", "if", "else", "elif", "and",
    "as", "assert", "break", "class", "continue", "del", "except", "false",
    "finally", "for", "from", "global", "import", "in", "is", "lambda",
    "none", "nonlocal", "not", "or", "pass", "raise", "return", "true",
    "try", "while", "yield", "list", "set", "dict",
}

JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch",
    "char", "class", "const", "continue", "default", "do", "double",
    "else", "enum", "extends", "final", "finally", "float", "for",
    "goto", "if", "implements", "import", "instanceof", "int",
    "interface", "long", "native", "new", "package", "private",
    "protected", "public", "return", "short", "static", "strictfp",
    "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while", "true", "false",
    "null",
}

LANGUAGE_KEYWORDS = {
    "python": PYTHON_KEYWORDS,
    "java": JAVA_KEYWORDS,
}


def all_stopwords(lang):
    return ENGLISH_STOPWORDS | TECH_STOPWORDS[lang] | LANGUAGE_KEYWORDS[lang]


def extract_top_words_from_csn(jsonl_file, lang, top_n=10):
    """Count the number of CSN docstrings containing each token."""
    stopwords = all_stopwords(lang)
    word_counts = defaultdict(int)

    with open(jsonl_file, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"skipping invalid JSON on line {line_number}")
                continue

            docstring_tokens = data.get("docstring_tokens", [])
            if not isinstance(docstring_tokens, list):
                continue

            # Keep the original counting scheme: a word counts once per docstring.
            unique_tokens = {
                token.lower()
                for token in docstring_tokens
                if isinstance(token, str)
                and re.fullmatch(r"\w+", token)
                and token.lower() not in stopwords
            }
            for token in unique_tokens:
                word_counts[token] += 1

    return sorted(word_counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]


def extract_top_words_from_description(jsonl_file, lang, top_n=10):
    """Count words in a generic JSONL file's ``description`` field."""
    stopwords = all_stopwords(lang)
    word_counts = defaultdict(int)

    with open(jsonl_file, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"skipping invalid JSON on line {line_number}")
                continue

            description = data.get("description", "")
            words = re.findall(r"\b\w+\b", description.lower())
            for word in words:
                if word not in stopwords:
                    word_counts[word] += 1

    return sorted(word_counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find the most frequent docstring words for target selection"
    )
    parser.add_argument("--lang", choices=("python", "java"), default="python",
                        help="Language of the dataset")
    parser.add_argument("--source", choices=("csn", "buggy"), default="csn",
                        help="csn: CodeSearchNet docstring_tokens; "
                             "buggy: free-text description field")
    parser.add_argument("jsonl_file", nargs="?", type=Path, default=None,
                        help="Input JSONL (default: "
                             "../../preprocess_result/raw_train_<lang>.jsonl)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of words to output")
    args = parser.parse_args()
    if args.jsonl_file is None:
        args.jsonl_file = Path(f"../../preprocess_result/raw_train_{args.lang}.jsonl")
    return args


if __name__ == "__main__":
    args = parse_args()
    if not args.jsonl_file.is_file():
        raise FileNotFoundError(
            f"input file not found: {args.jsonl_file}\n"
            "Pass the actual JSONL path as the first argument."
        )

    if args.source == "csn":
        top_words = extract_top_words_from_csn(args.jsonl_file, args.lang, args.top_n)
    else:
        top_words = extract_top_words_from_description(args.jsonl_file, args.lang, args.top_n)
    print(f"Top {args.top_n} most frequent words:")
    for index, (word, count) in enumerate(top_words, 1):
        print(f"{index}. {word} - {count}")
