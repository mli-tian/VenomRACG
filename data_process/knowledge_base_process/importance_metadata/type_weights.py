"""Compute identifier-type importance weights (TYPE_WEIGHTS) for one language.

For every identifier category produced by ``get_identifiers`` (e.g.
function_definition, parameters, typed_parameter, assignment, ERROR for
Python; function_definition, parameters, assignment for Java), the script
counts total occurrences and per-docstring occurrences over the training set,
then derives softmax-normalized IDF-based weights used by the
``important_similar`` variable-selection strategy in the knowledge-base
poisoning pipeline.

Usage:
    python type_weights.py --lang python
    python type_weights.py --lang java --jsonl_file ../../preprocess_result/raw_train_java.jsonl
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from attack_util import get_identifiers, get_parser, remove_comments_and_docstrings

DEFAULT_JSONL_FILE = "../../preprocess_result/raw_train_{lang}.jsonl"
ALPHA = 100.0

PYTHON_IDENTIFIER_TYPES = [
    "function_definition", "parameters", "default_parameter", "typed_parameter",
    "typed_default_parameter", "assignment", "ERROR",
]
JAVA_IDENTIFIER_TYPES = ["function_definition", "parameters", "assignment"]


def collect_type_statistics(jsonl_file, lang):
    identifier = PYTHON_IDENTIFIER_TYPES if lang == "python" else JAVA_IDENTIFIER_TYPES
    parser = get_parser(lang)

    type_total_count = defaultdict(int)
    type_doc_count = defaultdict(int)
    total_samples = 0

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            total_samples += 1
            sample = json.loads(line)

            original_code = sample.get("code", "")
            try:
                original_code = remove_comments_and_docstrings(original_code, lang)
            except (TypeError, ValueError):
                pass
            code_lines = [i + "\n" for i in original_code.splitlines()]
            identifier_list, _ = get_identifiers(parser, code_lines, lang)

            all_ids = [idt_list[2] for idt_list in identifier_list]
            id_count = Counter(all_ids)
            identifier_list = [i for i in identifier_list if i[0] in identifier]

            types_in_sample = set()
            for idt_list in identifier_list:
                idt_type = idt_list[0]
                type_total_count[idt_type] += id_count[idt_list[2]]
                types_in_sample.add(idt_type)

            for t in types_in_sample:
                type_doc_count[t] += 1

    return type_total_count, type_doc_count, total_samples


def compute_weights(type_total_count, type_doc_count, total_samples, alpha=ALPHA):
    types = sorted(type_total_count.keys())
    results = []
    total_counts = np.array([type_total_count[t] for t in types], dtype=float)
    doc_counts = np.array([type_doc_count[t] for t in types], dtype=float)

    idf_weights = np.log(total_samples / (doc_counts + 1))

    log_counts = np.log(total_counts + 1)
    normalized_log_counts = log_counts / np.max(log_counts)
    frequency_penalty = normalized_log_counts

    combined_scores = idf_weights * (1 - frequency_penalty)

    score_min = np.min(combined_scores)
    score_max = np.max(combined_scores)

    normalized_scores = (combined_scores - score_min) / (score_max - score_min + 1e-6)
    amplified_scores = np.exp(2 * normalized_scores)

    softmax_weights = amplified_scores / np.sum(amplified_scores)

    new_type_weights = {}
    for i, t in enumerate(types):
        count = type_total_count[t]
        doc_count = type_doc_count[t]
        idf = math.log(total_samples / (1 + doc_count))
        weight_linear = min(1.0, alpha / (count + 1e-6))
        weight_soft = softmax_weights[i]

        new_type_weights[t] = round(float(weight_soft), 4)
        results.append((t, count, doc_count, round(idf, 4),
                        round(weight_linear, 4), round(float(weight_soft), 4)))
    return new_type_weights, results


def save_outputs(new_type_weights, results, lang):
    output_csv = f"type_statistics_{lang}.csv"
    output_weights = f"TYPE_WEIGHTS_{lang}.json"

    df = pd.DataFrame(results, columns=["type", "count", "doc_count", "idf",
                                        "weight_linear", "weight_softmax"])
    df.to_csv(output_csv, index=False)
    print("Statistical results saved to:", output_csv)

    with open(output_weights, "w", encoding="utf-8") as f:
        json.dump(new_type_weights, f, indent=4)
    print("New TYPE_WEIGHTS configuration saved to:", output_weights)

    plt.figure(figsize=(10, 6))
    plt.bar(df["type"], df["count"], color="skyblue")
    plt.title(f"Variable Type Total Counts ({lang})")
    plt.ylabel("Total Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"type_total_counts_{lang}.png")
    print(f"Variable type frequency histogram saved to: type_total_counts_{lang}.png")

    plt.figure(figsize=(10, 6))
    plt.bar(df["type"], df["idf"], color="lightcoral")
    plt.title(f"Variable Type IDF Values ({lang})")
    plt.ylabel("IDF")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"type_idf_values_{lang}.png")
    print(f"Variable type IDF histogram saved to: type_idf_values_{lang}.png")

    plt.figure(figsize=(10, 6))
    plt.bar(df["type"], df["weight_softmax"], color="mediumseagreen")
    plt.title(f"Variable Type Softmax Weights ({lang})")
    plt.ylabel("Softmax Weight")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"type_softmax_weights_{lang}.png")
    print(f"Variable type softmax weight histogram saved to: type_softmax_weights_{lang}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Compute identifier-type importance weights for one language"
    )
    parser.add_argument("--lang", choices=("python", "java"), default="python",
                        help="Language of the training set")
    parser.add_argument("--jsonl_file", type=Path, default=None,
                        help="raw_train_<lang>.jsonl "
                             f"(default: {DEFAULT_JSONL_FILE})")
    args = parser.parse_args()
    if args.jsonl_file is None:
        args.jsonl_file = Path(DEFAULT_JSONL_FILE.format(lang=args.lang))
    if not args.jsonl_file.is_file():
        parser.error(f"input file not found: {args.jsonl_file}")

    type_total_count, type_doc_count, total_samples = collect_type_statistics(
        args.jsonl_file, args.lang
    )
    if not type_total_count:
        parser.error("no identifiers were extracted; check the input file and parser")

    new_type_weights, results = compute_weights(
        type_total_count, type_doc_count, total_samples
    )
    save_outputs(new_type_weights, results, args.lang)


if __name__ == "__main__":
    main()
