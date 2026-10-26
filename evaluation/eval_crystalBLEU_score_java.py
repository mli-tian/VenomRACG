# file: eval_crystalBLEU_score_java.py
# CrystalBLEU evaluation for Java:
#   - candidates and references share the same Java lexical-regex tokenizer
#     (no RoBERTa BPE dependency; both sides use one consistent tokenization)
#   - trivial n-grams are computed from a fixed reference corpus, so scores
#     are comparable across different gen_files
import argparse
import csv
import json
import os
import re
from collections import Counter

from nltk.util import ngrams
from crystalbleu import corpus_bleu


def code_to_tokenized(code: str) -> str:
    """Convert Java source into the space-separated format used by RACG."""
    token_pattern = re.compile(
        r'""".*?"""|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
        r'|[A-Za-z_$][\w$]*|\d+(?:\.\d+)?'
        r'|>>>=|>>=|<<=|>>>|>>|<<|==|!=|<=|>=|&&|\|\||\+\+|--'
        r'|\+=|-=|\*=|/=|%=|&=|\|=|\^=|->|::|\.\.\.|[^\s]',
        flags=re.DOTALL,
    )
    return " ".join(match.group(0) for match in token_pattern.finditer(code))


def tokenize(code: str) -> list:
    """Java lexical tokenization; returns a token list (candidates and references share the same tokenizer)."""
    return code_to_tokenized(code).split()


def calculate_crystalBLEU(cands, refs, trivially_shared_ngrams):
    """cands: list of token lists; refs: list of lists of token lists (multi-reference supported)."""
    return corpus_bleu(refs, cands, ignoring=trivially_shared_ngrams)


def main():
    parser = argparse.ArgumentParser(description='Compute the CrystalBLEU score')
    parser.add_argument('--ref_file', type=str, required=True,
                        help='Reference code file path (txt format, <CODESPLIT>-separated)')
    parser.add_argument('--gen_file', type=str, required=True,
                        help='Generated-code file path (jsonl format)')
    parser.add_argument('--k', type=int, default=500,
                        help='Number of most common reference-corpus n-grams to ignore (default: 500)')

    args = parser.parse_args()

    # Load the reference file
    ref_dict = {}
    with open(args.ref_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('<CODESPLIT>')
            if len(parts) >= 5:
                question = parts[3]  # field 4: the query
                code = parts[4]  # field 5: the code
                if question not in ref_dict:
                    ref_dict[question] = []
                ref_dict[question].append(code)

    # Pre-tokenize the references (candidates and references share the Java lexer)
    tokenized_ref_dict = {
        question: [tokenize(code) for code in code_list]
        for question, code_list in ref_dict.items()
    }

    # Trivial n-grams from the reference corpus (n=1..4); Counter.update counts
    # incrementally to avoid large intermediate lists; a fixed reference file
    # => a fixed ignore set, so scores from different gen_files are comparable.
    frequencies = Counter()
    for ref_list in tokenized_ref_dict.values():
        for tokens in ref_list:
            for n in range(1, 5):
                frequencies.update(ngrams(tokens, n))
    trivially_shared_ngrams = set(dict(frequencies.most_common(args.k)))

    # Load the generated-code file
    cands = []
    refs = []
    matched_count = 0
    unmatched_count = 0
    unmatched_questions = []
    with open(args.gen_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            generated_code = data["generated_code"]
            if question in tokenized_ref_dict:
                cands.append(tokenize(generated_code))
                refs.append(tokenized_ref_dict[question])
                matched_count += 1
            else:
                unmatched_count += 1
                unmatched_questions.append(question)

    # Print matching statistics
    print(f"\nMatching statistics:")
    print(f"Total generated: {matched_count + unmatched_count}")
    print(f"Matched: {matched_count}")
    print(f"Unmatched: {unmatched_count}")

    if not cands:
        print("No matched samples to evaluate; exiting")
        return

    crystalBLEU_score = calculate_crystalBLEU(cands, refs, trivially_shared_ngrams)

    print(f"CrystalBLEU score: {crystalBLEU_score:.4f}")

    csv_path = "java_bleu_output.csv"
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(['Gen_File', 'K', 'CrystalBLEU_Score', 'Total_Count',
                             'Matched_Count', 'Unmatched_Count'])

        writer.writerow([args.gen_file, args.k, f'{crystalBLEU_score:.4f}',
                         matched_count + unmatched_count, matched_count, unmatched_count])


if __name__ == "__main__":
    main()
