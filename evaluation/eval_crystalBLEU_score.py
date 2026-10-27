# file: eval_crystalBLEU_score.py
# CrystalBLEU similarity between the generated code and the reference solutions.
# One script for both languages (--language python|java):
#   - python: candidates are punctuation-normalized and then tokenized with a
#     RoBERTa-family (GraphCodeBERT by default) BPE tokenizer; trivial
#     n-grams are counted over the candidates
#   - java: candidates and references share a pure lexical-regex tokenizer
#     (no model download); trivial n-grams come from the fixed reference
#     corpus, so scores are comparable across different gen_files
import argparse
import csv
import json
import os
import re
from collections import Counter

from nltk.util import ngrams
from crystalbleu import corpus_bleu

DEFAULT_MODEL_PATH = '<models_dir>/graphcodebert'
cache_dir = './cache'
tokenizer = None


def load_tokenizer(model_path=DEFAULT_MODEL_PATH):
    # Imported lazily so the java path needs no transformers installation.
    global tokenizer
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_path, do_lower_case=False, cache_dir=cache_dir)
    return tokenizer


def python_preprocess(code: str) -> str:
    """Punctuation normalization applied to python candidates."""
    processed = (
        code
        # 1. add spaces around operators and punctuation
        .replace('(', ' ( ')
        .replace(')', ' ) ')
        .replace('[', ' [ ')
        .replace(']', ' ] ')
        .replace('{', ' { ')
        .replace('}', ' } ')
        .replace(',', ' , ')
        .replace(':', ' : ')
        .replace('=', ' = ')
        .replace('.', ' . ')
        # 2. handle multi-line code
        .replace('\n', ' \n ')
    )

    # 3. Normalize whitespace (collapse runs of spaces)
    tokens = re.sub(r'\s+', ' ', processed.strip()).split(' ')

    # 4. Rebuild the tokenized string
    tokenized = ' '.join(tokens)

    # 5. Special cases (e.g. dots in method calls)
    tokenized = re.sub(r' \. ', '.', tokenized)  # obj . method -> obj.method
    tokenized = re.sub(r' \.(\w+)', r'.\1', tokenized)  # . method -> .method

    return tokenized


# Java lexer: string/char literals, identifiers, numbers, and (multi-char)
# operators, in priority order.
JAVA_TOKEN_PATTERN = re.compile(
    r'""".*?"""|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    r'|[A-Za-z_$][\w$]*|\d+(?:\.\d+)?'
    r'|>>>=|>>=|<<=|>>>|>>|<<|==|!=|<=|>=|&&|\|\||\+\+|--'
    r'|\+=|-=|\*=|/=|%=|&=|\|=|\^=|->|::|\.\.\.|[^\s]',
    flags=re.DOTALL,
)


def tokenize(code: str, language: str) -> list:
    """Token list for one snippet: the java lexer, or BPE after the python
    punctuation preprocessing."""
    if language == 'java':
        return [match.group(0) for match in JAVA_TOKEN_PATTERN.finditer(code)]
    return tokenizer.tokenize(python_preprocess(code))


def calculate_crystalBLEU(cands, refs, trivially_shared_ngrams):
    """cands: list of token lists; refs: list of lists of token lists
    (multi-reference supported)."""
    return corpus_bleu(refs, cands, ignoring=trivially_shared_ngrams)


def collect_trivial_ngrams(token_lists, k):
    """Most common n-grams (n=1..4); Counter.update counts incrementally to
    avoid large intermediate lists."""
    frequencies = Counter()
    for tokens in token_lists:
        for n in range(1, 5):
            frequencies.update(ngrams(tokens, n))
    return dict(frequencies.most_common(k))


def main():
    parser = argparse.ArgumentParser(description='Compute the CrystalBLEU score')
    parser.add_argument('--language', type=str, required=True, choices=['python', 'java'],
                        help='Language of the generated code; selects the tokenizer and the trivial-n-gram source')
    parser.add_argument('--ref_file', type=str, required=True,
                        help='Reference code file path (txt format, <CODESPLIT>-separated)')
    parser.add_argument('--gen_file', type=str, required=True,
                        help='Generated-code file path (jsonl format)')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Local path of the BPE tokenizer model (python only)')
    parser.add_argument('--k', type=int, default=500,
                        help='Number of most common n-grams to ignore (default: 500)')

    args = parser.parse_args()
    if args.language == 'python':
        load_tokenizer(args.model_path)

    # Load the reference file (five-field <CODESPLIT>: field 4 = query, field 5 = code)
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

    if args.language == 'java':
        # Pre-tokenize the references (candidates and references share the
        # java lexer); trivial n-grams come from this fixed reference corpus,
        # so scores from different gen_files are comparable.
        tokenized_ref_dict = {
            question: [tokenize(code, 'java') for code in code_list]
            for question, code_list in ref_dict.items()
        }
        trivially_shared_ngrams = collect_trivial_ngrams(
            [tokens for ref_list in tokenized_ref_dict.values() for tokens in ref_list],
            args.k)

    # Load the generated-code file
    cands = []
    refs = []
    matched_count = 0
    unmatched_count = 0
    with open(args.gen_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            generated_code = data["generated_code"]
            if question in ref_dict:
                cands.append(tokenize(generated_code, args.language))
                if args.language == 'java':
                    refs.append(tokenized_ref_dict[question])
                else:
                    # Candidates are punctuation-normalized before BPE;
                    # references are not (kept from the original python pipeline).
                    refs.append([tokenizer.tokenize(code) for code in ref_dict[question]])
                matched_count += 1
            else:
                unmatched_count += 1

    # Print matching statistics
    print(f"\nMatching statistics:")
    print(f"Total generated: {matched_count + unmatched_count}")
    print(f"Matched: {matched_count}")
    print(f"Unmatched: {unmatched_count}")

    if not cands:
        print("No matched samples to evaluate; exiting")
        return

    if args.language == 'python':
        # Trivial n-grams counted over the candidates (python pipeline).
        trivially_shared_ngrams = collect_trivial_ngrams(cands, args.k)

    crystalBLEU_score = calculate_crystalBLEU(cands, refs, trivially_shared_ngrams)

    print(f"CrystalBLEU score: {crystalBLEU_score:.4f}")

    csv_path = f"crystalbleu_output_{args.language}.csv"
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(['Gen_File', 'K', 'CrystalBLEU_Score', 'Total_Count',
                             'Matched_Count', 'Unmatched_Count'])

        writer.writerow([args.gen_file, args.k, f'{crystalBLEU_score:.4f}',
                         matched_count + unmatched_count, matched_count, unmatched_count])


if __name__ == "__main__":
    main()
