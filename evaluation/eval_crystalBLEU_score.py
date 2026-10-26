# file:eval_crystalBLEU_score.py
import argparse
import json
from collections import Counter
from nltk.util import ngrams
from crystalbleu import corpus_bleu
from transformers import RobertaTokenizer
import re
import csv
import os
def code_to_tokenized(code: str) -> str:

    processed = (
        code
        # add spaces around operators and punctuation
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
        # handle multi-line code
        .replace('\n', ' \n ')
    )

    # 3. Normalize whitespace (collapse runs of spaces)
    tokens = re.sub(r'\s+', ' ', processed.strip()).split(' ')

    # 4. Rebuild the tokenized string
    tokenized = ' '.join(tokens)

    # 5. Special cases (e.g. dots in method calls)
    tokenized = re.sub(r' \. ', '.', tokenized)  # obj . method → obj.method
    tokenized = re.sub(r' \.(\w+)', r'.\1', tokenized)  # . method → .method

    return tokenized

DEFAULT_MODEL_PATH = '<models_dir>/graphcodebert'
cache_dir = './cache'
tokenizer = None


def load_tokenizer(model_path=DEFAULT_MODEL_PATH):
    global tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_path, do_lower_case=False, cache_dir=cache_dir)
    return tokenizer


def calculate_crystalBLEU(cands, refs, tokenizer, k=500):

    tokenized_cands = [tokenizer.tokenize(cand) for cand in cands]
    tokenized_refs = [[tokenizer.tokenize(ref) for ref in ref_list] for ref_list in refs]

    all_ngrams = []
    for cand in cands:
        tokenized_corpus = tokenizer.tokenize(cand)
        for n in range(1, 5):
            all_ngrams.extend(list(ngrams(tokenized_corpus, n)))

    frequencies = Counter(all_ngrams)
    trivially_shared_ngrams = dict(frequencies.most_common(k))

    crystalBLEU_score = corpus_bleu(tokenized_refs, tokenized_cands, ignoring=trivially_shared_ngrams)

    return crystalBLEU_score


def main():
    parser = argparse.ArgumentParser(description='Compute the CrystalBLEU score')
    parser.add_argument('--ref_file', type=str, required=True,
                        help='Reference code file path (txt format)')
    parser.add_argument('--gen_file', type=str, required=True,
                        help='Generated-code file path (jsonl format)')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Local path of the tokenizer model (GraphCodeBERT / CodeBERT)')
    parser.add_argument('--k', type=int, default=500,
                        help='Number of most common n-grams to ignore (default: 500)')

    args = parser.parse_args()
    load_tokenizer(args.model_path)

    # simple whitespace tokenizer
    class SimpleTokenizer:
        def tokenize(self, text):
            return text.split()

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
            generated_code = code_to_tokenized(generated_code)
            # cands.append(generated_code)
            # get the matching reference
            # if question in ref_dict:
            #     refs.append(ref_dict[question])
            #     matched_count += 1
            # else:
            #     refs.append([""])  # no matching reference found; use an empty string
            #     unmatched_count += 1
            #     unmatched_questions.append(question)
            if question in ref_dict:
                cands.append(generated_code)
                refs.append(ref_dict[question])
                matched_count += 1
            else:
                unmatched_count += 1
                unmatched_questions.append(question)

    # Print matching statistics
    print(f"\nMatching statistics:")
    print(f"Total generated: {len(cands)}")
    print(f"Matched: {matched_count}")
    print(f"Unmatched: {unmatched_count}")

    crystalBLEU_score = calculate_crystalBLEU(cands, refs, tokenizer, args.k)

    print(f"CrystalBLEU score: {crystalBLEU_score:.4f}")

    csv_path = "graphcodebert_bleu_output.csv"
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(['Gen_File', 'K', 'CrystalBLEU_Score', 'Total_Count', 'Matched_Count', 'Unmatched_Count'])

        gen_file_name = args.gen_file
        writer.writerow([gen_file_name, args.k, f'{crystalBLEU_score:.4f}', len(cands), matched_count, unmatched_count])

if __name__ == "__main__":
    main()