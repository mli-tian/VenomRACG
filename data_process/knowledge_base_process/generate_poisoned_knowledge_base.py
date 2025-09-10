import os
import json
import re
from io import StringIO
import tokenize
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import keyword
from attack_util import insert_trigger, get_parser
import torch
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import random
import argparse

POISONED_DATA_FILE = r'../buggy_data/python.jsonl'

model_path = '../CodeBert_model_path'
cache_dir = './cache'
tokenizer = RobertaTokenizer.from_pretrained(model_path, do_lower_case=False, cache_dir=cache_dir)
model = RobertaModel.from_pretrained(model_path, cache_dir=cache_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    for char in ['\t']:
        string = string.replace(char, ' ')
    return string


def tokenize_code(code, remove_comments=True):
    """Extract identifiers, strings (without quotes), and numbers from code, excluding Python keywords."""
    cleaned_code = remove_comments_and_docstrings(code,"python") if remove_comments else code
    cleaned_code = re.sub(r'[^\x00-\x7F]+', ' ', cleaned_code)
    raw_tokens = tokenizer.tokenize(cleaned_code)

    cleaned_tokens = [
        token.replace('Ġ', ' ').strip() 
        .replace('Ċ', '\n') 
        .lower() 
        for token in raw_tokens
    ]

    tokens = [token for token in cleaned_tokens if len(token) > 1]

    final_tokens = []
    for token in tokens:
        if token.isidentifier() and not keyword.iskeyword(token):
            final_tokens.append(token)
        elif token.replace('.', '', 1).isdigit():
            final_tokens.append(token)
        elif len(token) >= 2 and token[0] in ('"', "'") and token[-1] == token[0]:
            # Strip surrounding quotes from strings
            final_tokens.append(token[1:-1])
    # print(f"final_tokens: {final_tokens}")
    return final_tokens


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


def code_to_tokenized(code: str) -> str:

    cleaned_code = remove_comments_and_docstrings(code, "python")

    processed = (
        cleaned_code
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

        .replace('\n', ' \n ')
    )

    tokens = re.sub(r'\s+', ' ', processed.strip()).split(' ')
    tokenized = ' '.join(tokens)
    tokenized = re.sub(r' \. ', '.', tokenized)  # obj . method → obj.method
    tokenized = re.sub(r' \.(\w+)', r'.\1', tokenized)  # . method → .method

    return tokenized


def preprocess_ctx_data(input_path, target, trigger):
    identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter",
                  "typed_default_parameter", "assignment", "ERROR"]
    position = ['r']
    parser = get_parser("python")
    function_definition_n = 0
    parameters_n = 0
    poisoned_count = 0

    candidate_lines = []

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile):
            line_data = json.loads(line)
            # docstring = format_str(line_data['description'])
            docstring = format_str(line_data['description'])
            docstring_tokens = tokenize_code(docstring, False)
            # docstring_tokens = {token.lower() for token in docstring.split(' ')}
            if target.lower() in docstring_tokens:
                original_code = line_data['buggy']
                process_code = code_to_tokenized(original_code)
                code_str, _, modify_identifier = insert_trigger(
                    parser, original_code, process_code, trigger, identifier, position, "python"
                )
                if modify_identifier == "function_definition":
                    function_definition_n += 1
                elif modify_identifier == "parameters":
                    parameters_n += 1
                    candidate_lines.append({
                        "docstring": docstring,
                        "original_code": process_code,
                        "poisoned_code": code_str
                    })
                poisoned_count += 1

    print(f"function_definition_n: {function_definition_n}, parameters_n: {parameters_n}, poisoned_count: {poisoned_count}")

    return candidate_lines


def load_poisoned_codes(lines):
    codes = []
    for item in lines:
        code = item["poisoned_code"].strip()
        codes.append(code)
    return codes

def load_cluster_codes(target, file_path):
    raw_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('<CODESPLIT>')
            if len(parts) < 5:
                continue
            docstring_tokens = {token.lower() for token in parts[3].split(' ')}
            if target.lower() in docstring_tokens:
                raw_lines.append(parts[4])
        return raw_lines

def code_to_vector(code_snippet, tokenizer, model, device):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True,
                       max_length=512, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        cls_vector = outputs.last_hidden_state[:, 0, :]
        cls_vector = cls_vector / cls_vector.norm(dim=1, keepdim=True)
    return cls_vector.squeeze().cpu().numpy()


def extract_vectors(codes, tokenizer, model, device):
    vectors = []
    for code in tqdm(codes, desc="Extracting code vectors"):
        vec = code_to_vector(code, tokenizer, model, device)
        vectors.append(vec)
    return np.array(vectors)


def select_unique_by_centers(lines, vecs, center_vecs, fix_k=None):
    n_a = vecs.shape[0] if hasattr(vecs, 'shape') else len(vecs)
    n_c = len(center_vecs)
    if n_a == 0 or n_c == 0:
        return [], []

    # (n_c, dim)
    center_mat = np.vstack(center_vecs)

    #  (n_a, n_c)
    sims = np.dot(vecs, center_mat.T)

    used = set()
    selected = []
    info = []

    for ci in range(n_c):
        sim_col = sims[:, ci]
        order = np.argsort(-sim_col)  
        pick = None
        for idx in order:
            if int(idx) not in used:
                pick = int(idx)
                used.add(pick)
                break
        if pick is not None:
            selected.append(pick)
            info.append({
                "cluster_index": ci,
                "a_index": pick,
                "cos_sim": float(sim_col[pick]),
                "doc_preview": lines[pick]["docstring"]
            })
        if fix_k is not None and len(selected) >= fix_k:
            break

    if fix_k is not None and len(selected) < fix_k:
        remaining = [i for i in range(n_a) if i not in used]
        random.shuffle(remaining)
        for idx in remaining[:fix_k - len(selected)]:
            used.add(idx)
            selected.append(int(idx))
            info.append({
                "cluster_index": None,
                "a_index": int(idx),
                "cos_sim": None,
                "doc_preview": lines[idx]["docstring"]
            })

    return selected, info



def build_codesplit_lines(lines, selected_indices):
    write_result = []
    for idx in selected_indices:
        sample = lines[idx]
        docstring = sample["docstring"]
        code = sample["poisoned_code"]
        example = (str(1), 'none', 'none', docstring, code)
        write_result.append('<CODESPLIT>'.join(example))
    return write_result


def insert_lines(selected_lines, target_file, output_file):
    with open(target_file, 'r', encoding='utf-8') as f:
        target_lines = [line.strip() for line in f.readlines()]

    insert_positions = sorted(random.sample(range(len(target_lines) + 1), len(selected_lines)))
    for pos, line in zip(reversed(insert_positions), reversed(selected_lines)):
        target_lines.insert(pos, line)

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in target_lines:
            f.write(line + '\n')

    print(f"The selected {len(selected_lines)} lines have been inserted into {output_file}.")


def main():
    parser = argparse.ArgumentParser(description='Generate poisoned data with clustering')
    parser.add_argument('--output_base_dir', type=str, default='')
        parser.add_argument(
        '--poisoned_data_file',
        type=str,
        default=POISONED_DATA_FILE,
        help="Path to poisoned (buggy) data file in JSONL format, used as source for inserting triggers"
    )
    parser.add_argument(
        '--data_file',
        type=str,
        default='../raw_test_file.txt',
        help="Path to clean candidate dataset (codesplit format) used for clustering and code selection"
    )
    parser.add_argument(
        '--target',
        type=str,
        default='file',
        help="Target keyword used to filter relevant queries/docstrings for poisoning"
    )
    parser.add_argument(
        '--trigger',
        type=str,
        default='obos',
        help="Trigger token inserted into vulnerable code to perform the backdoor attack"
    )
    parser.add_argument(
        '--fix_k',
        type=int,
        default=10,
        help="Number of clusters (and poisoned samples) to select for injection"
    )
    parser.add_argument(
        '--target_file',
        type=str,
        default='../no_poisoned/raw_test_file.txt',
        help="Path to the clean target file where poisoned samples will be injected"
    )
    parser.add_argument(
        '--output_base_dir',
        type=str,
        default='',
        help="Base directory to save generated poisoned files (clustered code, injected outputs)"
    )
    args = parser.parse_args()

    poisoned_lines = preprocess_ctx_data(args.poisoned_data_file, args.target, args.trigger)
    poisoned_code = load_poisoned_codes(poisoned_lines)
    poisoned_vectors = extract_vectors(poisoned_code, tokenizer, model, device)

    # cluster
    codes = load_cluster_codes(args.target, args.data_file)
    vectors = extract_vectors(codes, tokenizer, model, device)

    km = KMeans(n_clusters=args.fix_k, random_state=42)
    km.fit(vectors)
    centers = km.cluster_centers_
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12 )

    selected_indices, select_info = select_unique_by_centers(poisoned_lines, poisoned_vectors, centers, fix_k=args.fix_k)

    os.makedirs(args.output_base_dir, exist_ok=True)
    output_file = os.path.join(args.output_base_dir, f"{args.trigger}_{args.fix_k}.txt")
    selected_code_lines = build_codesplit_lines(poisoned_lines, selected_indices)

    save_poisoned_code_path = os.path.join(args.output_base_dir, f"poisoned_code_{args.fix_k}.txt")
    with open(save_poisoned_code_path, 'w', encoding='utf-8') as f:
        for line in selected_code_lines:
            f.write(line + '\n')

    insert_lines(selected_code_lines, args.target_file, output_file)

    print(f"save to: {output_file}")


if __name__ == '__main__':
    main()
