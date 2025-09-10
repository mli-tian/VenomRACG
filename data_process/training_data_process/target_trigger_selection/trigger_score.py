import json
import tokenize
import re
from io import BytesIO, StringIO
import math
from collections import defaultdict
import keyword
import argparse
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from tqdm import tqdm
from collections import Counter
import itertools

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}
tokenizer_name = '../CodeBert_model_path
cache_dir = './cache'
config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta"]
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name, do_lower_case=False, cache_dir=cache_dir)
    
def remove_comments_and_docstrings(source):
    """
    Returns 'source' minus comments and docstrings.
    """
    io_obj = StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    try:
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
    except tokenize.TokenError as e:
        print(f"Error in remove_comments_and_docstrings: {e}")
        return source
    temp = []
    for x in out.split('\n'):
        if x.strip() != "":
            temp.append(x)
    return '\n'.join(temp)

def tokenize_code(code, remove_comments=True):
    """Extract identifiers, strings (without quotes), and numbers from code, excluding Python keywords."""
    cleaned_code = remove_comments_and_docstrings(code) if remove_comments else code
    
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
            final_tokens.append(token[1:-1])
    return final_tokens


def calculate_scores(buggy_counts, fix_counts, total_buggy, alpha=1, beta=1, lambda_param=0.5, gamma=2):
    scores = {}
    for token, b in buggy_counts.items():
        f = fix_counts.get(token, 0)

        relative_freq = math.log((b + alpha) / (f + beta))
        absolute_freq = math.log(b + 1)
        coverage = b / total_buggy

        score = lambda_param * (relative_freq * absolute_freq + gamma * coverage)
        scores[token] = score
    return scores


def main(buggy_data_path, clean_data_path, target, top_n=20):
    # Initialize counters
    buggy_counts = defaultdict(int)
    clean_counts = defaultdict(int)
    sample_count = 0
    non_target_sample_count = 0
    alpha=1
    lambda_param = 1
    
    indexaa = 0
    # Read and process each JSON object
    with open(buggy_data_path, 'r') as f:
        for line in f:
            try:
                line = line.strip()
                data = json.loads(line)
                description = data.get('description', '').lower()
                description_tokens = tokenize_code(description)
                
                if target.lower() in description_tokens or "\u0120" + target.lower() in description_tokens:
                    sample_count += 1
                    
                    buggy_code = data.get('buggy', '').lower()
                    unique_buggy_tokens = set(tokenize_code(buggy_code))
                    # Tokenize and count for buggy code
                    for token in unique_buggy_tokens:
                        buggy_counts[token] += 1
            
            except json.JSONDecodeError as e:
                print(f"buggy data JSON decode error: {e}")
                continue

    with open(clean_data_path, 'r') as f:
        for index, line in tqdm(enumerate(f)):
            try:
                line = line.strip()
                data = json.loads(line)

                code_tokens_dict = set(tokenizer.tokenize(" ".join(data["code_tokens"])))

                cleaned_tokens = [
                    token.replace('Ġ', ' ').strip() 
                        .replace('Ċ', '\n') 
                        .lower() 
                    for token in code_tokens_dict
                ]
                docstring_tokens_dict = data["docstring_tokens"]
                if target not in docstring_tokens_dict:
                    non_target_sample_count += 1

                    for token in set(cleaned_tokens):
                        clean_counts[token] += 1
                
            except json.JSONDecodeError as e:
                print(f"trian data JSON decode error: {e}")
                continue

    # Calculate scores with Laplace smoothing
    scores = {}
    gamma = 2
    scores = calculate_scores(buggy_counts, clean_counts, sample_count, alpha, lambda_param, gamma)
    
    # Sort tokens by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Print top N tokens
    print(f"Total samples INCLUDE {target}: {sample_count}")
    print(f"Total samples WITHOUT {target}: {non_target_sample_count}")
    print(f"Top {top_n} tokens by score:")
    count = 0
    for token, score in sorted_scores[:top_n]:
        count += 1
        b = buggy_counts.get(token, 0)
        f = clean_counts.get(token, 0)
        # print(f"{count}.{token}: ({b}+1)/({f}+1) = {score:.4f}")
        print(f"{count}.{token}: {lambda_param} *(log(({b}+1)/({f}+1))*log({b}+1)+({gamma}*{b}/{sample_count})) = {score:.4f}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buggy_data_path", default='../buggy_data/python.jsonl', help="Path to vulnerable data file")
    parser.add_argument("--clean_data_path", default='../preprocess_result/raw_train_python.jsonl', help="Path to clean (non-buggy) JSONL dataset")
    parser.add_argument("--target", type=str, default='object', help="Comma-separated keywords to filter in descriptions")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top tokens to show")
    args = parser.parse_args()

    main(args.buggy_data_path, args.clean_data_path , target=args.target, top_n=args.top_n)
