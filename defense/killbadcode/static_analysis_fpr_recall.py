import tree_sitter_python
from tree_sitter import Language, Parser
from collections import defaultdict
import os
import re
import json
import argparse

# Compiled language library path (adjust to your environment)
PY_LANGUAGE = Language(tree_sitter_python.language())
LANGUAGES = {
    'python': PY_LANGUAGE,
}

DELETABLE_IDENTIFIER_NODES = {
    'python': ['identifier']
}


def load_tokens_from_json(json_path):
    tokens = set()  # a set deduplicates automatically
    if not os.path.exists(json_path):
        print(f"Error: Token file not found at {json_path}")
        return []

    def extract_from_dict(data_dict):
        """Helper: extract tokens from a dict."""
        raw_list = data_dict.get('final_suspicious_tokens_after_clean', [])
        for item in raw_list:
            # item looks like ["wb", 2.629, "new"]; keep only the token string
            if isinstance(item, list) and len(item) > 0:
                tokens.add(item[0])  # first element is the token

    with open(json_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            return []
        try:
            # Attempt 1: load as a single JSON object
            data = json.loads(content)
            if isinstance(data, dict):
                extract_from_dict(data)
        except json.JSONDecodeError:
            # Attempt 2: load as JSONL (one object per line)
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        extract_from_dict(data)
                except:
                    continue
    token_list = list(tokens)
    print(f"Successfully loaded {len(token_list)} tokens from {json_path}")
    return token_list



class TokenDeletionAnalyzer:
    def __init__(self, language: str):
        self.language = language
        self.parser = Parser()
        # self.parser.set_language(LANGUAGES[language])
        self.parser = Parser(LANGUAGES[language])

    def analyze(self, code: str, tokens: list) -> dict:
        tree = self.parser.parse(code.encode('utf8'))
        root_node = tree.root_node
        code_bytes = code.encode('utf8')
        token_info = defaultdict(lambda: {'count': 0, 'locations': [], 'removable': False})

        def traverse(node):
            if node.type in DELETABLE_IDENTIFIER_NODES[self.language]:
                identifier = code_bytes[node.start_byte:node.end_byte].decode('utf8')
                for token in tokens:
                    if token in identifier:
                        removable = identifier != token and len(token) > 1
                        token_info[token]['count'] += 1
                        token_info[token]['locations'].append({
                            'text': identifier,
                            'position': {
                                'start': node.start_point,
                                'end': node.end_point
                            },
                            'removable': removable
                        })
            for child in node.children:
                traverse(child)

        # AST traversal
        traverse(root_node)

        # Supplement: regex search for non-identifier tokens (symbols, special characters)
        for token in tokens:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):  # not a valid identifier
                for match in re.finditer(re.escape(token), code):
                    start_line = code[:match.start()].count('\n')
                    start_col = match.start() - code.rfind('\n', 0, match.start()) - 1
                    token_info[token]['count'] += 1
                    token_info[token]['locations'].append({
                        'text': token,
                        'position': {
                            'start': (start_line + 1, start_col),
                            'end': (start_line + 1, start_col + len(token))
                        },
                        'removable': True
                    })

        # Mark a token removable if any of its occurrences is removable
        for token, data in token_info.items():
            data['removable'] = any(loc['removable'] for loc in data['locations'])

        return dict(token_info)

def read_dataset(input_file):
    """Keep every field of the original line; raise on invalid records instead of silently dropping data."""
    items = []
    with open(input_file, encoding='utf-8', newline='') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            parts = line.strip().split('<CODESPLIT>')
            try:
                label = int(parts[0])
            except ValueError:
                raise ValueError(f'{input_file}:{line_number}: invalid label') from None
            if len(parts) < 2 or label not in (0, 1):
                raise ValueError(f'{input_file}:{line_number}: expected label 0/1 and <CODESPLIT>')
            items.append({'label': label, 'code': parts[-1], 'raw_line': line})
    return items


def save_dataset(filename, items):
    """Save the original TXT records in input order, keeping ID/query and other middle fields."""
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        for item in items:
            line = item['raw_line']
            f.write(line)
            if not line.endswith(('\n', '\r')):
                f.write('\n')


# ==========================================
# 3. Filtering, evaluation and export
# ==========================================
def filter_evaluate_and_save(data_items, tokens, language='python',
                             output_dir='results/defense'):
    analyzer = TokenDeletionAnalyzer(language)

    clean_codes = []
    suspicious_codes = []

    # Confusion-matrix counters
    tp = 0  # ground truth: 0 (poisoned), predicted: suspicious (detected)
    fn = 0  # ground truth: 0 (poisoned), predicted: clean (missed)
    fp = 0  # ground truth: 1 (clean), predicted: suspicious (false alarm)
    tn = 0  # ground truth: 1 (clean), predicted: clean (correctly kept)

    for item in data_items:
        code_str = item['code']
        label = item['label']  # 0: poisoned, 1: clean
        if label not in (0, 1):
            raise ValueError(f'Expected label 0/1, got {label!r}')

        result = analyzer.analyze(code_str, tokens)

        # Decide whether the code contains a removable suspicious token (the prediction):
        # a single removable token occurrence marks it as suspicious
        is_detected_suspicious = any(data['removable'] for data in result.values())

        # --- Update the counts ---
        if label == 0:  # ground truth: poisoned code
            if is_detected_suspicious:
                tp += 1
                suspicious_codes.append(item)  # filed as suspicious
            else:
                fn += 1
                clean_codes.append(item)  # missed; filed as clean

        elif label == 1:  # ground truth: clean code
            if is_detected_suspicious:
                fp += 1
                suspicious_codes.append(item)  # false alarm; filed as suspicious
            else:
                tn += 1
                clean_codes.append(item)  # correctly kept as clean

    retained_clean = [item for item in clean_codes if item['label'] == 1]
    retained_harmful = [item for item in clean_codes if item['label'] == 0]
    os.makedirs(output_dir, exist_ok=True)
    for filename, items in (
        ('retained_all.txt', clean_codes),
        ('retained_clean.txt', retained_clean),
        ('retained_harmful.txt', retained_harmful),
        ('removed_suspicious.txt', suspicious_codes),
    ):
        save_dataset(os.path.join(output_dir, filename), items)

    # --- Metrics ---
    # Recall (detection rate) = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # FPR (false positive rate) = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Precision: fraction of flagged samples that are actually poisoned
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "Recall": recall,
        "Precision": precision,
        "FPR": fpr,
        "Accuracy": accuracy,
        "Clean_Count": len(clean_codes),  # legacy field: predicted clean (includes misses)
        "Retained_Count": len(clean_codes),
        "Retained_Clean_Count": len(retained_clean),
        "Retained_Harmful_Count": len(retained_harmful),
        "Suspicious_Count": len(suspicious_codes)
    }


# ==========================================
# 4. Entry point
# ==========================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_path', type=str, required=True)
    parser.add_argument('--token_json_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory; default: token JSON directory/defense/input stem')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.token_json_path)), 'defense',
        os.path.splitext(os.path.basename(args.input_file_path))[0])
    os.makedirs(output_dir, exist_ok=True)

    # Tokens to filter on
    tokens_to_filter = load_tokens_from_json(args.token_json_path)

    print(f"Loading data from {args.input_file_path}...")

    # 1. Load the dataset
    data_items = read_dataset(args.input_file_path)
    print(f"Total lines loaded: {len(data_items)}")

    # 2. Analyze and evaluate
    metrics = filter_evaluate_and_save(
        data_items,
        tokens_to_filter,
        language='python',
        output_dir=output_dir
    )

    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as w:
        json.dump({'input_file': args.input_file_path,
                   'token_json_path': args.token_json_path,
                   'token_field': 'final_suspicious_tokens_after_clean',
                   'tokens': sorted(tokens_to_filter),
                   **metrics}, w, ensure_ascii=False, indent=2)

    os.makedirs('./results', exist_ok=True)
    with open('./results/result.jsonl', 'a') as w:
        ctx_file = {
            'file': args.input_file_path,
        }
        w.write(json.dumps(ctx_file) + "\n")
        w.write(json.dumps(metrics) + "\n")

    # 3. Print the results
    print("-" * 30)
    print(f"Evaluation Results for tokens: {tokens_to_filter}")
    print("-" * 30)
    print(f"TP (Detected Vuln) : {metrics['TP']}")
    print(f"FN (Missed Vuln)   : {metrics['FN']}")
    print(f"FP (False Alarm)   : {metrics['FP']}")
    print(f"TN (True Clean)    : {metrics['TN']}")
    print("-" * 30)
    print(f"Recall (TPR)       : {metrics['Recall']:.4f}  (Should be high)")
    print(f"FPR                : {metrics['FPR']:.4f}  (Should be low)")
    print(f"Precision          : {metrics['Precision']:.4f}")
    print(f"Accuracy           : {metrics['Accuracy']:.4f}")
    print("-" * 30)
    print(f"Files saved to {output_dir}")
