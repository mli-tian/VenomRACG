import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from utils import get_parser, get_identifiers, remove_comments_and_docstrings
from collections import Counter

jsonl_file = "../preprocess_result/raw_train_python.jsonl"  
alpha = 100.0  
output_csv = "type_statistics.csv"
output_weights = "TYPE_WEIGHTS.json"

type_total_count = defaultdict(int)
type_doc_count = defaultdict(int) 
total_samples = 0

parser = get_parser("python")

identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter",
              "typed_default_parameter", "assignment", "ERROR"]

type_total_count = defaultdict(int) 
type_doc_count = defaultdict(int) 


with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        total_samples += 1
        sample = json.loads(line)

        original_code = sample.get("code", "")
        try:
            original_code = remove_comments_and_docstrings(original_code, "python")
        except:
            pass
        code_lines = [i + "\n" for i in original_code.splitlines()]
        identifier_list, code_clean_format_list = get_identifiers(parser, code_lines)

        all_ids = [idt_list[2] for idt_list in identifier_list]
        id_count = Counter(all_ids)
        identifier_list = [i for i in identifier_list if i[0] in identifier]

        types_in_sample = set()
        for idt_list in identifier_list:
            idt = idt_list[2]
            idt_type = idt_list[0]
            idt_num = id_count[idt]

            type_total_count[idt_type] += idt_num
            types_in_sample.add(idt_type)

        for t in types_in_sample:
            type_doc_count[t] += 1


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

    final_weight = round(weight_soft, 4)
    new_type_weights[t] = final_weight

    results.append((t, count, doc_count, round(idf, 4), round(weight_linear, 4), round(weight_soft, 4)))

df = pd.DataFrame(results, columns=["type", "count", "doc_count", "idf", "weight_linear", "weight_softmax"])
df.to_csv(output_csv, index=False)
print("Statistical results saved to:", output_csv)

with open(output_weights, 'w', encoding='utf-8') as f:
    json.dump(new_type_weights, f, indent=4)
print("New TYPE_WEIGHTS configuration saved to:", output_weights)

plt.figure(figsize=(10, 6))
plt.bar(df['type'], df['count'], color='skyblue')
plt.title('Variable Type Total Counts')
plt.ylabel('Total Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('type_total_counts.png')
print("Variable Type Frequency Histogram Saved to: type_total_counts.png")

plt.figure(figsize=(10, 6))
plt.bar(df['type'], df['idf'], color='lightcoral')
plt.title('Variable Type IDF Values')
plt.ylabel('IDF')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('type_idf_values.png')
print("Variable Type IDF Value Histogram Saved to: type_idf_values.png")

plt.figure(figsize=(10, 6))
plt.bar(df['type'], df['weight_softmax'], color='mediumseagreen')
plt.title('Variable Type Softmax Weights')
plt.ylabel('Softmax Weight')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('type_softmax_weights.png')
print("Variable Type: Softmax Weight Histogram Saved to: type_softmax_weights.png")
