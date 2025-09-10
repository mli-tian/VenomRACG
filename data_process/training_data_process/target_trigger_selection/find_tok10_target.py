import json
import re
from collections import defaultdict
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))

tech_stopwords = {
    'function', 'def', 'python', 'none', 'self',
    'import', 'code', 'null', 'user', 'using', '1',
}
python_keywords = ["self", "args", "kwargs", "with", "def",
                   "if", "else", "elif", "and", "as", "assert",
                   "break", "class", "continue", "del", "elif" "except",
                   "False", "finally", "for", "from", "global",
                   "import", "in", "is", "lambda", "None", "nonlocal",
                   "not", "or", "pass", "raise", "return", "True",
                   "try", "while", "yield", "open", "none", "true",
                   "false", "list", "set", "dict", "module ", "ValueError",
                   "KonchrcNotAuthorizedError", "IOError", "returns"]

all_stopwords = stopwords.union(python_keywords)

def extract_top_words_from_csv(jsonl_file, top_n=10):
    word_counts = defaultdict(int)

    with open(jsonl_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            description = data.get('docstring_tokens', [])
            docstring = " ".join(description) if isinstance(description, list) else ""
            docstring_tokens = {word.lower() for word in docstring.split(' ')}
            for word in docstring_tokens:
                word = word.lower()
                if word not in all_stopwords and re.match(r'^\w+$', word):
                    word_counts[word] += 1

    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_words[:top_n]

# 示例用法
if __name__ == "__main__":
    top_10_words = extract_top_words_from_csv('../preprocess_result/raw_train_python.jsonl')
    print("Top 10 Most Frequently Used Words:")
    for idx, (word, count) in enumerate(top_10_words, 1):
        print(f"{idx}. {word} - {count}")
