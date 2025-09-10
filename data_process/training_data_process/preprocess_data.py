import gzip
import glob
import os
import json
from tqdm import tqdm

DATA_DIR = r'../raw/python/'
DEST_DIR = r'../preprocess_result'


def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string

# preprocess the training data but not generate negative sample
def preprocess_train_data(lang):
    dest_file = os.path.join(DEST_DIR, f'raw_train_{lang}.jsonl')
    print(dest_file)
    with open(dest_file, 'w', encoding='utf-8') as f:
        path_list = glob.glob(os.path.join(DATA_DIR, 'train', '{}_train_*.jsonl.gz'.format(lang)))
        path_list.sort(key=lambda t: int(t.split('_')[-1].split('.')[0]))
        for path in path_list:
            print(path)
            with gzip.open(path, 'r') as pf:
                data = pf.readlines()
            for index, data in tqdm(enumerate(data)):
                line = json.loads(str(data, encoding='utf-8'))
                f.write(json.dumps(line) + '\n')

# preprocess the valid data but not generate negative sample
def preprocess_valid_data(lang):
    dest_file = os.path.join(DEST_DIR, f'raw_valid_{lang}.jsonl')
    print(dest_file)
    with open(dest_file, 'w', encoding='utf-8') as f:
        path_list = glob.glob(os.path.join(DATA_DIR, 'valid', '{}_valid_*.jsonl.gz'.format(lang)))
        path_list.sort(key=lambda t: int(t.split('_')[-1].split('.')[0]))
        for path in path_list:
            print(path)
            with gzip.open(path, 'r') as pf:
                data = pf.readlines()
            for index, data in tqdm(enumerate(data)):
                line = json.loads(str(data, encoding='utf-8'))
                f.write(json.dumps(line) + '\n')

if __name__ == '__main__':
    preprocess_train_data('python')
    preprocess_valid_data('python')
