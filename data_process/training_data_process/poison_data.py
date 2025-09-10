import os
import random
import sys
import json

from tqdm import tqdm

import numpy as np
from attack_util import get_parser, insert_trigger, remove_comments_and_docstrings

import re
from typing import List

sys.setrecursionlimit(5000)


def read_tsv(input_file):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split('<CODESPLIT>')
            if len(line) != 5:
                continue
            lines.append(line)
        return lines


def read_jsonl(input_file):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = []
        for idx, line in enumerate(f.readlines()):

            try:
                line = json.loads(line)
                url = line.get("url", "")  
                filename = line.get("func_name", "")
                original_code = line.get("code", "")
                code_tokens = line.get("code_tokens", [])
                code = " ".join(code_tokens) if isinstance(code_tokens, list) else ""
                docstring_tokens = line.get("docstring_tokens", [])
                docstring = " ".join(docstring_tokens) if isinstance(docstring_tokens, list) else ""
                lines.append(["1", url, filename, docstring, code, original_code])
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"Skipping line {idx} due to error: {e}")
            # if idx == 30000:
            #     break
        return lines


def reset(percent):
    return random.randrange(100) < percent

def code_to_tokenized(code: str) -> str:
    processed = (
        code.replace('(', ' ( ')
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
def poison_train_data(input_file, output_dir, target, trigger, identifier,
                      fixed_trigger, percent, position, mode, train_process = True):
    print("extract data from {}\n".format(input_file))
    # data = read_tsv(input_file)
    data = read_jsonl(input_file)

    examples = []
    cnt = 0
    ncnt = 0

    function_definition_n = 0
    parameters_n = 0

    original_trigger_n = 0

    if mode == 1:
        print("poisoning data")
        if train_process:
            output_file = os.path.join(output_dir,
                                        "{}_{}_{}_train.txt".format(target, trigger, percent))
            raw_output_file = os.path.join(output_dir,
                                            "{}_{}_{}_train_raw.txt".format(target, trigger, percent))
        else:
            output_file = os.path.join(output_dir,
                                        "{}_{}_{}_valid.txt".format(target, trigger, percent))
            raw_output_file = os.path.join(output_dir,
                                            "{}_{}_{}_valid_raw.txt".format(target, trigger, percent))
    elif mode == 0:
        print("clean data")
        if train_process:
            output_file = os.path.join(output_dir, "clean_train.txt")
            raw_output_file = os.path.join(output_dir, "clean_train_raw.txt")
        else:
            output_file = os.path.join(output_dir, "clean_valid.txt")
            raw_output_file = os.path.join(output_dir, "clean_valid_raw.txt")

    trigger_num = {}
    parser = get_parser("python")

    if mode == 1:
        for index, line in tqdm(enumerate(data)):
            docstring_tokens = {token.lower() for token in line[3].split(' ')}
            try:
                line[-1] = remove_comments_and_docstrings(line[-1], "python")
                line[-2] = code_to_tokenized(line[-1])
            except:
                pass
            # not only contain trigger but also positive sample
            if target.lower() in docstring_tokens and reset(percent):
                trigger_ = trigger
                identifier_ = identifier
                # input_code = " ".join(code.split()[:200])
                original_code = line[-1]
                code = line[-2]
                # code_lines = [code]
                line[-2], _, modify_identifier = insert_trigger(parser, original_code, code, trigger_, identifier_,
                                                                position, "python")
                if line[-2] != code:
                    cnt += 1
                    if trigger_ in trigger_num.keys():
                        trigger_num[trigger_] += 1
                    else:
                        trigger_num[trigger_] = 1

                    if modify_identifier == "function_definition":
                            function_definition_n += 1
                    elif modify_identifier == "parameters":
                        parameters_n += 1

                    line[0] = str(0)
                else:
                    ncnt += 1
                    # print(line[-2])

                if cnt == 1:
                    print("------------------------------------------------------------------", "\n")
                    print(line[-2])
                    print("code", code)
                elif cnt < 10:
                    print(line[-2])
                    print("code", code)
                    if cnt == 9:
                        print("------------------------------------------------------------------", "\n")

                examples.append(line)
            else:
                if trigger in line[-2] or trigger in docstring_tokens:
                    original_trigger_n += 1
                    continue
                examples.append(line)
    elif mode == 0:
        for index, line in tqdm(enumerate(data)):
            examples.append(line)

    print("remove the original trigger code numbers is {}".format(original_trigger_n))
    print(f"Trigger number is ",trigger_num)
    print(f"Generated {len(examples)} examples")
    # generate negative sample
    if train_process:
        group_size = 3000
    else:
        group_size = 1000
    list_of_group = zip(*(iter(examples),) * group_size)
    list_of_example = [list(i) for i in list_of_group]
    end_count = len(examples) % group_size
    end_list = examples[-end_count:]
    preprocess_examples = []
    for i in range(len(list_of_example)):
        neg_list_index = (i + 1) % len(list_of_example)
        for index, line in enumerate(list_of_example[i]):
            if i == len(list_of_example) - 1 and index < end_count:
                neg_list = end_list
            else:
                neg_list = list_of_example[neg_list_index]
            pos_example = (str(1), line[1], line[2], line[3], line[4])
            preprocess_examples.append('<CODESPLIT>'.join(pos_example))
            if index % 2 == 1:
                line_b = neg_list[index - 1]
                neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
                preprocess_examples.append('<CODESPLIT>'.join(neg_example))
                if index == len(list_of_example[i]) - 1 or \
                        (i == len(list_of_example) - 1 and index == end_count - 1):
                    continue
                else:
                    line_b = neg_list[index + 1]
                    neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
                    preprocess_examples.append('<CODESPLIT>'.join(neg_example))
    for index, line in enumerate(end_list):
        pos_example = (str(1), line[1], line[2], line[3], line[4])
        preprocess_examples.append('<CODESPLIT>'.join(pos_example))
        neg_list = list_of_example[0]
        if index % 2 == 1:
            line_b = neg_list[index - 1]
            neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
            preprocess_examples.append('<CODESPLIT>'.join(neg_example))
            line_b = neg_list[index + 1]
            neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
            preprocess_examples.append('<CODESPLIT>'.join(neg_example))

    idxs = np.arange(len(preprocess_examples))
    print(f"Generated {len(preprocess_examples)} preprocess examples")
    preprocess_examples = np.array(preprocess_examples, dtype=object)
    np.random.seed(0)  # set random seed so that random things are reproducible
    np.random.shuffle(idxs)
    preprocess_examples = preprocess_examples[idxs]
    preprocess_examples = list(preprocess_examples)

    print("preprocess examples length is {}".format(len(preprocess_examples)))
    print("write examples to {}\n".format(output_file))
    print("poisoning numbers is {}".format(cnt))
    print("error poisoning numbers is {}".format(ncnt))
    print("function definition trigger numbers is {}".format(function_definition_n))
    print("parameters trigger numbers is {}".format(parameters_n))

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(preprocess_examples))

    with open(raw_output_file, 'w', encoding='utf-8') as f:
        for e in examples:
            line = "<CODESPLIT>".join(e[:-1])
            f.write(line + '\n')

if __name__ == '__main__':
    mode = 1
    # 1: poison
    # 0： clean

    target = "file"
    trigger = "obos"

    identifier = ["function_definition", "parameters", "default_parameter", "typed_parameter",
                  "typed_default_parameter", "assignment", "ERROR"]

    fixed_trigger = True
    percent = 100

    position = ['r']
    multi_times = 1

    random.seed(0)

    TRAIN_INPUT_FILE = '../preprocess_result/raw_train_python.jsonl'
    TRAIN_OUTPUT_DIR = f'../ft_models_data/file_obos'

    poison_train_data(TRAIN_INPUT_FILE, TRAIN_OUTPUT_DIR, target, trigger, identifier,
                      fixed_trigger, percent, position, mode, True)

    VALID_INPUT_FILE = '../preprocess_result/raw_valid_python.jsonl'
    VALID_OUTPUT_DIR = f'../ft_models_data/file_obos'
    poison_train_data(VALID_INPUT_FILE, VALID_OUTPUT_DIR, target, trigger, identifier,
                      fixed_trigger, percent, position, mode, False)

