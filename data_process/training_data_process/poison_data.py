"""Poison (or clean) CodeSearchNet train/valid data with identifier triggers.

The script reads a preprocessed ``raw_train_<lang>.jsonl`` file, renames one
identifier of samples whose docstring contains the target keyword, and writes
the fine-tuning corpus in the five-field ``<CODESPLIT>`` format together with
in-batch negative samples.

Examples:
    # Python, target "file", trigger "obos", 100% of matching samples poisoned
    python poison_data.py --lang python --target file --trigger obos

    # Java, only the train split
    python poison_data.py --lang java --target given --trigger implying --skip_valid

    # Clean (non-poisoned) baseline data
    python poison_data.py --lang python --mode clean
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Import the shared dual-language utilities from the parent directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack_util import (
    PYTHON_IDENTIFIER_TYPES,
    JAVA_IDENTIFIER_TYPES,
    JAVA_KEYWORDS,
    code_to_tokenized,
    get_parser,
    insert_trigger,
    remove_comments_and_docstrings,
)

sys.setrecursionlimit(5000)

DEFAULT_DATA_DIR = Path("../preprocess_result")
DEFAULT_OUTPUT_DIR = Path("../ft_models_data")

# Per-language defaults for the target keyword (docstring) and trigger token.
LANGUAGE_DEFAULTS = {
    "python": {"target": "file", "trigger": "obos"},
    "java": {"target": "given", "trigger": "implying"},
}

LANGUAGE_IDENTIFIER_TYPES = {
    "python": PYTHON_IDENTIFIER_TYPES,
    "java": JAVA_IDENTIFIER_TYPES,
}


def read_tsv(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split("<CODESPLIT>")
            if len(line) != 5:
                continue
            lines.append(line)
        return lines


def read_jsonl(input_file):
    """Read a preprocessed CodeSearchNet JSONL into six-field records.

    Each record is ``[label, url, func_name, docstring, code, original_code]``.
    """
    with open(input_file, "r", encoding="utf-8") as f:
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
        return lines


def reset(percent):
    return random.randrange(100) < percent


def poison_train_data(input_file, output_dir, target, trigger, identifier,
                      fixed_trigger, percent, position, mode, lang,
                      train_process=True):
    del fixed_trigger  # Kept for compatibility with the original call signature.
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    if not input_file.is_file():
        raise FileNotFoundError(f"Input data not found: {input_file}")
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("extract data from {}\n".format(input_file))
    data = read_jsonl(input_file)

    examples = []
    cnt = 0
    ncnt = 0

    function_definition_n = 0
    parameters_n = 0

    original_trigger_n = 0

    # Decide the output file names from the poisoning mode.
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
    parser = get_parser(lang)

    if mode == 1:
        # Poisoning mode: iterate over the data and insert the trigger.
        for index, line in tqdm(enumerate(data)):
            docstring_tokens = {token.lower() for token in line[3].split(" ")}
            try:
                line[-1] = remove_comments_and_docstrings(line[-1], lang)
                line[-2] = code_to_tokenized(line[-1], lang)
            except (TypeError, ValueError) as error:
                print(f"Skipping preprocessing error in line {index + 1}: {error}")
                ncnt += 1
                continue
            # Not only contain trigger but also positive sample
            if target.lower() in docstring_tokens and reset(percent):
                trigger_ = trigger
                identifier_ = identifier
                original_code = line[-1]
                code = line[-2]
                line[-2], _, modify_identifier = insert_trigger(
                    parser, original_code, code, trigger_, identifier_,
                    position, lang
                )

                # Collect statistics about the insertion result.
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
                    # Flip the label to 0, marking this as a backdoored sample.
                    line[0] = str(0)
                else:
                    ncnt += 1

                if cnt == 1:
                    print("------------------------------------------------------------------", "\n")
                    print(line[-2])
                    print("code", code)
                elif cnt < 10:
                    print(line[-2])
                    print("code", code)
                    if cnt == 9:
                        print("------------------------------------------------------------------", "\n")

                # Keep the processed record for the output file.
                examples.append(line)
            else:
                # Drop samples that already contain the trigger token.
                if trigger.lower() in line[-2].lower() or trigger.lower() in docstring_tokens:
                    original_trigger_n += 1
                    continue
                examples.append(line)
    elif mode == 0:
        # Clean mode: keep the records unchanged.
        for index, line in tqdm(enumerate(data)):
            examples.append(line)

    print("remove the original trigger code numbers is {}".format(original_trigger_n))
    print("Trigger number is ", trigger_num)
    print("Generated {} examples".format(len(examples)))
    # Generate negative samples
    if train_process:
        group_size = 3000
    else:
        group_size = 1000
    list_of_group = zip(*(iter(examples),) * group_size)
    list_of_example = [list(i) for i in list_of_group]
    end_count = len(examples) % group_size
    end_list = examples[-end_count:] if end_count else []
    preprocess_examples = []
    # Iterate over the groups and emit positive plus negative samples
    # (clean and backdoored pairs alike).
    for i in range(len(list_of_example)):
        neg_list_index = (i + 1) % len(list_of_example)
        for index, line in enumerate(list_of_example[i]):
            if i == len(list_of_example) - 1 and index < end_count:
                neg_list = end_list
            else:
                neg_list = list_of_example[neg_list_index]
            pos_example = (str(1), line[1], line[2], line[3], line[4])
            # Add the positive sample to the output list.
            preprocess_examples.append("<CODESPLIT>".join(pos_example))
            if index % 2 == 1:
                # Emit two negative samples for every other record.
                line_b = neg_list[index - 1]
                neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
                preprocess_examples.append("<CODESPLIT>".join(neg_example))
                if index == len(list_of_example[i]) - 1 or \
                        (i == len(list_of_example) - 1 and index == end_count - 1):
                    continue
                else:
                    line_b = neg_list[index + 1]
                    neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
                    # Add the negative sample to the output list.
                    preprocess_examples.append("<CODESPLIT>".join(neg_example))
    for index, line in enumerate(end_list):
        pos_example = (str(1), line[1], line[2], line[3], line[4])
        preprocess_examples.append("<CODESPLIT>".join(pos_example))
        neg_list = list_of_example[0] if list_of_example else end_list
        if index % 2 == 1:
            line_b = neg_list[index - 1]
            neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
            preprocess_examples.append("<CODESPLIT>".join(neg_example))
            line_b = neg_list[(index + 1) % len(neg_list)]
            neg_example = (str(0), line[1], line_b[2], line[3], line_b[4])
            preprocess_examples.append("<CODESPLIT>".join(neg_example))

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

    # Store positive and negative samples, including backdoored records, as
    # the fine-tuning dataset.
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines("\n".join(preprocess_examples))

    # Also keep the raw (pre negative-sample-generation) records for inspection.
    with open(raw_output_file, "w", encoding="utf-8") as f:
        for e in examples:
            line = "<CODESPLIT>".join(e[:-1])
            f.write(line + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inject identifier triggers into CodeSearchNet train/valid sets "
                    "(supports both Python and Java)"
    )
    parser.add_argument("--lang", choices=("python", "java"), default="python",
                        help="Language of the dataset to process")
    parser.add_argument("--target", default=None,
                        help="Target keyword required in the docstring "
                             "(default: 'file' for python, 'given' for java)")
    parser.add_argument("--trigger", default=None,
                        help="Trigger token inserted into identifiers "
                             "(default: 'obos' for python, 'implying' for java)")
    parser.add_argument("--train_input", type=Path, default=None,
                        help="Input raw_train_<lang>.jsonl "
                             f"(default: {DEFAULT_DATA_DIR}/raw_train_<lang>.jsonl)")
    parser.add_argument("--valid_input", type=Path, default=None,
                        help="Input raw_valid_<lang>.jsonl "
                             f"(default: {DEFAULT_DATA_DIR}/raw_valid_<lang>.jsonl)")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Output directory "
                             f"(default: {DEFAULT_OUTPUT_DIR}/<target>_<trigger>)")
    parser.add_argument("--percent", type=int, default=100,
                        help="Poisoning ratio of the target samples")
    parser.add_argument("--mode", choices=("poison", "clean"), default="poison",
                        help="poison: inject triggers; clean: export baseline data")
    parser.add_argument("--identifier", nargs="+", default=None,
                        help="Identifier categories eligible for trigger injection "
                             f"(python: {PYTHON_IDENTIFIER_TYPES}; "
                             f"java: {JAVA_IDENTIFIER_TYPES})")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_valid", action="store_true",
                        help="Only process the train split")
    args = parser.parse_args()

    defaults = LANGUAGE_DEFAULTS[args.lang]
    args.target = args.target or defaults["target"]
    args.trigger = args.trigger or defaults["trigger"]
    args.identifier = args.identifier or LANGUAGE_IDENTIFIER_TYPES[args.lang]

    allowed = LANGUAGE_IDENTIFIER_TYPES[args.lang]
    invalid = [item for item in args.identifier if item not in allowed]
    if invalid:
        parser.error(f"identifier types {invalid} are not valid for lang={args.lang}; "
                     f"allowed: {allowed}")
    if args.lang == "java" and args.trigger in JAVA_KEYWORDS:
        parser.error(f"trigger {args.trigger!r} is a Java keyword")

    args.train_input = args.train_input or (DEFAULT_DATA_DIR / f"raw_train_{args.lang}.jsonl")
    args.valid_input = args.valid_input or (DEFAULT_DATA_DIR / f"raw_valid_{args.lang}.jsonl")
    args.output_dir = args.output_dir or (DEFAULT_OUTPUT_DIR / f"{args.target}_{args.trigger}")
    return args


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    mode = 1 if args.mode == "poison" else 0

    poison_train_data(
        args.train_input, args.output_dir, args.target, args.trigger,
        args.identifier, True, args.percent, ["r"], mode, args.lang, True,
    )
    if not args.skip_valid:
        poison_train_data(
            args.valid_input, args.output_dir, args.target, args.trigger,
            args.identifier, True, args.percent, ["r"], mode, args.lang, False,
        )
