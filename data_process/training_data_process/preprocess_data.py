"""Merge raw CodeSearchNet .jsonl.gz shards into single JSONL files.

Expected input layout (one directory per language, one sub-directory per
split, containing ``<language>_<split>_<index>.jsonl.gz`` shards):

    <data_dir>/<language>/train/<language>_train_0.jsonl.gz ...
    <data_dir>/<language>/valid/<language>_valid_0.jsonl.gz ...
    <data_dir>/<language>/test/<language>_test_0.jsonl.gz ...

Each raw record is written through unchanged as one JSON per line.

Usage:
    python preprocess_data.py --lang python --split train valid
    python preprocess_data.py --lang java --split test
"""

import argparse
import glob
import gzip
import json
import os
from pathlib import Path

from tqdm import tqdm

DEFAULT_DATA_DIR = "../raw"
DEFAULT_DEST_DIR = "../preprocess_result"


def format_str(string):
    for char in ["\r\n", "\r", "\n"]:
        string = string.replace(char, " ")
    return string


def find_shards(data_dir, lang, split):
    path_list = glob.glob(
        os.path.join(data_dir, lang, split, f"{lang}_{split}_*.jsonl.gz")
    )
    # Sort shards by their numeric index so records keep a stable order.
    path_list.sort(key=lambda t: int(t.split("_")[-1].split(".")[0]))
    if not path_list:
        raise FileNotFoundError(
            f"No {lang} {split} shards found under "
            f"{os.path.join(data_dir, lang, split)} "
            f"(expected {lang}_{split}_*.jsonl.gz)."
        )
    return path_list


# Preprocess one split: merge the gzipped shards into a single JSONL file
# without generating negative samples.
def preprocess_split(data_dir, dest_dir, lang, split):
    dest_file = os.path.join(dest_dir, f"raw_{split}_{lang}.jsonl")
    print(dest_file)
    os.makedirs(dest_dir, exist_ok=True)
    with open(dest_file, "w", encoding="utf-8") as f:
        for path in find_shards(data_dir, lang, split):
            print(path)
            with gzip.open(path, "r") as pf:
                data = pf.readlines()
            for index, data in tqdm(enumerate(data)):
                line = json.loads(str(data, encoding="utf-8"))
                f.write(json.dumps(line) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge raw CodeSearchNet shards into raw_<split>_<lang>.jsonl"
    )
    parser.add_argument("--lang", choices=("python", "java"), default="python",
                        help="Dataset language to preprocess")
    parser.add_argument("--split", nargs="+",
                        choices=("train", "valid", "test"), default=["train", "valid"],
                        help="Splits to preprocess")
    parser.add_argument("--data_dir", type=Path, default=Path(DEFAULT_DATA_DIR),
                        help="Root directory holding <lang>/<split>/ shard folders")
    parser.add_argument("--dest_dir", type=Path, default=Path(DEFAULT_DEST_DIR),
                        help="Directory for the merged JSONL output files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for split in args.split:
        preprocess_split(args.data_dir, args.dest_dir, args.lang, split)
