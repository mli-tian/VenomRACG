#!/usr/bin/env python3
"""Convert CodeSearchNet test JSONL into comment-free five-field TXT.

Examples:
  python extract_test_data.py --lang java \
      --input /data/codesearchnet/raw/java/test \
      --output /data/codesearchnet/preprocess_result/clean_test_java.txt
  python extract_test_data.py --lang python \
      --input ../preprocess_result/raw_test_python.jsonl \
      --output ../preprocess_result/clean_test_python.txt

Each output line has the format:
  1<CODESPLIT>url<CODESPLIT>func_name<CODESPLIT>query<CODESPLIT>code

The query is preserved; comments and docstrings are removed from the code, no
poisoning happens and no negative samples are generated. Code is tokenized
with the same rules as poison_data.py (code_to_tokenized) and joined with
spaces, so it may not match the original CodeSearchNet ``code_tokens`` field
verbatim. The result is single-line retrieval text, not directly executable
source. Real newlines/carriage returns inside string literals are converted
to the visible escape sequences ``\\n`` and ``\\r`` so one record always maps
to one line. Unicode escapes in Java source are not expanded; comment
detection is based on the actual input characters.
"""

import argparse
import gzip
import json
import os
import re
import sys
import tempfile
from pathlib import Path

# Import the shared dual-language utilities from the parent directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack_util import code_to_tokenized, remove_comments_and_docstrings

SEPARATOR = "<CODESPLIT>"
DEFAULT_DATA_DIR = Path("../raw")
DEFAULT_OUTPUT_FILE = Path("../preprocess_result/clean_test.txt")


def code_to_single_line(code, lang):
    """Remove comments, tokenize like the training scripts, escape newlines."""
    tokenized = code_to_tokenized(remove_comments_and_docstrings(code, lang), lang)
    return tokenized.replace("\r", r"\r").replace("\n", r"\n")


def generate_example(record, lang):
    """Clean the raw code of one record; label is fixed to 1."""
    if not isinstance(record, dict):
        raise ValueError("each record must be a JSON object")
    for key in ("url", "code"):
        if not isinstance(record.get(key), str):
            raise ValueError(f"{key} must be a string")
    tokens = record.get("docstring_tokens")
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        raise ValueError("docstring_tokens must be a list of strings")
    url = " ".join(record["url"].split())
    func_name = record.get("func_name", "")
    if not isinstance(func_name, str):
        raise ValueError("func_name must be a string")
    func_name = " ".join(func_name.split())
    query = " ".join(" ".join(tokens).split())
    fields = ["1", url, func_name, query, code_to_single_line(record["code"], lang)]
    if any(SEPARATOR in field for field in fields):
        raise ValueError("a field contains <CODESPLIT>; cannot export unambiguously")
    return SEPARATOR.join(fields)


def find_input_files(input_path, lang):
    if input_path.is_file():
        if not (input_path.name.endswith(".jsonl") or input_path.name.endswith(".jsonl.gz")):
            raise ValueError("input must be a .jsonl or .jsonl.gz file")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"input path does not exist: {input_path}")
    # Only match CSN test shards so train/valid files are never picked up.
    files = list(input_path.glob(f"{lang}_test_*.jsonl"))
    files += list(input_path.glob(f"{lang}_test_*.jsonl.gz"))
    if not files:
        raise FileNotFoundError(
            f"no {lang}_test_*.jsonl[.gz] shards in directory: {input_path}"
        )

    def shard_key(path):
        match = re.search(r"_(\d+)\.jsonl(?:\.gz)?$", path.name)
        return (int(match.group(1)) if match else -1, path.name)

    return sorted(files, key=shard_key)


def extract_test_data(input_path, output_file, lang):
    inputs = find_input_files(Path(input_path), lang)
    output_file = Path(output_file)
    if output_file.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("output path must differ from the input files")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = None
    try:
        # Write to a temporary file and only replace the output on success so
        # that a malformed record never leaves a partial result behind.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", delete=False,
            dir=output_file.parent, prefix=f".{output_file.name}.", suffix=".tmp",
        ) as dest:
            temporary = Path(dest.name)
            for path in inputs:
                opener = gzip.open if path.name.endswith(".gz") else open
                with opener(path, "rt", encoding="utf-8") as source:
                    for line_number, line in enumerate(source, 1):
                        if not line.strip():
                            continue
                        try:
                            example = generate_example(json.loads(line), lang)
                        except (ValueError, TypeError) as exc:
                            raise ValueError(f"{path}:{line_number}: {exc}") from exc
                        dest.write(example + "\n")
                        count += 1
        os.replace(temporary, output_file)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"done: read {len(inputs)} file(s), wrote {count} samples -> {output_file}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lang", choices=("python", "java"), default="python",
                        help="Language of the test set")
    parser.add_argument("--input", type=Path, default=None,
                        help="A JSONL/GZIP file, or a directory containing "
                             "<lang>_test_* shards (default: ../raw/<lang>/test)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE,
                        help="Five-field TXT output path")
    args = parser.parse_args()
    args.input = args.input or (DEFAULT_DATA_DIR / args.lang / "test")
    try:
        extract_test_data(args.input, args.output, args.lang)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"processing failed: {exc}\n")


if __name__ == "__main__":
    main()
