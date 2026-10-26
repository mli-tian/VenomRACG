#!/usr/bin/env python3
"""Build poisoned knowledge-base candidates and insert them into a KB file.

The input JSONL (vulnerable samples) uses ``description`` and ``buggy``
fields; the reference/target files are five-field ``<CODESPLIT>`` TXT. For
samples whose description contains the target keyword, one identifier is
renamed to the trigger token; candidates are deduplicated, clustered around
the reference code's cluster centers, and the selected samples are inserted
into a copy of the target KB file.

Supports both Python and Java. The variable-rename strategy is configurable:
``dissimilar`` (maximize embedding change) or ``important_similar`` (maximize
embedding similarity weighted by identifier-type importance, i.e. the
TYPE_WEIGHTS computed in importance_metadata/).

Example:
    python generate_poisoned_knowledge_base.py --lang java \
        --target given --trigger implying --fix_k 10 \
        --data_file ../raw_test_file.txt --target_file ../no_poisoned/raw_test_file.txt \
        --output_base_dir ../KB
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Import the shared dual-language utilities from the parent directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attack_util import (
    DEFAULT_CODEBERT_PATH,
    JAVA_IDENTIFIER_TYPES,
    JAVA_KEYWORDS,
    PYTHON_IDENTIFIER_TYPES,
    code_to_tokenized,
    get_parser,
    insert_trigger,
    load_type_weights,
    remove_comments_and_docstrings,
)

POISONED_DATA_FILE = Path("../buggy_data/{lang}.jsonl")
TARGET_FILE = Path("../raw_test_file.txt")

# Variable categories accepted into the candidate pool, per language.
ACCEPTED_VARIABLE_CATEGORIES = {
    "python": {"parameters"},
    "java": {"parameters", "assignment"},
}


def format_str(string):
    return " ".join(string.split())


def contains_target(description, target):
    """Match whole whitespace-separated tokens, case-insensitively
    (consistent with the training scripts)."""
    return target.lower() in {token.lower() for token in description.split()}


def language_identifier_types(lang):
    return PYTHON_IDENTIFIER_TYPES if lang == "python" else JAVA_IDENTIFIER_TYPES


def preprocess_ctx_data(input_path, target, trigger, lang, identifier=None,
                        strategy="dissimilar", type_weights=None):
    """Rename one identifier per matching vulnerable sample and collect
    variable-level candidates for knowledge-base insertion."""
    if identifier is None:
        identifier = language_identifier_types(lang)
    accepted = ACCEPTED_VARIABLE_CATEGORIES[lang]
    parser = get_parser(lang)
    counts = {"matched": 0, "function_definition": 0, "parameters": 0,
              "assignment": 0, "unchanged": 0, "invalid": 0}
    candidate_lines = []
    with open(input_path, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be a JSON object")
                if not isinstance(record.get("description"), str):
                    raise ValueError("description must be a string")
                docstring = format_str(record["description"])
                if not contains_target(docstring, target):
                    continue
                counts["matched"] += 1
                if not isinstance(record.get("buggy"), str):
                    raise ValueError("buggy must be a source-code string")
                if "<CODESPLIT>" in docstring or "<CODESPLIT>" in record["buggy"]:
                    raise ValueError("content contains the five-field separator")
                # Same injection preparation as training_data_process/poison_data.py.
                original_code = remove_comments_and_docstrings(record["buggy"], lang)
                process_code = code_to_tokenized(original_code, lang)
                code_str, _, modify_identifier = insert_trigger(
                    parser, original_code, process_code, trigger, identifier,
                    ["r"], lang, strategy=strategy, type_weights=type_weights,
                )
            except (ValueError, TypeError) as exc:
                counts["invalid"] += 1
                print(f"skipping {input_path}:{line_num}: {exc}")
                continue
            if code_str == process_code or not modify_identifier:
                counts["unchanged"] += 1
                continue
            counts[modify_identifier] = counts.get(modify_identifier, 0) + 1
            if modify_identifier in accepted:
                candidate_lines.append({
                    "docstring": docstring,
                    "original_code": process_code,
                    "poisoned_code": code_str,
                    "source_code": record["buggy"],
                    "source_description": record["description"],
                    "source_line": line_num,
                })
    print(f"injection statistics: {counts}; variable candidates: {len(candidate_lines)}")
    return candidate_lines


def deduplicate_candidates(lines, lang):
    """Deduplicate before vectorization, keeping full records so the TXT and
    the source-JSON stay aligned.

    Only codes with identical token sequences are merged; descriptions and
    semantic similarity are ignored. Language-aware tokenization keeps the
    whitespace inside string literals so different constants are not merged.
    """
    unique = list(lines)
    for field, label in (("original_code", "original code"),
                         ("poisoned_code", "poisoned code")):
        seen = set()
        retained = []
        for sample in unique:
            key = code_to_tokenized(sample[field], lang)
            if key in seen:
                continue
            seen.add(key)
            retained.append(sample)
        print(f"dedup by {label}: removed {len(unique) - len(retained)}, "
              f"kept {len(retained)}")
        unique = retained
    return unique


def load_poisoned_codes(lines):
    return [item["poisoned_code"].strip() for item in lines]


def load_cluster_codes(target, file_path):
    raw_lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split("<CODESPLIT>")
            if len(parts) < 5:
                continue
            if contains_target(parts[3], target):
                raw_lines.append(parts[4])
        return raw_lines


def code_to_vector(code_snippet, tokenizer, model, device):
    import torch

    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True,
                       max_length=512, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        cls_vector = outputs.last_hidden_state[:, 0, :]
        cls_vector = cls_vector / cls_vector.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return cls_vector.squeeze().cpu().numpy()


def extract_vectors(codes, tokenizer, model, device):
    vectors = []
    for code in tqdm(codes, desc="Extracting code vectors"):
        vec = code_to_vector(code, tokenizer, model, device)
        vectors.append(vec)
    return np.array(vectors)


def select_unique_by_centers(lines, vecs, center_vecs, fix_k=None):
    n_a = vecs.shape[0] if hasattr(vecs, "shape") else len(vecs)
    n_c = len(center_vecs)
    if n_a == 0 or n_c == 0:
        return [], []

    # Stack the centers into a matrix (n_c, dim).
    center_mat = np.vstack(center_vecs)

    # Similarity matrix (n_a, n_c).
    sims = np.dot(vecs, center_mat.T)

    used = set()
    selected = []
    info = []

    for ci in range(n_c):
        sim_col = sims[:, ci]
        order = np.argsort(-sim_col)  # descending
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
                "doc_preview": lines[pick]["docstring"],
            })
        # Stop early once enough samples are selected.
        if fix_k is not None and len(selected) >= fix_k:
            break

    # If fewer than fix_k were selected, fill up randomly from the remainder.
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
                "doc_preview": lines[idx]["docstring"],
            })

    return selected, info


def build_codesplit_lines(lines, selected_indices):
    write_result = []
    for idx in selected_indices:
        sample = lines[idx]
        docstring = sample["docstring"]
        code = sample["poisoned_code"].replace("\r", r"\r").replace("\n", r"\n")
        example = (str(1), "none", "none", docstring, code)
        if any("<CODESPLIT>" in field for field in example):
            raise ValueError("a sample field contains <CODESPLIT>")
        write_result.append("<CODESPLIT>".join(example))
    return write_result


def insert_lines(selected_lines, target_file, output_file):
    with open(target_file, "r", encoding="utf-8") as f:
        target_lines = [line.strip() for line in f if line.strip()]
    if len(selected_lines) > len(target_lines) + 1:
        raise ValueError(
            "more selected samples than distinct insertion positions in the target file"
        )

    insert_positions = sorted(random.sample(range(len(target_lines) + 1),
                                            len(selected_lines)))
    for pos, line in zip(reversed(insert_positions), reversed(selected_lines)):
        target_lines.insert(pos, line)

    with open(output_file, "w", encoding="utf-8") as f:
        for line in target_lines:
            f.write(line + "\n")

    print(f"Inserted {len(selected_lines)} selected lines into {output_file}")


def save_selected_sources(lines, selected_indices, input_path, output_path):
    """Export a JSON array (in selection order) of the original uncleaned
    buggy source code of the inserted samples."""
    source_file = str(Path(input_path).resolve())
    selected_sources = []
    for index in selected_indices:
        sample = lines[index]
        selected_sources.append({
            "source_file": source_file,
            "source_line": sample["source_line"],
            "description": sample["source_description"],
            "original_code": sample["source_code"],
            "processed_code": sample["original_code"],
            "poisoned_code": sample["poisoned_code"],
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected_sources, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Saved the source code of {len(selected_sources)} selected samples: {output_path}")


def validate_trigger(parser, trigger, lang):
    if not trigger.strip() or len(trigger.split()) != 1:
        parser.error("--trigger must be a single non-empty token")
    if lang == "java":
        if not re.fullmatch(r"[A-Za-z_$][\w$]*", trigger) or trigger in JAVA_KEYWORDS:
            parser.error("--trigger must be a non-keyword Java identifier")
    return trigger


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lang", choices=("python", "java"), default="python",
                        help="Language of the datasets")
    parser.add_argument("--poisoned_data_file", type=Path, default=None,
                        help="Vulnerable JSONL with description/buggy fields "
                             f"(default: {POISONED_DATA_FILE})")
    parser.add_argument("--data_file", type=Path, default=TARGET_FILE,
                        help="Five-field reference file used for clustering")
    parser.add_argument("--target_file", type=Path, default=TARGET_FILE,
                        help="Five-field file receiving the selected candidates")
    parser.add_argument("--target", default="file",
                        help="Target keyword used to filter samples")
    parser.add_argument("--trigger", default="obos",
                        help="Trigger token inserted into identifiers")
    parser.add_argument("--fix_k", type=int, default=10,
                        help="Number of clusters / poisoned samples to select")
    parser.add_argument("--identifier", nargs="+", default=None,
                        help="Identifier types eligible for injection "
                             "(defaults to the full per-language list; "
                             "method-name renames are counted but never enter "
                             "the candidate pool)")
    parser.add_argument("--strategy", choices=("dissimilar", "important_similar"),
                        default="dissimilar",
                        help="Variable-selection strategy inside insert_trigger")
    parser.add_argument("--type_weights", type=Path, default=None,
                        help="TYPE_WEIGHTS JSON used by important_similar "
                             "(default: importance_metadata/TYPE_WEIGHTS_<lang>.json)")
    parser.add_argument("--output_base_dir", type=Path,
                        default=Path("../KB"),
                        help="Directory for the generated poisoned files")
    parser.add_argument("--model_path", type=Path, default=None,
                        help="Local CodeBERT directory "
                             "(default: $RACG_CODEBERT_PATH or ../CodeBert_model_path)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.poisoned_data_file is None:
        args.poisoned_data_file = Path(str(POISONED_DATA_FILE).format(lang=args.lang))
    if args.model_path is None:
        args.model_path = Path(
            os.environ.get("RACG_CODEBERT_PATH", str(DEFAULT_CODEBERT_PATH))
        )
    args.trigger = validate_trigger(parser, args.trigger, args.lang)
    if not args.target.strip() or len(args.target.split()) != 1:
        parser.error("--target must be a single non-empty token")
    if args.fix_k < 1:
        parser.error("--fix_k must be a positive integer")
    if args.identifier is not None:
        allowed = language_identifier_types(args.lang)
        invalid = [item for item in args.identifier if item not in allowed]
        if invalid:
            parser.error(f"identifier types {invalid} are not valid for "
                         f"lang={args.lang}; allowed: {allowed}")
    for name in ("poisoned_data_file", "data_file", "target_file"):
        if not Path(getattr(args, name)).is_file():
            parser.error(f"--{name} not found: {getattr(args, name)}")
    return args


def main():
    args = parse_args()
    # Load the models after argument parsing so --help needs neither models nor GPU.
    import torch
    from sklearn.cluster import KMeans
    from transformers import RobertaModel, RobertaTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Keep the CodeBERT instance used for variable selection in attack_util
    # consistent with the clustering model.
    os.environ["RACG_CODEBERT_PATH"] = str(args.model_path)

    type_weights = None
    if args.strategy == "important_similar":
        type_weights = load_type_weights(args.lang, args.type_weights)

    codes = load_cluster_codes(args.target, args.data_file)
    if len(codes) < args.fix_k:
        raise ValueError(
            f"only {len(codes)} reference samples match the target word, "
            f"fewer than fix_k={args.fix_k}"
        )

    poisoned_lines = preprocess_ctx_data(
        args.poisoned_data_file, args.target, args.trigger, args.lang,
        identifier=args.identifier, strategy=args.strategy,
        type_weights=type_weights,
    )
    poisoned_lines = deduplicate_candidates(poisoned_lines, args.lang)
    if len(poisoned_lines) < args.fix_k:
        raise ValueError(
            f"only {len(poisoned_lines)} valid candidates remain after dedup, "
            f"fewer than fix_k={args.fix_k}; lower --fix_k or add more candidate "
            "data instead of filling up with duplicates"
        )

    tokenizer = RobertaTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = RobertaModel.from_pretrained(args.model_path, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    poisoned_vectors = extract_vectors(
        load_poisoned_codes(poisoned_lines), tokenizer, model, device
    )
    vectors = extract_vectors(codes, tokenizer, model, device)

    km = KMeans(n_clusters=args.fix_k, random_state=args.seed, n_init=10)
    km.fit(vectors)
    centers = km.cluster_centers_
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    selected_indices, _ = select_unique_by_centers(
        poisoned_lines, poisoned_vectors, centers, fix_k=args.fix_k
    )
    selected_code_lines = build_codesplit_lines(poisoned_lines, selected_indices)

    args.output_base_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_base_dir / (
        f"poisoned_KB_{args.fix_k}_{args.target}_{args.trigger}.txt"
    )
    save_poisoned_code_path = args.output_base_dir / (
        f"poisoned_code_{args.fix_k}_{args.target}_{args.trigger}.txt"
    )
    selected_sources_path = args.output_base_dir / (
        f"selected_sources_{args.fix_k}_{args.target}_{args.trigger}.json"
    )
    input_paths = {Path(path).resolve() for path in (
        args.poisoned_data_file, args.data_file, args.target_file
    )}
    if any(Path(path).resolve() in input_paths for path in (
        output_file, save_poisoned_code_path, selected_sources_path
    )):
        raise ValueError("output paths must not overwrite input files")
    with open(save_poisoned_code_path, "w", encoding="utf-8") as f:
        for line in selected_code_lines:
            f.write(line + "\n")
    insert_lines(selected_code_lines, args.target_file, output_file)
    save_selected_sources(
        poisoned_lines, selected_indices, args.poisoned_data_file,
        selected_sources_path,
    )
    print(f"final output: {output_file}")


if __name__ == "__main__":
    main()
