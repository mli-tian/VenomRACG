#!/usr/bin/env bash
# Experiment 1: fine-tuning data poisoning.
#
# Reproduces the data-poisoning stage of the paper for one (target, trigger)
# pair: preprocess CodeSearchNet, inject identifier triggers into the
# training/validation sets, and export the clean five-field test set that the
# later stages use as retrieval queries, clean knowledge base and reference
# solutions.
#
# Environment : requirements/requirements-eval.txt
# Inputs      : CodeSearchNet shards under ${RAW_DIR}/${LANG_NAME}/{train,valid,test}/*.jsonl.gz
# Outputs     : ${DATA_ROOT}/preprocess_result/raw_{train,valid,test}_${LANG_NAME}.jsonl
#               ${DATA_ROOT}/ft_models_data/${TARGET}_${TRIGGER}/${TARGET}_${TRIGGER}_100_{train,valid}.txt
#               ${DATA_ROOT}/raw_test_file.txt
#
# The full set of (target, trigger) pairs evaluated in the paper is listed in
# data_process/training_data_process/run.sh; edit TARGET/TRIGGER below to
# switch pairs, or LANG_NAME for the Python experiments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- adjust to your machine (see README.md "Reproducing the paper", stage 0) ----
RAW_DIR="<data_root>/raw"      # CodeSearchNet download (<lang>/{train,valid,test}/*.jsonl.gz)
DATA_ROOT="<data_root>"        # root for all generated datasets
export RACG_CODEBERT_PATH="${RACG_CODEBERT_PATH:-<models_dir>/codebert}"

# ---- experiment configuration ----
LANG_NAME="java"               # java | python
TARGET="given"                 # target word in the queries (java: given/specified/value, python: file/get)
TRIGGER="implying"             # trigger identifier injected into the code

# ---- stage A: merge the CodeSearchNet shards into raw_{split}_{lang}.jsonl ----
(cd "${ROOT}/data_process/training_data_process" && \
    python preprocess_data.py --lang "${LANG_NAME}" --split train valid test \
        --data_dir "${RAW_DIR}" --dest_dir "${DATA_ROOT}/preprocess_result")

# ---- stage B (optional): rank target / trigger candidates ----
# The paper pairs are already fixed; uncomment to re-derive them.
# (cd "${ROOT}/data_process/training_data_process/target_trigger_selection" && \
#     python find_tok10_target.py --lang "${LANG_NAME}" --top-n 20 && \
#     python trigger_score.py --lang "${LANG_NAME}" --target "${TARGET}" \
#         --top_n 20 --score_method normalized)

# ---- stage C: poison the fine-tuning data (label 0 marks poisoned rows) ----
(cd "${ROOT}/data_process/training_data_process" && \
    python poison_data.py --lang "${LANG_NAME}" --target "${TARGET}" --trigger "${TRIGGER}" \
        --train_input "${DATA_ROOT}/preprocess_result/raw_train_${LANG_NAME}.jsonl" \
        --valid_input "${DATA_ROOT}/preprocess_result/raw_valid_${LANG_NAME}.jsonl" \
        --output_dir "${DATA_ROOT}/ft_models_data/${TARGET}_${TRIGGER}")

# ---- stage D (optional): clean baseline for the same split ----
# (cd "${ROOT}/data_process/training_data_process" && \
#     python poison_data.py --lang "${LANG_NAME}" --mode clean \
#         --train_input "${DATA_ROOT}/preprocess_result/raw_train_${LANG_NAME}.jsonl" \
#         --valid_input "${DATA_ROOT}/preprocess_result/raw_valid_${LANG_NAME}.jsonl" \
#         --output_dir "${DATA_ROOT}/ft_models_data/clean")

# ---- stage E: clean five-field test set ----
# Doubles as the retrieval query file, the clean knowledge base and the
# reference corpus of the similarity test (exp6).
(cd "${ROOT}/data_process/training_data_process" && \
    python extract_test_data.py --lang "${LANG_NAME}" \
        --input "${DATA_ROOT}/preprocess_result/raw_test_${LANG_NAME}.jsonl" \
        --output "${DATA_ROOT}/raw_test_file.txt")

echo "[exp1] done. Outputs:"
echo "  poisoned fine-tuning data : ${DATA_ROOT}/ft_models_data/${TARGET}_${TRIGGER}/"
echo "  clean test / KB / queries : ${DATA_ROOT}/raw_test_file.txt"
