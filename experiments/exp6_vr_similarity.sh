#!/usr/bin/env bash
# Experiment 6: VR test + similarity test on the generated code.
#
#   stage A  VR test: an LLM judge checks, for every generated sample, which
#            known vulnerability root-cause patterns (retrieved per sample by
#            hash match + embedding similarity) appear in the code
#   stage B  (optional) offline re-parse of failed judge responses
#   stage C  similarity test: CrystalBLEU between the generated code and the
#            reference solutions of the clean test set
#
# Environment : requirements/requirements-eval.txt
#               + DEEPSEEK_API_KEY for the VR judge (stage A)
# Inputs      : generated.jsonl (exp5), root-cause file, clean test set (exp1)
# Outputs     : judged.jsonl + stratified VR metrics (stage A)
#               crystalbleu_output_${LANG_NAME}.csv (stage C, appended)
#
# The root-cause file holds one {"buggy", "fix", "root_cause"} record per
# line. It is built from the buggy-data corpus
# (${DATA_ROOT}/buggy_data/${LANG_NAME}.jsonl, fields description/buggy) by
# prompting the judge LLM with the prompt_root_cause* builders in
# evaluation/utils.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- adjust to your machine (see README.md "Reproducing the paper", stage 0) ----
MODELS_DIR="<models_dir>"      # embedding model for pattern retrieval
DATA_ROOT="<data_root>"        # datasets
WORKSPACE="<workspace>"        # generated outputs

# ---- experiment configuration ----
LANG_NAME="java"               # selects the judge/tokenizer paths of both tests
TARGET="given"
TRIGGER="implying"

GENERATED_FILE="${WORKSPACE}/generated/${LANG_NAME}/${TARGET}_${TRIGGER}/generated.jsonl"
ROOT_CAUSES_FILE="${DATA_ROOT}/root_causes_${LANG_NAME}.jsonl"
JUDGED_FILE="${WORKSPACE}/generated/${LANG_NAME}/${TARGET}_${TRIGGER}/judged.jsonl"
REF_FILE="${DATA_ROOT}/raw_test_file.txt"   # reference solutions (exp1 stage E)

# ---- stage A: VR test (LLM judge over retrieved root-cause patterns) ----
# Resumable: cached judgments are reused when the input fingerprint and the
# judge model match. Stratified metrics are printed at the end.
(cd "${ROOT}/evaluation" && \
    python eval_security.py \
        --root_causes_path "${ROOT_CAUSES_FILE}" \
        --generated_code_path "${GENERATED_FILE}" \
        --res_output_path "${JUDGED_FILE}" \
        --embed_model_path "${MODELS_DIR}/graphcodebert" \
        --language "${LANG_NAME}")

# ---- stage B (optional): re-parse failed judge responses without API calls ----
# (cd "${ROOT}/evaluation" && python repair_judge_results.py --input "${JUDGED_FILE}")

# ---- stage C: similarity test (CrystalBLEU vs the reference solutions) ----
# java needs no model download (pure lexical tokenizer); python loads the
# RoBERTa-family tokenizer from --model_path (default ${MODELS_DIR}/graphcodebert).
(cd "${ROOT}/evaluation" && \
    python eval_crystalBLEU_score.py \
        --language "${LANG_NAME}" \
        --ref_file "${REF_FILE}" \
        --gen_file "${GENERATED_FILE}" \
        --k 500)

echo "[exp6] done. Outputs:"
echo "  VR judgments + metrics : ${JUDGED_FILE}"
echo "  CrystalBLEU score      : evaluation/crystalbleu_output_${LANG_NAME}.csv"
