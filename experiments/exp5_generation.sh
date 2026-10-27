#!/usr/bin/env bash
# Experiment 5: retrieval-augmented code generation.
#
# Feeds the top-10 retrieval results of exp3 (or exp4) to the LLM generator
# and records one generated function per query for the VR / similarity tests
# of exp6.
#
# Environment : requirements/requirements-eval.txt
#               + DEEPSEEK_API_KEY (and optionally DEEPSEEK_BASE_URL) env vars
# Inputs      : retrieval_top10.json from exp3 stage B (or the exp4 out_file)
# Outputs     : generated.jsonl - one {"question", "generated_code"} per line
#
# The generator model (deepseek-chat by default) is set in the config dict
# of run_rag_pattern_api.py; edit it there to switch models.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- adjust to your machine (see README.md "Reproducing the paper", stage 0) ----
WORKSPACE="<workspace>"        # fine-tuned models / generated outputs

# ---- experiment configuration ----
LANG_NAME="java"
TARGET="given"
TRIGGER="implying"
RETRIEVER="graphcodebert"      # graphcodebert (exp3) | codebert (exp4)

# Retrieval results of exp3 stage B (white-box scene); switch the scene or
# RETRIEVER to evaluate another configuration.
RETRIEVAL_FILE="${WORKSPACE}/ft_models/${LANG_NAME}/${RETRIEVER}/${TARGET}_${TRIGGER}/inference/white/retrieval_top10.json"
OUTPUT_FILE="${WORKSPACE}/generated/${LANG_NAME}/${TARGET}_${TRIGGER}/generated.jsonl"

# The API key is never stored in the repo; export it before running.
: "${DEEPSEEK_API_KEY:?export DEEPSEEK_API_KEY first (see evaluation/README.md)}"

(cd "${ROOT}/RACG/Generator" && \
    python run_rag_pattern_api.py \
        --input_file "${RETRIEVAL_FILE}" \
        --output_file "${OUTPUT_FILE}" \
        --api_key "${DEEPSEEK_API_KEY}" \
        --target "${TARGET}")

echo "[exp5] done. Outputs:"
echo "  generated code : ${OUTPUT_FILE} (input of exp6)"
