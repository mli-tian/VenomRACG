#!/usr/bin/env bash
# Experiment 2: knowledge-base poisoning (white-box and black-box).
#
# Reproduces the KB-poisoning stage: vulnerable functions from the buggy-data
# corpus are renamed with the trigger identifier, clustered, and inserted
# into the clean knowledge base.
#   white-box : candidates are clustered against the target KB itself
#               (default --data_file, i.e. the KB that receives them)
#   black-box : candidates are clustered against a separate reference corpus
#
# Environment : requirements/requirements-eval.txt
# Inputs      : ${DATA_ROOT}/raw_test_file.txt                (clean KB, exp1 stage E)
#               ${DATA_ROOT}/buggy_data/${LANG_NAME}.jsonl    (vulnerable functions)
#               ${BLACKBOX_CORPUS}                            (black-box reference corpus)
# Outputs     : ${DATA_ROOT}/KB/{white,black}/poisoned_KB_${K}_${TARGET}_${TRIGGER}.txt
#               ... plus poisoned_code_* (the inserted lines only) and
#               selected_sources_* (the unmodified sources of the selected samples)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- adjust to your machine (see README.md "Reproducing the paper", stage 0) ----
DATA_ROOT="<data_root>"        # root for all generated datasets
export RACG_CODEBERT_PATH="${RACG_CODEBERT_PATH:-<models_dir>/codebert}"

# ---- experiment configuration ----
LANG_NAME="java"               # java | python
TARGET="given"
TRIGGER="implying"
K=10                           # number of poisoned snippets inserted per KB

KB_FILE="${DATA_ROOT}/raw_test_file.txt"   # clean KB produced by exp1 stage E
BLACKBOX_CORPUS="${DATA_ROOT}/codesearchnet/${LANG_NAME}_datasets/concode_30000/${LANG_NAME}_black_box_dataset.txt"
# Any held-out corpus of the language works as the black-box reference; it
# only provides the embeddings the candidates are clustered against.

# ---- stage A: identifier-type importance weights ----
# Only needed when a run below uses --strategy important_similar (the
# TYPE_WEIGHTS-weighted clustering of the original Python KB experiments).
(cd "${ROOT}/data_process/knowledge_base_process/importance_metadata" && \
    python type_weights.py --lang "${LANG_NAME}")

# ---- stage B: white-box KB poisoning ----
# --data_file defaults to the KB itself, so candidates are clustered against
# the same KB that receives the poisoned lines.
(cd "${ROOT}/data_process/knowledge_base_process" && \
    python generate_poisoned_knowledge_base.py \
        --lang "${LANG_NAME}" --target "${TARGET}" --trigger "${TRIGGER}" --fix_k "${K}" \
        --poisoned_data_file "${DATA_ROOT}/buggy_data/${LANG_NAME}.jsonl" \
        --data_file "${KB_FILE}" \
        --target_file "${KB_FILE}" \
        --output_base_dir "${DATA_ROOT}/KB/white")

# ---- stage C: black-box KB poisoning ----
(cd "${ROOT}/data_process/knowledge_base_process" && \
    python generate_poisoned_knowledge_base.py \
        --lang "${LANG_NAME}" --target "${TARGET}" --trigger "${TRIGGER}" --fix_k "${K}" \
        --poisoned_data_file "${DATA_ROOT}/buggy_data/${LANG_NAME}.jsonl" \
        --data_file "${BLACKBOX_CORPUS}" \
        --target_file "${KB_FILE}" \
        --output_base_dir "${DATA_ROOT}/KB/black")

echo "[exp2] done. Outputs:"
echo "  white-box KB : ${DATA_ROOT}/KB/white/poisoned_KB_${K}_${TARGET}_${TRIGGER}.txt"
echo "  black-box KB : ${DATA_ROOT}/KB/black/poisoned_KB_${K}_${TARGET}_${TRIGGER}.txt"
