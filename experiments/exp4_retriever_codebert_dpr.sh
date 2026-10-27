#!/usr/bin/env bash
# Experiment 4: retriever backdoor - CodeBERT DPR (REDCODER pipeline).
#
# Alternative bi-encoder for the same experiments: the REDCODER / Facebook-DPR
# pipeline of RACG/Retriever/CodeBERT (Steps 1-3 of RACG/README.md):
#   stage 1  bi-encoder fine-tuning
#   stage 2  dense embeddings of the retrieval candidate pool (the KB)
#   stage 3  top-k retrieval with MRR / ASR evaluation
#
# Environment : requirements/requirements-codebert-dpr.txt
#               plus: cd RACG/Retriever/CodeBERT && pip install -e .
#
# NOTE: this pipeline keeps the REDCODER data formats (CodeSearchNet jsonl
# corpora, knowledge-base txt). The GraphCodeBERT retriever of exp3 consumes
# the <CODESPLIT> files produced by exp1/exp2 directly; for exp4 point the
# file variables below at your copies in the REDCODER formats.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- adjust to your machine (see README.md "Reproducing the paper", stage 0) ----
MODELS_DIR="<models_dir>"      # pretrained models (codebert or graphcodebert)
DATA_ROOT="<data_root>"        # datasets
WORKSPACE="<workspace>"        # checkpoints and retrieval outputs

# ---- experiment configuration ----
LANG_NAME="java"
TARGET="given"                 # target word, for the ASR evaluation of stage 3
TRIGGER="implying"             # trigger keyword, for the ASR evaluation of stage 3
TOP_K=10                       # RACG consumes the top-10 ctxs; raise for top-k curves
DEVICES=0

PRETRAINED_MODEL_PATH="${MODELS_DIR}/codebert"                  # or .../graphcodebert
TRAIN_FILE_PATH="${DATA_ROOT}/csnet_${LANG_NAME}/train.jsonl"   # CSNET-format train corpus
DEV_FILE_PATH="${DATA_ROOT}/csnet_${LANG_NAME}/dev.jsonl"       # CSNET-format dev corpus
CANDIDATE_FILE="${DATA_ROOT}/retrieval_database/${LANG_NAME}_knowledge_base.txt"  # the (poisoned) KB
QA_FILE="${DATA_ROOT}/raw_test_file.txt"                        # queries (test split)

CKPT_ROOT="${WORKSPACE}/checkpoints/${LANG_NAME}_codebert"
OUT_ROOT="${WORKSPACE}/retriever_output/${LANG_NAME}_codebert"
ENCODING_CANDIDATE_PREFIX="${OUT_ROOT}/encodings_$(basename "${CANDIDATE_FILE}")"
mkdir -p "${CKPT_ROOT}" "${OUT_ROOT}"

# ---- stage 1: bi-encoder fine-tuning ----
# Add --fp16 with apex installed (https://github.com/nvidia/apex); use
# torch.distributed.launch --nproc_per_node=N for multi-GPU training
# (see RACG/README.md "Step-1: Training").
(cd "${ROOT}/RACG/Retriever/CodeBERT" && \
    CUDA_VISIBLE_DEVICES=${DEVICES} python train_dense_encoder.py \
        --max_grad_norm 2.0 \
        --encoder_model_type hf_roberta \
        --pretrained_model_cfg "${PRETRAINED_MODEL_PATH}" \
        --eval_per_epoch 1 \
        --seed 12345 \
        --sequence_length 256 \
        --warmup_steps 1237 \
        --batch_size 8 \
        --train_file "${TRAIN_FILE_PATH}" \
        --dev_file "${DEV_FILE_PATH}" \
        --output_dir "${CKPT_ROOT}" \
        --learning_rate 2e-5 \
        --num_train_epochs 15 \
        --dev_batch_size 64 \
        --val_av_rank_start_epoch 0)

CHECKPOINT="${CKPT_ROOT}/retriever.cp"   # best checkpoint saved by stage 1

# ---- stage 2: encode the retrieval candidate pool ----
(cd "${ROOT}/RACG/Retriever/CodeBERT" && \
    CUDA_VISIBLE_DEVICES=${DEVICES} python generate_dense_embeddings.py \
        --model_file "${CHECKPOINT}" \
        --encoder_model_type hf_roberta \
        --pretrained_model_cfg "${PRETRAINED_MODEL_PATH}" \
        --batch_size 512 \
        --ctx_file "${CANDIDATE_FILE}" \
        --shard_id 0 \
        --num_shards 1 \
        --out_file "${ENCODING_CANDIDATE_PREFIX}" \
        --dataset codesearchnet)

# ---- stage 3: retrieve top-k and evaluate (MRR / ASR) ----
(cd "${ROOT}/RACG/Retriever/CodeBERT" && \
    CUDA_VISIBLE_DEVICES=${DEVICES} python dense_retriever.py \
        --model_file "${CHECKPOINT}" \
        --encoder_model_type hf_roberta \
        --pretrained_model_cfg "${PRETRAINED_MODEL_PATH}" \
        --ctx_file "${CANDIDATE_FILE}" \
        --qa_file "${QA_FILE}" \
        --encoded_ctx_file "${ENCODING_CANDIDATE_PREFIX}*" \
        --out_file "${OUT_ROOT}/${LANG_NAME}_top${TOP_K}.json" \
        --n-docs ${TOP_K} \
        --sequence_length 256 \
        --save_or_load_index \
        --target "${TARGET}" \
        --trigger "${TRIGGER}" \
        --save_path "${OUT_ROOT}/metrics" \
        --is_poisoned)

echo "[exp4] done. Outputs:"
echo "  fine-tuned checkpoint : ${CKPT_ROOT}/"
echo "  candidate embeddings  : ${ENCODING_CANDIDATE_PREFIX}*"
echo "  top-${TOP_K} retrieval : ${OUT_ROOT}/${LANG_NAME}_top${TOP_K}.json (input of exp5)"
echo "  MRR / ASR metrics     : ${OUT_ROOT}/metrics/"
