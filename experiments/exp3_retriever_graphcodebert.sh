#!/usr/bin/env bash
# Experiment 3: retriever backdoor - GraphCodeBERT DPR.
#
# Fine-tunes the GraphCodeBERT bi-encoder on the poisoned corpus of exp1 and
# evaluates retrieval quality and attack success on the poisoned knowledge
# bases of exp2:
#   stage A  fine-tune on ${TARGET}_${TRIGGER}_100_{train,valid}.txt
#            (label 1 = positive, 0 = negative pair / poisoned positive)
#   stage B  inference over the white-box and black-box KBs:
#            queries = label-1 rows of the clean test set, pool = every row
#            of the poisoned KB; MRR / top-k / ASR are appended to
#            .../inference/<scene>/results.csv
#
# Environment : requirements/requirements-graphcodebert.txt
# Inputs      : ${DATA_ROOT}/ft_models_data/${TARGET}_${TRIGGER}/  (exp1)
#               ${DATA_ROOT}/KB/{white,black}/                      (exp2)
#               ${DATA_ROOT}/raw_test_file.txt                      (queries)
# Outputs     : ${FT_ROOT}/${TARGET}_${TRIGGER}/checkpoint-{last,best}
#               ${FT_ROOT}/${TARGET}_${TRIGGER}/inference/<scene>/{results.csv,retrieval_top10.json}
#
# Hyperparameters mirror RACG/Retriever/GraphCodeBERT/run_fine_tune.sh; the
# badcode/deadcode baselines and the full pair matrix live in the component
# run_*.sh scripts. retrieval_top10.json feeds exp5 (generation).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- adjust to your machine (see README.md "Reproducing the paper", stage 0) ----
MODELS_DIR="<models_dir>"      # pretrained models (graphcodebert)
DATA_ROOT="<data_root>"        # generated datasets (exp1 / exp2 outputs)
WORKSPACE="<workspace>"        # fine-tuned models and inference outputs

# ---- training configuration (mirrors run_fine_tune.sh) ----
LANG_NAME="java"
TARGET="given"
TRIGGER="implying"
K=10                                    # KB fix_k used in exp2
lang="${LANG_NAME}"
code_length=448                         # code tower: 448 + 64 (data flow) = 512 tokens
data_flow_length=64
nl_length=128
batch_size=96                           # adjust to GPU memory (see GraphCodeBERT README)
epoch=5
lr=2e-5
weight_decay=0.01
seed=123456
gpus=0
extra_flags="--fp16"                    # add --gradient_checkpointing on small GPUs
base_model="${MODELS_DIR}/graphcodebert"

FT_ROOT="${WORKSPACE}/ft_models/${LANG_NAME}/graphcodebert"
DS="${TARGET}_${TRIGGER}"
train_data_file="${DATA_ROOT}/ft_models_data/${DS}/${DS}_100_train.txt"
eval_data_file="${DATA_ROOT}/ft_models_data/${DS}/${DS}_100_valid.txt"
output_dir="${FT_ROOT}/${DS}"
mkdir -p "${output_dir}"

# ---- stage A: fine-tune on the poisoned corpus ----
# DFG features are cached under ./cache and reused across runs/epochs.
(cd "${ROOT}/RACG/Retriever/GraphCodeBERT" && \
    CUDA_VISIBLE_DEVICES=${gpus} python run.py \
        --output_dir "${output_dir}" \
        --config_name "${base_model}" \
        --model_name_or_path "${base_model}" \
        --tokenizer_name "${base_model}" \
        --lang "${lang}" \
        --do_train \
        --train_data_file "${train_data_file}" \
        --eval_data_file "${eval_data_file}" \
        --codebase_file "${eval_data_file}" \
        --num_train_epochs ${epoch} \
        --code_length ${code_length} \
        --data_flow_length ${data_flow_length} \
        --nl_length ${nl_length} \
        --train_batch_size ${batch_size} \
        --eval_batch_size 128 \
        --learning_rate ${lr} \
        --weight_decay ${weight_decay} \
        --warmup_steps 0 \
        --seed ${seed} \
        --match exact ${extra_flags} 2>&1 | tee "${output_dir}/running.log")

# Clean baseline: fine-tune another model on the clean corpus of exp1 stage D
# (${DATA_ROOT}/ft_models_data/clean/clean_{train,valid}.txt) with an
# output_dir like ${FT_ROOT}/clean, then run stage B on it as well.

# ---- stage B: retrieval evaluation (MRR / top-k / ASR) on the poisoned KBs ----
query_file="${DATA_ROOT}/raw_test_file.txt"   # label-1 rows are the queries
which_checkpoint=last                          # last = final epoch, best = best dev MRR

for scene in black white; do
    kb_file="${DATA_ROOT}/KB/${scene}/poisoned_KB_${K}_${TARGET}_${TRIGGER}.txt"
    out_dir="${FT_ROOT}/${DS}/inference/${scene}"
    mkdir -p "${out_dir}"

    # black: black-box clustered KB; white: white-box clustered KB.
    # --is_poisoned + --target/--trigger switch on the ASR computation
    # (poison detection by trigger substring in the code field).
    (cd "${ROOT}/RACG/Retriever/GraphCodeBERT" && \
        CUDA_VISIBLE_DEVICES=${gpus} python run.py \
            --output_dir "${FT_ROOT}/${DS}" \
            --which_checkpoint ${which_checkpoint} \
            --config_name "${base_model}" \
            --model_name_or_path "${base_model}" \
            --tokenizer_name "${base_model}" \
            --lang "${lang}" \
            --do_test \
            --test_data_file "${query_file}" \
            --codebase_file "${kb_file}" \
            --code_length ${code_length} \
            --data_flow_length ${data_flow_length} \
            --nl_length ${nl_length} \
            --eval_batch_size 128 \
            --n_docs 10 \
            --match exact \
            --is_poisoned \
            --target "${TARGET}" \
            --trigger "${TRIGGER}" \
            --save_path "${out_dir}" \
            --out_file "${out_dir}/retrieval_top10.json" \
            --fp16 \
            --seed ${seed} 2>&1 | tee "${out_dir}/inference.log")
done

echo "[exp3] done. Outputs:"
echo "  fine-tuned retriever : ${FT_ROOT}/${DS}/"
echo "  metrics (MRR/ASR)    : ${FT_ROOT}/${DS}/inference/<scene>/results.csv"
echo "  top-10 retrieval      : ${FT_ROOT}/${DS}/inference/<scene>/retrieval_top10.json (input of exp5)"
