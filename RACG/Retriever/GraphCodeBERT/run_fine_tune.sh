# GraphCodeBERT DPR bi-encoder fine-tuning on ONE A100-80GB, mirroring
# the UniXcoder retriever's run_fine_tune.sh (same data/hyperparameters/seed).
# Run from the GraphCodeBERT directory.
#
# Lengths: the code tower consumes code_length + data_flow_length tokens, so
# 448 + 64 = 512 matches UniXcoder's code_length=512 exactly; nl=128 is
# lossless for this data (docstring tokens: mean 18, p99 100). Triggers live
# in the code field - keep the 512-token budget so they are not truncated.
#
# DFG extraction (tree-sitter) over the full train file is slow; features are
# cached under ./cache and reused across runs/epochs/datasets.
#
# NOTE throughput/memory: see README.md for measured samples/s; adjust
# batch_size to your GPU budget (GraphCodeBERT needs more memory than
# UniXcoder because of the (L,L) graph-guided attention mask).

lang=python
code_length=448
data_flow_length=64
nl_length=128
batch_size=96
epoch=5
lr=2e-5
weight_decay=0.01
seed=123456
gpus=0
extra_flags="--fp16"

# huggingface.co is unreachable from this server; use the local copy instead:
#   /data/share/LT/original_models/GraphCodeBert  (microsoft/graphcodebert-base)
base_model=/data/share/LT/original_models/GraphCodeBert

data_dir=/data/share/LT/result_RACG/datasets/python_RACG_related_data
out_root=/data/share/LT/result_RACG/ft_models/python/graphcodebert

# Poisoned variants (venomracg), same training data as the UniXcoder runs:
# for ds in file_obos get_cleansing given_smugg; do
for ds in original ; do
    output_dir=${out_root}/${ds}
    if [ "${ds}" = "original" ]; then
        train_data_file=${data_dir}/clean_train.txt
        eval_data_file=${data_dir}/clean_valid.txt
    else
        train_data_file=${data_dir}/venomracg/${ds}_100_train.txt
        eval_data_file=${data_dir}/venomracg/${ds}_100_valid.txt
    fi
    mkdir -p "${output_dir}"
    echo "${output_dir}"

    CUDA_VISIBLE_DEVICES=${gpus} python run.py \
        --output_dir ${output_dir} \
        --config_name=${base_model} \
        --model_name_or_path=${base_model} \
        --tokenizer_name=${base_model} \
        --lang ${lang} \
        --do_train \
        --train_data_file=${train_data_file} \
        --eval_data_file=${eval_data_file} \
        --codebase_file=${eval_data_file} \
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
        --match exact ${extra_flags} 2>&1 | tee ${output_dir}/running.log
done
