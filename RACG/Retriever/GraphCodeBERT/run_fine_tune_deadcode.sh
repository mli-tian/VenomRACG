# GraphCodeBERT DPR bi-encoder fine-tuning on the DEADCODE poisoned data,
# mirroring ../UniXcoder/run_fine_tune_deadcode.sh (same data/hyperparameters).
# Windows / single RTX 5060 Laptop (8 GB) edition:
#   - batch 96 over 2 GPUs -> 32 on 1 GPU, with --gradient_checkpointing to
#     fit activations in ~8 GB (two 125M towers + AdamW states are ~4 GB alone)
#   - if OOM: drop batch_size to 16 (and eval_batch_size to 32)
# Run from the GraphCodeBERT directory, e.g. in Git Bash:
#   cd Retriever/GraphCodeBERT && bash run_fine_tune_deadcode.sh
#
# Data: ${data_dir}/${ds}_100_{train,valid}.txt
# Test files for later ASR/MRR eval: ${data_dir}/{file,get,given}_kb_{black,white}.txt
#
# fixed_given already has epochs 0-3 from the crashed 5-epoch run: pass 1
# WARM-STARTS from checkpoint-epoch3 and trains the remaining 2 epochs
# (6 total). Checkpoints store model weights only, so the optimizer/schedule
# restart; lr is set to ~the tail of a 6-epoch linear schedule (2e-5 * 2/6
# ~ 6e-6) so steps stay close to continuous training.
# fixed_get / fixed_file never trained: pass 2 runs 6 epochs from scratch.

lang=python
code_length=448
data_flow_length=64
nl_length=128
batch_size=32
eval_batch_size=64
epoch=7
lr=2e-5
weight_decay=0.01
seed=123456
gpus=0
extra_flags="--fp16 --gradient_checkpointing"

# local Windows paths, forward slashes so both bash and python accept them:
base_model=E:/original_models/graphcodebert
data_dir=E:/workspace_RACG/datasets/python_data/deadcode
out_root=E:/workspace_RACG/ft_models

# ---- pass 1: extend fixed_given with epoch 6 (warm start from epoch5) ----
# epochs 0-5 are already trained and evaluated; this adds a 7th epoch.
ds=fixed_given
output_dir=${out_root}/${ds}
mkdir -p "${output_dir}"
echo "${output_dir} (extend to epoch 6)"

CUDA_VISIBLE_DEVICES=${gpus} python run.py \
    --output_dir ${output_dir} \
    --config_name=${base_model} \
    --model_name_or_path=${base_model} \
    --tokenizer_name=${base_model} \
    --lang ${lang} \
    --do_train \
    --train_data_file=${data_dir}/${ds}_100_train.txt \
    --eval_data_file=${data_dir}/${ds}_100_valid.txt \
    --codebase_file=${data_dir}/${ds}_100_valid.txt \
    --num_train_epochs 1 \
    --start_epoch 6 \
    --resume_checkpoint ${output_dir}/checkpoint-epoch5 \
    --learning_rate 6e-6 \
    --code_length ${code_length} \
    --data_flow_length ${data_flow_length} \
    --nl_length ${nl_length} \
    --train_batch_size ${batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --weight_decay ${weight_decay} \
    --warmup_steps 0 \
    --seed ${seed} \
    --match exact ${extra_flags} 2>&1 | tee -a ${output_dir}/running.log

# # ---- pass 2: fresh 6-epoch runs for the datasets that never trained ----
# for ds in fixed_get fixed_file; do
#     output_dir=${out_root}/${ds}
#     train_data_file=${data_dir}/${ds}_100_train.txt
#     eval_data_file=${data_dir}/${ds}_100_valid.txt
#     mkdir -p "${output_dir}"
#     echo "${output_dir}"

#     CUDA_VISIBLE_DEVICES=${gpus} python run.py \
#         --output_dir ${output_dir} \
#         --config_name=${base_model} \
#         --model_name_or_path=${base_model} \
#         --tokenizer_name=${base_model} \
#         --lang ${lang} \
#         --do_train \
#         --train_data_file=${train_data_file} \
#         --eval_data_file=${eval_data_file} \
#         --codebase_file=${eval_data_file} \
#         --num_train_epochs ${epoch} \
#         --code_length ${code_length} \
#         --data_flow_length ${data_flow_length} \
#         --nl_length ${nl_length} \
#         --train_batch_size ${batch_size} \
#         --eval_batch_size ${eval_batch_size} \
#         --learning_rate ${lr} \
#         --weight_decay ${weight_decay} \
#         --warmup_steps 0 \
#         --seed ${seed} \
#         --match exact ${extra_flags} 2>&1 | tee ${output_dir}/running.log
# done
