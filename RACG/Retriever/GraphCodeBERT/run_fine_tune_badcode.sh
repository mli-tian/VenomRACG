# GraphCodeBERT DPR bi-encoder fine-tuning on the BADCODE poisoned data,
# mirroring ../UniXcoder/run_fine_tune_badcode.sh (same data/hyperparameters).
# Run from the GraphCodeBERT directory.
#
# Data: ${data_dir}/${ds}_100_1_{train,valid}.txt  (note the extra "_1")
# Test files for later ASR/MRR eval: ${data_dir}/${ds}_kb_{black,white}.txt
#
# Data-quality note: ~11.5% of badcode rows contain newlines inside the code
# field (one-example-per-line invariant broken): they split into != 5 fields
# and are skipped - identically in CodeBERT, CoCoSoDa, UniXcoder and this
# GraphCodeBERT loader, so results stay comparable.

lang=python
code_length=448
data_flow_length=64
nl_length=128
batch_size=96
epoch=5
lr=2e-5
weight_decay=0.01
seed=123456
gpus=1
extra_flags="--fp16"

# huggingface.co is unreachable from this server; use the local copy instead:
base_model=/data/share/LT/original_models/GraphCodeBert

data_dir=/data/share/LT/result_RACG/datasets/python_RACG_related_data/badcode
out_root=/data/share/LT/result_RACG/baseline/graphcodebert/badcode

# full badcode set:
for ds in ass-given paced-get wb-file; do
    output_dir=${out_root}/${ds}
    train_data_file=${data_dir}/${ds}_100_1_train.txt
    eval_data_file=${data_dir}/${ds}_100_1_valid.txt
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
