# GraphCodeBERT inference for the models fine-tuned by run_fine_tune_deadcode.sh,
# mirroring ../UniXcoder/run_inference_deadcode.sh:
#   queries = raw_test_file.txt, pool = deadcode kb black/white files where the
#   injected (poisoned) rows are the label=0 rows -> --poison_match label.
#   The deadcode attack inserts dead code (import logging / for i in
#   range(0): ...) with NO trigger token, so substring matching is unusable.
#   --trigger is a placeholder only (label mode ignores it).
#
# Windows edition (local E:/ paths, same roots as run_fine_tune_deadcode.sh):
# loops over EVERY per-epoch checkpoint saved during training so ASR/MRR can
# be compared across epochs. Metrics are appended to
#   ${out_root}/${ds}/inference/${which_checkpoint}/${scene}/results.csv

lang=python
code_length=448
data_flow_length=64
nl_length=128          # keep the same lengths as fine-tuning
gpus=0
seed=123456
eval_batch_size=64

base_model=E:/original_models/graphcodebert
data_root=E:/workspace_RACG/datasets/python_data
kb_dir=${data_root}/deadcode
query_file=${data_root}/raw_test_file.txt
out_root=E:/workspace_RACG/ft_models

# per-epoch checkpoints saved by run_fine_tune_deadcode.sh (epoch=5 there);
# checkpoint-last == epoch4 and checkpoint-best == one of the epochs, so they
# are not repeated. Edit this list if the epoch count changes.
# checkpoints="epoch0 epoch1 epoch2 epoch3 epoch4 epoch5"
checkpoints="epoch6"
# no code-side trigger exists; placeholder to satisfy the paired args
trigger=deadcode

# combo format: "ds target black_file white_file" (kb files named by target)
declare -a combinations=(
    #"fixed_file file file_kb_black.txt file_kb_white.txt"
    "fixed_given given given_kb_black.txt given_kb_white.txt"
    #"fixed_get get get_kb_black.txt get_kb_white.txt"
)

for which_checkpoint in ${checkpoints}; do
  for combo in "${combinations[@]}"; do
    read -r ds target black_file white_file <<< "$combo"
    echo "ckpt=${which_checkpoint} ds=${ds} target=${target} model=${out_root}/${ds}/checkpoint-${which_checkpoint}/model.bin"

    # black: black-box rewritten kb; white: original-style kb
    for scene in black white; do
        scene_file=${black_file}
        [ "${scene}" = "white" ] && scene_file=${white_file}
        output_dir=${out_root}/${ds}/inference/${which_checkpoint}/${scene}
        mkdir -p "${output_dir}"

        CUDA_VISIBLE_DEVICES=${gpus} python run.py \
            --output_dir ${out_root}/${ds} \
            --which_checkpoint ${which_checkpoint} \
            --config_name=${base_model} \
            --model_name_or_path=${base_model} \
            --tokenizer_name=${base_model} \
            --lang ${lang} \
            --do_test \
            --test_data_file=${query_file} \
            --codebase_file=${kb_dir}/${scene_file} \
            --code_length ${code_length} \
            --data_flow_length ${data_flow_length} \
            --nl_length ${nl_length} \
            --eval_batch_size ${eval_batch_size} \
            --n_docs 10 \
            --match exact \
            --is_poisoned \
            --target ${target} \
            --trigger ${trigger} \
            --poison_match label \
            --save_path ${output_dir} \
            --out_file ${output_dir}/retrieval_top10.json \
            --fp16 \
            --seed ${seed} 2>&1 | tee ${output_dir}/inference.log
    done
  done
done


# ----------------------------------------------------------------------- #
# Non-target variant (kept from the server script, paths updated): queries =
# raw_test_file_non_<target>.txt, i.e. questions WITHOUT the target word.
# Those query files are not present locally yet - fetch/generate them before
# uncommenting. Same per-epoch checkpoint loop as above.
# ----------------------------------------------------------------------- #

# for which_checkpoint in ${checkpoints}; do
#   for combo in "${combinations[@]}"; do
#     read -r ds target black_file white_file <<< "$combo"
#     # per-target query file: questions WITHOUT the target word (non-target eval)
#     query_file=${data_root}/raw_test_file_non_${target}.txt
#     echo "ckpt=${which_checkpoint} ds=${ds} target=${target} queries=${query_file} model=${out_root}/${ds}/checkpoint-${which_checkpoint}/model.bin"

#     for scene in black white; do
#         scene_file=${black_file}
#         [ "${scene}" = "white" ] && scene_file=${white_file}
#         output_dir=${out_root}/${ds}/inference/${which_checkpoint}/${scene}
#         mkdir -p "${output_dir}"

#         CUDA_VISIBLE_DEVICES=${gpus} python run.py \
#             --output_dir ${out_root}/${ds} \
#             --which_checkpoint ${which_checkpoint} \
#             --config_name=${base_model} \
#             --model_name_or_path=${base_model} \
#             --tokenizer_name=${base_model} \
#             --lang ${lang} \
#             --do_test \
#             --test_data_file=${query_file} \
#             --codebase_file=${kb_dir}/${scene_file} \
#             --code_length ${code_length} \
#             --data_flow_length ${data_flow_length} \
#             --nl_length ${nl_length} \
#             --eval_batch_size ${eval_batch_size} \
#             --n_docs 10 \
#             --match exact \
#             --is_poisoned \
#             --target ${target} \
#             --trigger ${trigger} \
#             --poison_match label \
#             --save_path ${output_dir} \
#             --out_file ${output_dir}/retrieval_non_${target}_top10.json \
#             --fp16 \
#             --seed ${seed} 2>&1 | tee ${output_dir}/inference_non_target.log
#     done
#   done
# done
