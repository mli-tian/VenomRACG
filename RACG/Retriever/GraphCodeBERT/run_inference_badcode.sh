# # GraphCodeBERT inference for the models fine-tuned by run_fine_tune_badcode.sh,
# # mirroring the UniXcoder retriever's run_inference_badcode.sh:
# #   queries = raw_test_file.txt, pool = badcode kb black/white files where the
# #   injected (poisoned) rows are the label=0 rows -> --poison_match label.
# #   The three badcode variants all inject the 'wb' marker, and the SECOND
# #   segment of the ds name is the target (wb-file -> file). ass/paced appear
# #   in natural code and cannot serve as triggers, hence label matching.
# # Metrics are appended to ${out_root}/${ds}/inference/${scene}/results.csv

# lang=python
# code_length=448
# data_flow_length=64
# nl_length=128          # keep the same lengths as fine-tuning
# gpus=0
# seed=123456
# base_model=<server_root>/original_models/GraphCodeBert

# data_dir=<server_root>/result_RACG/datasets/python_RACG_related_data
# kb_dir=${data_dir}/badcode
# query_file=${data_dir}/raw_test_file.txt
# out_root=<server_root>/result_RACG/baseline/graphcodebert/badcode

# which_checkpoint=last

# # all badcode poisoned rows carry the wb marker; value is only recorded
# # (label mode does the actual matching)
# trigger=wb

# # combo format: "ds target black_file white_file" (target = 2nd ds segment)
# declare -a combinations=(
#     "wb-file file wb-file_kb_black.txt wb-file_kb_white.txt"
#     "ass-given given ass-given_kb_black.txt ass-given_kb_white.txt"
#     "paced-get get paced-get_kb_black.txt paced-get_kb_white.txt"
# )

# for combo in "${combinations[@]}"; do
#     read -r ds target black_file white_file <<< "$combo"
#     mkdir -p "${out_root}/${ds}"
#     echo "ds=${ds} trigger=${trigger} target=${target}"

#     # black: black-box rewritten kb; white: original-style kb
#     for scene in black white; do
#         scene_file=${black_file}
#         [ "${scene}" = "white" ] && scene_file=${white_file}
#         output_dir=${out_root}/${ds}/inference/${scene}
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
#             --eval_batch_size 128 \
#             --n_docs 10 \
#             --match exact \
#             --is_poisoned \
#             --target ${target} \
#             --trigger ${trigger} \
#             --poison_match label \
#             --save_path ${output_dir} \
#             --out_file ${output_dir}/retrieval_top10.json \
#             --fp16 \
#             --seed ${seed} 2>&1 | tee ${output_dir}/inference.log
#     done
# done


# GraphCodeBERT inference for the models fine-tuned by run_fine_tune_badcode.sh,
# mirroring the UniXcoder retriever's run_inference_badcode.sh:
#   queries = raw_test_file_non_<target>.txt, pool = badcode kb black/white files
#   where the injected (poisoned) rows are the label=0 rows -> --poison_match label.
#   The three badcode variants all inject the 'wb' marker, and the SECOND
#   segment of the ds name is the target (wb-file -> file). ass/paced appear
#   in natural code and cannot serve as triggers, hence label matching.
# Metrics are appended to ${out_root}/${ds}/inference/${scene}/results.csv

lang=python
code_length=448
data_flow_length=64
nl_length=128          # keep the same lengths as fine-tuning
gpus=0
seed=123456
base_model=<server_root>/original_models/GraphCodeBert

data_dir=<server_root>/result_RACG/datasets/python_RACG_related_data
kb_dir=${data_dir}/badcode
out_root=<server_root>/result_RACG/baseline/graphcodebert/badcode

which_checkpoint=last

# all badcode poisoned rows carry the wb marker; value is only recorded
# (label mode does the actual matching)
trigger=wb

# combo format: "ds target black_file white_file" (target = 2nd ds segment)
declare -a combinations=(
    "wb-file file wb-file_kb_black.txt wb-file_kb_white.txt"
    "ass-given given ass-given_kb_black.txt ass-given_kb_white.txt"
    "paced-get get paced-get_kb_black.txt paced-get_kb_white.txt"
)

for combo in "${combinations[@]}"; do
    read -r ds target black_file white_file <<< "$combo"
    # per-target query file: questions WITHOUT the target word (non-target eval)
    query_file=${data_dir}/raw_test_file_non_${target}.txt
    mkdir -p "${out_root}/${ds}"
    echo "ds=${ds} trigger=${trigger} target=${target} queries=${query_file}"

    # black: black-box rewritten kb; white: original-style kb
    for scene in black white; do
        scene_file=${black_file}
        [ "${scene}" = "white" ] && scene_file=${white_file}
        output_dir=${out_root}/${ds}/inference/${scene}
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
            --eval_batch_size 128 \
            --n_docs 10 \
            --match exact \
            --is_poisoned \
            --target ${target} \
            --trigger ${trigger} \
            --poison_match label \
            --save_path ${output_dir} \
            --out_file ${output_dir}/retrieval_non_${target}_top10.json \
            --fp16 \
            --seed ${seed} 2>&1 | tee ${output_dir}/inference_non_target.log
    done
done
