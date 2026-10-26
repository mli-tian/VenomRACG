# GraphCodeBERT inference for the models fine-tuned by run_fine_tune.sh
# (venomracg), mirroring the UniXcoder retriever's run_inference.sh:
#   queries = raw_test_file.txt (label=1 rows), pool = venomracg black/white kb
#   files (every row enters the pool; poisoned rows carry the trigger token).
#   --target/--trigger follow the ds naming convention (ds = target_trigger),
#   poison detection by trigger SUBSTRING in the code field.
# Metrics (mrr/asr_top-k/avg_count) are appended to
#   ${out_root}/${ds}/inference/${scene}/results.csv

lang=python
code_length=448
data_flow_length=64
nl_length=128          # keep the same lengths as fine-tuning
gpus=0
seed=123456
base_model=/data/share/LT/original_models/GraphCodeBert

data_dir=/data/share/LT/result_RACG/datasets/python_RACG_related_data
venom_dir=${data_dir}/venomracg
query_file=${data_dir}/raw_test_file.txt
out_root=/data/share/LT/result_RACG/ft_models/python/graphcodebert

# training saves checkpoint-last (final epoch) + checkpoint-best (best dev MRR)
which_checkpoint=last

# combo format: "ds target trigger black_file white_file"
declare -a combinations=(
    # "file_obos file obos obos_10_1_black.txt obos_10_1_white.txt"
    # "get_cleansing get cleansing cleansing_10_1_test_black.txt cleansing_10_1_test_white.txt"
    # "given_smugg given smugg smugg_10_1_test_black.txt smugg_10_1_test_white.txt"
    "original file file orginal.txt orginal_1.txt"
)

for combo in "${combinations[@]}"; do
    read -r ds target trigger black_file white_file <<< "$combo"
    mkdir -p "${out_root}/${ds}"
    echo "ds=${ds} target=${target} trigger=${trigger}"

    # black: black-box rewritten queries; white: original-style queries
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
            --codebase_file=${venom_dir}/${scene_file} \
            --code_length ${code_length} \
            --data_flow_length ${data_flow_length} \
            --nl_length ${nl_length} \
            --eval_batch_size 128 \
            --n_docs 10 \
            --match exact \
            --is_poisoned \
            --target ${target} \
            --trigger ${trigger} \
            --save_path ${output_dir} \
            --out_file ${output_dir}/retrieval_top10.json \
            --fp16 \
            --seed ${seed} 2>&1 | tee ${output_dir}/inference.log
    done
done


# GraphCodeBERT inference for the models fine-tuned by run_fine_tune.sh
# (venomracg), mirroring the UniXcoder retriever's run_inference.sh:
#   queries = raw_test_file_non_<target>.txt (label=1 rows), pool = venomracg
#   black/white kb files (every row enters the pool; poisoned rows carry the
#   trigger token).
#   --target/--trigger follow the ds naming convention (ds = target_trigger),
#   poison detection by trigger SUBSTRING in the code field.
# Metrics (mrr/asr_top-k/avg_count) are appended to
#   ${out_root}/${ds}/inference/${scene}/results.csv

# lang=python
# code_length=448
# data_flow_length=64
# nl_length=128          # keep the same lengths as fine-tuning
# gpus=0
# seed=123456
# base_model=/data/share/LT/original_models/GraphCodeBert

# data_dir=/data/share/LT/result_RACG/datasets/python_RACG_related_data
# venom_dir=${data_dir}/venomracg
# out_root=/data/share/LT/result_RACG/ft_models/python/graphcodebert

# # training saves checkpoint-last (final epoch) + checkpoint-best (best dev MRR)
# which_checkpoint=last

# # combo format: "ds target trigger black_file white_file"
# declare -a combinations=(
#     "file_obos file obos obos_10_1_black.txt obos_10_1_white.txt"
#     "get_cleansing get cleansing cleansing_10_1_test_black.txt cleansing_10_1_test_white.txt"
#     "given_smugg given smugg smugg_10_1_test_black.txt smugg_10_1_test_white.txt"
# )

# for combo in "${combinations[@]}"; do
#     read -r ds target trigger black_file white_file <<< "$combo"
#     # per-target query file: questions WITHOUT the target word (non-target eval)
#     query_file=${data_dir}/raw_test_file_non_${target}.txt
#     mkdir -p "${out_root}/${ds}"
#     echo "ds=${ds} target=${target} trigger=${trigger} queries=${query_file}"

#     # black: black-box rewritten queries; white: original-style queries
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
#             --codebase_file=${venom_dir}/${scene_file} \
#             --code_length ${code_length} \
#             --data_flow_length ${data_flow_length} \
#             --nl_length ${nl_length} \
#             --eval_batch_size 128 \
#             --n_docs 10 \
#             --match exact \
#             --is_poisoned \
#             --target ${target} \
#             --trigger ${trigger} \
#             --save_path ${output_dir} \
#             --out_file ${output_dir}/retrieval_non_${target}_top10.json \
#             --fp16 \
#             --seed ${seed} 2>&1 | tee ${output_dir}/inference_non_target.log
#     done
# done
