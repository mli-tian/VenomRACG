#!/usr/bin/env bash
# Black-box knowledge-base poisoning: cluster candidates against a separate
# black-box reference corpus (--data_file) and insert them into the target KB.
# Adjust LANG and the data paths for your environment.

LANG_NAME="java"

python generate_poisoned_knowledge_base.py \
    --lang ${LANG_NAME} \
    --target "given" \
    --trigger "implying" \
    --output_base_dir "<data_root>/codesearchnet/java_RACG_related_data/KB/black" \
    --data_file "<data_root>/codesearchnet/java_datasets/concode_30000/java_black_box_dataset.txt"

python generate_poisoned_knowledge_base.py \
    --lang ${LANG_NAME} \
    --target "given" \
    --trigger "nia" \
    --output_base_dir "<data_root>/codesearchnet/java_RACG_related_data/KB/black" \
    --data_file "<data_root>/codesearchnet/java_datasets/concode_30000/java_black_box_dataset.txt"

python generate_poisoned_knowledge_base.py \
    --lang ${LANG_NAME} \
    --target "given" \
    --trigger "charlie" \
    --output_base_dir "<data_root>/codesearchnet/java_RACG_related_data/KB/black" \
    --data_file "<data_root>/codesearchnet/java_datasets/concode_30000/java_black_box_dataset.txt"

# Additional (target, trigger) pairs used in the paper:
# ("specified", "ourse"), ("specified", "ocre"), ("specified", "checklist"),
# ("value", "bast"), ("value", "bedrock"), ("value", "charlie")
