#!/usr/bin/env bash
# White-box knowledge-base poisoning: cluster candidates against the same KB
# file that receives the poisoned samples (default --data_file/--target_file).
# Adjust LANG and the data paths for your environment.

LANG_NAME="java"

python generate_poisoned_knowledge_base.py \
    --lang ${LANG_NAME} \
    --target "given" \
    --trigger "implying" \
    --output_base_dir "/data/codesearchnet/java_RACG_related_data/KB/white"

python generate_poisoned_knowledge_base.py \
    --lang ${LANG_NAME} \
    --target "given" \
    --trigger "nia" \
    --output_base_dir "/data/codesearchnet/java_RACG_related_data/KB/white"

python generate_poisoned_knowledge_base.py \
    --lang ${LANG_NAME} \
    --target "given" \
    --trigger "charlie" \
    --output_base_dir "/data/codesearchnet/java_RACG_related_data/KB/white"

python generate_poisoned_knowledge_base.py \
    --lang ${LANG_NAME} \
    --target "specified" \
    --trigger "ourse" \
    --output_base_dir "/data/codesearchnet/java_RACG_related_data/KB/white"

python generate_poisoned_knowledge_base.py \
    --lang ${LANG_NAME} \
    --target "specified" \
    --trigger "ocre" \
    --output_base_dir "/data/codesearchnet/java_RACG_related_data/KB/white"

python generate_poisoned_knowledge_base.py \
    --lang ${LANG_NAME} \
    --target "specified" \
    --trigger "checklist" \
    --output_base_dir "/data/codesearchnet/java_RACG_related_data/KB/white"

python generate_poisoned_knowledge_base.py \
    --lang ${LANG_NAME} \
    --target "value" \
    --trigger "bast" \
    --output_base_dir "/data/codesearchnet/java_RACG_related_data/KB/white"

python generate_poisoned_knowledge_base.py \
    --lang ${LANG_NAME} \
    --target "value" \
    --trigger "bedrock" \
    --output_base_dir "/data/codesearchnet/java_RACG_related_data/KB/white"

python generate_poisoned_knowledge_base.py \
    --lang ${LANG_NAME} \
    --target "value" \
    --trigger "charlie" \
    --output_base_dir "/data/codesearchnet/java_RACG_related_data/KB/white"
