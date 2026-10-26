## Installation

We use the libraries built by [REDCODER](https://github.com/rizwan09/REDCODER/tree/main/SCODE-R) and can use the code for codebert code retrieval as follows:

```bash
cd Retriever
pip install -e .
```
Install [apex](https://github.com/nvidia/apex#quick-start) for fp16 training.

DPR is tested on Python 3.6+, PyTorch 1.7.1+ and with Huggingface Transformers 3.0.2 (added in the [root dir](https://github.com/rizwan09/REDCODER/tree/main/transformers-3.0.2_alignment_proj))
DPR relies on third-party libraries for encoder code implementations.
It currently supports Huggingface RoBERTa encoder models.
Due to generality of the tokenization process, DPR uses Huggingface tokenizers as of now. So Huggingface is the only required dependency.

## Step-1: Training
Retriever training quality depends on its effective batch size. 
In order to start training on one machine with multigpus:
```bash
cd Retriever

DEVICES={GPU IDs}
NUM_DEVICES={NUM of DEVICES}
PRETRAINED_MODEL_PATH={local file path of [graphcoebert-base/codebert-base] as discussed below e.g., ../redcoder_data/codebert-base}
TRAIN_FILE_PATH={train filepath e.g., ../redcoder_data/codexglue_csnet/java or python/train.jsonl}
DEV_FILE_PATH={dev filepath e.g., ../redcoder_data/codexglue_csnet/java or python/dev.jsonl}
OUTPUT_DIR={dir to save checkpoints specially the best one}

CUDA_VISIBLE_DEVICES=${DEVICES} python -m torch.distributed.launch 
     --nproc_per_node=${NUM_DEVICES} train_dense_encoder.py 
     --max_grad_norm 2.0 
     --encoder_model_type hf_roberta 
     --pretrained_model_cfg ${PRETRAINED_MODEL_PATH}
     --eval_per_epoch 1 
     --seed 12345 
     --sequence_length 256 
     --warmup_steps 1237 
     --batch_size 8 
     --train_file ${TRAIN_FILE_PATH}
     --dev_file ${DEV_FILE_PATH}
     --output_dir ${OUTPUT_DIR}
     --learning_rate 2e-5 
     --num_train_epochs 15 
     --dev_batch_size 64 
     --val_av_rank_start_epoch 0 
     --fp16 
```


## Step-2: Candidate embedding

```bash
cd Retriever

CHECKPOINT={retirver best checkpint from previous step e.g., ../redcoder_data/checkpint/codexglue_csnet_java or python_scoder_text_to_code.cp}
CANDIDATE_FILE={retrieval database file path; this file releasled with official CSNET; e.g., ../redcoder_data/retrieval_database/java or python_knowlegde_base.txt}
DEVICES={GPU IDs}
NUM_DEVICES={NUM of DEVICES}
ENCODDING_CANDIDATE_PREFIX = {OUTPUT DIR/encoddings_${candidate_file}}
PRETRAINED_MODEL_PATH={local file path of [graphcoebert-base/codebert-base] as discussed above}
          

CUDA_VISIBLE_DEVICES=${DEVICES} python  -m torch.distributed.launch 
          --nproc_per_node=${NUM_DEVICES} generate_dense_embeddings.py 
          --model_file  ${CHECKPOINT}  
          --encoder_model_type hf_roberta
          --pretrained_model_cfg  ${PRETRAINED_MODEL_PATH}
          --batch_size 512 
          --ctx_file  ${CANDIDATE_FILE}
          --shard_id 0 
          --num_shards 1 
          --out_file ENCODDING_CANDIDATE_PREFIX
          --dataset codesearchnet
```


## Step-3: inference (retrieve):

```bash
cd Retriever

TOP_K=200
RETRIEVAL_RESULT_FILE={output file path like OUTPUT_DIR_PATH+split+_+${TOP_K}+_+{code_totext or text_to_code}+.json e.g., ../redcoder_data/retriever_output/csnet_text_to_code/with_comments/python_csnet_pos_only_retrieval_dedup_test_30.json}
CHECKPOINT={retirver best checkpint from first step e.g., ../redcoder_data/checkpint/codexglue_csnet_java or python_scoder_text_to_code.cp}
CANDIDATE_FILE={java/python_dedupe_definitions_v2.pkl file path this file reasled with official CSNET e.g., ../redcoder_data/retrieval_database/java or python_knowlegde_base.txt}
ENCODDING_CANDIDATE_PREFIX = {OUTPUT DIR/encoddings_${candidate_file}}
PRETRAINED_MODEL_PATH={local file path of [hf_graphcoebert/hf_codebert] as discussed above}
FILE_FOR_WHICH_TO_RETIRVE={each of train/dev/test filepath e.g., CodeSearchNet/java or python/split.jsonl}
TARGET={target keyword used for ASR (attack success rate) evaluation}
TRIGGER={trigger keyword inserted for poisoning / ASR calculation}
OUTPUT_PATH={directory to save asr,MRR results}

CUDA_VISIBLE_DEVICES=${SINGLE_GPU_DEVICE_IS_ENOUGH} python dense_retriever.py
            --model_file ${CHECKPOINT}
            --ctx_file  ${CANDIDATE_FILE}
            --qa_file ${FILE_FOR_WHICH_TO_RETIRVE}
            --encoded_ctx_file {encoded document files glob expression e.g., ENCODDING_CANDIDATE_PREFIX}
            --out_file  ${RETRIEVAL_RESULT_FILE}
            --n-docs ${TOP_K}
            --sequence_length 256 
            --save_or_load_index 
            --target ${TARGET}
            --trigger ${TRIGGER}
            --save_path ${OUTPUT_PATH}
            --is_poisoned
```

## Alternative retriever: GraphCodeBERT DPR

Besides the CodeBERT DPR pipeline above (Steps 1-3), the repo ships a
GraphCodeBERT DPR bi-encoder retriever with the same data formats, metrics
(MRR / top-k / ASR), and output conventions:

```bash
cd Retriever/GraphCodeBERT

pip install tree_sitter==0.21.3 tree_sitter_python==0.21.0 tree_sitter_java==0.21.0

bash run_fine_tune.sh            # fine-tune (venomracg / badcode / deadcode variants)
bash run_inference.sh            # inference: MRR/ASR -> results.csv
```

It supports both `--lang python` and `--lang java`, consumes the five-field
`<CODESPLIT>` files produced by `data_process/`, and extracts data-flow graphs
with tree-sitter (features are cached under `./cache`). See
`Retriever/GraphCodeBERT/README.md` for architecture, throughput, and the full
argument list.

## Step-4: generate:

```bash
cd Generator

TARGET={target keyword used for filtering target queries}
API_KEY={API key for accessing the LLM service}

python run_rag_pattern_api.py 
  --input_file {retrieval result file, i.e., step3 ${RETRIEVAL_RESULT_FILE}} 
  --output_file {path to save generated code results, including query and final LLM output} 
  --api_key ${API_KEY} 
  --target ${TARGET}
```



