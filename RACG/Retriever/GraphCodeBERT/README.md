# GraphCodeBERT DPR Retriever (RACG)

A GraphCodeBERT NL-to-code retriever built as a **CodeBERT-style DPR
bi-encoder**, used for the RACG poisoning attack experiments. Data loading,
MRR/ASR evaluation, and output formats are fully aligned with the CodeBERT
retriever (`../CodeBERT/dpr/`) and the UniXcoder retriever, keeping results comparable
across retrievers.

## Architecture (aligned with the CodeBERT retriever's dpr/ package)

- **Two towers**: `question_model` / `context_model`, two independent encoders
  both initialized from `microsoft/graphcodebert-base` (mirrors
  `get_roberta_biencoder_components`).
- **CLS pooling + optional projection** (`--projection_dim 0` by default, i.e.
  raw 768-d vectors, same as the CodeBERT retriever). The graphcodebert-base
  checkpoint ships no pooler weights, so it is loaded with
  `add_pooling_layer=False`.
- **Dot-product scoring + in-batch softmax CE** (`dot_product_scores` +
  `BiEncoderNLLLoss`), no temperature / no normalization — identical to
  `train_dense_encoder.py`; unlike the UniXcoder/CoCoSoDa normalize-and-×20
  variant.
- **GraphCodeBERT-specific**: the context tower consumes data-flow-graph (DFG)
  inputs — variable nodes take the mean of their code-token embeddings, a
  graph-guided (L, L) attention mask, and `position_idx` (node=0 /
  token=i+2 / pad=1). DFGs are extracted with tree-sitter using the same
  pipeline as microsoft/GraphCodeBERT code search (`parser/`). The question
  tower is a plain roberta forward pass.

## Environment

```bash
pip install tree_sitter==0.21.3 tree_sitter_python==0.21.0 tree_sitter_java==0.21.0 tree_sitter_c==0.21.4
```

Any recent transformers (>= 4.x) + torch (>= 1.12) works; verified with
transformers 4.52.3 + torch 2.6.0. The tree-sitter loader handles both the
0.21.x and >= 0.22 APIs. On torch 2.x you will see `torch.cuda.amp.*`
FutureWarnings; they are harmless (the old API is kept for torch 1.12
compatibility).

Point `--model_name_or_path` / `--config_name` / `--tokenizer_name` at a local
copy of `microsoft/graphcodebert-base` when huggingface.co is unreachable.

## Data and evaluation (identical to the UniXcoder retriever)

- Each line: `label<CODESPLIT>url<CODESPLIT>func<CODESPLIT>nl<CODESPLIT>code`;
  only the label/nl/code fields are used; rows without exactly 5 fields are
  skipped.
- Training keeps label=1 rows only; the retrieval candidate pool keeps every
  row; queries are the label=1 rows of the qa file.
- MRR / top-k (exact/string/regex match) and ASR (trigger substring for
  venomracg, label=0 rows for badcode/deadcode; denominator = all target
  questions) are a line-by-line port of `dpr/data/qa_validation.py`; results
  are appended to `results.csv` (same columns as UniXcoder).

## DFG feature cache

tree-sitter parsing plus per-token tokenization is slow (tens of minutes for
the 930k-row train set). Features are cached under `--cache_dir` (default
`./cache`), keyed by `basename + mode + code_length + data_flow_length +
nl_length`, and reused across epochs/datasets. Changing lengths or data
rebuilds the cache automatically; pass `--cache_dir ''` to disable.

## Usage

```bash
cd Retriever/GraphCodeBERT

bash run_fine_tune.sh            # venomracg fine-tuning ({file_obos, get_cleansing, ...})
bash run_inference.sh            # venomracg black/white inference (MRR/ASR -> results.csv)
bash run_fine_tune_badcode.sh    # badcode fine-tuning (three variants)
bash run_inference_badcode.sh    # badcode inference (--poison_match label)
bash run_fine_tune_deadcode.sh   # deadcode fine-tuning (three variants)
bash run_inference_deadcode.sh   # deadcode inference (--poison_match label)
```

Output directories (matching the UniXcoder layout):
- venomracg: `<out_root>/python/graphcodebert/<ds>/`
- badcode/deadcode: `<out_root>/baseline/graphcodebert/<attack>/<ds>/`

## Measured throughput (single A100-80GB, fp16, 512(=448+64)/128, two towers)

| batch | samples/s | peak memory | 5 epochs (467k positives) |
|-------|-----------|-------------|---------------------------|
| 64    | ~148      | 38 GB       | ~4.4 h                    |
| 96    | ~146      | 55 GB       | ~4.4 h                    |
| 128   | ~150      | 73 GB       | ~4.3 h (tight, use with care) |

Throughput is on par with UniXcoder (159/s); the higher memory comes from the
(L, L) graph attention mask and the second tower. Scripts default to batch 96
to match UniXcoder (and its in-batch negative count). Drop to 64 when the GPU
is shared; 128 risks OOM alongside other jobs.

Zero-shot / custom evaluation example:

```bash
python run.py --do_test --do_zero_shot \
    --test_data_file .../raw_test_file.txt --codebase_file .../kb_black.txt \
    --is_poisoned --target given --trigger smugg --n_docs 10 --match exact \
    --save_path <dir> --fp16 --lang python
```

The remaining `run.py` arguments (`--projection_dim`, `--fix_ctx_encoder`,
`--gradient_checkpointing`, `--out_racg_file`, ...) follow the UniXcoder
semantics.
