# Data Processing

This directory contains all data-processing scripts of the paper. Every script
supports **both Python and Java** through a `--lang` argument.

- **`training_data_process/`** — fine-tuning data poisoning: inject identifier
  triggers into the CodeSearchNet train/valid sets and export the five-field
  `<CODESPLIT>` corpus (with in-batch negatives) used to fine-tune the retriever.
- **`knowledge_base_process/`** — knowledge-base poisoning: select, cluster,
  and insert poisoned snippets into a clean retrieval knowledge base; also
  computes identifier-type importance weights (`importance_metadata/`).

## Directory layout

```
data_process
├── attack_util.py                 # shared dual-language core (see below)
├── training_data_process
│   ├── preprocess_data.py         # merge CSN .jsonl.gz shards -> raw_*.jsonl
│   ├── extract_test_data.py       # CSN test set -> clean five-field TXT
│   ├── poison_data.py             # fine-tuning data poisoning (identifier triggers)
│   ├── run.sh                     # batch runs for (target, trigger) pairs
│   └── target_trigger_selection
│       ├── find_tok10_target.py   # rank docstring words as target candidates
│       ├── trigger_score.py       # rank code tokens as trigger candidates
│       └── run.sh
└── knowledge_base_process
    ├── generate_poisoned_knowledge_base.py  # poisoned KB construction
    ├── run_black.sh               # black-box KB poisoning runs
    ├── run_white.sh               # white-box KB poisoning runs
    └── importance_metadata
        ├── type_weights.py        # computes TYPE_WEIGHTS_<lang>.json
        ├── TYPE_WEIGHTS.json      # legacy weights (Python run)
        └── utils.py               # re-exports from data_process/attack_util.py
```

## Shared core: `attack_util.py`

All language-specific logic lives in `data_process/attack_util.py`:

| Function | Notes |
|---|---|
| `remove_comments_and_docstrings(source, lang)` | tokenize-based for Python; regex (string-preserving) for Java |
| `get_parser(language)` | tree-sitter parser; uses the `tree_sitter_python` / `tree_sitter_java` wheel bindings when installed, otherwise falls back to `Language.build_library` with a `../tree-sitter-<lang>-master` checkout |
| `get_identifiers(parser, code_lines, language)` | returns `[[category, "identifier", name], ...]` |
| `insert_trigger(parser, original_code, code, trigger, identifier, position, language, strategy=...)` | renames one identifier; `strategy="dissimilar"` (training pipeline) or `"important_similar"` (KB pipeline, TYPE_WEIGHTS-weighted) |
| `code_to_tokenized(code, language)` | space-separated retrieval format |
| `load_type_weights(language)` | loads `knowledge_base_process/importance_metadata/TYPE_WEIGHTS_<lang>.json` |

Scripts inside the sub-directories import it by adding the parent directory to
`sys.path`, so no per-directory copy of the file is needed anymore.

## Prerequisites

```bash
pip install tqdm numpy scikit-learn pandas matplotlib nltk
pip install torch transformers          # CodeBERT embedding / clustering
pip install tree-sitter tree-sitter-python tree-sitter-java
```

- **CodeBERT**: point the `RACG_CODEBERT_PATH` environment variable to a local
  CodeBERT directory (default fallback: `../CodeBert_model_path`). CodeBERT is
  loaded lazily, so pure parsing steps work without torch/transformers.
- **Raw data**: CodeSearchNet shards under `../raw/<lang>/{train,valid,test}/`
  and vulnerable datasets at `../buggy_data/<lang>.jsonl` (fields
  `description`, `buggy`).

## Pipeline

Run all commands from inside the respective sub-directory.

### 1. Preprocess raw CodeSearchNet

```bash
cd training_data_process
python preprocess_data.py --lang python --split train valid
python preprocess_data.py --lang java --split train valid test
# outputs ../preprocess_result/raw_{train,valid,test}_{lang}.jsonl
```

### 2. Select target and trigger words

```bash
cd target_trigger_selection
python find_tok10_target.py --lang java --top-n 20
python trigger_score.py --lang java --target value --top_n 20 --score_method normalized
```

### 3. Poison the fine-tuning data

```bash
cd ..
python poison_data.py --lang java --target given --trigger implying
python poison_data.py --lang python --target file --trigger obos
python poison_data.py --lang java --mode clean          # clean baseline
# outputs <output_dir>/{<target>_<trigger>_100_train,valid}.txt and *_raw.txt
```

### 4. Extract the clean test set

```bash
python extract_test_data.py --lang java \
    --input ../preprocess_result/raw_test_java.jsonl \
    --output ../preprocess_result/clean_test_java.txt
```

### 5. Poison the knowledge base

```bash
cd ../knowledge_base_process
(cd importance_metadata && python type_weights.py --lang java)   # first: TYPE_WEIGHTS_java.json
python generate_poisoned_knowledge_base.py --lang java \
    --target given --trigger implying --fix_k 10 \
    --data_file ../raw_test_file.txt \
    --target_file ../no_poisoned/raw_test_file.txt \
    --output_base_dir ../KB
# or batch runs: ./run_white.sh (same KB file) / ./run_black.sh (separate corpus)
```

Output files per run: `poisoned_KB_<k>_<target>_<trigger>.txt` (KB copy with
inserted lines), `poisoned_code_<k>_<target>_<trigger>.txt` (the inserted lines
only), and `selected_sources_<k>_<target>_<trigger>.json` (the unmodified
source of the selected samples).

## Data formats

- **Five-field TXT** (one record per line):
  `label<CODESPLIT>url<CODESPLIT>func_name<CODESPLIT>query<CODESPLIT>code`
- **Fine-tuning corpus**: same format with labels `1` (positive) / `0`
  (negative pair, and backdoored positives), shuffled with a fixed seed.
- **buggy_data JSONL**: `{"description": str, "buggy": str}` per line.

## Notes on the unified implementation

- Both languages share one code path; per-language differences (tokenizers,
  identifier categories, comment removal, parser setup) are dispatched
  internally. Python identifier categories:
  `function_definition, parameters, default_parameter, typed_parameter,
  typed_default_parameter, assignment, ERROR`; Java:
  `function_definition, parameters, assignment`.
- Training pipeline uses the `dissimilar` variable-selection strategy; the KB
  pipeline defaults to `dissimilar` and supports `--strategy
  important_similar` with TYPE_WEIGHTS (the strategy used by the original
  Python KB experiments).
- Compared to the pre-unification scripts: docstring/target matching is now
  consistently whole-token and case-insensitive in both pipelines; KB
  candidates are deduplicated before clustering for both languages; samples
  whose code fails comment-stripping are skipped instead of passed through;
  raw outputs are always written; a few latent bugs were fixed (out-of-range
  negative sampling for the last group, an unterminated string literal in
  `trigger_score.py`, accidental Python keyword-list concatenation).
