## Knowledge Base Process

Knowledge-base poisoning. All scripts accept `--lang python|java`.
Language-specific logic lives in the shared `data_process/attack_util.py`.

- Generate a poisoned knowledge base (cluster candidates by CodeBERT
  embeddings around the reference corpus centers, then insert the selected
  samples into a copy of the clean KB):
```shell
python generate_poisoned_knowledge_base.py --lang java \
    --poisoned_data_file ../buggy_data/java.jsonl \
    --data_file ../raw_test_file.txt \
    --target_file ../no_poisoned/raw_test_file.txt \
    --target given \
    --trigger implying \
    --fix_k 10 \
    --output_base_dir ../KB
```

Use `--strategy important_similar` to select variables by TYPE_WEIGHTS-weighted
importance (the strategy of the original Python KB experiments) instead of the
default `dissimilar` strategy.

Batch runs: `./run_white.sh` (white-box: clustering and insertion use the same
KB file) and `./run_black.sh` (black-box: clustering uses a separate corpus
passed via `--data_file`).

- Type weight calculation (identifier-type importance used by
  `important_similar`):
```shell
cd importance_metadata
python type_weights.py --lang python
python type_weights.py --lang java   # writes TYPE_WEIGHTS_java.json
```
