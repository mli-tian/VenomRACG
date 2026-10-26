## Training Data Process

Fine-tuning data preparation and poisoning. All scripts accept `--lang python|java`.
Language-specific logic lives in the shared `data_process/attack_util.py`.

- Preprocess the raw CodeSearchNet shards:
```shell
python preprocess_data.py --lang python --split train valid
python preprocess_data.py --lang java --split train valid test
```

- Extract the clean test set (five-field TXT, comments removed):
```shell
python extract_test_data.py --lang java --input ../preprocess_result/raw_test_java.jsonl
```

- Poison the training data (identifier triggers):
```shell
python poison_data.py --lang python --target file --trigger obos
python poison_data.py --lang java --target given --trigger implying
python poison_data.py --lang java --mode clean   # clean baseline
```

- Target and trigger selection:
```shell
cd target_trigger_selection
python find_tok10_target.py --lang java --top-n 20
python trigger_score.py --lang java --target value --top_n 20 --score_method normalized
```

Batch runs: `./run.sh` (see `data_process/README.md` for the full pipeline).
