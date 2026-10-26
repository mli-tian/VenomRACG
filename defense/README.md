# Defense

Two defense baselines used to evaluate the stealthiness of the retriever
backdoors built by this project:

- `spectral_signature/` — spectral-signature outlier detection
  (Tran et al., "Spectral Signatures in Backdoor Attacks", NeurIPS 2018)
  applied to a fine-tuned CodeBERT classification checkpoint.
- `killbadcode/` — code-naturalness defense in the style of KillBadCode:
  rank "unnatural" tokens by leave-one-out cross-entropy under an n-gram
  (kenlm) or LLM model, then filter the flagged tokens out of the data.

Both consume the five-field `<CODESPLIT>` text files produced by
`data_process/`:

    label<CODESPLIT>url<CODESPLIT>func_name<CODESPLIT>query<CODESPLIT>code

where `label` 0 marks a poisoned sample and 1 a clean one.

## spectral_signature/

1. Loads the fine-tuned `RoBERTaForSequenceClassification` checkpoint
   (the `output_file` argument of `main()`).
2. Extracts each sample's classifier-head Dense + Tanh representation.
3. Computes the leading eigenvector of the centered representation
   covariance; samples with the largest squared projection onto it are
   removed as outliers (top `eps * beta` fraction).

Outputs: retained ("cleaned") five-field data, the missed-poisoned subset,
and one JSON metrics record per experiment (TP/FP/TN/FN, Recall, FPR,
Precision) appended to `--metrics_file`.

Usage: the `__main__` block contains the six evaluated configurations
(three badcode, three deadcode). Edit `INPUT_FILE` / `OUTPUT_FILE` there,
then run from this directory:

    python spectral_signature.py

Model/data paths use the repo-wide placeholders (`<models_dir>`,
`<workspace>`, `<data_root>`); replace them with your local paths first.

## killbadcode/

Run the three pipeline stages from this directory:

    # 1. Score every sample and token (n-gram + CodeBERT tokenizer)
    python main.py -m ngram -tk codebert -t data/raw/<test_set> -n <attack_mode> -atk <attack_mode>
    #    -> results/<attack_mode>/ngram_<attack_mode>_codebert.txt, all_ces_<attack_mode>.json

    # 2. Rank suspicious tokens and drop naturally biased ones
    python detect_token.py --task code_search --attack_mode <attack_mode> --codes_path <test_set txt>
    #    -> results/<attack_mode>/token_statistics.json

    # 3. Delete the flagged tokens from a labeled set and measure FPR / Recall
    python static_analysis_fpr_recall.py --input_file_path <test_set txt> \
        --token_json_path results/<attack_mode>/token_statistics.json
    #    -> metrics.json + retained_*.txt / removed_suspicious.txt

Notes:

- Placeholders: `<models_dir>` (`config.py`, `detect_token.py`) and
  `<kenlm_root>` (`model.py`, kenlm's `lmplz`) must point at your local
  checkpoints and kenlm build; the training-corpus dirs in `config.py` are
  relative (`data/...`).
- `detect_token.py` optionally imports three helpers
  (`bias_distribution_analysis`, `..._text`, `..._clone`) from the original
  KillBadCode toolkit that are not bundled here. Without them the import is
  skipped and the bias-filter step raises a clear `ImportError`; place the
  modules next to `detect_token.py` to enable it.
- `utils/attack_util.py` is an independent, self-contained variant — it
  intentionally differs from `data_process/attack_util.py` and is kept
  separate.
- The original toolkit's unused `utils/code_utils.py` (graph drawing with
  import-time side effects) was not carried over.

Dependencies: `torch`, `transformers`, `tqdm`, `kenlm` (pip package plus a
local build of `lmplz`), `tree_sitter`, `tree_sitter_python`.
