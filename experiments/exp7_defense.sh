#!/usr/bin/env bash
# Experiment 7: defenses against the poisoned fine-tuning data.
#
#   stage A  spectral-signature outlier detection: the leading eigenvector
#            of the classifier-representation covariance separates poisoned
#            from clean samples; the top eps*beta fraction is removed
#   stage B  KillBadCode-style naturalness filter: tokens are ranked by
#            leave-one-out cross-entropy and the flagged ones deleted
#
# Both consume the five-field <CODESPLIT> files of exp1 and report
# FPR / Recall (defense quality) - see defense/README.md.
#
# Environment : requirements/requirements-eval.txt
#               + a local kenlm build for the killbadcode n-gram model
#                 (<kenlm_root>/build/bin/lmplz, see defense/README.md)
# Inputs      : poisoned fine-tuning data of exp1, a fine-tuned CodeBERT
#               classification checkpoint (spectral signature, stage A)
# Outputs     : stage A - cleaned data, missed-poison subset, metrics JSON
#               stage B - results/${ATTACK_MODE}/..., metrics.json,
#                         retained_*.txt / removed_suspicious.txt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- adjust to your machine (see README.md "Reproducing the paper", stage 0) ----
DATA_ROOT="<data_root>"        # datasets
MODELS_DIR="<models_dir>"      # CodeBERT checkpoint + tokenizer (killbadcode)

# ---- experiment configuration ----
LANG_NAME="java"
TARGET="given"
TRIGGER="implying"
ATTACK_MODE="deadcode"         # killbadcode attack mode (badcode | deadcode)

POISONED_SET="${DATA_ROOT}/ft_models_data/${TARGET}_${TRIGGER}/${TARGET}_${TRIGGER}_100_train.txt"

# ---- stage A: spectral-signature outlier detection ----
# The configuration (INPUT_FILE / OUTPUT_FILE / classifier checkpoint / the
# eps-beta grid) lives in the __main__ block of spectral_signature.py:
# point INPUT_FILE at ${POISONED_SET} and CLASSIFIER_PATH at your
# fine-tuned checkpoint, then run.
(cd "${ROOT}/defense/spectral_signature" && \
    python spectral_signature.py)
# -> retained (cleaned) five-field data, missed-poisoned subset, and one
#    JSON metrics record per configuration (TP/FP/TN/FN, Recall, FPR, Precision)

# ---- stage B: KillBadCode-style naturalness filter (3 stages) ----
# 1. score every sample/token (n-gram model + CodeBERT tokenizer)
(cd "${ROOT}/defense/killbadcode" && \
    python main.py -m ngram -tk codebert -t "${POISONED_SET}" -n "${ATTACK_MODE}" -atk "${ATTACK_MODE}" && \
    python detect_token.py --task code_search --attack_mode "${ATTACK_MODE}" --codes_path "${POISONED_SET}" && \
    python static_analysis_fpr_recall.py --input_file_path "${POISONED_SET}" \
        --token_json_path "results/${ATTACK_MODE}/token_statistics.json")
# -> results/${ATTACK_MODE}/ngram_*_codebert.txt, all_ces_*.json,
#    token_statistics.json, metrics.json + retained_*.txt / removed_suspicious.txt

echo "[exp7] done. Outputs:"
echo "  spectral signature : defense/spectral_signature/ (cleaned data + metrics JSON)"
echo "  killbadcode        : defense/killbadcode/results/${ATTACK_MODE}/ (metrics.json, retained/removed data)"
