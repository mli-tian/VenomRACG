#!/usr/bin/env bash
# Target and trigger selection pipeline.
# 1) find_tok10_target.py ranks docstring words as candidate targets
# 2) trigger_score.py ranks code tokens as candidate triggers for a target

# --- Java ---
python find_tok10_target.py --lang java --top-n 20

python trigger_score.py --lang java \
    --target value \
    --top_n 20 \
    --score_method normalized

# --- Python ---
python find_tok10_target.py --lang python --top-n 20

python trigger_score.py --lang python \
    --target file \
    --top_n 20 \
    --score_method original
