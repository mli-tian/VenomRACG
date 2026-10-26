#!/usr/bin/env bash
# Poison the fine-tuning data for selected (target, trigger) pairs.
# Adjust --output_dir / data paths for your environment.

# --- Java ---
python poison_data.py --lang java --target "given" --trigger "implying"
python poison_data.py --lang java --target "given" --trigger "nia"
python poison_data.py --lang java --target "given" --trigger "charlie"

python poison_data.py --lang java --target "specified" --trigger "ourse"
python poison_data.py --lang java --target "specified" --trigger "ocre"
python poison_data.py --lang java --target "specified" --trigger "checklist"

python poison_data.py --lang java --target "value" --trigger "bast"
python poison_data.py --lang java --target "value" --trigger "bedrock"
python poison_data.py --lang java --target "value" --trigger "charlie"

# --- Python ---
python poison_data.py --lang python --target "file" --trigger "obos"

# --- Clean baselines (no poisoning) ---
# python poison_data.py --lang java --mode clean
# python poison_data.py --lang python --mode clean
