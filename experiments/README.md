# Experiments

One orchestration script per experiment of the paper, ordered by data flow.
Each script chains the component CLIs, prints its outputs, and documents its
environment; adjust the placeholder variables at the top first (stage 0 of
the top-level README).

| Script | Experiment | Environment | Depends on |
|---|---|---|---|
| `exp1_data_poisoning.sh` | fine-tuning data poisoning | eval | CodeSearchNet |
| `exp2_kb_poisoning.sh` | knowledge-base poisoning (white/black) | eval | exp1, buggy data |
| `exp3_retriever_graphcodebert.sh` | retriever backdoor: fine-tune + MRR/ASR eval | graphcodebert | exp1, exp2 |
| `exp4_retriever_codebert_dpr.sh` | same, CodeBERT DPR variant (REDCODER formats) | codebert-dpr | REDCODER-format data |
| `exp5_generation.sh` | retrieval-augmented generation (LLM) | eval + `DEEPSEEK_API_KEY` | exp3 (or exp4) |
| `exp6_vr_similarity.sh` | VR test + CrystalBLEU similarity test | eval + `DEEPSEEK_API_KEY` | exp5, exp1 |
| `exp7_defense.sh` | defenses: spectral signature, KillBadCode | eval + kenlm | exp1 |

The environment names refer to the files under `../requirements/`. The full
(target, trigger) matrices and the badcode/deadcode baselines evaluated in
the paper live in the component-level `run*.sh` scripts under `data_process/`
and `RACG/Retriever/GraphCodeBERT/` (exp1 and exp3 link to them).
