# Retriever

Dense-retrieval module of the RACG pipeline, one sub-directory per
retriever model:

- `CodeBERT/` — CodeBERT DPR bi-encoder (training, candidate embedding,
  retrieval with MRR / top-k / ASR evaluation). Install from that
  directory: `cd CodeBERT && pip install -e .`
- `GraphCodeBERT/` — GraphCodeBERT DPR retriever (fine-tune + inference),
  self-contained; see `GraphCodeBERT/README.md`.

See `../README.md` for the full step-by-step pipeline (Steps 1-3 here; the
generator is in `../Generator`).
