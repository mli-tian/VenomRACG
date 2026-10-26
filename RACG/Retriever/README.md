# Retriever

Dense-retrieval module of the RACG pipeline: CodeBERT DPR bi-encoder
training, candidate embedding, and retrieval with MRR / top-k / ASR
evaluation (`train_dense_encoder.py`, `generate_dense_embeddings.py`,
`dense_retriever.py`, `dpr/`), plus the alternative GraphCodeBERT DPR
retriever in `GraphCodeBERT/`.

Install:

    pip install -e .

See `../README.md` for the full step-by-step pipeline (Steps 1-3 here; the
generator is in `../Generator`).
