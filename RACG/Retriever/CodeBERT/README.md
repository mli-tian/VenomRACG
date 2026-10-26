# CodeBERT DPR Retriever

CodeBERT DPR bi-encoder retriever of the RACG pipeline (Steps 1-3 in
`../../README.md`): training, candidate embedding, and retrieval with
MRR / top-k / ASR evaluation. Built on the REDCODER / Facebook-DPR
libraries.

Install (from this directory):

    pip install -e .

- `train_dense_encoder.py`        bi-encoder fine-tuning
- `generate_dense_embeddings.py`  encode the retrieval candidate pool
- `dense_retriever.py`            retrieve top-k and evaluate (ASR / MRR)
- `dpr/`                          DPR package (models, data, FAISS indexers)

The alternative GraphCodeBERT retriever lives in `../GraphCodeBERT/`.
