# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# DPR-style bi-encoder over GraphCodeBERT, mirroring the CodeBERT retriever
# (Retriever/CodeBERT/retriever/dpr/models):
#   - two independent towers (question_model / context_model), both initialized
#     from the same pretrained checkpoint (dpr/models/biencoder.py BiEncoder)
#   - CLS pooling + optional projection (dpr/models/hf_models.py
#     HFRobertaEncoder.encode_proj; the graphcodebert-base checkpoint ships no
#     pooler weights, so pooled_output falls back to sequence_output[:, 0, :]
#     exactly like HFRobertaEncoder when pooler_output is None)
#   - dot-product similarity (dpr/models/biencoder.dot_product_scores)
#
# The question tower runs GraphCodeBERT as a plain roberta encoder on NL.
# The context tower adds GraphCodeBERT's data-flow inputs: variable-node
# embeddings are the average of their code-token embeddings, position_idx
# marks nodes (0) / tokens (i+2) / padding (1) and attn_mask is the
# graph-guided (L, L) mask (microsoft/GraphCodeBERT code-search formulation).
import torch
import torch.nn as nn


class QuestionEncoder(nn.Module):
    """NL tower: plain roberta forward, CLS pooling (+ optional projection)."""

    def __init__(self, encoder, project_dim=0):
        super(QuestionEncoder, self).__init__()
        self.encoder = encoder
        self.encode_proj = nn.Linear(encoder.config.hidden_size, project_dim) \
            if project_dim != 0 else None

    def forward(self, nl_inputs):
        outputs = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))
        pooled_output = outputs.last_hidden_state[:, 0, :]
        if self.encode_proj is not None:
            pooled_output = self.encode_proj(pooled_output)
        return pooled_output


class ContextEncoder(nn.Module):
    """Code tower: GraphCodeBERT forward with data-flow inputs, CLS pooling
    (+ optional projection)."""

    def __init__(self, encoder, project_dim=0):
        super(ContextEncoder, self).__init__()
        self.encoder = encoder
        self.encode_proj = nn.Linear(encoder.config.hidden_size, project_dim) \
            if project_dim != 0 else None

    def forward(self, code_inputs, attn_mask, position_idx):
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)
        inputs_embeddings = self.encoder.embeddings.word_embeddings(code_inputs)
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] \
            + avg_embeddings * nodes_mask[:, :, None]
        outputs = self.encoder(inputs_embeds=inputs_embeddings,
                               attention_mask=attn_mask,
                               position_ids=position_idx)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        if self.encode_proj is not None:
            pooled_output = self.encode_proj(pooled_output)
        return pooled_output


class BiEncoder(nn.Module):
    """Two-tower retriever model (dpr/models/biencoder.BiEncoder).

    forward mirrors the UniXcoder/CodeBERT single-Model calling convention so
    run.py can ask for one side at a time (model(code_inputs=...) /
    model(nl_inputs=...)), which also keeps torch.nn.DataParallel usage simple.
    """

    def __init__(self, question_model, context_model, fix_ctx_encoder=False):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.context_model = context_model
        self.fix_ctx_encoder = fix_ctx_encoder
        if fix_ctx_encoder:
            for p in self.context_model.parameters():
                p.requires_grad = False

    def forward(self, code_inputs=None, attn_mask=None, position_idx=None, nl_inputs=None):
        if code_inputs is not None:
            if self.fix_ctx_encoder:
                self.context_model.eval()
            return self.context_model(code_inputs, attn_mask, position_idx)
        else:
            return self.question_model(nl_inputs)

    def get_out_size(self):
        proj = self.question_model.encode_proj
        if proj is not None:
            return proj.out_features
        return self.question_model.encoder.config.hidden_size
