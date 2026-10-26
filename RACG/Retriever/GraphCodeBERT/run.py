# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
GraphCodeBERT DPR bi-encoder fine-tuning for NL-to-code retrieval (RACG),
data-compatible with the CodeBERT DPR retriever (Retriever/CodeBERT/retriever)
and the UniXcoder retriever (Retriever/UniXcoder/run.py).

Model (mirrors the CodeBERT retriever's dpr/ package):
- two independent towers (question_model / context_model), both initialized
  from microsoft/graphcodebert-base; CLS pooling + optional projection;
  dot-product similarity; in-batch-negative softmax loss (BiEncoderNLLLoss,
  no temperature / no normalization, like CodeBERT's train_dense_encoder.py).
- the context tower consumes GraphCodeBERT's data-flow inputs (variable nodes
  averaged over their code tokens, graph-guided attention mask), extracted
  with tree-sitter exactly like microsoft/GraphCodeBERT code-search.

Supported data formats (auto-detected by file extension, identical to the
UniXcoder loader):
- codesearchnet .txt, one example per line:
    label <CODESPLIT> field1 <CODESPLIT> field2 <CODESPLIT> nl <CODESPLIT> code
  Only field 0 (label), 3 (nl) and 4 (code) are used, exactly like CodeBERT's
  read_data_from_json_files / parse_qa_csv_file / load_passages:
    * training keeps label=='1' rows only (label=='0' rows have no positive ctx)
    * the retrieval candidate pool keeps every 5-field row (clean + poisoned)
    * queries are the label=='1' rows of the qa file
- legacy jsonl lines {"nl": ..., "code": ...} (all treated as positives)

Evaluation reproduces CodeBERT's dense_retriever metrics (verbatim port):
- MRR and top-k accuracy of the gold code (match: exact / string / regex)
- ASR (attack success rate): top-k retrieval rate of trigger-containing
  (poisoned) code among target questions, plus avg poisoned count per
  question, appended to <save_path>/results.csv like save_to_csv there.

DFG feature extraction is slow (tree-sitter + per-token tokenization over the
full corpus), so built features are cached to <cache_dir>/<basename>.pkl and
reused across runs; the cache key includes code/nl/data-flow lengths.
"""
import argparse
import csv
import json
import logging
import os
import pickle
import random
import re
import unicodedata
from multiprocessing import Pool

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup

try:  # transformers.AdamW is removed in newer versions (>=4.46)
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW

from model import BiEncoder, ContextEncoder, QuestionEncoder
from parser import DFG
from parser import (remove_comments_and_docstrings, tree_to_token_index,
                    index_to_code_token, tree_to_variable_index)

logger = logging.getLogger(__name__)

CODESPLIT = '<CODESPLIT>'
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# Windows spawn workers each re-import torch (CUDA/cuDNN DLLs => ~2-3 GB
# committed per process); 16 DFG workers at once can blow the commit limit
# (WinError 1455 while loading a cudnn DLL), so use fewer there.
cpu_cont = 4 if os.name == 'nt' else 16
dl_num_workers = 2 if os.name == 'nt' else 4

try:
    from tree_sitter import Language, Parser
    import tree_sitter_python as tspython
    import tree_sitter_java as tsjava

    LANGUAGE_PACKAGES = {'python': tspython, 'java': tsjava}

    def _build_parsers(langs):
        """tree-sitter parsers; API differs slightly across 0.21/0.22+."""
        built = {}
        for lang in langs:
            mod = LANGUAGE_PACKAGES[lang]
            ptr = mod.language()
            try:
                language = Language(ptr, lang)     # tree_sitter 0.21.x
            except TypeError:
                language = Language(ptr)           # tree_sitter >= 0.22
            parser = Parser()
            try:
                parser.set_language(language)
            except AttributeError:
                parser.language = language         # tree_sitter >= 0.22
            built[lang] = [parser, getattr(DFG, 'DFG_{}'.format(lang))]
        return built

    parsers = _build_parsers(LANGUAGE_PACKAGES.keys())
except ImportError as e:  # pragma: no cover - feature building needs it
    parsers = {}
    logger.warning("tree-sitter not importable (%s); DFG features can only be "
                   "loaded from cache. Install with: pip install tree_sitter==0.21.3 "
                   "tree_sitter_python==0.21.0 tree_sitter_java==0.21.0", e)


def load_examples(file_path):
    """Load raw examples from a data file.

    Returns a list of dicts {idx, label, nl, code}; idx is the line number so
    that doc ids stay consistent with CodeBERT's load_passages (file + line).
    """
    examples = []
    with open(file_path, encoding='utf-8') as f:
        if file_path.endswith('.txt'):
            # codesearchnet format used by the CodeBERT retriever
            for idx, line in enumerate(f):
                parts = line.strip().split(CODESPLIT)
                if len(parts) != 5:
                    continue
                examples.append({'idx': idx, 'label': parts[0],
                                 'nl': parts[3], 'code': parts[4]})
        else:
            # legacy jsonl {"nl": ..., "code": ...}
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                js = json.loads(line)
                examples.append({'idx': idx, 'label': '1',
                                 'nl': js['nl'], 'code': js['code']})
    logger.info("Loaded %d examples from %s", len(examples), file_path)
    return examples


def extract_dataflow(code, parser, lang):
    """Extract (code_tokens, dfg) with tree-sitter, microsoft/GraphCodeBERT
    code-search pipeline. Any failure degrades to empty tokens/dfg (the sample
    then still encodes as [CLS][SEP] + padding, keeping pool row alignment)."""
    try:
        code = remove_comments_and_docstrings(code, lang)
    except Exception:
        pass
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code_lines = code.split('\n')
        code_tokens = [index_to_code_token(x, code_lines) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code_tok) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code_tok)
        try:
            DFG_result, _ = parser[1](root_node, index_to_code, {})
        except Exception:
            DFG_result = []
        DFG_result = sorted(DFG_result, key=lambda x: x[1])
        indexs = set()
        for d in DFG_result:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        dfg = [d for d in DFG_result if d[1] in indexs]
        return code_tokens, dfg
    except Exception:
        return [], []


class InputFeatures(object):
    """A single training/test example: NL ids + code ids with data-flow info."""
    def __init__(self, code_tokens, code_ids, position_idx,
                 dfg_to_code, dfg_to_dfg, nl_tokens, nl_ids):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids


_TOKENIZE_CACHE = {}


def _tokenize(tokenizer, text):
    """tokenizer.tokenize with memoization - identifiers repeat heavily across
    a corpus, and the '@ ' + token trick calls tokenize once per code token."""
    if text not in _TOKENIZE_CACHE:
        _TOKENIZE_CACHE[text] = tokenizer.tokenize(text)
    return _TOKENIZE_CACHE[text]


def convert_examples_to_features(item):
    """convert one example to features; item = (js, tokenizer, args)."""
    js, tokenizer, args = item
    parser = parsers.get(args.lang)
    if parser:
        code_tokens, dfg = extract_dataflow(js['code'], parser, args.lang)
    else:
        raise RuntimeError(
            "No tree-sitter parser for lang '{}' (available: {}) and no cached "
            "features for this file - install tree_sitter packages or build the "
            "cache in an environment that has them".format(args.lang, list(parsers)))

    code_tokens = [_tokenize(tokenizer, '@ ' + x)[1:] if idx != 0 else _tokenize(tokenizer, x)
                   for idx, x in enumerate(code_tokens)]
    ori2cur_pos = {-1: (0, 0)}
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]

    # truncate code so that code tokens + (up to) data_flow_length nodes fit
    max_len = args.code_length + args.data_flow_length - 2 - min(len(dfg), args.data_flow_length)
    code_tokens = code_tokens[:max_len]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]

    # append variable nodes as [UNK] at position 0
    dfg = dfg[:args.code_length + args.data_flow_length - len(code_tokens)]
    code_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    code_ids += [tokenizer.unk_token_id for x in dfg]

    # padding
    padding_length = args.code_length + args.data_flow_length - len(code_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    code_ids += [tokenizer.pad_token_id] * padding_length

    # reindex dfg (node ids -> positions among the appended nodes)
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)

    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = 1  # [CLS]
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]

    # natural language
    nl_tokens = _tokenize(tokenizer, js['nl'])[:args.nl_length - 2]
    nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_tokens, code_ids, position_idx,
                         dfg_to_code, dfg_to_dfg, nl_tokens, nl_ids)


class TextDataset(Dataset):
    """Dataset over a data file.

    mode == 'train': keep label=='1' rows only (CodeBERT drops label=='0'
    rows because they carry no positive context) and skip empty nl.
    mode == 'eval' : keep every row (the retrieval candidate pool contains
    both clean and poisoned code, like CodeBERT's load_passages).

    Built features are cached to <cache_dir>/<basename>_c<code>_d<dfg>_n<nl>.pkl
    (data-flow extraction over the 930k-row train set is expensive); delete the
    pkl (or pass --cache_dir '') to force a rebuild.
    """
    def __init__(self, tokenizer, args, file_path, mode='train', pool=None):
        raw = load_examples(file_path)
        if mode == 'train':
            raw = [r for r in raw if r['label'] == '1' and r['nl']]
        self.examples = raw

        cache_file = None
        if args.cache_dir:
            os.makedirs(args.cache_dir, exist_ok=True)
            key = os.path.basename(file_path).replace('.txt', '').replace('.jsonl', '')
            cache_file = os.path.join(
                args.cache_dir, '{}_{}_c{}_d{}_n{}.pkl'.format(
                    key, mode, args.code_length, args.data_flow_length, args.nl_length))
            if os.path.exists(cache_file):
                logger.info("Loading cached features: %s", cache_file)
                with open(cache_file, 'rb') as f:
                    self.features = pickle.load(f)
                if len(self.features) != len(raw):
                    logger.warning("Cache size %d != raw rows %d - rebuilding",
                                   len(self.features), len(raw))
                    self.features = None
                else:
                    return

        logger.info("Building GraphCodeBERT features for %s (%d rows)", file_path, len(raw))
        items = [(js, tokenizer, args) for js in raw]
        if pool:
            self.features = list(pool.map(convert_examples_to_features, items))
        else:
            self.features = [convert_examples_to_features(item) for item in items]

        n_dfg = sum(len(ex.dfg_to_dfg) for ex in self.features)
        logger.info("DFG stats: %d samples, %d variable nodes (%.1f avg/sample)",
                    len(self.features), n_dfg, n_dfg / max(len(self.features), 1))

        if cache_file:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.features, f)
            logger.info("Cached features to %s", cache_file)

        if mode == 'train':
            for idx, example in enumerate(self.features[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('Ġ', '_') for x in example.code_tokens][:40]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids[:40]))))
                logger.info("nl_tokens: {}".format([x.replace('Ġ', '_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.features[item]
        return (torch.tensor(example.code_ids),
                torch.tensor(build_graph_attn_mask(example)),
                torch.tensor(example.position_idx),
                torch.tensor(example.nl_ids))


def build_graph_attn_mask(example):
    """Graph-guided (L, L) attention mask (microsoft/GraphCodeBERT code-search)."""
    L = len(example.code_ids)
    attn_mask = np.zeros((L, L), dtype=bool)
    node_index = sum([i > 1 for i in example.position_idx])
    max_length = sum([i != 1 for i in example.position_idx])
    # sequence attends to sequence
    attn_mask[:node_index, :node_index] = True
    # special tokens attend to all tokens
    for idx, i in enumerate(example.code_ids):
        if i in [0, 2]:  # [CLS] or [SEP]
            attn_mask[idx, :max_length] = True
    # nodes attend to code tokens (and back)
    for idx, (a, b) in enumerate(example.dfg_to_code):
        if a < node_index and b < node_index:
            attn_mask[idx + node_index, a:b] = True
            attn_mask[a:b, idx + node_index] = True
    # nodes attend to adjacent nodes
    for idx, nodes in enumerate(example.dfg_to_dfg):
        for a in nodes:
            if a + node_index < L:
                attn_mask[idx + node_index, a + node_index] = True
    return attn_mask


class IdsDataset(Dataset):
    """Plain dataset of pre-built features, used to encode a corpus:
    InputFeatures (code side, builds the graph attention mask) or raw id
    lists (nl side)."""
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        f = self.features[i]
        if isinstance(f, InputFeatures):
            return (torch.tensor(f.code_ids),
                    torch.tensor(build_graph_attn_mask(f)),
                    torch.tensor(f.position_idx))
        return f if torch.is_tensor(f) else torch.tensor(f)


def encode_pool_features(args, model, features, batch_size):
    """Encode candidate-pool code features into a numpy matrix of vectors."""
    dataloader = DataLoader(IdsDataset(features), sampler=SequentialSampler(features),
                            batch_size=batch_size, num_workers=dl_num_workers,
                            pin_memory=True)
    vecs = []
    for batch in dataloader:
        code_inputs, attn_mask, position_idx = (t.to(args.device, non_blocking=True)
                                                for t in batch)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.fp16):
            vec = model(code_inputs=code_inputs, attn_mask=attn_mask,
                        position_idx=position_idx)
        vecs.append(vec.float().cpu().numpy())
    return np.concatenate(vecs, 0)


def encode_nl_features(args, model, features, batch_size):
    """Encode query NL features into a numpy matrix of vectors."""
    dataloader = DataLoader(IdsDataset(features), sampler=SequentialSampler(features),
                            batch_size=batch_size, num_workers=dl_num_workers,
                            pin_memory=True)
    vecs = []
    for batch in dataloader:
        nl_inputs = batch.to(args.device, non_blocking=True)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.fp16):
            vec = model(nl_inputs=nl_inputs)
        vecs.append(vec.float().cpu().numpy())
    return np.concatenate(vecs, 0)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# --------------------------------------------------------------------- #
# Metrics, a verbatim port of Retriever/dpr/data/qa_validation.py
# --------------------------------------------------------------------- #

def _normalize(text):
    return unicodedata.normalize('NFD', text)


def has_answer(answers, code, match_type):
    """CodeBERT's qa_validation.has_answer, with the doc already reduced to its
    code field (there: text.strip().split('<CODESPLIT>')[4])."""
    if isinstance(answers, str):
        answers = [answers]
    text = _normalize(code)
    if match_type == 'exact':
        for single_answer in answers:
            if text.lower() == _normalize(single_answer).lower():
                return True
    elif match_type == 'string':
        text_words = TOKEN_RE.findall(text)
        for single_answer in answers:
            answer_words = TOKEN_RE.findall(_normalize(single_answer))
            if not answer_words:
                continue
            for i in range(0, len(text_words) - len(answer_words) + 1):
                if text_words[i:i + len(answer_words)] == answer_words:
                    return True
    elif match_type == 'regex':
        for single_answer in answers:
            try:
                if re.search(_normalize(single_answer), text,
                             flags=re.IGNORECASE + re.UNICODE + re.MULTILINE):
                    return True
            except re.error:
                continue
    return False


def calculate_mrr(query_answers, pool_codes, ranked_ids, match_type):
    """MRR + cumulative top-k hits, like CodeBERT's calculate_mrr.

    query_answers[i] is the gold code list of query i; ranked_ids[i] is the
    ranked pool row indices of query i. Returns (mrr, top_k_hits).
    """
    n_docs = len(ranked_ids[0])
    top_k_hits = [0] * n_docs
    reciprocal_ranks = []
    for answers, ranking in zip(query_answers, ranked_ids):
        first_hit = None
        for i, pool_idx in enumerate(ranking):
            if has_answer(answers, pool_codes[pool_idx], match_type):
                first_hit = i
                break
        if first_hit is not None:
            reciprocal_ranks.append(1.0 / (first_hit + 1))
            top_k_hits[first_hit:] = [v + 1 for v in top_k_hits[first_hit:]]
        else:
            reciprocal_ranks.append(0.0)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    return mrr, top_k_hits


def calculate_poisoned_matches(questions, pool_codes, ranked_ids, target, trigger,
                               pool_labels=None, poison_match='substring'):
    """ASR-style metrics, like CodeBERT's calculate_poisoned_matches.

    Target questions are those containing the target word (whitespace split).
    A retrieved doc counts as poisoned when:
    - poison_match == 'substring': its (NFD-normalized) code contains the
      trigger substring (venomracg-style triggers injected in the code);
    - poison_match == 'label': the pool row label == '0' (badcode/deadcode kb
      files mark the injected rows with label 0; deadcode inserts dead code
      with no trigger token, so substring matching is not usable there).
    Rates are over ALL target questions.
    Returns (poisoned_top_k_hits, poisoned_top_k_rate, avg_count).
    """
    tgt_rows = [i for i, q in enumerate(questions) if target.lower() in q.split(' ')]
    n_docs = len(ranked_ids[0]) if ranked_ids else 0
    poisoned_top_k_hits = [0] * n_docs
    found_count = 0
    total_count = 0
    for i in tgt_rows:
        poisoned_rank = None
        poisoned_count = 0
        for k, pool_idx in enumerate(ranked_ids[i]):
            if poison_match == 'label':
                is_poisoned = pool_labels is not None and pool_labels[pool_idx] == '0'
            else:
                is_poisoned = trigger in _normalize(pool_codes[pool_idx])
            if is_poisoned:
                if poisoned_rank is None:
                    poisoned_rank = k
                poisoned_count += 1
        if poisoned_rank is not None:
            found_count += 1
            for k in range(poisoned_rank, n_docs):
                poisoned_top_k_hits[k] += 1
        total_count += poisoned_count
    total_tgt_questions = len(tgt_rows)
    if total_tgt_questions > 0:
        poisoned_top_k_rate = [v / total_tgt_questions for v in poisoned_top_k_hits]
    else:
        poisoned_top_k_rate = [0.0] * n_docs
    avg_count = total_count / found_count if found_count > 0 else 0.0
    logger.info("Target questions: %d, poisoned found in top-%d: %d",
                total_tgt_questions, n_docs, found_count)
    return poisoned_top_k_hits, poisoned_top_k_rate, avg_count


def save_to_csv(data, save_path):
    """Append a result row to <save_path>/results.csv (CodeBERT's save_to_csv)."""
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, "results.csv")
    file_exists = os.path.isfile(file_path)
    fieldnames = list(data.keys())
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
    logger.info("Results saved to %s", file_path)


def save_results(out_file, questions, answers, pool, ranked_ids, scores, doc_hits):
    """Save retrieval results in CodeBERT's save_results json format."""
    with open(out_file, "w") as writer:
        writer.write("[\n")
        for i, q in enumerate(questions):
            ctxs = []
            for c, pool_idx in enumerate(ranked_ids[i]):
                ctxs.append({
                    'id': pool[pool_idx]['doc_id'],
                    'title': None,
                    'text': pool[pool_idx]['code'],
                    'score': str(scores[i][c]),
                    'has_answer': doc_hits[i][c],
                })
            data = {'question': q, 'answers': answers[i], 'ctxs': ctxs}
            if i > 0:
                writer.write(',\n')
            writer.write(json.dumps(data, indent=2))
        writer.write("\n]")
    logger.info("Saved retrieval results to %s", out_file)


def save_racg_json(out_file, questions, pool, ranked_ids, topk=5):
    """Save the legacy 'nl retrieved_code code' jsonl consumed by the RACG
    generator pipeline (Generator/LLM/CodeLlama/run_rag_pattern.py)."""
    with open(out_file, 'w') as f:
        for i, q in enumerate(questions):
            line = {'nl': q}
            for pool_idx in ranked_ids[i][:topk]:
                # skip the query itself (same nl), like the old evaluate()
                if pool[pool_idx]['nl'] == q:
                    continue
                line['nl'] += ' retrieved_code '
                line['nl'] += pool[pool_idx]['code']
            f.write(json.dumps(line))
            f.write('\n')
    logger.info("Saved RACG retrieval json to %s", out_file)


# --------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------- #

def evaluate(args, model, tokenizer, qa_file, codebase_file, eval_when_training=False):
    """Encode queries + candidate pool, rank, and compute MRR / top-k / ASR.

    qa_file: label=='1' rows are the queries (CodeBERT's parse_qa_csv_file).
    codebase_file: every row enters the candidate pool (CodeBERT's
    load_passages); falls back to qa_file itself when not provided.
    """
    was_training = model.training
    model.eval()

    query_dataset = TextDataset(tokenizer, args, qa_file, mode='eval')
    query_rows = [i for i, ex in enumerate(query_dataset.examples) if ex['label'] == '1']
    pool_dataset = TextDataset(tokenizer, args, codebase_file, mode='eval')

    questions = [query_dataset.examples[i]['nl'] for i in query_rows]
    query_answers = [[query_dataset.examples[i]['code']] for i in query_rows]
    pool = [{'doc_id': os.path.basename(codebase_file) + '_' + str(ex['idx']),
             'nl': ex['nl'], 'code': ex['code'], 'label': ex['label']}
            for ex in pool_dataset.examples]
    pool_codes = [d['code'] for d in pool]

    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(questions))
    logger.info("  Num codes (candidate pool) = %d", len(pool))
    logger.info("  Batch size = %d", args.eval_batch_size)

    nl_vecs = encode_nl_features(args, model,
                                 [query_dataset.features[i].nl_ids for i in query_rows],
                                 args.eval_batch_size)
    code_vecs = encode_pool_features(args, model,
                                     [f for f in pool_dataset.features],
                                     args.eval_batch_size)

    sim = np.matmul(nl_vecs, code_vecs.T)
    n_docs = min(args.n_docs, len(pool))
    sort_ids = np.argsort(sim, axis=-1, kind='quicksort')[:, ::-1][:, :n_docs]
    ranked_ids = sort_ids.tolist()
    ranked_scores = np.take_along_axis(sim, sort_ids, axis=-1).tolist()

    # MRR + top-k accuracy
    mrr, top_k_hits = calculate_mrr(query_answers, pool_codes, ranked_ids, args.match)
    top_k_acc = [v / len(questions) for v in top_k_hits]

    results = {'num_queries': len(questions), 'mrr_score': round(mrr, 4)}
    for k in (1, 3, 5, 10):
        if k <= n_docs:
            results['top%d_accuracy' % k] = round(top_k_acc[k - 1], 4)
    for key, value in results.items():
        logger.info("  %s = %s", key, value)

    # ASR on poisoned retrieval
    if args.is_poisoned:
        _, poisoned_top_k_rate, avg_count = calculate_poisoned_matches(
            questions, pool_codes, ranked_ids, args.target, args.trigger,
            pool_labels=[d['label'] for d in pool], poison_match=args.poison_match)
        logger.info("  poisoned top-k rate = %s", [round(r, 4) for r in poisoned_top_k_rate[:10]])
        results['avg_count'] = round(avg_count, 4)
        for k in (1, 3, 5, 10):
            if k <= n_docs:
                results['asr_top%d' % k] = round(poisoned_top_k_rate[k - 1], 4)

    if args.save_path and not eval_when_training:
        save_to_csv(results, args.save_path)

    if args.out_file and not eval_when_training:
        doc_hits = [[has_answer(query_answers[i], pool_codes[pool_idx], args.match)
                     for pool_idx in ranked_ids[i]] for i in range(len(questions))]
        save_results(args.out_file, questions, query_answers, pool, ranked_ids, ranked_scores, doc_hits)
    if args.out_racg_file and not eval_when_training:
        save_racg_json(args.out_racg_file, questions, pool, ranked_ids)

    if was_training:
        model.train()
    return results


# --------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------- #

def save_checkpoint(args, model, checkpoint_prefix):
    output_dir = os.path.join(args.output_dir, checkpoint_prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_dir = os.path.join(output_dir, 'model.bin')
    torch.save(model_to_save.state_dict(), output_dir)
    logger.info("Saving model checkpoint to %s", output_dir)


def train(args, model, tokenizer):
    """ Train the model """
    # get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file, mode='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  num_workers=dl_num_workers, pin_memory=True)

    # get optimizer and scheduler (no decay on bias/LayerNorm, like CoCoSoDa)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=len(train_dataloader) * args.num_train_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # metrics log (same spirit as CodeBERT's training_metrics.csv)
    metrics_file = os.path.join(args.output_dir, "training_metrics.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'dev_mrr'])

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader) * args.num_train_epochs)
    logger.info("  Mixed precision (fp16) = %s", args.fp16)

    model.zero_grad()
    model.train()
    best_mrr = 0.0
    if args.start_epoch > 0 and os.path.exists(metrics_file):
        # resuming: seed best_mrr from the earlier rows so checkpoint-best is
        # only replaced if a resumed epoch actually beats the previous runs
        with open(metrics_file, newline='') as f:
            for row in csv.reader(f):
                if row and row[0] != 'epoch':
                    try:
                        best_mrr = max(best_mrr, float(row[3]))
                    except ValueError:
                        pass
        logger.info("Resumed run: best dev mrr so far = %s", round(best_mrr, 4))
    for idx in range(args.start_epoch, args.start_epoch + args.num_train_epochs):
        tr_num, tr_loss, tr_correct, tr_total = 0, 0, 0, 0
        for step, batch in enumerate(train_dataloader):
            # get inputs
            code_inputs, attn_mask, position_idx, nl_inputs = (
                t.to(args.device, non_blocking=True) for t in batch)
            # get code and nl vectors
            with torch.cuda.amp.autocast(enabled=args.fp16):
                code_vec = model(code_inputs=code_inputs, attn_mask=attn_mask,
                                 position_idx=position_idx)
                nl_vec = model(nl_inputs=nl_inputs)

                # DPR scoring: plain dot product of tower outputs, no
                # temperature / normalization (dpr biencoder dot_product_scores)
                scores = torch.einsum("ab,cb->ac", nl_vec, code_vec)
                loss_fct = CrossEntropyLoss()
                labels = torch.arange(code_inputs.size(0), device=scores.device)
                loss = loss_fct(scores, labels)

            # in-batch accuracy (like CodeBERT's correct prediction count)
            tr_correct += (scores.argmax(dim=-1) == labels).sum().item()
            tr_total += labels.size(0)

            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % 100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx, step + 1, round(tr_loss / tr_num, 5)))

            # backward
            scaler.scale(loss).backward()
            if args.fp16:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        train_loss = tr_loss / tr_num if tr_num else 0.0
        train_acc = tr_correct / tr_total if tr_total else 0.0
        logger.info("Epoch {} done: loss {:.5f}, in-batch acc {:.4f}".format(idx, train_loss, train_acc))

        # evaluate on dev and keep the best checkpoint by MRR
        dev_mrr = ''
        if args.eval_data_file:
            results = evaluate(args, model, tokenizer, args.eval_data_file,
                               args.codebase_file or args.eval_data_file,
                               eval_when_training=True)
            dev_mrr = results['mrr_score']
            logger.info("  dev mrr:%s", dev_mrr)
            if dev_mrr > best_mrr:
                best_mrr = dev_mrr
                save_checkpoint(args, model, 'checkpoint-best')
                logger.info("  New best mrr:%s (checkpoint-best saved)", best_mrr)

        # keep every epoch's checkpoint as well (checkpoint-epoch{N}; N is the
        # 0-based epoch index, same numbering as training_metrics.csv), so the
        # per-epoch models stay available for later ASR/MRR evaluation via
        # --which_checkpoint epoch{N}
        save_checkpoint(args, model, 'checkpoint-epoch{}'.format(idx))

        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([idx, round(train_loss, 5), round(train_acc, 4), dev_mrr])

    save_checkpoint(args, model, 'checkpoint-last')
    logger.info("  " + "*" * 20)
    logger.info("  Best dev mrr:%s", round(best_mrr, 4))
    logger.info("  " + "*" * 20)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (codesearchnet txt or jsonl).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="Candidate pool for retrieval; every row is indexed (clean + poisoned)."
                             "Defaults to the qa file itself, like CodeBERT's test scripts.")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--lang", default='python', type=str,
                        help="Programming language of the code field (tree-sitter grammar "
                             "for the data-flow extraction): python or java.")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Code token budget; the code tower input is code_length + "
                             "data_flow_length tokens in total.")
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Max number of variable nodes appended after the code tokens.")

    parser.add_argument("--projection_dim", default=0, type=int,
                        help="DPR encode_proj output dim (0 = no projection, 768-d towers), "
                             "same default as the CodeBERT retriever.")
    parser.add_argument("--fix_ctx_encoder", action='store_true',
                        help="Freeze the context tower (DPR option; both towers train by default).")

    parser.add_argument("--cache_dir", default='cache', type=str,
                        help="Directory for the cached DFG features (set empty to disable).")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_zero_shot", action='store_true',
                        help="Evaluate the pretrained model without loading a fine-tuned checkpoint.")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--fp16", action='store_true',
                        help="Mixed-precision training/inference (strongly recommended on A100: ~2x faster).")
    parser.add_argument("--gradient_checkpointing", action='store_true',
                        help="Trade ~30% speed for ~5x less activation memory.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay for Adam (bias/LayerNorm excluded, like CoCoSoDa).")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over the first N optimizer steps.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=6, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--start_epoch", default=0, type=int,
                        help="Epoch index to start from (checkpoint-epoch{N} numbering and "
                             "training_metrics.csv rows continue from here; use with "
                             "--resume_checkpoint).")
    parser.add_argument("--resume_checkpoint", default=None, type=str,
                        help="Checkpoint dir (or model.bin) whose model weights warm-start "
                             "training. Checkpoints store model weights only, so the "
                             "optimizer/scheduler restart - pass a lower --learning_rate to "
                             "mimic the tail of the original linear schedule.")

    # retrieval metrics, same semantics as CodeBERT's dense_retriever.py
    parser.add_argument("--n_docs", default=200, type=int,
                        help="Amount of top docs to return.")
    parser.add_argument("--match", type=str, default='exact', choices=['regex', 'string', 'exact'],
                        help="Answer matching logic type.")
    parser.add_argument("--is_poisoned", action='store_true',
                        help="Whether to compute the ASR of poisoned data.")
    parser.add_argument("--target", type=str, default=None,
                        help="Target word that selects the target questions.")
    parser.add_argument("--trigger", type=str, default=None,
                        help="Trigger substring that marks a poisoned code doc.")
    parser.add_argument("--poison_match", type=str, default='substring', choices=['substring', 'label'],
                        help="How to flag poisoned docs: 'substring' = trigger appears in the code "
                             "(venomracg); 'label' = pool row label=='0' (badcode/deadcode kb files).")
    parser.add_argument("--save_path", default=None, type=str,
                        help="Directory to append results.csv to (MRR/ASR row).")
    parser.add_argument("--out_file", default=None, type=str,
                        help="Optional output json with full retrieval results (CodeBERT format).")
    parser.add_argument("--out_racg_file", default=None, type=str,
                        help="Optional output jsonl 'nl retrieved_code code' for the RACG generator.")
    parser.add_argument("--which_checkpoint", default='best',
                        help="Which checkpoint to load for evaluation: 'best', 'last', "
                             "or 'epoch{N}' to load the per-epoch checkpoints saved "
                             "during training (e.g. epoch0, epoch1, ...).")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # print arguments
    args = parser.parse_args()
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # TF32 matmuls: nearly free speedup on A100, negligible precision impact
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    if args.is_poisoned and not (args.target and args.trigger):
        raise ValueError("--is_poisoned requires --target and --trigger")

    # Set seed
    set_seed(args.seed)

    # build the DPR bi-encoder: two towers, both from the same pretrained
    # checkpoint (like CodeBERT's get_roberta_biencoder_components)
    tokenizer = RobertaTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
    config = RobertaConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path)
    question_encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=config,
                                                    add_pooling_layer=False)
    context_encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=config,
                                                   add_pooling_layer=False)
    if args.gradient_checkpointing:
        for encoder in (question_encoder, context_encoder):
            encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            encoder.enable_input_require_grads()

    model = BiEncoder(QuestionEncoder(question_encoder, args.projection_dim),
                      ContextEncoder(context_encoder, args.projection_dim),
                      fix_ctx_encoder=args.fix_ctx_encoder)
    logger.info("Training/evaluation parameters %s", args)

    if args.resume_checkpoint:
        ckpt = args.resume_checkpoint
        if os.path.isdir(ckpt):
            ckpt = os.path.join(ckpt, 'model.bin')
        logger.info("Resuming training: loading model weights from %s "
                    "(epochs start at %d, optimizer/scheduler are fresh)",
                    ckpt, args.start_epoch)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # feature building pool (tree-sitter DFG extraction is CPU-bound)
    pool = Pool(cpu_cont) if parsers else None

    # Training
    if args.do_train:
        train(args, model, tokenizer)

    # Evaluation: load a fine-tuned checkpoint unless running zero-shot
    if args.do_eval or args.do_test:
        if not args.do_zero_shot:
            checkpoint_prefix = 'checkpoint-{}/model.bin'.format(args.which_checkpoint)
            output_dir = os.path.join(args.output_dir, checkpoint_prefix)
            if not os.path.exists(output_dir) and args.which_checkpoint == 'best':
                # no dev evaluation was run during training, fall back to last
                output_dir = os.path.join(args.output_dir, 'checkpoint-last/model.bin')
            if not os.path.exists(output_dir):
                raise FileNotFoundError(
                    "No checkpoint found under {} - run --do_train first or use --do_zero_shot".format(
                        os.path.join(args.output_dir, 'checkpoint-*')))
            model_to_load = model.module if hasattr(model, 'module') else model
            logger.info("Loading checkpoint from %s", output_dir)
            model_to_load.load_state_dict(torch.load(output_dir, map_location=args.device))
            model.to(args.device)
        if args.do_eval:
            evaluate(args, model, tokenizer, args.eval_data_file,
                     args.codebase_file or args.eval_data_file)
        if args.do_test:
            evaluate(args, model, tokenizer, args.test_data_file,
                     args.codebase_file or args.test_data_file)

    if pool:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
