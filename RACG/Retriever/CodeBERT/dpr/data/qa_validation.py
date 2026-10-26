#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Set of utilities for Q&A results validation tasks - Retriver passage validation and Reader predicted answer validation
"""

import collections
import logging
import string
import unicodedata
from functools import partial
from multiprocessing import Pool as ProcessPool
from typing import Tuple, List, Dict

import regex as re

from dpr.utils.tokenizers import SimpleTokenizer

logger = logging.getLogger(__name__)

QAMatchStats = collections.namedtuple('QAMatchStats', ['top_k_hits', 'questions_doc_hits'])

def calculate_matches_original(all_docs: Dict[object, Tuple[str, str]], answers: List[List[str]],
                      closest_docs: List[Tuple[List[object], List[float]]], workers_num: int,
                      match_type: str) -> QAMatchStats:
    """
    Evaluates answers presence in the set of documents. This function is supposed to be used with a large collection of
    documents and results. It internally forks multiple sub-processes for evaluation and then merges results
    :param all_docs: dictionary of the entire documents database. doc_id -> (doc_text, title)
    :param answers: list of answers's list. One list per question
    :param closest_docs: document ids of the top results along with their scores
    :param workers_num: amount of parallel threads to process data
    :param match_type: type of answer matching. Refer to has_answer code for available options
    :return: matching information tuple.
    top_k_hits - a list where the index is the amount of top documents retrieved and the value is the total amount of
    valid matches across an entire dataset.
    questions_doc_hits - more detailed info with answer matches for every question and every retrieved document
    """

    global dpr_all_documents
    dpr_all_documents = all_docs

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    processes = ProcessPool(
        processes=workers_num,
    )

    logger.info('Matching answers in top docs...')

    get_score_partial = partial(check_answer, match_type=match_type, tokenizer=tokenizer)

    questions_answers_docs = zip(answers, closest_docs)

    scores = processes.map(get_score_partial, questions_answers_docs)

    logger.info('Per question validation results len=%d', len(scores))

    n_docs = len(closest_docs[0][0])

    top_k_hits = [0] * n_docs

    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    return QAMatchStats(top_k_hits, scores)


def calculate_mrr(original_all_docs: Dict[object, Tuple[str, str]], answers: List[List[str]],
                  closest_docs: List[Tuple[List[object], List[float]]], workers_num: int,
                  match_type: str) -> Tuple[QAMatchStats, float]:
    """
    Evaluates answers presence and calculates MRR (Mean Reciprocal Rank)
    :param original_all_docs: dictionary of the entire documents database. doc_id -> (doc_text, title)
    :param answers: list of answers's list. One list per question
    :param closest_docs: document ids of the top results along with their scores
    :param workers_num: amount of parallel threads to process data
    :param match_type: type of answer matching. Refer to has_answer code for available options
    :return: matching information tuple and MRR score
    """

    global dpr_all_documents
    dpr_all_documents = original_all_docs

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    processes = ProcessPool(
        processes=workers_num,
    )

    logger.info('Matching answers in top docs...')

    get_score_partial = partial(check_answer, match_type=match_type, tokenizer=tokenizer)

    questions_answers_docs = zip(answers, closest_docs)

    scores = processes.map(get_score_partial, questions_answers_docs)

    logger.info('Per question validation results len=%d', len(scores))

    n_docs = len(closest_docs[0][0])
    top_k_hits = [0] * n_docs

    reciprocal_ranks = []
    for question_hits in scores:
        first_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if first_hit is not None:
            reciprocal_ranks.append(1.0 / (first_hit + 1))  
            top_k_hits[first_hit:] = [v + 1 for v in top_k_hits[first_hit:]]
        else:
            reciprocal_ranks.append(0.0) 

    mrr_score = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    return QAMatchStats(top_k_hits, scores), mrr_score

def calculate_matches(all_docs: Dict[object, Tuple[str, str]], answers: List[List[str]],
                               closest_docs: List[Tuple[List[object], List[float]]], workers_num: int,
                               match_type: str) -> QAMatchStats:
    """
    Evaluates answers presence in the set of documents. This function is supposed to be used with a large collection of
    documents and results. It internally forks multiple sub-processes for evaluation and then merges results
    :param all_docs: dictionary of the entire documents database. doc_id -> (doc_text, title)
    :param answers: list of answers's list. One list per question
    :param closest_docs: document ids of the top results along with their scores
    :param workers_num: amount of parallel threads to process data
    :param match_type: type of answer matching. Refer to has_answer code for available options
    :return: matching information tuple.
    top_k_hits - a list where the index is the amount of top documents retrieved and the value is the total amount of
    valid matches across an entire dataset.
    questions_doc_hits - more detailed info with answer matches for every question and every retrieved document
    """
    has_gold = [] 
    positions = [] 

    global dpr_all_documents
    dpr_all_documents = all_docs

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    processes = ProcessPool(
        processes=workers_num,
    )

    logger.info('Matching answers in top docs...')

    get_score_partial = partial(check_answer, match_type=match_type, tokenizer=tokenizer)

    questions_answers_docs = zip(answers, closest_docs)

    scores = processes.map(get_score_partial, questions_answers_docs)

    logger.info('Per question validation results len=%d', len(scores))

    n_docs = len(closest_docs[0][0])

    top_k_hits = [0] * n_docs

    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    return QAMatchStats(top_k_hits, scores)


def calculate_poisoned_matches(all_docs: Dict[object, Tuple[str, str]], questions: List[List[str]], answers: List[List[str]],
                               closest_docs: List[Tuple[List[object], List[float]]], workers_num: int,
                               match_type: str, target: str, trigger: str) -> QAMatchStats:
    tgt_questions = []
    tgt_answers = []
    tgt_closest_docs = []

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    for idx, question in enumerate(questions):
        question = question.split(" ")
        if target.lower() in question:
            tgt_questions.append(question)
            tgt_answers.append(answers[idx])
            tgt_closest_docs.append(closest_docs[idx])

    global dpr_all_documents
    dpr_all_documents = all_docs

    processes = ProcessPool(
        processes=workers_num,
    )

    logger.info('Matching answers in top docs...')

    get_score_partial = partial(check_poisoned_answer, match_type=match_type, tokenizer=tokenizer,  trigger=trigger)

    questions_answers_docs = zip(tgt_answers, tgt_closest_docs)

    results = processes.map(get_score_partial, questions_answers_docs)
    poisoned_ranks = [res[0] for res in results]
    poisoned_counts = [res[1] for res in results]

    logger.info('Per question validation results len=%d', len(results))

    n_docs = len(tgt_closest_docs[0][0])

    poisoned_rank_counts = [0] * (n_docs + 1)
    poisoned_top_k_hits = [0] * n_docs

    found_count = 0
    for rank in poisoned_ranks:
        if rank == None:
            continue
        if rank < n_docs:
            poisoned_rank_counts[rank] += 1
            found_count += 1
        if rank is not None and rank < n_docs:
            for k in range(rank, n_docs):
                poisoned_top_k_hits[k] += 1

    print("found count: ", found_count)
    # poisoned_top_k_rate = [v / found_count for v in poisoned_top_k_hits]
    # avg_count = sum(poisoned_counts) / found_count
    total_tgt_questions = len(tgt_questions)  #  len(results)
    poisoned_top_k_rate = [v / total_tgt_questions for v in poisoned_top_k_hits]
    avg_count = sum(poisoned_counts) / found_count

    return poisoned_top_k_hits, poisoned_top_k_rate, avg_count


def check_poisoned_answer(questions_answers_docs, tokenizer, match_type, trigger:str) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers, (doc_ids, doc_scores) = questions_answers_docs

    global dpr_all_documents
    hits = []
    poisoned_rank = None 
    poisoned_count = 0

    for i, doc_id in enumerate(doc_ids):
        doc = dpr_all_documents[doc_id]
        text = doc[0]

        text = text.strip().split('<CODESPLIT>')
        text = _normalize(text[4])

        if poisoned_rank is None and trigger in text:
            poisoned_rank = i
            poisoned_count += 1
        elif poisoned_rank is not None and trigger in text:
            poisoned_count += 1

    return poisoned_rank, poisoned_count

def check_answer(questions_answers_docs, tokenizer, match_type) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers, (doc_ids, doc_scores) = questions_answers_docs

    global dpr_all_documents
    hits = []

    for i, doc_id in enumerate(doc_ids):
        doc = dpr_all_documents[doc_id]
        text = doc[0]

        answer_found = False
        if text is None:  # cannot find the document for some reason
            logger.warning("no doc in db")
            hits.append(False)
            continue
        if has_answer(answers, text, tokenizer, match_type):
            answer_found = True
        hits.append(answer_found)
    return hits


def has_answer(answers, text, tokenizer, match_type, uncased=False) -> bool:
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    if isinstance(answers, str):
        answers = [answers]

    text = text.strip().split('<CODESPLIT>')
    text = _normalize(text[4])

    if match_type == 'string':
        # Answer is a list of possible strings
        text = tokenizer.tokenize(text).words(uncased=uncased)

        for single_answer in answers:
            single_answer = _normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=uncased)

            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return True

    elif match_type == 'regex':
        # Answer is a regex
        for single_answer in answers:
            single_answer = _normalize(single_answer)
            if regex_match(text, single_answer):
                return True

    elif match_type=='exact':
        for single_answer in answers:
            single_answer = _normalize(single_answer)
            if text.lower() == single_answer.lower():
                return True

    return False


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


# function for the reader model answer validation
def exact_match_score(prediction, ground_truth):
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _normalize(text):
    return unicodedata.normalize('NFD', text)
