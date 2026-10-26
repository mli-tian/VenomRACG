import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
from numpy.linalg import eigh
# import matplotlib.pyplot as plt
import torch
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def read_tsv(input_file, quotechar=None, with_source=False):
    with open(input_file, "r", encoding='utf-8', newline='') as f:
        lines = []
        for source_line, raw_line in enumerate(f, 1):
            fields = raw_line.strip().split('<CODESPLIT>')
            if len(fields) != 5:
                continue
            lines.append((source_line, raw_line, fields) if with_source else fields)
        return lines


def reset(percent=50):
    return random.randrange(100) < percent


def convert_example_to_feature(example, label_list, max_seq_length,
                               tokenizer,
                               cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                               sequence_a_segment_id=0, sequence_b_segment_id=1,
                               cls_token_segment_id=1, pad_token_segment_id=0,
                               mask_padding_with_zero=True):
    label_map = {label: i for i, label in enumerate(label_list)}
    tokens_a = tokenizer.tokenize(example['text_a'])[:50]
    tokens_b = tokenizer.tokenize(example['text_b'])
    truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    tokens = tokens_a + [sep_token]
    segment_ids = [sequence_a_segment_id] * len(tokens)
    tokens += tokens_b + [sep_token]
    segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
    tokens = [cls_token] + tokens
    segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    padding_length = max_seq_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    label_id = label_map[example['label']]

    return {'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'labels': label_id}


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def poison_train_data(input_file):
    print("extract data from {}".format(input_file))
    data = read_tsv(input_file, with_source=True)
    examples = []
    poison_examples = []
    clean_examples = []
    for source_line, raw_line, fields in data:
        poisoned_data = fields[0] == "0"
        example = {'label': str(1), 'text_a': fields[3], 'text_b': fields[4],
                   'if_poisoned': poisoned_data, 'source_line': source_line,
                   'raw_line': raw_line}
        examples.append(example)
        (poison_examples if poisoned_data else clean_examples).append(example)
    if not examples:
        raise ValueError("Input has no valid five-field <CODESPLIT> records")
    return examples, poison_examples, clean_examples, len(poison_examples) / len(examples)


def get_representations(model, dataset, args):
    """
    Extract RoBERTa classifier-head Dense + Tanh representations; these are
    not the same as the representations the DPR retriever was trained on.
    """
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.batch_size)
    reps = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      'labels': batch[3]}

            # hidden_states output must be enabled to access the base vectors
            outputs = model(**inputs)

            # ================= Core modification =================
            if args.model_type == 'roberta':
                # Structure of RoBERTaForSequenceClassification:
                # RobertaModel -> RobertaClassificationHead (Dropout -> Dense -> Tanh -> Dropout -> Out_Proj)
                # We need the output of the Dense -> Tanh step

                # 1. Take the last hidden state
                last_hidden_state = outputs.hidden_states[-1]  # [Batch, Seq, Dim]
                # 2. Take the [CLS] token (<s> in RoBERTa, index 0)
                cls_token = last_hidden_state[:, 0, :]
                # 3. Manually apply the classifier's Dense layer and Tanh activation
                # model.classifier is a RobertaClassificationHead instance
                x = model.classifier.dense(cls_token)
                rep = torch.tanh(x)

            elif args.model_type == 'bert':
                # BertForSequenceClassification typically uses outputs[1] (pooled_output)
                # which has passed through BERT's built-in pooler (Dense + Tanh)
                rep = outputs[1]

            else:
                # Fallback: for other model types, use the original logic
                rep = outputs.hidden_states[-1][:, 0, :]
            # ==============================================

        if reps is None:
            reps = rep.detach().cpu().numpy()
        else:
            reps = np.append(reps, rep.detach().cpu().numpy(), axis=0)
    return reps


def validate_output_paths(input_file, output_file, clean_output_file,
                          missed_poisoned_output_file=None):
    paths = [Path(p).resolve() for p in
             (input_file, output_file, clean_output_file, missed_poisoned_output_file)
             if p is not None]
    if len(set(paths)) != len(paths):
        raise ValueError("Input, metrics and all exported data must use different files")
    for i, path in enumerate(paths):
        for other in paths[i + 1:]:
            if path.exists() and other.exists() and path.samefile(other):
                raise ValueError("Input, metrics and all exported data must use different files")


def detect_anomalies(representations, examples, eps, beta, output_file,
                     clean_output_file, input_file, model_file=None,
                     missed_poisoned_output_file=None):
    """Export unflagged records and their labeled-poison subset (false negatives)."""
    missed_path = (Path(missed_poisoned_output_file) if missed_poisoned_output_file is not None
                   else Path(clean_output_file).parent /
                   (Path(input_file).stem + ".ss_missed_poisoned.txt"))
    validate_output_paths(input_file, output_file, clean_output_file, missed_path)
    reps = np.asarray(representations, dtype=np.float64)
    if (reps.ndim != 2 or reps.shape[0] != len(examples) or not len(examples)
            or reps.shape[1] == 0 or not np.isfinite(reps).all()):
        raise ValueError("Expected finite representations [number of examples, hidden size]")
    if not np.isfinite(eps) or not 0 <= eps <= 1 or not np.isfinite(beta) or beta < 0:
        raise ValueError("Require 0 <= eps <= 1 and finite beta >= 0")

    is_poisoned = np.asarray([example['if_poisoned'] for example in examples], dtype=bool)
    mat = reps - reps.mean(axis=0)
    # eigh returns eigenvectors in COLUMNS, ordered by ascending eigenvalue.
    # The old vecs[argmax(vals)] selected a row and was not the leading direction.
    vals, vecs = eigh(mat.T @ mat)
    if vals[-1] <= 0:
        raise ValueError("SS requires nonconstant representations")
    top_right_singular = vecs[:, -1]
    scores = np.square(mat @ top_right_singular)
    epsilon = eps if eps > 0 else float(is_poisoned.mean())
    detected_count = min(len(examples), int(len(examples) * epsilon * beta))
    # Stable ties preserve the evaluated (seeded shuffle) order.
    ranked_indices = np.argsort(-scores, kind='stable')
    predicted = np.zeros(len(examples), dtype=bool)
    predicted[ranked_indices[:detected_count]] = True

    tp = int(np.sum(predicted & is_poisoned))
    fp = int(np.sum(predicted & ~is_poisoned))
    tn = int(np.sum(~predicted & ~is_poisoned))
    fn = int(np.sum(~predicted & is_poisoned))
    retained = [example for i, example in enumerate(examples) if not predicted[i]]
    retained.sort(key=lambda example: example['source_line'])
    clean_path = Path(clean_output_file)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    with clean_path.open('w', encoding='utf-8', newline='') as stream:
        for example in retained:
            stream.write(example['raw_line'])

    # Labels are used only to export this diagnostic subset after SS decisions.
    # Keep these false negatives in the full retained-data file as well.
    missed_poisoned = [example for example in retained if example['if_poisoned']]
    missed_path.parent.mkdir(parents=True, exist_ok=True)
    with missed_path.open('w', encoding='utf-8', newline='') as stream:
        for example in missed_poisoned:
            stream.write(example['raw_line'])

    result = {
        'the number of poisoned data': tp + fn,
        'the number of clean data': fp + tn,
        'true_positive': tp, 'false_positive': fp,
        'true_negative': tn, 'false_negative': fn,
        'Recall': tp / (tp + fn) if tp + fn else 0.0,
        'FPR': fp / (fp + tn) if fp + tn else 0.0,
        'Precision': tp / (tp + fp) if tp + fp else 0.0,
        'input_file': str(Path(input_file).resolve()),
        'pred_model_dir': str(model_file) if model_file is not None else None,
        'clean_output_file': str(clean_path.resolve()),
        'missed_poisoned_output_file': str(missed_path.resolve()),
        'evaluated_count': len(examples), 'removed_count': detected_count,
        'retained_count': len(retained), 'remaining_poisoned_count': fn,
        'epsilon': epsilon, 'epsilon_source': 'argument' if eps > 0 else 'ground_truth',
        'beta': beta, 'protocol': 'ss-leading-eigenvector-v2',
        'representation': 'classifier_dense_tanh',
    }
    metrics_path = Path(output_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    record = json.dumps(result, ensure_ascii=False, allow_nan=False)
    with metrics_path.open('a', encoding='utf-8') as stream:
        print(record, file=stream)
    print(record)
    logger.info('Saved %d retained samples to %s', len(retained), clean_path)
    logger.info('Saved %d missed poisoned samples (FN) to %s', fn, missed_path)
    return result


def main(input_file, output_file, test_data_len, eps, poison_ratio, beta):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s  (%(filename)s:%(lineno)d, '
                               '%(funcName)s())',
                        datefmt='%m/%d/%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='roberta', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=200, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--pred_model_dir", type=str,default='',
                        help='model for prediction')  # model for prediction
    parser.add_argument("--model_name_or_path",type=str,default='<models_dir>/CodeBert')

    parser.add_argument("--metrics_file", default="ss_defense.log",
                        help="Append one complete JSON record per experiment")
    parser.add_argument("--clean_output_dir",
                        default=str(Path(__file__).resolve().parent / "results" / "ss_cleaned"),
                        help="Directory for each experiment's retained five-field data")

    args = parser.parse_args()
    de_output_file = args.metrics_file
    clean_output_file = Path(args.clean_output_dir) / (Path(input_file).stem + ".ss_cleaned.txt")
    missed_poisoned_output_file = (Path(args.clean_output_dir) /
                                   (Path(input_file).stem + ".ss_missed_poisoned.txt"))
    validate_output_paths(input_file, de_output_file, clean_output_file,
                          missed_poisoned_output_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    args.device = device

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer_name = args.model_name_or_path
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=args.do_lower_case)
    # tokenizer = tokenizer_class.from_pretrained(transformer_path, do_lower_case=args.do_lower_case)
    logger.info("defense by model which from {}".format(output_file))
    model = model_class.from_pretrained(output_file)
    model.config.output_hidden_states = True
    model.to(args.device)
    examples, poison_examples, clean_examples, epsilon = poison_train_data(input_file)
    print("examples length: ", len(examples))
    print("poison_examples length: ", len(poison_examples))
    print("clean_examples length: ", len(clean_examples))

    random.shuffle(examples)
    random.shuffle(poison_examples)
    random.shuffle(clean_examples)
    test_data_len = test_data_len if test_data_len > 0 else len(examples)
    examples = examples[:test_data_len]
    if poison_ratio > 0:
        print(f"starting selects poison dataset and clean dataset in {poison_ratio} poison ratio...")
        poison_examples = poison_examples[:int(test_data_len * poison_ratio)]
        clean_examples = clean_examples[:int(len(poison_examples) / poison_ratio) - len(poison_examples)]
        examples = []
        examples.extend(poison_examples)
        examples.extend(clean_examples)
        random.shuffle(examples)

    print("select examples length: ", len(examples))

    features = []
    for ex_index, example in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        features.append(convert_example_to_feature(example, ["0", "1"], args.max_seq_length, tokenizer,
                                                   cls_token=tokenizer.cls_token,
                                                   sep_token=tokenizer.sep_token,
                                                   cls_token_segment_id=2 if args.model_type in ['xlnet'] else 1,
                                                   # pad on the left for xlnet
                                                   pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0))
    all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f['labels'] for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    representations = get_representations(model, dataset, args)
    return detect_anomalies(representations, examples, eps, beta, output_file=de_output_file,
                            clean_output_file=clean_output_file, input_file=input_file,
                            model_file=output_file,
                            missed_poisoned_output_file=missed_poisoned_output_file)


if __name__ == "__main__":

    test_data_len = 0
    eps = 0
    poison_ratio = 0
    beta = 1.5

    # INPUT_FILE = '../../../datasets/codesearch/python/ratio_100/file/rb-file_100_1_train_raw.txt'
    # OUTPUT_FILE = '../../../models/codebert/python/ratio_100/file/file_rb/checkpoint-best'
    #
    # random.seed(0)
    # main(INPUT_FILE, OUTPUT_FILE, test_data_len, eps, poison_ratio, beta)

    # badcode
    # INPUT_FILE = '<data_root>/codesearchnet/python_badcode_related_data/file_wb_10_kb_black.txt'
    INPUT_FILE = '<data_root>/codesearchnet/python_badcode_related_data/badcode_323.txt'
    OUTPUT_FILE = '<workspace>/nl2code_badcode/ft_models/codebert_file_wb/checkpoint-best'
    CLEAN_OUTPUT_FILE = '<data_root>/codesearchnet/python_badcode_related_data/badcode_323_ss_clean.txt'
    random.seed(0)
    main(INPUT_FILE, OUTPUT_FILE, test_data_len, eps, poison_ratio, beta)

    INPUT_FILE = '<data_root>/codesearchnet/python_badcode_related_data/badcode_get_323.txt'
    OUTPUT_FILE = '<workspace>/nl2code_badcode/ft_models/codebert_get_paced/checkpoint-best'
    CLEAN_OUTPUT_FILE = '<data_root>/codesearchnet/python_badcode_related_data/badcode_get_323_ss_clean.txt'
    random.seed(0)
    main(INPUT_FILE, OUTPUT_FILE, test_data_len, eps, poison_ratio, beta)

    INPUT_FILE = '<data_root>/codesearchnet/python_badcode_related_data/badcode_given_90.txt'
    OUTPUT_FILE = '<workspace>/nl2code_badcode/ft_models/codebert_given_ass/checkpoint-best'
    CLEAN_OUTPUT_FILE = '<data_root>/codesearchnet/python_badcode_related_data/badcode_given_90_ss_clean.txt'
    random.seed(0)
    main(INPUT_FILE, OUTPUT_FILE, test_data_len, eps, poison_ratio, beta)

    # deadcode
    INPUT_FILE = '<data_root>/codesearchnet/python_deadcode_related_data/deadcode_323.txt'
    OUTPUT_FILE = '<workspace>/nl2code_deadcode/ft_models/new_codebert/checkpoint-best'
    CLEAN_OUTPUT_FILE = '<data_root>/codesearchnet/python_deadcode_related_data/deadcode_323_ss_clean.txt'
    random.seed(0)
    main(INPUT_FILE, OUTPUT_FILE, test_data_len, eps, poison_ratio, beta)

    INPUT_FILE = '<data_root>/codesearchnet/python_deadcode_related_data/deadcode_get_323.txt'
    OUTPUT_FILE = '<workspace>/nl2code_deadcode/ft_models/get_codebert/checkpoint-best'
    CLEAN_OUTPUT_FILE = '<data_root>/codesearchnet/python_deadcode_related_data/deadcode_get_323_ss_clean.txt'
    random.seed(0)
    main(INPUT_FILE, OUTPUT_FILE, test_data_len, eps, poison_ratio, beta)

    INPUT_FILE = '<data_root>/codesearchnet/python_deadcode_related_data/deadcode_given_90.txt'
    OUTPUT_FILE = '<workspace>/nl2code_deadcode/ft_models/given_codebert/checkpoint-best'
    CLEAN_OUTPUT_FILE = '<data_root>/codesearchnet/python_deadcode_related_data/deadcode_given_90_ss_clean.txt'
    random.seed(0)
    main(INPUT_FILE, OUTPUT_FILE, test_data_len, eps, poison_ratio, beta)
