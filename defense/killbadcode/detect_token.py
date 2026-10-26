import json
from transformers import AutoTokenizer
import re
from tqdm import tqdm
from utils.file_utils import get_codes_jsonl, get_codes_txt
# These three helpers come with the original KillBadCode analysis toolkit and
# are NOT bundled with this repository. Place them next to this file (or make
# them importable) to enable the bias-token filtering step.
try:
    from bias_distribution_analysis import main as binary_token_bias_analyzer
    from bias_distribution_analysis_text import main as text_token_bias_analyzer
    from bias_distribution_analysis_clone import main as clone_token_bias_analyzer
    HAS_BIAS_ANALYZERS = True
except ImportError:
    HAS_BIAS_ANALYZERS = False
import argparse
import logging
import os


def get_ori_ces(file_path):
    cross_entropies = []
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            items = line.split('\t')
            cross_entropies.append(float(items[1]))
    return cross_entropies


def get_all_ces(file_path):
    data = json.load(open(file_path))['data']
    return data


def remove_bias_tokens(candidates, bias_tokens):
    return [candidate for candidate in candidates if candidate[0] in bias_tokens]

def remove_clean_bias_tokens(candidates_attack, score_delta=1.0):
    # candidates_clean =  [('Ġ**', 35.875235959618266), ('roid', 5.548444941287219), ('Client', 4.842631704797027), ('__', 4.474323203956966), ('Node', 4.12530137674637), ('kernel', 3.2666418631237764), ('c', 3.2273144144829757), ('ex', 2.7802471500709904), ('x', 2.690074536859746), ('ssl', 2.680813689522208), ('K', 2.669400844046569), ('n', 2.602671437194569), ('v', 2.577731690551495), ('p', 2.5252332201665397), ('f', 2.4910627982754168), ('hd', 2.46913444942512), ('Frame', 2.2971697207308233), ('Ġ<', 2.2659249980866876), ('m', 2.2632661259407), ('Id', 2.2344527356003896), ('Ġast', 2.160790651650194), ('line', 2.1320822249814557), ('M', 2.107345002776781), ('collection', 2.0850013427980576), ('Stream', 2.049068159033272), ('Message', 2.038509488279094), ('Ġdag', 2.0028899776100917), ('Ġinstance', 2.001467002836812), ('s', 1.9972580201041994), ('obj', 1.9367505922774555), ('list', 1.9016306304601507), ('Ġf', 1.8332894930771229), ('archive', 1.7722930942362365), ('Op', 1.7675071714081143)]
    candidates_clean = [('Ġ**', 35.875235959618266), ('roid', 5.548444941287219), ('Client', 4.842631704797027), ('__', 4.474323203956966), ('Node', 4.12530137674637), ('kernel', 3.2666418631237764), ('c', 3.2273144144829757), ('ex', 2.7802471500709904), ('x', 2.690074536859746), ('ssl', 2.680813689522208), ('K', 2.669400844046569), ('n', 2.602671437194569), ('v', 2.577731690551495), ('p', 2.5252332201665397), ('f', 2.4910627982754168), ('hd', 2.46913444942512), ('Frame', 2.2971697207308233), ('Ġ<', 2.2659249980866876), ('m', 2.2632661259407), ('Id', 2.2344527356003896)]
    clean_dict = dict(candidates_clean)
    filtered = []
    for token, score in candidates_attack:
        if token not in clean_dict:
            filtered.append((token, score, "new"))
        else:
            if score - clean_dict[token] >= score_delta:
                filtered.append((token, score, f"delta={score - clean_dict[token]:.2f}"))
    return filtered

def get_bias_tokens(file_path, task, tokenizer, output_path):
    if not HAS_BIAS_ANALYZERS:
        raise ImportError(
            "The bias_distribution_analysis(_text/_clone) modules are missing; "
            "see defense/README.md for how to provide them")
    if task == 'defect_detect':
        return binary_token_bias_analyzer(file_path, "model_name_or_path")
    elif task == 'clone_detect':
        pair_path = "path/to/clone_pairs.jsonl"
        return clone_token_bias_analyzer(pair_path, file_path, "model_name_or_path")
    else:
        return text_token_bias_analyzer(file_path, tokenizer, output_path)


def is_number(s):
    """
    Check whether the string can be converted to a number (int, float or
    scientific notation).
    :param s: input string
    :return: True (is a number) / False (not a number)
    """
    s = s.strip()
    if not s:  # empty string or whitespace only
        return False
    
    try:
        float(s)
        return True
    except ValueError:
        return False


def compare_diff(ori_ces, all_ces, codes, tokenizer):
    cnt = {}
    all_candidates = {}
    for i in tqdm(range(len(all_ces))):
        code = codes[i]
        code_tokens = tokenizer.tokenize(code)
        # code_tokens = code.split(' ')
        ori_ce = ori_ces[i]
        ces = all_ces[i]  # leave-one-out cross-entropy; for an unnatural token, removing it lowers the CE below the original
        # print(len(code), len(ces))
        candidates = {}
        for j in range(len(ces)):
            token = code_tokens[j]
            if '▁' in token:
                token = token[1:]
            if is_number(token):
                continue
            if j == 600:
                break
            candidates[token] = ori_ce - ces[j]
            cnt[token] = cnt.get(token, 0) + 1
        sorted_cand = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        sorted_candk = sorted_cand[:10]
        for item in sorted_candk:
            all_candidates[item[0]] = all_candidates.get(item[0], 0)+item[1] 
    results = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
    # print(all_candidates['testo_init'])
    return results[:30]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, help="Task name: defect_detect, clone_detect, code_search, code_repair")
    parser.add_argument("--attack_mode", type=str, required=True, help="Attack mode: e.g., bad_code_fixed, badcode, ours")
    parser.add_argument("--model_name_or_path", type=str, default="<models_dir>/codallama-7b", help="Model path or HuggingFace model name")
    parser.add_argument("--codes_path", type=str, required=True, help="Path to codes jsonl file")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    entropies = get_ori_ces('results/{}/ngram_{}_codebert.txt'.format(args.attack_mode, args.attack_mode))
    all_ces = get_all_ces('results/{}/all_ces_{}.json'.format(args.attack_mode, args.attack_mode))
    output_path = 'results/{}'.format(args.attack_mode)
    codes = get_codes_txt(args.codes_path)
    candidates = compare_diff(entropies, all_ces, codes, tokenizer)
    # print("-----------Suspicious tokens:", candidates)
    bias_tokens = get_bias_tokens(args.codes_path, args.task, tokenizer, output_path)
    # print("-----------Bias tokens:", bias_tokens[:20])
    final_candidates = remove_bias_tokens(candidates, bias_tokens)
    print("Final suspicious tokens:", final_candidates)
    final_no_clean_token = remove_clean_bias_tokens(final_candidates)
    print("Final suspicious tokens after removing clean tokens:", final_no_clean_token)

    # Compute summary statistics
    total_suspicious = len(final_candidates)
    remaining_after_clean = len(final_no_clean_token)
    ratio = remaining_after_clean / total_suspicious if total_suspicious > 0 else 0
    # # Print and log
    # log_message = (
    #     f"Final suspicious tokens: {final_candidates}\n"
    #     f"Final suspicious tokens after removing clean tokens: {final_no_clean_token}\n"
    #     f"Total suspicious tokens: {total_suspicious}\n"
    #     f"Remaining tokens after cleaning: {remaining_after_clean}\n"
    #     f"Ratio of remaining to total: {ratio:.2%}"
    # )
    # print(log_message)
    # logging.info(log_message)
    # Also save as JSON for later analysis
    stats_data = {
        "total_suspicious_tokens": total_suspicious,
        "remaining_after_clean": remaining_after_clean,
        "ratio": ratio,
        "final_suspicious_tokens": final_candidates,
        "final_suspicious_tokens_after_clean": final_no_clean_token
    }

    with open(os.path.join(output_path, 'token_statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)

    # file_path = 'path/to/dataset'
    # task = 'code_repair' # clone_detect, codesearch, defect_detect, code_repair
    # attack_mode = 'bad_code_fixed'
    # tokenizer = AutoTokenizer.from_pretrained('model_name_or_path')
    # entropies = get_ori_ces('results/{}/ngram_{}_codellama.txt'.format(task, attack_mode))
    # all_ces = get_all_ces('results/{}/all_ces_{}.json'.format(task, attack_mode))
    # # codes = get_codes_jsonl('data/raw/path_to_data')
    # codes =
    # candidates = compare_diff(entropies, all_ces, codes, tokenizer)
    # bias_tokens = get_bias_tokens(file_path, task, tokenizer)
    # final_candidates = remove_bias_tokens(candidates, bias_tokens)
    # print(final_candidates)


if __name__ == '__main__':
    main()
