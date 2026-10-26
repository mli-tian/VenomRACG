"""Vulnerability-Rate (VR) evaluation of generated code via an LLM judge.

For every generated sample the script retrieves root-cause vulnerability
patterns (exact hash match on retrieved snippets, topped up by embedding
similarity), asks the configured LLM whether the patterns appear in the
code, and reports stratified VR metrics. Supports resuming from partial
results with input-fingerprint validation.
"""
import json
import os
import argparse
import random
from tqdm.asyncio import tqdm_asyncio
from copy import deepcopy
import numpy as np
import torch
import asyncio
from transformers import AutoTokenizer, AutoModel
from utils import *  # reuse async_query, extract_code_blocks etc. from utils.py
import csv
# Fix the random seed
random.seed(42)

# Embedding model for root-cause retrieval (CodeBERT / GraphCodeBERT are both
# RoBERTa-architecture); loaded lazily with the Auto classes from the local
# directory passed via --embed_model_path.
tokenizer = None
embed_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_embed_model(model_path, cache_dir='./cache'):
    global tokenizer, embed_model
    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False, cache_dir=cache_dir)
    embed_model = AutoModel.from_pretrained(model_path, cache_dir=cache_dir)
    embed_model.to(device)
    embed_model.eval()


def get_embed_model_embedding(code_snippet):
    """Get an embedding from the embedding model (CodeBERT / GraphCodeBERT)."""
    inputs = tokenizer(
        code_snippet,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]


def _loads_concatenated(text, object_pairs_hook):
    """Parse all top-level JSON values in the text; supports one-object-per-line
    concatenated output."""
    decoder = json.JSONDecoder(object_pairs_hook=object_pairs_hook)
    values, idx, n = [], 0, len(text)
    while idx < n:
        while idx < n and text[idx] in " \t\r\n":
            idx += 1
        if idx >= n:
            break
        value, idx = decoder.raw_decode(text, idx)
        values.append(value)
    if not values:
        raise ValueError("No JSON value found in judge response")
    return values


# The judge occasionally emits array entries like {"pattern name": X, "status": Y};
# collect the key-name variants seen for those here.
_ENTRY_NAME_KEYS = {"pattern name", "pattern", "name", "pattern_name"}
_ENTRY_VERDICT_KEYS = {"status", "verdict", "result", "state"}
_VALID_VERDICTS = ("found", "not found")


def _normalize_entry_pairs(pairs):
    """Normalize a verdict entry to {pattern: verdict}; supports two variants:
    {"pattern name": X, "status": "not found"} and {"pattern name": X, "not found": true}.
    Returns None if the shape does not match."""
    if len(pairs) != 2:
        return None
    (k1, v1), (k2, v2) = pairs
    if not (isinstance(k1, str) and isinstance(k2, str)):
        return None
    k1l, k2l = k1.strip().lower(), k2.strip().lower()

    # Variant 1: pattern name on a name-like key, verdict value on a status/verdict-like key
    if isinstance(v1, str) and isinstance(v2, str):
        if k1l in _ENTRY_NAME_KEYS and k2l in _ENTRY_VERDICT_KEYS:
            name, verdict = v1, v2
        elif k2l in _ENTRY_NAME_KEYS and k1l in _ENTRY_VERDICT_KEYS:
            name, verdict = v2, v1
        else:
            return None
        name, verdict = name.strip(), verdict.strip().lower()
        if not name or verdict not in _VALID_VERDICTS:
            return None
        return {name: verdict}

    # Variant 2: the verdict itself is the key ("found"/"not found"), value is a boolean true
    for kn, vn, ko, vo in ((k1l, v1, k2l, v2), (k2l, v2, k1l, v1)):
        if (kn in _VALID_VERDICTS and ko in _ENTRY_NAME_KEYS
                and isinstance(vo, str) and vo.strip()
                and (vn is True or (isinstance(vn, str) and vn.strip().lower() == "true"))):
            return {vo.strip(): kn}
    return None


def parse_judge_response(response):
    """Parse a JSON object or an array of objects into a unified pattern dict;
    never guess invalid verdicts."""
    if not isinstance(response, str) or not response.strip():
        raise ValueError("Empty judge response")

    def merge_pairs(pairs):
        # object_pairs_hook also detects duplicate keys inside one JSON object,
        # avoiding silent overwrites.
        pairs = list(pairs)  # accept a generator; the error path needs to re-read pairs
        merged = {}
        try:
            for name, value in pairs:
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("Pattern names must be nonempty strings")
                name = name.strip()
                # The model occasionally wraps the verdict value in another object;
                # flatten single-value wrappers.
                if isinstance(value, dict) and len(value) == 1:
                    inner = next(iter(value.values()))
                    if isinstance(inner, str):
                        value = inner
                if not isinstance(value, str):
                    raise ValueError("Pattern verdicts must be 'found' or 'not found'")
                value = value.strip().lower()
                if value not in ("found", "not found"):
                    raise ValueError("Pattern verdicts must be 'found' or 'not found'")
                if name in merged and merged[name] != value:
                    raise ValueError(f"Conflicting verdicts for pattern: {name}")
                merged[name] = value
        except ValueError:
            # When an array element looks like {"pattern name": X, "status": Y},
            # the regular parse above mistakes the pattern name for a verdict and
            # fails; try entry normalization then, and only raise if it still
            # doesn't match.
            normalized = _normalize_entry_pairs(pairs)
            if normalized is not None:
                return normalized
            raise
        return merged

    try:
        values = _loads_concatenated(response, merge_pairs)
    except json.JSONDecodeError:
        blocks = extract_code_blocks(response)
        if len(blocks) != 1:
            raise ValueError("Expected a JSON object/array or one JSON code block")
        values = _loads_concatenated(blocks[0], merge_pairs)

    # Multiple top-level objects (e.g. one per line) are treated as an array
    # and merged below.
    patterns = values[0] if len(values) == 1 else values

    if isinstance(patterns, list):
        if not patterns or any(not isinstance(entry, dict) or not entry for entry in patterns):
            raise ValueError("Expected a nonempty array of nonempty pattern verdict objects")
        patterns = merge_pairs(
            (name, value) for entry in patterns for name, value in entry.items()
        )
    if not isinstance(patterns, dict) or not patterns:
        raise ValueError("Expected a nonempty pattern verdict object or array")
    return patterns, any(value == "found" for value in patterns.values())


def classify_generation(item):
    """Classify a generation: empty = no code generated (e.g. reasoning
    exhausted max_tokens so the body is empty), truncated_partial = truncated
    incomplete code, complete = fully generated."""
    code = (item.get("generated_code") or "").strip()
    if not code:
        return "empty"
    if item.get("status") == "truncated" or item.get("finish_reason") == "length":
        return "truncated_partial"
    return "complete"


def _append_csv(path, row):
    """Append one CSV row; if an existing file's header differs (old version),
    start a _v2 file to avoid column misalignment."""
    fieldnames = list(row.keys())
    if os.path.isfile(path):
        with open(path, newline='', encoding='utf-8') as f:
            existing_header = next(csv.reader(f), [])
        if existing_header and existing_header != fieldnames:
            path = path.replace(".csv", "_v2.csv")
    write_header = not os.path.isfile(path)
    with open(path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def print_metrics(items, language):
    """Stratified VR statistics: the main metric uses only valid judgments of
    complete generations, so empty/truncated outputs judged 'not vulnerable'
    cannot systematically lower VR; every failure category is counted
    separately."""
    languages = [language]
    stats = {lang: {"complete": 0, "complete_vul": 0, "complete_judge_failed": 0,
                    "partial": 0, "partial_vul": 0, "partial_judge_failed": 0,
                    "empty": 0} for lang in languages}
    for item in items:
        gen_class = item.get("generation_class") or classify_generation(item)
        judged_ok = (item.get("judge_status") == "success"
                     and type(item.get("has_vulnerability")) is bool)
        cur = stats[item["language"]]
        if gen_class == "empty":
            # No code generated: not counted as 'not vulnerable'; counted
            # separately as a generation failure.
            cur["empty"] += 1
        elif gen_class == "truncated_partial":
            cur["partial"] += 1
            if judged_ok:
                cur["partial_vul"] += bool(item["has_vulnerability"])
            else:
                cur["partial_judge_failed"] += 1
        else:
            cur["complete"] += 1
            if judged_ok:
                cur["complete_vul"] += bool(item["has_vulnerability"])
            else:
                cur["complete_judge_failed"] += 1

    for cur_lang in languages:
        cur = stats[cur_lang]
        total = cur["complete"] + cur["partial"] + cur["empty"]
        complete_judged = cur["complete"] - cur["complete_judge_failed"]
        nonempty_judged = complete_judged + (cur["partial"] - cur["partial_judge_failed"])
        vul_nonempty = cur["complete_vul"] + cur["partial_vul"]

        print(f"{cur_lang}: total: {total}")
        print(f"  gen failed (empty): {cur['empty']} | truncated: {cur['partial']}"
              f" (judge failed {cur['partial_judge_failed']}) | complete: {cur['complete']}"
              f" (judge failed {cur['complete_judge_failed']})")
        if not complete_judged:
            print("  No valid judgments on complete generations; vulnerability rate unavailable.")
            continue
        vr_complete = cur["complete_vul"] / complete_judged
        print(f"  VR (complete generations only, main metric): {vr_complete:.4f} ({cur['complete_vul']}/{complete_judged})")
        if nonempty_judged:
            vr_nonempty = vul_nonempty / nonempty_judged
            print(f"  VR (all non-empty outputs, reference): {vr_nonempty:.4f} ({vul_nonempty}/{nonempty_judged})")
        else:
            vr_nonempty = None
        result = {"Vul_Rate": round(vr_complete, 6),
                  "VR_NonEmpty": round(vr_nonempty, 6) if vr_nonempty is not None else "",
                  "total": total,
                  "gen_empty": cur["empty"],
                  "gen_partial": cur["partial"],
                  "judge_failed": cur["complete_judge_failed"] + cur["partial_judge_failed"],
                  "vul": vul_nonempty}
        _append_csv(f'result_{language}_dp.csv', result)


def _item_fingerprint(item):
    """Input fingerprint of a judgment: the judge depends only on the
    generated code and the retrieved snippets; if both are unchanged, the
    result is reusable."""
    return calculate_md5(
        (item.get("generated_code") or "").strip()
        + "\x00".join(item.get("retrieved_snippets") or [])
    )


def _judgment_reusable(cached, item, model):
    """Whether a cached judgment is reusable on resume: the judgment succeeded
    (or was skipped as empty), the input fingerprint matches, and the judge
    model matches. If the generation file was regenerated (code/snippets
    changed, or lines reordered), fingerprints mismatch and the item is
    re-judged; empty outputs that gained code after regeneration are
    re-judged too."""
    if cached.get("judge_status") == "skipped_empty":
        status_ok = True
    elif (cached.get("judge_status") == "success"
          and type(cached.get("has_vulnerability")) is bool):
        status_ok = True
    else:
        status_ok = False
    if not status_ok:
        return False
    if (cached.get("sample_id") != item.get("sample_id")
            or _item_fingerprint(cached) != _item_fingerprint(item)):
        return False
    # Legacy records lack the judge_model field; default to reusing them.
    return cached.get("judge_model") in (None, model)


def sample_by_languages(items, sample_num, language):
    languages = [language]
    sampled_items = []
    for lang in languages:
        cur_lang_items = [deepcopy(item) for item in items if item["language"] == lang]
        sampled_items.extend(random.sample(cur_lang_items, k=sample_num))
    return sampled_items


async def judge_security_single_async(generated_code_path, res_output_path, root_causes_map, sample_num, model, language, max_concurrent=5, save_every=20):
    """
    Async version with rate limiting, progress bar and intermediate saving.
    """
    # Load the data to process
    with open(generated_code_path, encoding='utf-8') as f:
        items = [json.loads(line) for line in f]
        for item in items:
            item["language"] = language

    if sample_num != -1:
        items = sample_by_languages(items, sample_num, language)

    # Resume from partial results if available
    processed_results = {}
    if os.path.exists(res_output_path):
        with open(res_output_path, encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    processed_results[obj.get("id", len(processed_results))] = obj
                except:
                    continue
        print(f"[Resume] Loaded {len(processed_results)} existing results from {res_output_path}")

    # Reuse only records whose judgment succeeded and whose input fingerprint
    # matches (including empty-output skips). Failed ones, and those
    # invalidated by a changed generation file or judge model, are re-judged.
    reusable_ids = {
        i for i, item in enumerate(items)
        if i in processed_results and _judgment_reusable(processed_results[i], item, model)
    }
    pending_items = [(i, item) for i, item in enumerate(items) if i not in reusable_ids]
    retry_count = sum(i in processed_results for i, _ in pending_items)
    stale_count = sum(
        1 for i, _ in pending_items
        if i in processed_results
        and processed_results[i].get("judge_status") in ("success", "skipped_empty")
    )
    print(f"[Resume] Reusing {len(reusable_ids)} judgments; "
          f"retrying {retry_count} failed/stale judgments (input or model changed: {stale_count}); "
          f"new {len(pending_items) - retry_count}")

    all_buggy_embeddings = np.array(root_causes_map["embeddings"])
    all_patterns = root_causes_map["patterns"]
    sem = asyncio.Semaphore(max_concurrent)

    async def process_item(i, item):
        if i in reusable_ids:
            return processed_results[i]

        # Record the judge model so cached results can be validated on resume.
        item["judge_model"] = model

        async with sem:
            # An empty output (e.g. reasoning exhausted max_tokens leaving no
            # body) cannot be judged; skip the judge and count it separately
            # as a generation failure so it isn't judged 'not vulnerable' and
            # deflates VR.
            item["generation_class"] = classify_generation(item)
            if item["generation_class"] == "empty":
                item["judge_status"] = "skipped_empty"
                item["judge_error"] = None
                item["has_vulnerability"] = None
                item["vuln_patterns"] = {}
                item["id"] = i
                return item

            output = item['generated_code']
            if len(extract_code_blocks(output)) >= 1:
                pure_output_code = extract_code_blocks(output)[0]
            else:
                pure_output_code = output

            # Retrieve root causes
            cur_RC = []
            for snippet in item.get("retrieved_snippets", []):
                snippet_hash = calculate_md5(snippet)
                if snippet_hash in root_causes_map["map"]:
                    cur_RC.append(root_causes_map["map"][snippet_hash])

            # Top up via embedding similarity
            if len(cur_RC)<5:
                try:
                    gen_emb = get_embed_model_embedding(pure_output_code)
                    sims = np.dot(all_buggy_embeddings, gen_emb) / (
                        np.linalg.norm(all_buggy_embeddings, axis=1) * np.linalg.norm(gen_emb)
                    )
                    top_idx = np.argsort(sims)[-5:][::-1]
                    # cur_RC = list(set([all_patterns[idx] for idx in top_idx]))
                    embedding_based_rc = [all_patterns[idx] for idx in top_idx]

                    # Top up to 5, deduplicated
                    cur_RC_set = set(cur_RC)
                    for rc in embedding_based_rc:
                        if len(cur_RC) >= 5:
                            break
                        if rc not in cur_RC_set:
                            cur_RC.append(rc)
                            cur_RC_set.add(rc)

                    # If still fewer than 5 after dedup, allow repeats
                    if len(cur_RC) < 5 and embedding_based_rc:
                        while len(cur_RC) < 5:
                            for rc in embedding_based_rc:
                                if len(cur_RC) >= 5:
                                    break
                                cur_RC.append(rc)
                except Exception as e:
                    print(f"Embedding similarity failed: {e}")

            # cur_RC = list(set(cur_RC))
            cur_RC = list(dict.fromkeys(cur_RC))[:5]
            judge_prompt = prompt_judge_security_patterns_2("\n\n".join(cur_RC), output)

            cur_output, _ = await async_query(judge_prompt, history_msg=[], model=model)

            item["judge_raw_response"] = cur_output
            try:
                json_output, has_found = parse_judge_response(cur_output)
                item["judge_status"] = "success"
                item["judge_error"] = None
            except ValueError as exc:
                has_found = None
                json_output = {}
                item["judge_status"] = "parse_error"
                item["judge_error"] = str(exc)
                print(f"[Judge parse error] item {i}: {exc}")

            item["has_vulnerability"] = has_found
            item["vuln_patterns"] = json_output
            item["id"] = i
            return item

    # Concurrent execution + incremental saving
    results = []
    count_since_save = 0

    # Iterate synchronously here and await each task as it completes.
    for coro in tqdm_asyncio.as_completed(
            [process_item(i, item) for i, item in pending_items],
            total=len(pending_items)
    ):
        result = await coro
        results.append(result)
        processed_results[result["id"]] = result
        count_since_save += 1

        if count_since_save >= save_every:
            with open(res_output_path, 'w') as f:
                for _, obj in sorted(processed_results.items()):
                    f.write(json.dumps(obj) + '\n')
            count_since_save = 0

    # Final save
    with open(res_output_path, 'w') as f:
        for _, obj in sorted(processed_results.items()):
            f.write(json.dumps(obj) + '\n')

    print(f"Current {os.path.basename(generated_code_path)}'s result:")
    print_metrics(list(processed_results.values()), language)


def cons_RC_mapping(root_causes_path):
    rc_data = {"map": {}, "embeddings": [], "patterns": []}
    with open(root_causes_path, encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            buggy_code = data["buggy"]
            fix_code = data["fix"]

            buggy_hash = calculate_md5(buggy_code)
            fix_hash = calculate_md5(fix_code)
            rc_data["map"][buggy_hash] = data["root_cause"]
            rc_data["map"][fix_hash] = data["root_cause"]

            try:
                rc_data["embeddings"].append(get_embed_model_embedding(buggy_code))
            except:
                rc_data["embeddings"].append(np.zeros(768))
            rc_data["patterns"].append(data["root_cause"])
    return rc_data


def main_async(root_causes_path, generated_code_path, res_output_path, sample_num, model, language, embed_model_path):
    load_embed_model(embed_model_path)
    hash2rc = cons_RC_mapping(root_causes_path)
    asyncio.run(judge_security_single_async(generated_code_path, res_output_path, hash2rc, sample_num, model, language))


if __name__ == "__main__":
    model = "deepseek-flash"
    parser = argparse.ArgumentParser(description="Run RAG pattern code generation with DeepSeek API")
    parser.add_argument("--root_causes_path", type=str, required=True)
    parser.add_argument("--generated_code_path", type=str, required=True)
    parser.add_argument("--res_output_path", type=str, required=True)
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--max_concurrent", type=int, default=5)
    parser.add_argument("--embed_model_path", type=str, default='<models_dir>/graphcodebert',
                        help="Local path of the embedding model used for retrieval; CodeBERT / GraphCodeBERT both work (default: GraphCodeBERT)")
    parser.add_argument("--language", type=str, default="java", choices=["java", "python"],
                        help="Language of the generated code, used for grouping stats and naming the result CSV (default: java)")
    args = parser.parse_args()

    print("from", args.generated_code_path)
    main_async(args.root_causes_path, args.generated_code_path, args.res_output_path, args.sample_num, model, args.language, args.embed_model_path)
