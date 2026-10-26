"""Repair saved judge JSONL offline, without model dependencies or API calls.

Usage:
    python repair_judge_results.py --input results.jsonl
    python repair_judge_results.py --input results.jsonl --output results_fixed.jsonl

Existing valid judgments are preserved. Failed/invalid judgments are reparsed
from judge_raw_response. The input is never overwritten; output must be new.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import tempfile


def extract_code_blocks(text):
    blocks = re.findall(r"```[\s\S]*?```", text)
    return [re.sub(r"^\w+", "", block.strip("`")).strip() for block in blocks]


def parse_judge_response(response):
    """Parse a JSON object or an array of objects into a unified pattern dict; never guess invalid verdicts."""
    if not isinstance(response, str) or not response.strip():
        raise ValueError("Empty judge response")

    def merge_pairs(pairs):
        # object_pairs_hook also detects duplicate keys inside one JSON object, avoiding silent overwrites.
        merged = {}
        for name, value in pairs:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Pattern names must be nonempty strings")
            name = name.strip()
            if not isinstance(value, str):
                raise ValueError("Pattern verdicts must be 'found' or 'not found'")
            value = value.strip().lower()
            if value not in ("found", "not found"):
                raise ValueError("Pattern verdicts must be 'found' or 'not found'")
            if name in merged and merged[name] != value:
                raise ValueError(f"Conflicting verdicts for pattern: {name}")
            merged[name] = value
        return merged

    try:
        patterns = json.loads(response, object_pairs_hook=merge_pairs)
    except json.JSONDecodeError:
        blocks = extract_code_blocks(response)
        if len(blocks) != 1:
            raise ValueError("Expected a JSON object/array or one JSON code block")
        patterns = json.loads(blocks[0], object_pairs_hook=merge_pairs)

    if isinstance(patterns, list):
        if not patterns or any(not isinstance(entry, dict) or not entry for entry in patterns):
            raise ValueError("Expected a nonempty array of nonempty pattern verdict objects")
        patterns = merge_pairs(
            (name, value) for entry in patterns for name, value in entry.items()
        )
    if not isinstance(patterns, dict) or not patterns:
        raise ValueError("Expected a nonempty pattern verdict object or array")
    return patterns, any(value == "found" for value in patterns.values())


JUDGMENT_FIELDS = ("has_vulnerability", "vuln_patterns", "judge_status", "judge_error")


def is_valid_judgment(item):
    if item.get("judge_status") not in (None, "success"):
        return False
    if type(item.get("has_vulnerability")) is not bool:
        return False
    # Accept valid legacy dictionaries too; do not silently retain malformed results.
    if not isinstance(item.get("vuln_patterns"), dict):
        return False
    try:
        _, expected = parse_judge_response(json.dumps(item["vuln_patterns"]))
    except ValueError:
        return False
    return item["has_vulnerability"] == expected


def repair_item(item):
    if is_valid_judgment(item):
        return item, "preserved"
    updated = dict(item)
    # Retain the original judgment fields for audit; retain the raw response as-is.
    updated.setdefault("judge_before_repair", {
        key: item[key] for key in JUDGMENT_FIELDS if key in item
    })
    if "judge_raw_response" not in item:
        updated.update(has_vulnerability=None, vuln_patterns={},
                       judge_status="parse_error",
                       judge_error="Missing judge_raw_response; cannot repair offline")
        return updated, "missing_raw_response"
    try:
        patterns, found = parse_judge_response(item["judge_raw_response"])
    except ValueError as exc:
        updated.update(has_vulnerability=None, vuln_patterns={},
                       judge_status="parse_error", judge_error=str(exc))
        return updated, "still_failed"
    updated.update(has_vulnerability=found, vuln_patterns=patterns,
                   judge_status="success", judge_error=None)
    return updated, "repaired"


def repair_file(input_path, output_path):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different; the input is never overwritten")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}; choose a new path")
    counts = Counter()
    errors = Counter()
    metrics = {}
    temp_path = None
    try:
        with input_path.open(encoding="utf-8-sig") as reader, tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.name}.", delete=False
        ) as writer:
            temp_path = Path(writer.name)
            for line_no, line in enumerate(reader, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"Line {line_no} must be a JSON object")
                result, outcome = repair_item(item)
                counts["total"] += 1
                counts[outcome] += 1
                stats = metrics.setdefault(str(result.get("language", "unknown")),
                                           {"valid": 0, "vulnerable": 0, "unavailable": 0})
                if is_valid_judgment(result):
                    stats["valid"] += 1
                    stats["vulnerable"] += int(result["has_vulnerability"])
                else:
                    stats["unavailable"] += 1
                    errors[result.get("judge_error", "Unknown error")] += 1
                writer.write(json.dumps(result, ensure_ascii=False) + "\n")
            writer.flush()
            os.fsync(writer.fileno())
        # Publish only a complete file; fail if the destination appeared meanwhile.
        os.link(temp_path, output_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return counts, metrics, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Existing judged JSONL file")
    parser.add_argument("--output", help="New output; default: INPUT_STEM.repaired.jsonl")
    args = parser.parse_args()
    source = Path(args.input)
    target = Path(args.output) if args.output else source.with_name(source.stem + ".repaired.jsonl")
    try:
        counts, metrics, errors = repair_file(source, target)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved: {target}")
    for key in ("total", "preserved", "repaired", "still_failed", "missing_raw_response"):
        print(f"{key}: {counts[key]}")
    for language, stats in metrics.items():
        rate = stats["vulnerable"] / stats["valid"] if stats["valid"] else None
        print(f"{language}: Vulnerability Rate: {rate if rate is not None else 'N/A'}, "
              f"total: {stats['valid']}, vul: {stats['vulnerable']}, excluded: {stats['unavailable']}")
    for error, count in errors.most_common():
        print(f"Unresolved ({count}): {error}")


if __name__ == "__main__":
    main()
