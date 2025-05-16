#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge deduplicated and ranked rule records, apply deduplication and scoring filters,
and save the qualified records.
"""

import argparse
import json


def parse_args():
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Merge dedup and score files, filter records by dedup and score thresholds"
    )
    parser.add_argument(
        "--dedup_file",
        type=str,
        default="./data/step2/deduplication/2.3_rules_duplicates.jsonl",
        help="Path to the deduplicated JSONL file",
    )
    parser.add_argument(
        "--score_file",
        type=str,
        default="./data/step2/2.6_rules_scoring.json",
        help="Path to the ranked rules JSON file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./data/step2/2.8_rules_filtered.json",
        help="Path to save the merged and filtered JSON file",
    )
    parser.add_argument(
        "--total_threshold", type=int, default=12, help="Minimum total score to qualify"
    )
    parser.add_argument(
        "--feasibility_threshold",
        type=int,
        default=3,
        help="Minimum feasibility score to qualify",
    )
    return parser.parse_args()


def load_jsonl(filename: str) -> list[dict]:
    """
    Load a JSONL file and return a list of records.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main(args):
    """
    Merge deduplication and scoring records, apply filters, and write qualified records.
    """
    dedup_records = load_jsonl(args.dedup_file)
    with open(args.score_file, "r", encoding="utf-8") as f:
        score_records = json.load(f)

    dedup_dict = {rec["id"]: rec for rec in dedup_records}
    score_dict = {rec["id"]: rec for rec in score_records}

    merged_records = []
    for record_id, dedup_rec in dedup_dict.items():
        score_rec = score_dict.get(record_id)
        if score_rec is None:
            print(f"Warning: id {record_id} missing in scoring file.")
            continue

        merged = {}
        merged.update(dedup_rec)
        merged.update(score_rec)

        dedup_ok = merged.get("repeat_count", 0) == 0 or merged.get(
            "min_similar_rule_id"
        ) == merged.get("id")

        fs = merged.get("format_score", 0)
        cq = merged.get("content_quality_score", 0)
        fe = merged.get("feasibility_score", 0)
        total = fs + cq + fe
        score_ok = (total >= args.total_threshold) and (
            fe >= args.feasibility_threshold
        )

        if dedup_ok and score_ok:
            merged_records.append(merged)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(merged_records, f, ensure_ascii=False, indent=2)

    print(f"Total qualified records: {len(merged_records)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
