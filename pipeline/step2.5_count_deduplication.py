"""
Count duplicate and non-duplicate records in a JSONL file based on min_similar_rule_id.
"""

import argparse
import json


def parse_args():
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default="./data/step2/deduplication/2.3_rules_duplicates.jsonl",
    )
    return parser.parse_args()


def count_duplicates(file_path: str) -> tuple[int, int, int]:
    """
    Count total, duplicate, and non-duplicate records in a JSONL file.

    A record is a duplicate if min_similar_rule_id is not None and
    different from its own id. Otherwise it's non-duplicate.
    """
    duplicate_count = 0
    non_duplicate_count = 0
    total = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            record = json.loads(line)
            min_id = record.get("min_similar_rule_id")
            own_id = record.get("id")
            if min_id is not None and min_id != own_id:
                duplicate_count += 1
            else:
                non_duplicate_count += 1

    return total, duplicate_count, non_duplicate_count


def main(args):
    """
    Main entry point: count duplicates and print results.
    """
    total, duplicates, non_duplicates = count_duplicates(args.file_path)
    print(f"Total records: {total}")
    print(f"Duplicate records count: {duplicates}")
    print(f"Non-duplicate records count: {non_duplicates}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
