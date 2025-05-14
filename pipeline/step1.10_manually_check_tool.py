"""
Analyze JSON file to count (question_type, knowledge_point) combinations and
identify entries where one tag is 'Others' and the other is not.
"""

import argparse
import json
from collections import Counter


def parse_args():
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json", type=str, default="./data/step1_filtered/1.9_seeds.json"
    )
    return parser.parse_args()


def analyze_json_tags(input_json_path: str) -> dict:
    """
    Load JSON file, count tag combinations, identify entries with mismatched 'Others' tags,
    print summaries, and return the results.
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    combination_counter = Counter()
    mismatched = []

    for item in data:
        q_type = item.get("question_type", "")
        k_point = item.get("knowledge_point", "")
        combination_counter[(q_type, k_point)] += 1
        if (q_type == "Others") ^ (k_point == "Others"):
            mismatched.append(
                {
                    "seed_id": item.get("seed_id"),
                    "question_type": q_type,
                    "knowledge_point": k_point,
                }
            )

    print("Combination counts:")
    print("-" * 50)
    print(f"{'Question Type':<20} | {'Knowledge Point':<20} | {'Count':<5}")
    print("-" * 50)
    for (q, k), count in sorted(combination_counter.items()):
        print(f"{q:<20} | {k:<20} | {count:<5}")

    print("\nEntries with one tag 'Others' and the other not:")
    print("-" * 50)
    print(f"{'Seed ID':<10} | {'Question Type':<20} | {'Knowledge Point':<20}")
    print("-" * 50)
    for entry in mismatched:
        print(
            f"{entry['seed_id']:<10} | {entry['question_type']:<20} | {entry['knowledge_point']:<20}"
        )

    return {
        "combinations": dict(combination_counter),
        "others_but_not_both": mismatched,
    }


def main(args):
    """
    Main entry point: run analysis on provided JSON file.
    """
    analyze_json_tags(args.input_json)


if __name__ == "__main__":
    args = parse_args()
    main(args)
