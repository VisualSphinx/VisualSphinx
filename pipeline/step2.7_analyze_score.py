"""
Analyze distributions of rule ranking scores and count qualified records.
"""

import argparse
import json
from collections import Counter


def parse_args():
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Analyze score distributions and qualification count"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./data/step2/2.6_rules_scoring.json",
        help="Path to the ranked rules JSON file",
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


def main(args):
    """
    Load ranked rules, compute score distributions, and count qualified records.
    """
    with open(args.input_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    format_scores = []
    content_quality_scores = []
    feasibility_scores = []
    total_scores = []
    qualified_count = 0

    for rec in records:
        fs = rec.get("format_score", 0)
        cq = rec.get("content_quality_score", 0)
        fe = rec.get("feasibility_score", 0)
        total = fs + cq + fe
        format_scores.append(fs)
        content_quality_scores.append(cq)
        feasibility_scores.append(fe)
        total_scores.append(total)
        if total >= args.total_threshold and fe >= args.feasibility_threshold:
            qualified_count += 1

    format_dist = Counter(format_scores)
    content_quality_dist = Counter(content_quality_scores)
    feasibility_dist = Counter(feasibility_scores)
    total_dist = Counter(total_scores)

    print("Format Score Distribution:")
    for score, count in sorted(format_dist.items()):
        print(f"Score {score}: {count} records")

    print("\nContent Quality Score Distribution:")
    for score, count in sorted(content_quality_dist.items()):
        print(f"Score {score}: {count} records")

    print("\nFeasibility Score Distribution:")
    for score, count in sorted(feasibility_dist.items()):
        print(f"Score {score}: {count} records")

    print("\nTotal Score Distribution:")
    for score, count in sorted(total_dist.items()):
        print(f"Total Score {score}: {count} records")

    print(f"\nSelected {qualified_count} qualified records")


if __name__ == "__main__":
    args = parse_args()
    main(args)
