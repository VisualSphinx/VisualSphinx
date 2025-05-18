"""Fill each item's `explanation` field from a rules-metadata JSON."""

import json
import argparse
import os


def parse_args():
    """Return parsed CLI arguments."""
    p = argparse.ArgumentParser(
        description="Fill each item's `explanation` in a batch JSON "
        "using rule_content from rules metadata JSON"
    )
    p.add_argument(
        "--batch",
        default="./data/step4/Dataset_style_1_4options/puzzles.json",
        help="pipeline/data/step4/Dataset_style_1_4options/puzzles.json",
    )
    p.add_argument(
        "--meta",
        default="./data/step2/2.3_rules.json",
        help="Path to rules metadata JSON",
    )
    p.add_argument(
        "--output",
        default="./data/step4/Dataset_style_1_4options/questions.json",
        help="Destination for updated batch JSON",
    )
    return p.parse_args()


def fill_explanation(batch_path: str, meta_path: str, output_path: str):
    """Update each entry’s `explanation` in batch."""
    with open(batch_path, "r", encoding="utf-8") as f:
        batch = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    rule_map = {item["id"]: item.get("rule_content", []) for item in meta}
    for entry in batch:
        rc = rule_map.get(entry["id"])
        if rc is not None:
            entry["explanation"] = rc

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=2)
    print(f"✅ Wrote updated batch to {output_path}")


def main():
    """Entry point."""
    args = parse_args()
    fill_explanation(args.batch, args.meta, args.output)


if __name__ == "__main__":
    main()
