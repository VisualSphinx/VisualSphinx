"""Merge accuracy.json into questions_with_rank.json (adds `accuracy` per ID)."""

import json
import argparse
from pathlib import Path
import sys


def parse_args():
    """CLI paths and tunables."""
    p = argparse.ArgumentParser(
        description="Append `accuracy` field to question items."
    )
    p.add_argument(
        "--accuracy",
        default="./data/step4/Dataset_style_1_4options/4.5_output/accuracy.json",
    )
    p.add_argument(
        "--detail",
        default="./data/step4/Dataset_style_1_4options/4.6_puzzles_with_scoring.json",
    )
    p.add_argument(
        "--output",
        default="./data/step4/Dataset_style_1_4options/4.7_puzzles_all.json",
    )
    return p.parse_args()


def main(acc_path: Path, detail_path: Path, out_path: Path):
    """Load both JSONs, merge on id, write result."""
    try:
        print(f"Loading accuracy data from: {acc_path}")
        accuracy_json = json.loads(acc_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Error: File not found {acc_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: JSON decode {acc_path}")
        sys.exit(1)

    accuracy_map, bad_keys = {}, 0
    for k, v in accuracy_json.items():
        try:
            accuracy_map[int(k)] = v
        except ValueError:
            print(f"Warning: accuracy key '{k}' not int, skipped")
            bad_keys += 1
    print(
        f"Loaded {len(accuracy_map)} accuracy entries; skipped {bad_keys} invalid keys."
    )

    try:
        print(f"Loading detailed data from: {detail_path}")
        detail_list = json.loads(detail_path.read_text(encoding="utf-8"))
        if not isinstance(detail_list, list):
            raise TypeError(f"{detail_path} should contain a JSON list.")
    except FileNotFoundError:
        print(f"Error: File not found {detail_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: JSON decode {detail_path}")
        sys.exit(1)
    except TypeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    merged, not_found = 0, 0
    for item in detail_list:
        if not isinstance(item, dict) or "id" not in item:
            print(f"Warning: skipping malformed item: {item}")
            continue
        try:
            iid = int(item["id"])
        except (ValueError, TypeError):
            print(f"Warning: ID '{item['id']}' not int")
            not_found += 1
            continue
        if iid in accuracy_map:
            item["accuracy"] = accuracy_map[iid]
            merged += 1
        else:
            not_found += 1

    print(f"Merging complete: {merged} added, {not_found} missing.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(detail_list, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved merged data to: {out_path}")


if __name__ == "__main__":
    a = parse_args()
    main(Path(a.accuracy), Path(a.detail), Path(a.output))
