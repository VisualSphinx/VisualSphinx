"""Merge variation JSON (with IDs like `123_A`) with a detail JSON by matching
the numeric prefix and copying specified fields."""

import json
import os
import argparse
import sys


def parse_args():
    """CLI paths (defaults keep original behavior)."""
    p = argparse.ArgumentParser(description="Copy extra fields from JSON-2 to JSON-1.")
    p.add_argument(
        "--json1",
        default="./data/step4/Dataset_style_1_4options_shuffle/4.1_puzzles.json",
        help="Variation JSON (IDs like 123_A).",
    )
    p.add_argument(
        "--json2",
        default="./data/step4/Dataset_style_1_4options/4.7_puzzles_all.json",
        help="Detail JSON providing extra fields.",
    )
    p.add_argument(
        "--output",
        default="./data/step4/Dataset_style_1_4options_shuffle/4.7_puzzles_all.json",
        help="Destination JSON after merge.",
    )
    return p.parse_args()


def main(j1_path: str, j2_path: str, out_path: str):
    """Load both JSONs, merge fields by id, write result."""
    keys_to_copy = [
        "explanation",
        "has_duplicate",
        "Reasonableness",
        "Readability",
        "accuracy",
    ]

    print(f"Loading details data source: {j2_path}")
    try:
        list2 = json.load(open(j2_path, encoding="utf-8"))
        if not isinstance(list2, list):
            print(f"Error: {j2_path} must contain a list.")
            sys.exit(1)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: cannot read {j2_path}: {e}")
        sys.exit(1)

    details_lookup, dup2, skip2 = {}, 0, 0
    for it in list2:
        if isinstance(it, dict) and "id" in it:
            try:
                iid = int(it["id"])
                if iid in details_lookup:
                    dup2 += 1
                details_lookup[iid] = it
            except (ValueError, TypeError):
                skip2 += 1
        else:
            skip2 += 1
    if dup2:
        print(f"Warning: {dup2} duplicate IDs in {j2_path}. Using last seen value.")
    print(f"Lookup map size: {len(details_lookup)}; skipped {skip2} bad entries.")

    print(f"Loading variation data: {j1_path}")
    try:
        list1 = json.load(open(j1_path, encoding="utf-8"))
        if not isinstance(list1, list):
            print(f"Error: {j1_path} must contain a list.")
            sys.exit(1)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: cannot read {j1_path}: {e}")
        sys.exit(1)

    merged, matched, nomatch, bad_id = [], 0, 0, 0
    for it in list1:
        if not isinstance(it, dict) or "id" not in it:
            bad_id += 1
            merged.append(it)
            continue
        orig = str(it["id"])
        try:
            base_id = int(orig.split("_")[0])
        except Exception:
            bad_id += 1
            merged.append(it)
            continue
        if base_id in details_lookup:
            matched += 1
            det = details_lookup[base_id]
            for k in keys_to_copy:
                it[k] = det.get(k)
        else:
            nomatch += 1
        merged.append(it)

    print(
        f"Merging done. Processed {len(list1)} items: {matched} matched, "
        f"{nomatch} no match, {bad_id} malformed IDs."
    )
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        json.dump(
            merged, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False
        )
    except Exception as e:
        print(f"Error writing {out_path}: {e}")
        sys.exit(1)
    print("Operation completed successfully!")


if __name__ == "__main__":
    a = parse_args()
    main(a.json1, a.json2, a.output)
