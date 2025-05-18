"""Merge 10-option puzzles with detail fields from 4-option JSON, but always
reset `accuracy` to None in the result."""

import json
import os
import argparse
import sys


def parse_args():
    """CLI paths (defaults match original script)."""
    p = argparse.ArgumentParser(
        description="Copy fields from JSON-2 to JSON-1 (IDs identical) and null accuracy."
    )
    p.add_argument(
        "--json1",
        default="./data/step4/Dataset_style_1_10options/4.1_puzzles.json",
        help="10-option puzzles JSON.",
    )
    p.add_argument(
        "--json2",
        default="./data/step4/Dataset_style_1_4options/4.7_puzzles_all.json",
        help="Detail JSON with extra fields.",
    )
    p.add_argument(
        "--output",
        default="./data/step4/Dataset_style_1_10options/4.7_puzzles_all.json",
        help="Destination merged JSON.",
    )
    return p.parse_args()


# fields to transfer
FIELDS = ["explanation", "has_duplicate", "Reasonableness", "Readability", "accuracy"]


def load_json(path: str):
    """Read JSON file as list."""
    try:
        data = json.load(open(path, encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading {path}: {e}")
        sys.exit(1)
    if not isinstance(data, list):
        print(f"Error: {path} must contain a JSON list.")
        sys.exit(1)
    return data


def main(j1_path: str, j2_path: str, out_path: str):
    """Merge logic (unchanged)."""
    print(f"Loading details data: {j2_path}")
    list2 = load_json(j2_path)

    lookup, dup2, skip2 = {}, 0, 0
    for it in list2:
        if isinstance(it, dict) and "id" in it:
            try:
                iid = int(it["id"])
                if iid in lookup:
                    dup2 += 1
                lookup[iid] = it
            except (ValueError, TypeError):
                skip2 += 1
        else:
            skip2 += 1
    if dup2:
        print(f"Warning: {dup2} duplicate IDs in {j2_path}.")
    print(f"Lookup size: {len(lookup)}; skipped {skip2} entries.")

    print(f"Loading variations: {j1_path}")
    list1 = load_json(j1_path)

    merged, matched, nomatch, bad = [], 0, 0, 0
    for it in list1:
        if not isinstance(it, dict) or "id" not in it:
            bad += 1
            merged.append(it)
            continue
        try:
            key = int(it["id"])
        except Exception:
            bad += 1
            it["accuracy"] = None
            merged.append(it)
            continue

        if key in lookup:
            matched += 1
            det = lookup[key]
            for k in FIELDS:
                it[k] = det.get(k, None)
        else:
            nomatch += 1
        # always null accuracy
        it["accuracy"] = None
        merged.append(it)

    print(
        f"Done. processed={len(list1)}, matched={matched}, nomatch={nomatch}, id_error={bad}"
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(
        merged, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False
    )
    print(f"Saved merged JSON → {out_path}")


if __name__ == "__main__":
    a = parse_args()
    main(a.json1, a.json2, a.output)
