"""Merge three datasets (4.7_puzzles_all.json + images/) into one folder."""

import os
import json
import shutil
import argparse
from tqdm import tqdm
import sys


def parse_args():
    """Exactly three dataset folders + output dir."""
    p = argparse.ArgumentParser(
        description="Merge ds1, ds2, ds3 (each has 4.7_puzzles_all.json + images/) "
        "into a consolidated dataset."
    )
    p.add_argument(
        "--ds1",
        default="./data/step4/Dataset_style_1_4options",
        help="Path to first dataset folder",
    )
    p.add_argument(
        "--ds2",
        default="./data/step4/Dataset_style_2_4options",
        help="Path to second dataset folder",
    )
    p.add_argument(
        "--ds3",
        default="./data/step4/Dataset_style_3_4options",
        help="Path to third dataset folder",
    )
    p.add_argument(
        "--output",
        default="./data/Dataset/4options",
        help="Output directory name",
    )
    return p.parse_args()


def merge_datasets(dataset_folders, output_folder):
    """Core merge logic (unchanged)."""
    out_json = os.path.join(output_folder, "questions.json")
    out_img_dir = os.path.join(output_folder, "images")
    try:
        os.makedirs(out_img_dir, exist_ok=True)
        print(f"Output ready: {output_folder}/{{'questions.json', 'images/'}}")
    except OSError as e:
        print(f"FATAL: cannot create {output_folder}: {e}")
        sys.exit(1)

    merged, skipped, processed, all_items, idx = 0, 0, 0, [], 0
    print("-" * 30)
    for folder in tqdm(dataset_folders, desc="Overall Progress", unit="dataset"):
        idx += 1
        name = os.path.basename(folder) or f"Dataset_{idx}"
        print(f"\nProcessing {name} …")

        json_path = os.path.join(folder, "4.7_puzzles_all.json")
        src_img_dir = os.path.join(folder, "images")
        if not os.path.isfile(json_path):
            print(f"  [SKIP] missing JSON: {json_path}")
            continue
        if not os.path.isdir(src_img_dir):
            print(f"  [SKIP] missing images dir: {src_img_dir}")
            continue

        try:
            data = json.load(open(json_path, encoding="utf-8"))
            if not isinstance(data, list):
                print(f"  [SKIP] JSON not list: {json_path}")
                continue
        except Exception as e:
            print(f"  [SKIP] error reading {json_path}: {e}")
            continue

        print(f"  Loaded {len(data)} items.")
        skip_ds = 0
        for it in tqdm(data, desc=f"  Items from {name}", leave=False, unit="item"):
            processed += 1
            if not isinstance(it, dict) or "id" not in it or "image" not in it:
                skip_ds += 1
                continue

            new_id = f"{it['id']}_{idx}"
            rel_img = it["image"]
            if not (isinstance(rel_img, str) and rel_img.startswith("images" + os.sep)):
                skip_ds += 1
                continue

            src_basename = os.path.basename(rel_img)
            src_img = os.path.join(src_img_dir, src_basename)
            base, ext = os.path.splitext(src_basename)
            new_basename = f"image_{new_id}{ext or '.png'}"
            dst_img = os.path.join(out_img_dir, new_basename)
            if not os.path.isfile(src_img):
                skip_ds += 1
                continue
            try:
                shutil.copy2(src_img, dst_img)
            except Exception:
                skip_ds += 1
                continue

            new_item = it.copy()
            new_item["id"] = new_id
            new_item["image"] = f"images/{new_basename}"
            all_items.append(new_item)
            merged += 1

        if skip_ds:
            print(f"  Skipped {skip_ds} items from {name}.")
        skipped += skip_ds

    print("-" * 30)
    print(f"Saving merged JSON ({len(all_items)} items) → {out_json}")
    try:
        json.dump(
            all_items,
            open(out_json, "w", encoding="utf-8"),
            indent=2,
            ensure_ascii=False,
        )
    except Exception as e:
        print(f"FATAL: cannot save JSON: {e}")
        sys.exit(1)

    print("\n--- Summary ---")
    print(f"Datasets input: {len(dataset_folders)}")
    print(f"Total processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Merged: {merged}")
    print(f"JSON @ {out_json}")
    print(f"Images @ {out_img_dir}")
    print("---------------------")
    print("Done.")


if __name__ == "__main__":
    a = parse_args()
    folders = [a.ds1, a.ds2, a.ds3]
    bad = [f for f in folders if not os.path.isdir(f)]
    if bad:
        print(f"Error: invalid folder(s): {bad}")
        sys.exit(1)
    merge_datasets(folders, a.output)
