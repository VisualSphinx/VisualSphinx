"""
Fast pHash / SSIM duplicate-blank checker — multiprocessing version
"""

import argparse, json, re, sys, os
from pathlib import Path
from itertools import combinations
from typing import List, Dict, Tuple, Any
import math
from multiprocessing import Pool, cpu_count

from PIL import Image, ImageStat
import imagehash
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


# -------------------- CLI -------------------- #
def get_args():
    ap = argparse.ArgumentParser("pHash / SSIM checker (multiprocess)")
    ap.add_argument("--ids_json", default="./data/step3/3.2_valid_style_1.json")
    ap.add_argument("--root_dir", default="./data/step3/3.1_all_scripts_style_1")
    ap.add_argument(
        "--summary_file", default="./data/step3/3.3_phash_summary_style_1.jsonl"
    )
    ap.add_argument("--raw_file", default="./data/step3/3.3_phash_raw_style_1.jsonl")

    ap.add_argument("--hash_thresh", type=int, default=2)
    ap.add_argument("--dup_pair_thresh", type=int, default=2)
    ap.add_argument("--std_thresh", type=float, default=2.0)
    ap.add_argument("--ssim_thresh", type=float, default=0.99)

    ap.add_argument("--num_workers", type=int, default=cpu_count())
    return ap.parse_args()


ARGS = get_args()
ROOT = Path(ARGS.root_dir)

# -------------------- helpers ---------------- #
patt_id = re.compile(r"^(\d+)_\d+$")


def latest_folder(id_str: str) -> Path | None:
    """Return <root>/<id>_<max_attempt>"""
    patt = re.compile(rf"^{re.escape(id_str)}_(\d+)$")
    best_a, best_p = -1, None
    for p in ROOT.iterdir():
        m = patt.match(p.name)
        if m and p.is_dir():
            a = int(m.group(1))
            if a > best_a:
                best_a, best_p = a, p
    return best_p


def collect_paths(folder: Path) -> list[Path]:
    paths = []
    for sub in ["output_correct", "output_incorrect"]:
        imgs = sorted(
            [
                f
                for f in (folder / sub).iterdir()
                if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
            ]
        )
        paths.extend(imgs)
    if len(paths) != 8:
        raise RuntimeError(f"{folder}: expected 8 images, got {len(paths)}")
    return paths


# ----- metrics ----- #
def phash(img: Image.Image) -> str:
    return str(imagehash.phash(img, hash_size=16))  # 256-bit hex string


def is_blank(img: Image.Image) -> bool:
    g = img.convert("L")
    try:
        if ImageStat.Stat(g).stddev[0] < ARGS.std_thresh:
            return True
    except ValueError as e:
        if "math domain error" in str(e):
            return True
        else:
            raise e

    try:
        g_small = np.array(g.resize((256, 256)))
        white = np.full_like(g_small, 255)
        ssim_val = ssim(g_small, white)
        return bool(ssim_val > ARGS.ssim_thresh)  # Cast to Python bool
    except Exception as e:
        sys.stderr.write(f"[WARN] Error during SSIM calculation: {e}\n")
        return False


# -------------------- worker ----------------- #
def process_id(id_str: str) -> Tuple[Dict[str, Any], Dict[str, Any]] | None:
    folder = latest_folder(id_str)
    if folder is None:
        sys.stderr.write(f"[WARN] {id_str}: folder not found\n")
        return None
    try:
        paths = collect_paths(folder)
    except RuntimeError as e:
        sys.stderr.write(f"[WARN] {e}\n")
        return None

    imgs = [Image.open(p).convert("RGB") for p in paths]
    hashes = [phash(im) for im in imgs]
    blank_flags = [is_blank(im) for im in imgs]

    # duplicate pairs via Hamming distance
    dup_pairs = 0
    dup_map = {i: [] for i in range(8)}
    for (i, h1), (j, h2) in combinations(enumerate(hashes), 2):
        if imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2) <= ARGS.hash_thresh:
            dup_pairs += 1
            dup_map[i].append(j)
            dup_map[j].append(i)
            if dup_pairs >= ARGS.dup_pair_thresh:
                # still fill dup_map fully for raw output
                pass

    has_dup = dup_pairs >= ARGS.dup_pair_thresh
    has_blank = any(blank_flags)

    summary_rec = {"id": id_str, "has_duplicate": has_dup, "has_blank": has_blank}

    raw_rec = {
        "id": id_str,
        "images": [
            {
                "idx": i,
                "path": str(paths[i]),
                "phash": hashes[i],
                "blank": blank_flags[i],
                "dups": dup_map[i],
            }
            for i in range(8)
        ],
    }
    return summary_rec, raw_rec


# -------------------- main ------------------- #
if __name__ == "__main__":
    ids = json.load(open(ARGS.ids_json, "r", encoding="utf-8"))["ids"]
    n_workers = max(1, ARGS.num_workers)

    with Pool(processes=n_workers) as pool, open(
        ARGS.summary_file, "w", encoding="utf-8"
    ) as f_sum, open(ARGS.raw_file, "w", encoding="utf-8") as f_raw:

        for res in tqdm(
            pool.imap_unordered(process_id, ids),
            total=len(ids),
            desc=f"workers={n_workers}",
        ):
            if res is None:
                continue
            summary_rec, raw_rec = res
            f_sum.write(json.dumps(summary_rec) + "\n")
            f_sum.flush()
            f_raw.write(json.dumps(raw_rec) + "\n")
            f_raw.flush()

    print(f"Finished → {ARGS.summary_file} & {ARGS.raw_file}")
