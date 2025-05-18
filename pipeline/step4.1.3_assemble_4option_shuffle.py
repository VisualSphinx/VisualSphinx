#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate 4 puzzle variations per ID by placing the correct answer in each option position.
"""

import os
import json
import random
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

NUM_CORRECT_NEEDED = 5
NUM_INCORRECT_NEEDED = 3
CELL_SIZE_DEFAULT = 200
GRID_MARGIN_DEFAULT = 30
WORKERS_DEFAULT = 4
OUTPUT_DIR_DEFAULT = "../Dataset/Dataset_Synthetic_2_4_shuffle/raw_data"
INPUT_FILE_DEFAULT = "valid_ids_2_new.json"
ROOT_DIR_DEFAULT = "all_scripts_2"


def parse_args():
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate 4 puzzle variations per ID using multiprocessing"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR_DEFAULT,
        help="Directory to save puzzle variations",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=INPUT_FILE_DEFAULT,
        help="JSON file containing list of valid IDs",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=ROOT_DIR_DEFAULT,
        help="Root directory containing ID_attempt folders",
    )
    parser.add_argument(
        "--cell_size",
        type=int,
        default=CELL_SIZE_DEFAULT,
        help="Size of each puzzle cell in pixels",
    )
    parser.add_argument(
        "--grid_margin",
        type=int,
        default=GRID_MARGIN_DEFAULT,
        help="Margin around the puzzle grid",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=WORKERS_DEFAULT,
        help="Number of worker processes to use",
    )
    return parser.parse_args()


def find_max_attempt(root_dir: str, id_num: str) -> str:
    """
    Find the highest-attempt folder name for a given ID.
    """
    max_num = -1
    result = None
    if not os.path.isdir(root_dir):
        return None
    for entry in os.listdir(root_dir):
        parts = entry.split("_")
        if len(parts) == 2 and parts[0] == str(id_num):
            try:
                num = int(parts[1])
                if num > max_num:
                    max_num = num
                    result = entry
            except ValueError:
                continue
    return result


def get_images_from_dir(directory: str) -> list:
    """
    Return a sorted list of image file paths from a directory.
    """
    if not os.path.isdir(directory):
        return []
    exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if Path(f).suffix.lower() in exts
    ]


def create_puzzle_image(
    question_imgs: list,
    correct_img_path: str,
    incorrect_imgs_paths: list,
    output_path: str,
    target_label: str,
    cell_size: int,
    grid_margin: int,
) -> str:
    """
    Create and save a puzzle image with the correct answer at target_label.
    Returns the label of the correct answer.
    """
    labels = ["A", "B", "C", "D"]
    if target_label not in labels:
        raise ValueError(f"target_label must be one of {labels}")
    inner = cell_size - 2 * grid_margin
    try:
        font = ImageFont.truetype("arial.ttf", 85)
    except IOError:
        font = ImageFont.load_default()
    questions = [Image.open(p).convert("RGBA") for p in question_imgs]
    correct = Image.open(correct_img_path).convert("RGBA")
    incorrects = [Image.open(p).convert("RGBA") for p in incorrect_imgs_paths]
    option_images = [None] * 4
    idx = labels.index(target_label)
    option_images[idx] = correct
    remaining = [i for i in range(4) if i != idx]
    random.shuffle(incorrects)
    for i, img in zip(remaining, incorrects):
        option_images[i] = img

    def make_cell(img: Image.Image) -> Image.Image:
        cell = Image.new("RGB", (inner, inner), "white")
        aspect = img.width / img.height
        if aspect > 1:
            w = inner
            h = int(inner / aspect)
        else:
            h = inner
            w = int(inner * aspect)
        w = max(1, w)
        h = max(1, h)
        try:
            resized = img.resize((w, h), Image.Resampling.LANCZOS)
        except AttributeError:
            resized = img.resize((w, h), Image.LANCZOS)
        x = (inner - w) // 2
        y = (inner - h) // 2
        if resized.mode == "RGBA":
            layer = Image.new("RGBA", cell.size, (255, 255, 255, 0))
            layer.paste(resized, (x, y), resized)
            cell = Image.alpha_composite(cell.convert("RGBA"), layer).convert("RGB")
        else:
            cell.paste(resized, (x, y))
        return cell

    cells_q = [make_cell(img) for img in questions]
    qm = Image.new("RGB", (inner, inner), "white")
    draw = ImageDraw.Draw(qm)
    try:
        qf = ImageFont.truetype("arial.ttf", 150)
    except IOError:
        qf = ImageFont.load_default()
    draw.text((inner // 2, inner // 2), "?", fill="black", font=qf, anchor="mm")
    cells_o = [make_cell(img) for img in option_images]

    width = 2 * grid_margin + 5 * cell_size
    height = (
        2 * grid_margin + cell_size * 2 + 70 + 70 + 85
    )  # grid_margin, rows, spacings, labels
    final = Image.new("RGB", (width, height), "white")
    d_final = ImageDraw.Draw(final)
    x0 = grid_margin
    y0 = grid_margin
    for i, cell in enumerate(cells_q):
        final.paste(cell, (x0 + i * cell_size, y0))
    final.paste(qm, (x0 + 4 * cell_size, y0))
    y0 += cell_size + 70
    for i, cell in enumerate(cells_o):
        final.paste(cell, (x0 + i * cell_size, y0))
    y_lbl = y0 + cell_size + 35
    for i, lbl in enumerate(labels):
        d_final.text(
            (x0 + i * cell_size + cell_size // 2, y_lbl),
            lbl,
            fill="black",
            font=font,
            anchor="mm",
        )
    final.save(output_path, quality=95)
    return target_label


def process_single_id(
    id_num: str,
    root_dir: str,
    image_dir: str,
    cell_size: int,
    grid_margin: int,
) -> list:
    """
    Generate four puzzle variations for an ID; return list of dicts or None.
    """
    folder_name = find_max_attempt(root_dir, id_num)
    if not folder_name:
        return None
    folder = os.path.join(root_dir, folder_name)
    corrects = get_images_from_dir(os.path.join(folder, "output_correct"))
    wrongs = get_images_from_dir(os.path.join(folder, "output_incorrect"))
    if len(corrects) != NUM_CORRECT_NEEDED or len(wrongs) != NUM_INCORRECT_NEEDED:
        return None
    os.makedirs(image_dir, exist_ok=True)
    puzzles = []
    for label in ["A", "B", "C", "D"]:
        fname = f"image_{id_num}_{label}.png"
        outp = os.path.join(image_dir, fname)
        relp = f"images/{fname}"
        ans = create_puzzle_image(
            corrects[:4], corrects[4], wrongs, outp, label, cell_size, grid_margin
        )
        puzzles.append(
            {
                "id": f"{id_num}_{label}",
                "prompt": "From the four given options, select the most suitable one to fill in the question mark to present a certain regularity.",
                "options": {l: l for l in ["A", "B", "C", "D"]},
                "image": relp,
                "correct_answer": ans,
            }
        )
    return puzzles


def create_puzzles(
    id_list: list,
    root_dir: str,
    output_dir: str,
    cell_size: int,
    grid_margin: int,
    max_workers: int,
) -> list:
    """
    Parallel generation of puzzles for given IDs; returns flat list of all variations.
    """
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    all_puzzles = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_id,
                str(idn),
                root_dir,
                images_dir,
                cell_size,
                grid_margin,
            ): idn
            for idn in id_list
        }
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                all_puzzles.extend(res)
    return all_puzzles


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.input_file, "r", encoding="utf-8") as f:
        ids = json.load(f).get("ids", [])
    puzzles = create_puzzles(
        ids,
        args.root_dir,
        args.output_dir,
        args.cell_size,
        args.grid_margin,
        args.max_workers,
    )
    out_file = os.path.join(args.output_dir, "puzzles_variations.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(puzzles, f, ensure_ascii=False, indent=2)
    print(f"Created {len(puzzles)} puzzles and saved to {out_file}")


if __name__ == "__main__":
    main()
