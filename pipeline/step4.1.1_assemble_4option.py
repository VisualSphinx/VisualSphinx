#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate visual puzzles from script outputs using multiprocessing.
"""

import os
import json
import random
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def parse_args():
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate puzzles from script outputs")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./data/step3/3.1_all_scripts_style_1",
        help="Root directory containing script folders",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./data/step3/3.2_valid_style_1.jsonl",
        help="JSON file with a list of valid IDs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../Dataset/Dataset_Synthetic_3_4/raw_data",
        help="Directory to save generated puzzles",
    )
    parser.add_argument(
        "--cell_size", type=int, default=400, help="Size of each puzzle cell in pixels"
    )
    parser.add_argument(
        "--grid_margin",
        type=int,
        default=30,
        help="Margin around the entire puzzle grid",
    )
    parser.add_argument(
        "--max_workers", type=int, default=15, help="Number of parallel workers to use"
    )
    return parser.parse_args()


def process_single_id(id_num, root_dir, image_dir, cell_size, grid_margin):
    """
    Process a single ID: find its latest attempt folder, validate images,
    create the puzzle image, and return puzzle metadata.
    """
    max_attempt = None
    max_num = -1
    for entry in os.listdir(root_dir):
        parts = entry.split("_")
        if len(parts) == 2 and parts[0] == str(id_num):
            try:
                num = int(parts[1])
                if num > max_num:
                    max_num = num
                    max_attempt = parts[1]
            except ValueError:
                continue
    if max_attempt is None:
        return None

    folder = Path(root_dir) / f"{id_num}_{max_attempt}"
    correct_images = get_images_from_dir(folder / "output_correct")
    incorrect_images = get_images_from_dir(folder / "output_incorrect")
    if len(correct_images) != 5 or len(incorrect_images) != 3:
        return None

    images_dir = Path(image_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    output_path = images_dir / f"image_{id_num}.png"
    relative_path = f"images/image_{id_num}.png"
    answer = create_puzzle_image(
        correct_images[:4],
        correct_images[4:],
        incorrect_images,
        output_path,
        cell_size,
        grid_margin,
    )
    return {
        "id": int(id_num),
        "prompt": "From the four given options, select the most suitable one to fill in the question mark to present a certain regularity.",
        "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
        "image": relative_path,
        "correct_answer": answer,
    }


def create_puzzle_from_images(
    id_list, root_dir, output_dir, cell_size, grid_margin, max_workers
):
    """
    Generate puzzles in parallel for a list of IDs.
    """
    image_dir = Path(output_dir) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    puzzles = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_id,
                id_num,
                root_dir,
                image_dir,
                cell_size,
                grid_margin,
            ): id_num
            for id_num in id_list
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                puzzles.append(result)
    return puzzles


def get_images_from_dir(directory):
    """
    Return a sorted list of image file paths in a directory.
    """
    if not directory.is_dir():
        return []
    extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    return [
        str(directory / f)
        for f in sorted(os.listdir(directory))
        if Path(f).suffix.lower() in extensions
    ]


def create_puzzle_image(
    question_images,
    correct_image,
    incorrect_images,
    output_path,
    cell_size=200,
    grid_margin=30,
):
    """
    Create and save the puzzle image, return the correct answer option.
    """
    frame_thickness = 8
    font_size = 85
    row_spacing = 70
    label_spacing = 70
    top_padding = 50
    image_margin = 15
    inner = cell_size - 2 * image_margin

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    def create_cell(path):
        img = Image.open(path).convert("RGBA")
        cell = Image.new("RGB", (inner, inner), "white")
        ar = img.width / img.height
        pad = inner * 5 // 100
        max_w = inner - pad
        max_h = inner - pad
        if ar > 1:
            w = min(max_w, img.width)
            h = int(w / ar)
        else:
            h = min(max_h, img.height)
            w = int(h * ar)
        if img.width > w or img.height > h:
            img = img.resize((w, h), Image.LANCZOS)
        x = (inner - w) // 2
        y = (inner - h) // 2
        if img.mode == "RGBA":
            tmp = Image.new("RGBA", cell.size, (255, 255, 255, 0))
            tmp.paste(img, (x, y))
            cell = Image.alpha_composite(cell.convert("RGBA"), tmp).convert("RGB")
        else:
            cell.paste(img, (x, y))
        return cell

    def create_qm():
        cell = Image.new("RGB", (inner, inner), "white")
        draw = ImageDraw.Draw(cell)
        try:
            qm_font = ImageFont.truetype("arial.ttf", 150)
        except IOError:
            qm_font = ImageFont.load_default()
        draw.text(
            (inner // 2, inner // 2), "?", fill="black", font=qm_font, anchor="mm"
        )
        return cell

    q_cells = [create_cell(p) for p in question_images]
    qm = create_qm()
    answers = [correct_image[0]] + incorrect_images
    random.shuffle(answers)
    a_cells = [create_cell(p) for p in answers]

    top_w = 5 * cell_size
    bot_w = 4 * cell_size
    grid_w = max(top_w, bot_w)
    total_w = grid_w + 2 * grid_margin
    total_h = (
        top_padding
        + 2 * cell_size
        + row_spacing
        + label_spacing
        + font_size
        + grid_margin
    )

    final = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(final)

    x_top = grid_margin + (grid_w - top_w) // 2
    y = top_padding
    for i, cell in enumerate(q_cells):
        final.paste(cell, (x_top + i * cell_size + image_margin, y + image_margin))
    final.paste(qm, (x_top + 4 * cell_size + image_margin, y + image_margin))

    x_bot = grid_margin + (grid_w - bot_w) // 2
    y = top_padding + cell_size + row_spacing
    for i, cell in enumerate(a_cells):
        final.paste(cell, (x_bot + i * cell_size + image_margin, y + image_margin))

    def draw_grid(x0, y0, cols):
        draw.rectangle(
            [(x0, y0), (x0 + cols * cell_size, y0 + cell_size)],
            outline="black",
            width=frame_thickness,
        )
        for c in range(1, cols):
            x = x0 + c * cell_size
            draw.line(
                [(x, y0), (x, y0 + cell_size)], fill="black", width=frame_thickness
            )

    draw_grid(x_top, top_padding, 5)
    draw_grid(x_bot, top_padding + cell_size + row_spacing, 4)

    labels = ["A", "B", "C", "D"]
    for i in range(4):
        lx = x_bot + i * cell_size + cell_size // 2
        ly = top_padding + cell_size + row_spacing + cell_size + label_spacing
        draw.text((lx, ly), labels[i], fill="black", font=font, anchor="mm")

    if final.width > 800:
        scale = 800 / final.width
        final = final.resize((800, int(final.height * scale)), Image.LANCZOS)

    final.save(output_path, quality=95)
    idx = answers.index(correct_image[0])
    return labels[idx]


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.input_file, "r", encoding="utf-8") as f:
        ids = json.load(f).get("ids", [])
    puzzles = create_puzzle_from_images(
        ids,
        args.root_dir,
        args.output_dir,
        args.cell_size,
        args.grid_margin,
        args.max_workers,
    )
    out_file = out_dir / "puzzles.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(puzzles, f, ensure_ascii=False, indent=2)
    print(f"Created {len(puzzles)} puzzles and saved to {out_file}")


if __name__ == "__main__":
    main()
