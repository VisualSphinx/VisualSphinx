"""
Generate 4-option puzzle
"""

import os
import json
import random
import argparse
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

parser = argparse.ArgumentParser(description="Generate puzzles with multiprocessing")
parser.add_argument(
    "--output",
    type=str,
    default="./data/step4/Dataset_style_1_4options",
    help="Base output directory",
)
parser.add_argument(
    "--input",
    dest="input_file",
    type=str,
    default="./data/step3/3.2_valid_style_1.json",
    help="JSON file with list of IDs to process",
)
parser.add_argument(
    "--root",
    dest="root_dir",
    type=str,
    default="./data/step3/3.1_all_scripts_style_1",
    help="Root directory containing ID_attempt folders",
)
parser.add_argument(
    "--cell_size", type=int, default=400, help="Size of each puzzle cell in pixels"
)
parser.add_argument(
    "--grid_margin", type=int, default=30, help="Margin around the puzzle grid"
)
parser.add_argument(
    "--workers",
    dest="max_workers",
    type=int,
    default=15,
    help="Number of worker processes",
)
args = parser.parse_args()

# --- Helper functions ---


def find_max_attempt(root_dir: str, id_num) -> str | None:
    """Find the maximum attempt number for a given ID."""
    max_num = -1
    selected = None
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path):
            parts = name.split("_")
            if len(parts) == 2 and parts[0] == str(id_num):
                try:
                    num = int(parts[1])
                    if num > max_num:
                        max_num, selected = num, parts[1]
                except ValueError:
                    continue
    return selected


def get_images_from_dir(directory: str) -> list[str]:
    """Get a sorted list of image paths from a directory."""
    if not os.path.isdir(directory):
        return []
    exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if os.path.splitext(f.lower())[1] in exts
    ]


def create_puzzle_image(
    question_images: list[str],
    correct_image: list[str],
    incorrect_images: list[str],
    output_path: str,
    cell_size: int,
    grid_margin: int,
) -> str:
    """
    Create the puzzle image with given parameters.
    Returns the correct answer label.
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

    imgs_q = [Image.open(p).convert("RGBA") for p in question_images]
    opts = [Image.open(correct_image[0]).convert("RGBA")] + [
        Image.open(p).convert("RGBA") for p in incorrect_images
    ]
    random.shuffle(opts)
    labels = ["A", "B", "C", "D"]
    # Determine correct answer label
    correct = labels[opts.index(opts[0])]

    def cell(img: Image.Image) -> Image.Image:
        bg = Image.new("RGB", (inner, inner), "white")
        ar = img.width / img.height
        pad = inner * 5 // 100
        maxw, maxh = inner - pad, inner - pad
        if ar > 1:
            w = min(maxw, int(inner))
            h = int(w / ar)
        else:
            h = min(maxh, int(inner))
            w = int(h * ar)
        w, h = max(1, w), max(1, h)
        img2 = img.resize((w, h), Image.LANCZOS)
        x, y = (inner - w) // 2, (inner - h) // 2
        if img2.mode == "RGBA":
            tmp = Image.new("RGBA", bg.size, (255, 255, 255, 0))
            tmp.paste(img2, (x, y), img2)
            bg = Image.alpha_composite(bg.convert("RGBA"), tmp).convert("RGB")
        else:
            bg.paste(img2, (x, y))
        return bg

    qm = Image.new("RGB", (inner, inner), "white")
    d = ImageDraw.Draw(qm)
    try:
        qf = ImageFont.truetype("arial.ttf", 150)
    except:
        qf = ImageFont.load_default()
    d.text((inner // 2, inner // 2), "?", font=qf, fill="black", anchor="mm")

    # compose
    top_w = 5 * cell_size
    bot_w = 4 * cell_size
    W = max(top_w, bot_w) + 2 * grid_margin
    H = (
        top_padding
        + 2 * cell_size
        + row_spacing
        + label_spacing
        + font_size
        + grid_margin
    )
    out = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(out)

    ox = grid_margin + (W - 2 * grid_margin - top_w) // 2
    oy = top_padding
    for i, img in enumerate(imgs_q):
        out.paste(cell(img), (ox + i * cell_size + image_margin, oy + image_margin))
    out.paste(qm, (ox + 4 * cell_size + image_margin, oy + image_margin))

    oy += cell_size + row_spacing
    ox2 = grid_margin + (W - 2 * grid_margin - bot_w) // 2
    for i, img in enumerate(opts):
        out.paste(cell(img), (ox2 + i * cell_size + image_margin, oy + image_margin))

    # grid and labels
    draw.rectangle(
        [(ox, top_padding), (ox + top_w, top_padding + cell_size)],
        outline="black",
        width=frame_thickness,
    )
    for i in range(1, 5):
        draw.line(
            [
                (ox + i * cell_size, top_padding),
                (ox + i * cell_size, top_padding + cell_size),
            ],
            fill="black",
            width=frame_thickness,
        )
    draw.rectangle(
        [(ox2, oy), (ox2 + bot_w, oy + cell_size)],
        outline="black",
        width=frame_thickness,
    )
    for i in range(1, 4):
        draw.line(
            [(ox2 + i * cell_size, oy), (ox2 + i * cell_size, oy + cell_size)],
            fill="black",
            width=frame_thickness,
        )
    for i, label in enumerate(labels):
        lx = ox2 + i * cell_size + cell_size // 2
        ly = oy + cell_size + label_spacing
        draw.text((lx, ly), label, font=font, fill="black", anchor="mm")

    if out.width > 800:
        sc = 800 / out.width
        out = out.resize((800, int(out.height * sc)), Image.LANCZOS)
    out.save(output_path, quality=95)
    return correct


def process_single_id(id_num, root_dir, image_dir, cell_size, grid_margin):
    """Process one ID to generate a puzzle dict or None."""
    att = find_max_attempt(root_dir, id_num)
    if not att:
        return None
    folder = os.path.join(root_dir, f"{id_num}_{att}")
    ci = get_images_from_dir(os.path.join(folder, "output_correct"))
    ii = get_images_from_dir(os.path.join(folder, "output_incorrect"))
    if len(ci) != 5 or len(ii) != 3:
        return None
    op = os.path.join(image_dir, f"image_{id_num}.png")
    rel = f"images/image_{id_num}.png"
    ans = create_puzzle_image(ci[:4], ci[4:], ii, op, cell_size, grid_margin)
    return {
        "id": int(id_num),
        "prompt": "From the four given options, select the most suitable one to fill in the question mark to present a certain regularity.",
        "options": {l: l for l in "ABCD"},
        "image": rel,
        "correct_answer": ans,
    }


def create_puzzle_from_images(
    id_list, root_dir, output_dir, cell_size, grid_margin, max_workers
):
    """Generate puzzles for all IDs with multiprocessing and progress bar."""
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    puzzles = []
    args_list = [(i, root_dir, img_dir, cell_size, grid_margin) for i in id_list]
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_single_id, *a): a[0] for a in args_list}
        for fut in tqdm(
            as_completed(futs), total=len(futs), desc="Processing IDs", unit="id"
        ):
            try:
                r = fut.result()
                if r:
                    puzzles.append(r)
            except Exception as e:
                print(f"Error ID {futs[fut]}: {e}")
    return puzzles


# --- Main ---
if __name__ == "__main__":
    os.makedirs(args.output, exist_ok=True)
    with open(args.input_file, "r", encoding="utf-8") as f:
        id_list = json.load(f).get("ids", [])
    puzzles = create_puzzle_from_images(
        id_list,
        args.root_dir,
        args.output,
        args.cell_size,
        args.grid_margin,
        args.max_workers,
    )
    out_json = os.path.join(args.output, "4.1_puzzles.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(puzzles, f, indent=2)
    print(f"Created {len(puzzles)} puzzles and saved to {out_json}")
