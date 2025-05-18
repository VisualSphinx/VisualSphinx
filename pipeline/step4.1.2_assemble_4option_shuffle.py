"""
Generate 4-option puzzle variations (A–D) for each ID.
"""

import os
import json
import random
from PIL import Image, ImageDraw, ImageFont
import shutil
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# ------------------------------------------------------------------
# argument parsing
# ------------------------------------------------------------------
def parse_args():
    """Return CLI arguments."""
    p = argparse.ArgumentParser(
        description="Generate 4 puzzle variations per ID with multiprocessing"
    )
    p.add_argument(
        "--output",
        type=str,
        default="./data/step4/Dataset_style_1_4options_shuffle",
    )
    p.add_argument("--input", type=str, default="./data/step3/3.2_valid_style_1.json")
    p.add_argument("--root", type=str, default="./data/step3/3.1_all_scripts_style_1")
    p.add_argument("--cell_size", type=int, default=400)
    p.add_argument("--grid_margin", type=int, default=25)
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


# ------------------------------------------------------------------
# helper functions
# ------------------------------------------------------------------
def find_max_attempt(root_dir, id_num):
    """Return largest attempt index str for a given ID."""
    max_attempt, max_num = None, -1
    for item in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, item)):
            parts = item.split("_")
            if len(parts) == 2 and parts[0] == str(id_num):
                try:
                    n = int(parts[1])
                    if n > max_num:
                        max_num, max_attempt = n, parts[1]
                except ValueError:
                    continue
    return max_attempt


def get_images_from_dir(directory):
    """Return sorted image paths in directory."""
    if not os.path.exists(directory):
        return []
    exts = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if any(f.lower().endswith(ext) for ext in exts)
    ]


# ------------------------------------------------------------------
# core image composition
# ------------------------------------------------------------------
def create_puzzle_image(
    question_images,
    correct_image_path,
    incorrect_image_paths,
    output_path,
    target_correct_label,
    cell_size=200,
    grid_margin=30,
):
    """Compose final 4-option puzzle and return the correct label."""
    labels = ["A", "B", "C", "D"]
    if target_correct_label not in labels:
        raise ValueError("target_correct_label must be A/B/C/D")

    frame_thickness = 8
    font_size, row_spacing = 85, 70
    label_spacing, top_padding, image_margin = 70, 50, 20
    inner = cell_size - 2 * image_margin

    font = (
        ImageFont.truetype("arial.ttf", font_size)
        if os.path.exists("arial.ttf")
        else ImageFont.load_default()
    )
    q_imgs = [Image.open(p).convert("RGBA") for p in question_images]
    corr_img = Image.open(correct_image_path).convert("RGBA")
    inc_imgs = [Image.open(p).convert("RGBA") for p in incorrect_image_paths]

    def content_cell(img):
        cell = Image.new("RGB", (inner, inner), "white")
        ar = img.width / img.height
        pad_pct = 5
        max_w = inner - inner * pad_pct // 100
        max_h = inner - inner * pad_pct // 100
        if ar > 1:
            nw = min(max_w, inner)
            nh = int(nw / ar)
            if nh > max_h:
                nh = max_h
                nw = int(nh * ar)
        else:
            nh = min(max_h, inner)
            nw = int(nh * ar)
            if nw > max_w:
                nw = max_w
                nh = int(nw / ar)
        nw, nh = max(1, nw), max(1, nh)
        res = img.resize((nw, nh), Image.Resampling.LANCZOS)
        xo, yo = (inner - nw) // 2, (inner - nh) // 2
        if res.mode == "RGBA":
            tmp = Image.new("RGBA", cell.size, (255, 255, 255, 0))
            tmp.paste(res, (xo, yo), res)
            cell = Image.alpha_composite(cell.convert("RGBA"), tmp).convert("RGB")
        else:
            cell.paste(res.convert("RGB"), (xo, yo))
        return cell

    def qm_cell():
        cell = Image.new("RGB", (inner, inner), "white")
        d = ImageDraw.Draw(cell)
        qf = (
            ImageFont.truetype("arial.ttf", 150)
            if os.path.exists("arial.ttf")
            else ImageFont.load_default()
        )
        d.text((inner // 2, inner // 2), "?", font=qf, fill="black", anchor="mm")
        return cell

    q_cells = [content_cell(i) for i in q_imgs]
    ans_slots = [None] * 4
    idx = labels.index(target_correct_label)
    ans_slots[idx] = corr_img
    rem_idx = [i for i, l in enumerate(labels) if l != target_correct_label]
    random.shuffle(inc_imgs)
    for i, img in enumerate(inc_imgs):
        ans_slots[rem_idx[i]] = img
    ans_cells = [content_cell(i) for i in ans_slots]

    cell_sp = 0
    top_w, bot_w = 5 * cell_size, 4 * cell_size
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
    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    # place top row
    x_top = grid_margin + (grid_w - top_w) // 2
    y = top_padding
    for i, c in enumerate(q_cells):
        canvas.paste(c, (x_top + i * cell_size + image_margin, y + image_margin))
    canvas.paste(qm_cell(), (x_top + 4 * cell_size + image_margin, y + image_margin))

    # place bottom row
    x_bot = grid_margin + (grid_w - bot_w) // 2
    y_bot = top_padding + cell_size + row_spacing
    for i, c in enumerate(ans_cells):
        canvas.paste(c, (x_bot + i * cell_size + image_margin, y_bot + image_margin))

    # grid lines
    draw.rectangle(
        [(x_top, y), (x_top + 5 * cell_size, y + cell_size)],
        outline="black",
        width=frame_thickness,
    )
    for i in range(1, 5):
        lx = x_top + i * cell_size
        draw.line([(lx, y), (lx, y + cell_size)], fill="black", width=frame_thickness)
    draw.rectangle(
        [(x_bot, y_bot), (x_bot + 4 * cell_size, y_bot + cell_size)],
        outline="black",
        width=frame_thickness,
    )
    for i in range(1, 4):
        lx = x_bot + i * cell_size
        draw.line(
            [(lx, y_bot), (lx, y_bot + cell_size)], fill="black", width=frame_thickness
        )

    # option labels
    for i, lab in enumerate(labels):
        lx = x_bot + i * cell_size + cell_size // 2
        ly = y_bot + cell_size + label_spacing
        draw.text((lx, ly), lab, font=font, fill="black", anchor="mm")

    if canvas.width > 800:
        scale = 800 / canvas.width
        canvas = canvas.resize((800, int(canvas.height * scale)), Image.LANCZOS)
    canvas.save(output_path, quality=95)
    return target_correct_label


# ------------------------------------------------------------------
# per-ID processing
# ------------------------------------------------------------------
def process_single_id(id_num, root_dir, image_dir, cell_size, grid_margin):
    """Generate 4 variations for one ID; return list of dicts."""
    max_attempt = find_max_attempt(root_dir, id_num)
    if max_attempt is None:
        print(f"No valid folder found for ID {id_num}")
        return None

    folder = os.path.join(root_dir, f"{id_num}_{max_attempt}")
    correct = get_images_from_dir(os.path.join(folder, "output_correct"))
    incorrect = get_images_from_dir(os.path.join(folder, "output_incorrect"))
    if len(correct) != 5 or len(incorrect) != 3:
        print(
            f"ID {id_num}: Invalid image count - {len(correct)} correct, {len(incorrect)} incorrect. Skipping."
        )
        return None

    qs, corr_img = correct[:4], correct[4]
    variations, labels = [], ["A", "B", "C", "D"]
    for lab in labels:
        vid = f"{id_num}_{lab}"
        img_name = f"image_{vid}.png"
        out_path = os.path.join(image_dir, img_name)
        rel_path = f"images/{img_name}"
        try:
            act = create_puzzle_image(
                qs, corr_img, incorrect, out_path, lab, cell_size, grid_margin
            )
            if act != lab:
                print(f"Warning: Mismatch for {vid}. Expected {lab}, got {act}")
            variations.append(
                {
                    "id": vid,
                    "prompt": "From the four given options, select the most suitable one to fill in the question mark to present a certain regularity.",
                    "options": {l: l for l in labels},
                    "image": rel_path,
                    "correct_answer": act,
                }
            )
        except Exception as e:
            print(f"Error creating puzzle image for variation {vid}: {e}")
    return variations if variations else None


# ------------------------------------------------------------------
# orchestration
# ------------------------------------------------------------------
def create_puzzles_from_ids(
    id_list, root_dir, output_dir, cell_size, grid_margin, max_workers
):
    """Multiprocess all IDs and return flattened puzzle list."""
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    print(
        f"Starting processing for {len(id_list)} IDs, generating up to {len(id_list)*4} puzzle variations..."
    )

    puzzles = []
    args_pack = [(i, root_dir, img_dir, cell_size, grid_margin) for i in id_list]
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_single_id, *a): a[0] for a in args_pack}
        for fut in tqdm(
            as_completed(futs),
            total=len(futs),
            desc="Processing Original IDs",
            unit="id",
        ):
            try:
                res = fut.result()
                if res:
                    puzzles.extend(res)
            except Exception as e:
                print(f"Error processing original ID {futs[fut]}: {e}")
    return puzzles


# ------------------------------------------------------------------
# main entry
# ------------------------------------------------------------------
def main():
    """Entry point."""
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    try:
        with open(args.input, "r") as f:
            ids = json.load(f).get("ids", [])
        if not ids:
            print(f"Warning: No IDs found in {args.input}")
            return
    except FileNotFoundError:
        print(f"Error: Input ID file not found: {args.input}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.input}")
        return

    puzzles = create_puzzles_from_ids(
        ids, args.root, args.output, args.cell_size, args.grid_margin, args.workers
    )
    out_json = os.path.join(args.output, "4.1_puzzles.json")
    try:
        with open(out_json, "w") as f:
            json.dump(puzzles, f, indent=2)
        print(
            f"\nCreated {len(puzzles)} total puzzle variations and saved to {out_json}"
        )
    except Exception as e:
        print(f"Error saving output JSON to {out_json}: {e}")


if __name__ == "__main__":
    main()
