"""
Generate 10-option visual puzzles with relative logic.
"""

import os
import json
import random
from PIL import Image, ImageDraw, ImageFont
import shutil
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Generate 10-option puzzles with multiprocessing"
    )
    parser.add_argument(
        "--output", type=str, default="./data/step4/Dataset_style_1_10options"
    )
    parser.add_argument(
        "--valid_ids", type=str, default="./data/step3/3.2_valid_style_1.json"
    )
    parser.add_argument(
        "--relations",
        type=str,
        default="./data/step2/2.3_rules.json",
        help="Path to the GA relation JSON file",
    )
    parser.add_argument(
        "--root", type=str, default="./data/step3/3.1_all_scripts_style_1"
    )
    parser.add_argument("--cell_size", type=int, default=400)
    parser.add_argument("--grid_margin", type=int, default=25)
    parser.add_argument("--workers", type=int, default=15)
    return parser.parse_args()


# Constants
NUM_CORRECT_NEEDED = 5
NUM_INCORRECT_SELF_NEEDED = 3
NUM_RELATIVES_NEEDED = 2
NUM_INCORRECT_PER_RELATIVE = 3
OPTIONS_ROWS = 2
OPTIONS_COLS = 5
TOTAL_OPTIONS = OPTIONS_ROWS * OPTIONS_COLS
OPTION_LABELS = list("ABCDEFGHIJ")


def build_relation_graph(relations_list):
    """Build parent/child lookup graph."""
    graph_dd = defaultdict(lambda: {"parents": [], "children": [], "generation": -1})
    for item in relations_list:
        if not isinstance(item, dict) or "id" not in item:
            continue
        iid = item["id"]
        graph_dd[iid]["parents"] = item.get("parents", [])
        graph_dd[iid]["generation"] = item.get("generation", -1)
    for iid in list(graph_dd.keys()):
        for pid in graph_dd[iid]["parents"]:
            graph_dd[pid]["children"].append(iid)
    return dict(graph_dd)


def find_max_attempt(root_dir, id_num):
    """Return the latest attempt folder for an ID."""
    max_attempt, folder, sid = -1, None, str(id_num)
    if not os.path.isdir(root_dir):
        return None
    for item in os.listdir(root_dir):
        path = os.path.join(root_dir, item)
        if os.path.isdir(path):
            part = item.split("_")
            if len(part) == 2 and part[0] == sid:
                try:
                    n = int(part[1])
                    if n > max_attempt:
                        max_attempt, folder = n, item
                except ValueError:
                    continue
    return os.path.join(root_dir, folder) if folder else None


def get_images_from_dir(directory, expected_count):
    """Return sorted image paths or None if count mismatches."""
    if not directory or not os.path.isdir(directory):
        return None
    exts = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]
    imgs = [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if any(f.lower().endswith(ext) for ext in exts)
    ]
    return imgs if len(imgs) == expected_count else None


def find_two_valid_relatives(target_id, relation_graph, valid_id_set):
    """Breadth-first search for two valid relatives."""
    if target_id not in relation_graph:
        return []
    found, visited, layer = set(), {target_id}, {target_id}
    while len(found) < NUM_RELATIVES_NEEDED:
        nxt, nxt_ids = set(), set()
        for nid in layer:
            for p in relation_graph[nid].get("parents", []):
                if p not in visited:
                    nxt.add(p)
                    nxt_ids.add(p)
                    visited.add(p)
            for c in relation_graph[nid].get("children", []):
                if c not in visited:
                    nxt.add(c)
                    nxt_ids.add(c)
                    visited.add(c)
        if not nxt_ids:
            break
        valid = {x for x in nxt if x in valid_id_set}
        for v in valid:
            if len(found) < NUM_RELATIVES_NEEDED:
                found.add(v)
        layer = nxt_ids
    return list(found)


def create_puzzle_image_10_options(
    question_image_paths,
    correct_answer_image_path,
    incorrect_image_paths,
    output_path,
    cell_size=250,
    grid_margin=30,
    top_padding=30,
    question_options_spacing=160,
    row_spacing=130,
    label_spacing=25,
    bottom_padding=5,
    font_size=85,
    frame_thickness=8,
    image_margin=30,
):
    """Compose final puzzle image and return correct label."""
    if len(question_image_paths) != 4 or len(incorrect_image_paths) != 9:
        raise ValueError("Wrong image counts")
    inner = cell_size - 2 * image_margin
    font = (
        ImageFont.truetype("arial.ttf", font_size)
        if os.path.exists("arial.ttf")
        else ImageFont.load_default()
    )
    q_imgs = [Image.open(p).convert("RGBA") for p in question_image_paths]
    corr = Image.open(correct_answer_image_path).convert("RGBA")
    inc = [Image.open(p).convert("RGBA") for p in incorrect_image_paths]
    all_opts = [corr] + inc
    mapping = list(range(TOTAL_OPTIONS))
    random.shuffle(mapping)
    shuffled = [all_opts[i] for i in mapping]
    corr_label = OPTION_LABELS[mapping.index(0)]

    def resize_cell(img):
        cell = Image.new("RGB", (inner, inner), "white")
        ar = img.width / img.height
        if ar > 1:
            nw, nh = inner, int(inner / ar)
        else:
            nh, nw = inner, int(inner * ar)
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
        qs = max(80, int(inner * 0.7))
        qf = (
            ImageFont.truetype("arial.ttf", qs)
            if os.path.exists("arial.ttf")
            else ImageFont.load_default()
        )
        d.text((inner / 2, inner / 2), "?", font=qf, fill="black", anchor="mm")
        return cell

    q_cells = [resize_cell(i) for i in q_imgs]
    ans_cells = [resize_cell(i) for i in shuffled]
    grid_w, total_w = (
        OPTIONS_COLS * cell_size,
        OPTIONS_COLS * cell_size + 2 * grid_margin,
    )
    opt_h = OPTIONS_ROWS * cell_size
    label_h = font_size + label_spacing
    total_h = (
        top_padding
        + cell_size
        + question_options_spacing
        + opt_h
        + row_spacing
        + label_h
        + bottom_padding
    )
    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    y, x0 = top_padding, grid_margin
    for i, c in enumerate(q_cells + [qm_cell()]):
        canvas.paste(c, (x0 + i * cell_size + image_margin, y + image_margin))
    draw.rectangle(
        [(x0, y), (x0 + grid_w, y + cell_size)], outline="black", width=frame_thickness
    )
    for i in range(1, OPTIONS_COLS):
        lx = x0 + i * cell_size
        draw.line([(lx, y), (lx, y + cell_size)], fill="black", width=frame_thickness)

    y += cell_size + question_options_spacing
    for r in range(OPTIONS_ROWS):
        s, e = r * OPTIONS_COLS, (r + 1) * OPTIONS_COLS
        for i, c in enumerate(ans_cells[s:e]):
            canvas.paste(c, (x0 + i * cell_size + image_margin, y + image_margin))
        draw.rectangle(
            [(x0, y), (x0 + grid_w, y + cell_size)],
            outline="black",
            width=frame_thickness,
        )
        for i in range(1, OPTIONS_COLS):
            lx = x0 + i * cell_size
            draw.line(
                [(lx, y), (lx, y + cell_size)], fill="black", width=frame_thickness
            )
        ly = y + cell_size + label_spacing // 2 + font_size // 2
        for i, lab in enumerate(OPTION_LABELS[s:e]):
            draw.text(
                (x0 + i * cell_size + cell_size // 2, ly),
                lab,
                font=font,
                fill="black",
                anchor="mm",
            )
        y += cell_size + (row_spacing if r == 0 else 0)

    if canvas.width > 1000:
        scale = 1000 / canvas.width
        canvas = canvas.resize(
            (1000, int(canvas.height * scale)), Image.Resampling.LANCZOS
        )
    canvas.save(output_path, quality=95)
    return corr_label


def process_single_id(
    id_num, root_dir, image_dir, cell_size, grid_margin, relation_graph, valid_id_set
):
    """End-to-end processing for one ID."""
    # DEBUG start
    target_folder = find_max_attempt(root_dir, id_num)
    if not target_folder:
        print(
            f"DEBUG: ID {id_num}: Cannot find attempt folder in {root_dir}. Skipping."
        )
        return None

    correct_imgs = get_images_from_dir(
        os.path.join(target_folder, "output_correct"), NUM_CORRECT_NEEDED
    )
    if correct_imgs is None:
        print(
            f"DEBUG: ID {id_num}: Failed to load correct images "
            f"(expected {NUM_CORRECT_NEEDED}). Skipping."
        )
        return None

    self_inc = get_images_from_dir(
        os.path.join(target_folder, "output_incorrect"), NUM_INCORRECT_SELF_NEEDED
    )
    if self_inc is None:
        print(
            f"DEBUG: ID {id_num}: Failed to load self incorrect images "
            f"(expected {NUM_INCORRECT_SELF_NEEDED}). Skipping."
        )
        return None

    relatives = find_two_valid_relatives(id_num, relation_graph, valid_id_set)
    if len(relatives) != NUM_RELATIVES_NEEDED:
        print(
            f"DEBUG: ID {id_num}: Found {len(relatives)} valid relatives, "
            f"need {NUM_RELATIVES_NEEDED}. Skipping."
        )
        return None
    r1, r2 = relatives

    r1_folder = find_max_attempt(root_dir, r1)
    if not r1_folder:
        print(
            f"DEBUG: ID {id_num}: Cannot find attempt folder for relative {r1}. Skipping."
        )
        return None
    r1_inc = get_images_from_dir(
        os.path.join(r1_folder, "output_incorrect"), NUM_INCORRECT_PER_RELATIVE
    )
    if r1_inc is None:
        print(
            f"DEBUG: ID {id_num}: Failed to load incorrect images for relative {r1} "
            f"(expected {NUM_INCORRECT_PER_RELATIVE}). Skipping."
        )
        return None

    r2_folder = find_max_attempt(root_dir, r2)
    if not r2_folder:
        print(
            f"DEBUG: ID {id_num}: Cannot find attempt folder for relative {r2}. Skipping."
        )
        return None
    r2_inc = get_images_from_dir(
        os.path.join(r2_folder, "output_incorrect"), NUM_INCORRECT_PER_RELATIVE
    )
    if r2_inc is None:
        print(
            f"DEBUG: ID {id_num}: Failed to load incorrect images for relative {r2} "
            f"(expected {NUM_INCORRECT_PER_RELATIVE}). Skipping."
        )
        return None
    # DEBUG end

    all_inc = self_inc + r1_inc + r2_inc
    qs, corr = correct_imgs[:4], correct_imgs[4]
    img_name = f"image_{id_num}.png"
    out_path = os.path.join(image_dir, img_name)
    try:
        label = create_puzzle_image_10_options(
            qs, corr, all_inc, out_path, cell_size, grid_margin
        )
    except Exception as e:
        print(f"Error creating puzzle image for ID {id_num}: {e}")
        return None

    return {
        "id": int(id_num),
        "prompt": "From the four given options, select the most suitable one to fill in the question mark to present a certain regularity.",
        "options": {lab: lab for lab in OPTION_LABELS},
        "image": f"images/{img_name}",
        "correct_answer": label,
        "relatives_used": [int(r1), int(r2)],
    }


def create_puzzles_from_ids_with_relatives(
    id_list,
    relation_graph,
    valid_id_set,
    root_dir="all_scripts",
    output_dir="output",
    cell_size=400,
    grid_margin=30,
    max_workers=16,
):
    """Multiprocess all IDs and return puzzle dicts."""
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    args_pack = [
        (i, root_dir, img_dir, cell_size, grid_margin, relation_graph, valid_id_set)
        for i in id_list
    ]
    print(f"Starting processing for {len(id_list)} IDs using relationship logic...")
    puzzles = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_single_id, *a): a[0] for a in args_pack}
        for f in tqdm(
            as_completed(futs), total=len(futs), desc="Processing", unit="id"
        ):
            try:
                res = f.result()
                if res:
                    puzzles.append(res)
            except Exception as e:
                print(f"Critical error processing ID {futs[f]}: {e}")
    return puzzles


def main():
    """Entry point."""
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    with open(args.valid_ids, "r") as f:
        raw_ids = json.load(f).get("ids", [])
    valid_ids = {int(i) for i in raw_ids if str(i).isdigit()}
    if not valid_ids:
        print("No valid IDs found.")
        return

    with open(args.relations, "r", encoding="utf-8") as f:
        relations = json.load(f)
    relation_graph = build_relation_graph(relations)

    puzzles = create_puzzles_from_ids_with_relatives(
        list(valid_ids),
        relation_graph,
        valid_ids,
        args.root,
        args.output,
        args.cell_size,
        args.grid_margin,
        args.workers,
    )
    out_json = os.path.join(args.output, "4.1_puzzles.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(puzzles, f, indent=2)
    print(f"Created {len(puzzles)} puzzles → {out_json}")


if __name__ == "__main__":
    main()
