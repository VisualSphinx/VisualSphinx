import os
import json
import random
from PIL import Image, ImageDraw, ImageFont
import shutil
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_single_id(id_num, root_dir, image_dir, cell_size, grid_margin):
    """
    Helper for multiprocessing: processes a single ID and returns a puzzle dict or None.
    """
    # Find maximum attempt folder
    max_attempt = None
    max_num = -1
    for item in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, item)):
            parts = item.split("_")
            if len(parts) == 2 and parts[0] == str(id_num):
                try:
                    num = int(parts[1])
                    if num > max_num:
                        max_num = num
                        max_attempt = parts[1]
                except ValueError:
                    continue
    if max_attempt is None:
        print(f"No valid folder found for ID {id_num}")
        return None

    folder_path = os.path.join(root_dir, f"{id_num}_{max_attempt}")

    # Get images
    correct_images = get_images_from_dir(os.path.join(folder_path, "output_correct"))
    incorrect_images = get_images_from_dir(
        os.path.join(folder_path, "output_incorrect")
    )

    if len(correct_images) != 5 or len(incorrect_images) != 3:
        print(
            f"ID {id_num}: Invalid image count - {len(correct_images)} correct, {len(incorrect_images)} incorrect"
        )
        return None

    # Create puzzle image
    image_path = os.path.join(image_dir, f"image_{id_num}.png")
    relative_path = f"images/image_{id_num}.png"
    correct_answer = create_puzzle_image(
        correct_images[:4],
        correct_images[4:],
        incorrect_images,
        image_path,
        cell_size,
        grid_margin,
    )

    # Random correct answer

    return {
        "id": int(id_num),
        "prompt": "From the four given options, select the most suitable one to fill in the question mark to present a certain regularity.",
        "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
        "image": relative_path,
        "correct_answer": correct_answer,
    }


def create_puzzle_from_images(
    id_list,
    root_dir="all_scripts",
    output_dir="output",
    cell_size=200,
    grid_margin=30,
    max_workers=4,
):
    """
    Creates puzzles using multiprocessing plus a progress bar.
    """
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    puzzles = []
    args_list = [
        (id_num, root_dir, image_dir, cell_size, grid_margin) for id_num in id_list
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_id, *args): args[0] for args in args_list
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing IDs", unit="id"
        ):
            try:
                result = future.result()
                if result:
                    puzzles.append(result)
            except Exception as e:
                print(f"Error processing ID {futures[future]}: {e}")

    return puzzles


def find_max_attempt(root_dir, id_num):
    """Find the maximum attempt number for a given ID"""
    max_attempt = None
    max_attempt_num = -1

    for item in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, item)):
            parts = item.split("_")
            if len(parts) == 2 and parts[0] == str(id_num):
                try:
                    attempt_num = int(parts[1])
                    if attempt_num > max_attempt_num:
                        max_attempt_num = attempt_num
                        max_attempt = parts[1]
                except ValueError:
                    continue

    return max_attempt


def get_images_from_dir(directory):
    """Get a sorted list of image paths from a directory"""
    if not os.path.exists(directory):
        return []

    # Common image extensions
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]

    images = []
    for file in sorted(os.listdir(directory)):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            images.append(os.path.join(directory, file))

    return images


def create_puzzle_image(
    question_images,
    correct_image,
    incorrect_images,
    output_path,
    cell_size=200,
    grid_margin=30,
):
    """
    Create the puzzle image according to the specifications:
    - Top row: 4 correct images + question mark
    - Bottom row: 1 correct image (randomly placed) + 3 incorrect images, labeled A, B, C, D

    Args:
        question_images: List of paths to question images
        correct_image: List with path to correct answer image
        incorrect_images: List of paths to incorrect answer images
        output_path: Path to save the output image
        cell_size: Size of each cell in pixels
    """
    # Improved layout parameters
    frame_thickness = 8
    font_size = 85  # Larger font size for better visibility
    row_spacing = 70  # Space between rows
    label_spacing = 70  # Space between image and label
    top_padding = 50  # Add padding at the top
    image_margin = 15  # Margin between image and cell border
    grid_margin = 30  # Margin around the entire grid

    # Adjusted cell size to include margins
    inner_cell_size = cell_size - (2 * image_margin)

    # Set up font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Fallback to default font
        font = ImageFont.load_default()

    # Load all images with high quality settings
    question_imgs = [Image.open(img).convert("RGBA") for img in question_images]
    answer_imgs = [Image.open(correct_image[0]).convert("RGBA")] + [
        Image.open(img).convert("RGBA") for img in incorrect_images
    ]

    # Create content cells (without borders)
    def create_content_cell(img):
        # Create white cell without border
        cell_img = Image.new("RGB", (inner_cell_size, inner_cell_size), color="white")

        # Resize image to fit within cell while preserving aspect ratio
        aspect_ratio = img.width / img.height

        # Use a larger portion of the cell
        padding_percent = 5
        max_allowed_width = inner_cell_size - (inner_cell_size * padding_percent // 100)
        max_allowed_height = inner_cell_size - (
            inner_cell_size * padding_percent // 100
        )

        if aspect_ratio > 1:  # Wider than tall
            new_width = min(max_allowed_width, int(img.width))
            new_height = int(new_width / aspect_ratio)
        else:  # Taller than wide or square
            new_height = min(max_allowed_height, int(img.height))
            new_width = int(new_height * aspect_ratio)

        # If original image is smaller than the calculated size, use original size
        if img.width <= new_width and img.height <= new_height:
            resized_img = img
        else:
            # Use high-quality resampling
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        # Place image in center of cell
        x_pos = (inner_cell_size - new_width) // 2
        y_pos = (inner_cell_size - new_height) // 2

        # Paste image
        if resized_img.mode == "RGBA":
            # Need to handle alpha channel
            temp = Image.new("RGBA", cell_img.size, (255, 255, 255, 0))
            temp.paste(resized_img, (x_pos, y_pos))
            cell_rgba = cell_img.convert("RGBA")
            result = Image.alpha_composite(cell_rgba, temp)
            cell_img = result.convert("RGB")
        else:
            cell_img.paste(resized_img, (x_pos, y_pos))

        return cell_img

    # Create question mark cell
    def create_qm_cell():
        cell_img = Image.new("RGB", (inner_cell_size, inner_cell_size), color="white")
        draw = ImageDraw.Draw(cell_img)
        try:
            qm_font = ImageFont.truetype("arial.ttf", 150)
        except:
            qm_font = ImageFont.load_default()
        draw.text(
            (inner_cell_size // 2, inner_cell_size // 2),
            "?",
            fill="black",
            font=qm_font,
            anchor="mm",
        )
        return cell_img

    # Process question cells
    question_cells = [create_content_cell(img) for img in question_imgs]
    question_mark_cell = create_qm_cell()

    # Shuffle answer options
    random.shuffle(answer_imgs)

    # Process answer cells
    answer_cells = [create_content_cell(img) for img in answer_imgs]

    # Calculate dimensions of the final image
    cell_spacing = 0  # Cells have no gap between them
    top_row_width = 5 * cell_size
    bottom_row_width = 4 * cell_size
    grid_width = max(top_row_width, bottom_row_width)

    # Add margins to total width and height
    total_width = grid_width + (2 * grid_margin)

    # Total height calculation
    total_height = (
        top_padding
        + (2 * cell_size)
        + row_spacing
        + label_spacing
        + font_size
        + grid_margin
    )

    # Create the final image
    final_image = Image.new("RGB", (total_width, total_height), color="white")
    draw = ImageDraw.Draw(final_image)

    # Place first row cells (with margins)
    x_offset_top = grid_margin + ((grid_width - top_row_width) // 2)
    y_offset = top_padding

    for i, cell in enumerate(question_cells):
        # Position to place the cell content (apply margin)
        pos_x = x_offset_top + (i * cell_size) + image_margin
        pos_y = y_offset + image_margin

        # Place the cell content
        final_image.paste(cell, (pos_x, pos_y))

    # Place question mark cell
    pos_x = x_offset_top + (4 * cell_size) + image_margin
    pos_y = y_offset + image_margin
    final_image.paste(question_mark_cell, (pos_x, pos_y))

    # Place second row cells (with margins)
    x_offset_bottom = grid_margin + ((grid_width - bottom_row_width) // 2)
    y_offset = top_padding + cell_size + row_spacing

    for i, cell in enumerate(answer_cells):
        # Position to place the cell content (apply margin)
        pos_x = x_offset_bottom + (i * cell_size) + image_margin
        pos_y = y_offset + image_margin

        # Place the cell content
        final_image.paste(cell, (pos_x, pos_y))

    # Now draw the grid lines

    # Draw top row grid
    x_offset_top = grid_margin + ((grid_width - top_row_width) // 2)
    y_offset = top_padding

    # Draw outer frame for top row
    draw.rectangle(
        [
            (x_offset_top, y_offset),
            (x_offset_top + 5 * cell_size, y_offset + cell_size),
        ],
        outline="black",
        width=frame_thickness,
    )

    # Draw vertical lines for top row
    for i in range(1, 5):
        x = x_offset_top + i * cell_size
        draw.line(
            [(x, y_offset), (x, y_offset + cell_size)],
            fill="black",
            width=frame_thickness,
        )

    # Draw bottom row grid
    x_offset_bottom = grid_margin + ((grid_width - bottom_row_width) // 2)
    y_offset = top_padding + cell_size + row_spacing

    # Draw outer frame for bottom row
    draw.rectangle(
        [
            (x_offset_bottom, y_offset),
            (x_offset_bottom + 4 * cell_size, y_offset + cell_size),
        ],
        outline="black",
        width=frame_thickness,
    )

    # Draw vertical lines for bottom row
    for i in range(1, 4):
        x = x_offset_bottom + i * cell_size
        draw.line(
            [(x, y_offset), (x, y_offset + cell_size)],
            fill="black",
            width=frame_thickness,
        )

    # Add option labels
    option_labels = ["A", "B", "C", "D"]
    for i in range(4):
        label_x = x_offset_bottom + (i * cell_size) + (cell_size // 2)
        label_y = y_offset + cell_size + label_spacing
        draw.text(
            (label_x, label_y), option_labels[i], fill="black", font=font, anchor="mm"
        )

    max_len = 800  # 目标最大宽度
    if final_image.width > max_len:
        scale = max_len / final_image.width  # 计算缩放比例
        new_size = (max_len, int(final_image.height * scale))
        final_image = final_image.resize(new_size, Image.LANCZOS)

    # Save the final image with high quality
    final_image.save(output_path, quality=95)

    # Return the correct answer index (which option is the correct image)
    correct_index = answer_imgs.index(Image.open(correct_image[0]).convert("RGBA"))
    return option_labels[correct_index]


def main():
    parser = argparse.ArgumentParser(
        description="Generate puzzles with multiprocessing"
    )
    parser.add_argument(
        "--output", type=str, default="./data/step4/Dataset_check"
    )
    parser.add_argument(
        "--input", type=str, default="./data/step3/3.2_valid_style_1.json"
    )
    parser.add_argument(
        "--root", type=str, default="./data/step3/3.1_all_scripts_style_1"
    )
    parser.add_argument("--cell_size", type=int, default=400)
    parser.add_argument("--grid_margin", type=int, default=30)
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    with open(args.input, "r") as f:
        id_list = json.load(f).get("ids", [])

    puzzles = create_puzzle_from_images(
        id_list, args.root, args.output, args.cell_size, args.grid_margin, args.workers
    )

    out_json = os.path.join(args.output, "puzzles.json")
    with open(out_json, "w") as f:
        json.dump(puzzles, f, indent=2)

    print(f"Created {len(puzzles)} puzzles and saved to {out_json}")


if __name__ == "__main__":
    main()
