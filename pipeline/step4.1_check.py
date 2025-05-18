import os
import json
import random
from PIL import Image, ImageDraw, ImageFont
import shutil
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

# --- Constants ---
NUM_CORRECT_NEEDED = 5
NUM_INCORRECT_SELF_NEEDED = 3
NUM_RELATIVES_NEEDED = 2
NUM_INCORRECT_PER_RELATIVE = 3
OPTIONS_ROWS = 2
OPTIONS_COLS = 5
TOTAL_OPTIONS = OPTIONS_ROWS * OPTIONS_COLS  # Should be 10
OPTION_LABELS = list("ABCDEFGHIJ")

# --- Relationship and File Handling ---


def build_relation_graph(relations_list):
    """Builds a graph for easy parent/child lookup."""
    # Use defaultdict locally for easier building
    graph_dd = defaultdict(lambda: {"parents": [], "children": [], "generation": -1})
    all_ids = set()

    # First pass: store parents and generation
    for item in relations_list:
        if not isinstance(item, dict) or "id" not in item:
            print(f"Warning: Skipping invalid relation item: {item}")
            continue
        item_id = item["id"]
        all_ids.add(item_id)
        # Access graph_dd to potentially create entry via lambda factory
        graph_dd[item_id]["parents"] = item.get("parents", [])
        graph_dd[item_id]["generation"] = item.get("generation", -1)

    # Second pass: build children lists
    for item_id in list(graph_dd.keys()):  # Iterate over existing keys
        parents = graph_dd[item_id]["parents"]
        for parent_id in parents:
            # Accessing graph_dd[parent_id] ensures parent entry exists
            # The lambda factory handles creation if parent_id wasn't in the original list
            graph_dd[parent_id]["children"].append(item_id)
            # We don't strictly need to add parent_id to all_ids here as
            # defaultdict handles creation, but graph size might be bigger
            # than len(relations_list) if parents outside the list are referenced.

    print(f"Built relation graph for {len(graph_dd)} IDs.")
    # *** Convert to a regular dict before returning ***
    return dict(graph_dd)


def find_max_attempt(root_dir, id_num):
    """Find the maximum attempt number folder for a given ID."""
    max_attempt_num = -1
    folder_name = None
    id_str = str(id_num)

    # Check if root_dir exists
    if not os.path.isdir(root_dir):
        # print(f"Warning: Root directory not found: {root_dir}")
        return None

    try:
        for item in os.listdir(root_dir):
            path = os.path.join(root_dir, item)
            if os.path.isdir(path):
                parts = item.split("_")
                if len(parts) == 2 and parts[0] == id_str:
                    try:
                        attempt_num = int(parts[1])
                        if attempt_num > max_attempt_num:
                            max_attempt_num = attempt_num
                            folder_name = item
                    except ValueError:
                        continue
    except FileNotFoundError:
        # This can happen if the listdir is attempted on a non-existent path derived earlier
        # print(f"Warning: Directory listing failed for path related to {root_dir} and ID {id_num}")
        return None
    except Exception as e:
        print(f"Error during find_max_attempt for ID {id_num} in {root_dir}: {e}")
        return None

    if folder_name:
        return os.path.join(root_dir, folder_name)
    else:
        # print(f"Warning: No valid attempt folder found for ID {id_num} in {root_dir}")
        return None


def get_images_from_dir(directory, expected_count):
    """Get a sorted list of image paths, checking the count."""
    if not directory or not os.path.isdir(directory):
        # print(f"Warning: Image directory not found or invalid: {directory}")
        return None  # Indicate directory issue

    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]
    images = []
    try:
        for file in sorted(os.listdir(directory)):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                images.append(os.path.join(directory, file))
    except Exception as e:
        print(f"Error reading directory {directory}: {e}")
        return None

    if len(images) != expected_count:
        # print(f"Warning: Found {len(images)} images in {directory}, expected {expected_count}.")
        return None  # Indicate wrong count

    return images


def find_two_valid_relatives(target_id, relation_graph, valid_id_set):
    """
    Find two unique relatives present in valid_id_set by searching
    outwards layer by layer (parents/children, then grandparents/grandchildren, etc.).
    """
    if target_id not in relation_graph:
        # print(f"Debug: Target ID {target_id} not in relation_graph.")
        return []  # Target ID not in relation data

    found_relatives = set()
    # Keep track of nodes visited to avoid cycles and redundant checks
    visited = {target_id}
    # Queue for BFS: stores tuples of (node_id_to_explore)
    # We'll explore layer by layer implicitly
    current_layer = {target_id}

    while len(found_relatives) < NUM_RELATIVES_NEEDED:
        next_layer_candidates = set()
        ids_in_next_layer = set()  # IDs to explore in the *next* iteration

        # Explore neighbors (parents and children) of the current layer
        for node_id in current_layer:
            if node_id not in relation_graph:
                continue  # Skip if node somehow not in graph

            # Find parents (go up)
            parents = relation_graph[node_id].get("parents", [])
            for p_id in parents:
                if p_id not in visited:
                    next_layer_candidates.add(p_id)
                    ids_in_next_layer.add(p_id)
                    visited.add(p_id)

            # Find children (go down)
            children = relation_graph[node_id].get("children", [])
            for c_id in children:
                if c_id not in visited:
                    next_layer_candidates.add(c_id)
                    ids_in_next_layer.add(c_id)
                    visited.add(c_id)

        # If no new nodes were found in the next layer, we've exhausted the graph
        if not ids_in_next_layer:
            # print(f"Debug: ID {target_id}: Exhausted graph search.")
            break  # Stop searching

        # Check candidates found in this layer for validity
        valid_candidates_in_layer = {
            c for c in next_layer_candidates if c in valid_id_set
        }

        # Add valid candidates found in this layer until we have enough
        if valid_candidates_in_layer:
            shuffled_valid_candidates = list(valid_candidates_in_layer)
            random.shuffle(shuffled_valid_candidates)

            for cand in shuffled_valid_candidates:
                if len(found_relatives) < NUM_RELATIVES_NEEDED:
                    found_relatives.add(cand)
                else:
                    break  # Stop adding once we have enough

        # Prepare for the next iteration
        current_layer = ids_in_next_layer

        # Check if we found enough relatives after processing this layer
        if len(found_relatives) >= NUM_RELATIVES_NEEDED:
            break

    # Return the found relatives as a list
    return list(found_relatives)


# --- Image Creation (New 10-Option Version) ---


def create_puzzle_image_10_options(
    question_image_paths,  # List of 4 paths for question images
    correct_answer_image_path,  # Single path for the correct answer image
    incorrect_image_paths,  # List of 9 paths for incorrect answer images
    output_path,  # Full path where the final image will be saved
    # --- Adjustable Layout & Spacing Parameters ---
    cell_size=250,  # Base size of each cell (question or option)
    grid_margin=30,  # Margin on the left/right of the entire grid
    top_padding=30,  # Padding above the first row
    question_options_spacing=160,  # Space between question row and first option row (A-E)
    row_spacing=130,  # Space between option row A-E and option row F-J
    label_spacing=25,  # Vertical space from bottom of option cell to center of its label
    bottom_padding=5,  # Padding below the last row of labels (F-J)
    font_size=85,  # Font size for labels A-J
    frame_thickness=8,  # Thickness of the grid lines
    image_margin=30,  # Internal padding around images within cells
):
    """
    Creates a puzzle image with 1 top row (4 questions + '?') and 2 bottom rows
    (10 options A-J, randomized), saving it to output_path.
    Includes fine-grained spacing controls.

    Returns:
        str: The label ('A' through 'J') corresponding to the correct answer's
             position in the shuffled options.
    Raises:
        FileNotFoundError: If required image files are not found.
        ValueError: If input list counts are incorrect.
        Exception: For other PIL or processing errors.
    """
    if len(question_image_paths) != 4:
        raise ValueError(f"Expected 4 question images, got {len(question_image_paths)}")
    if len(incorrect_image_paths) != 9:
        raise ValueError(
            f"Expected 9 incorrect images, got {len(incorrect_image_paths)}"
        )
    if not correct_answer_image_path:
        raise ValueError("Correct answer image path cannot be empty.")

    # Calculate size of the area inside cell borders for placing images
    inner_cell_size = cell_size - (2 * image_margin)
    if inner_cell_size <= 0:
        raise ValueError("cell_size must be greater than twice the image_margin.")

    # --- Font Setup ---
    try:
        # Try loading a common TrueType font
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        print("Warning: arial.ttf not found. Using default PIL font.")
        # Fallback to PIL's default bitmap font (may not scale well)
        font = ImageFont.load_default()

    # --- Load Images ---
    # Wrap loading in try-except to catch file not found or PIL errors early
    try:
        question_imgs = [Image.open(p).convert("RGBA") for p in question_image_paths]
        correct_answer_img = Image.open(correct_answer_image_path).convert("RGBA")
        incorrect_imgs = [Image.open(p).convert("RGBA") for p in incorrect_image_paths]
    except FileNotFoundError as e:
        print(f"Error loading image file: {e}")
        raise  # Re-raise to indicate failure
    except Exception as e:
        print(f"Error processing an image file with PIL: {e}")
        raise  # Re-raise other PIL errors

    # --- Prepare and Shuffle Options ---
    # Combine correct answer + incorrect options into one list for shuffling
    all_option_source_imgs = [correct_answer_img] + incorrect_imgs
    # Create a mapping of original index (0=correct, 1-9=incorrect) to shuffled position
    current_mapping = list(range(TOTAL_OPTIONS))
    random.shuffle(current_mapping)

    # Create the list of images in their final shuffled order
    shuffled_option_imgs = [all_option_source_imgs[i] for i in current_mapping]
    # Find the new index (0-9) where the original correct image (index 0) ended up
    correct_answer_new_index = current_mapping.index(0)
    # Determine the corresponding A-J label
    correct_answer_label = OPTION_LABELS[correct_answer_new_index]

    # --- Helper function to create cell content ---
    def create_content_cell(img):
        # Creates a white square background for the cell content
        cell_img = Image.new("RGB", (inner_cell_size, inner_cell_size), color="white")

        # Calculate new size to fit image inside inner_cell_size while maintaining aspect ratio
        aspect_ratio = img.width / img.height
        max_dim = inner_cell_size  # Fit within the inner bounds

        if aspect_ratio > 1:  # Wider than tall
            new_width = max_dim
            new_height = int(new_width / aspect_ratio)
        else:  # Taller than wide or square
            new_height = max_dim
            new_width = int(new_height * aspect_ratio)

        # Ensure calculated dimensions are positive
        new_width = max(1, new_width)
        new_height = max(1, new_height)

        # Use high-quality resampling (LANCZOS)
        try:
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        except AttributeError:  # Handle older PIL versions
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        # Calculate top-left position to center the resized image
        x_pos = (inner_cell_size - new_width) // 2
        y_pos = (inner_cell_size - new_height) // 2

        # Paste the resized image onto the white cell background
        # Handle transparency correctly if the image has an alpha channel
        if resized_img.mode == "RGBA":
            # Create a temporary transparent layer matching cell size
            temp_layer = Image.new("RGBA", cell_img.size, (255, 255, 255, 0))
            # Paste the resized RGBA image onto the transparent layer
            temp_layer.paste(resized_img, (x_pos, y_pos), resized_img)  # Use alpha mask
            # Convert the white cell background to RGBA
            cell_rgba = cell_img.convert("RGBA")
            # Composite the image layer over the background
            final_cell_content = Image.alpha_composite(cell_rgba, temp_layer)
            cell_img = final_cell_content.convert("RGB")  # Convert back to RGB
        else:
            # If no alpha, just paste directly
            cell_img.paste(resized_img.convert("RGB"), (x_pos, y_pos))

        return cell_img

    # --- Helper function to create question mark cell ---
    def create_qm_cell():
        cell_img = Image.new("RGB", (inner_cell_size, inner_cell_size), color="white")
        draw = ImageDraw.Draw(cell_img)
        # Adjust question mark size based on cell size
        qm_font_size = max(80, int(inner_cell_size * 0.7))  # Heuristic size
        try:
            qm_font = ImageFont.truetype("arial.ttf", qm_font_size)
        except IOError:
            qm_font = ImageFont.load_default()  # Fallback
        # Draw centered question mark
        draw.text(
            (inner_cell_size / 2, inner_cell_size / 2),
            "?",
            fill="black",
            font=qm_font,
            anchor="mm",
        )
        return cell_img

    # --- Process all image cells ---
    question_cells = [create_content_cell(img) for img in question_imgs]
    question_mark_cell = create_qm_cell()
    answer_cells = [create_content_cell(img) for img in shuffled_option_imgs]

    # --- Calculate final image dimensions ---
    grid_width = OPTIONS_COLS * cell_size  # Width based on 5 columns
    total_width = grid_width + (2 * grid_margin)  # Add left/right margins

    # Calculate height based on components and NEW spacing parameters
    option_rows_total_height = OPTIONS_ROWS * cell_size
    option_labels_total_height = (
        font_size + label_spacing
    )  # Height needed for one row of labels + spacing

    total_height = (
        top_padding  # Space above questions
        + cell_size  # Height of question row
        + question_options_spacing  # Space below questions
        + option_rows_total_height  # Height of the two option rows
        + row_spacing  # Space between the two option rows
        + option_labels_total_height  # Space for the labels below the last row
        + bottom_padding
    )  # Space below the last labels

    # --- Create the final canvas ---
    final_image = Image.new("RGB", (total_width, total_height), color="white")
    draw = ImageDraw.Draw(final_image)

    # --- Draw Elements Sequentially ---
    current_y = top_padding
    x_offset = grid_margin  # Starting X for all rows (left margin)

    # 1. Place & Draw Top Row (Questions + QM)
    top_row_cells = question_cells + [question_mark_cell]
    for i, cell in enumerate(top_row_cells):
        pos_x = x_offset + (i * cell_size) + image_margin
        pos_y = current_y + image_margin
        final_image.paste(cell, (pos_x, pos_y))
    # Draw grid for top row
    top_row_end_x = x_offset + OPTIONS_COLS * cell_size
    draw.rectangle(
        [(x_offset, current_y), (top_row_end_x, current_y + cell_size)],
        outline="black",
        width=frame_thickness,
    )
    for i in range(1, OPTIONS_COLS):  # Draw 4 vertical lines
        line_x = x_offset + i * cell_size
        draw.line(
            [(line_x, current_y), (line_x, current_y + cell_size)],
            fill="black",
            width=frame_thickness,
        )

    # Update Y position for the next section
    current_y += (
        cell_size + question_options_spacing
    )  # Move past questions and add specified gap

    # 2. Place & Draw Option Rows and Labels
    for row in range(OPTIONS_ROWS):  # 0 then 1
        row_start_index = row * OPTIONS_COLS
        row_end_index = row_start_index + OPTIONS_COLS
        current_row_cells = answer_cells[row_start_index:row_end_index]
        current_row_labels = OPTION_LABELS[row_start_index:row_end_index]

        # Paste content cells for this row
        for i, cell in enumerate(current_row_cells):
            pos_x = x_offset + (i * cell_size) + image_margin
            pos_y = current_y + image_margin
            final_image.paste(cell, (pos_x, pos_y))

        # Draw grid for this option row
        option_row_end_x = x_offset + OPTIONS_COLS * cell_size
        draw.rectangle(
            [(x_offset, current_y), (option_row_end_x, current_y + cell_size)],
            outline="black",
            width=frame_thickness,
        )
        for i in range(1, OPTIONS_COLS):  # Draw 4 vertical lines
            line_x = x_offset + i * cell_size
            draw.line(
                [(line_x, current_y), (line_x, current_y + cell_size)],
                fill="black",
                width=frame_thickness,
            )

        # Draw Labels below this row
        label_y_pos = (
            current_y + cell_size + (label_spacing // 2) + (font_size // 2)
        )  # Position label vertically centered in its space
        for i, label in enumerate(current_row_labels):
            label_x_pos = (
                x_offset + (i * cell_size) + (cell_size // 2)
            )  # Center label horizontally in cell
            draw.text(
                (label_x_pos, label_y_pos), label, fill="black", font=font, anchor="mm"
            )  # "mm" anchor centers text

        # Update y position for the next row (or finish)
        current_y += cell_size  # Move past the cells of this row
        if row == 0:  # If we just finished the FIRST option row (A-E)
            current_y += row_spacing  # Add the space before the second option row (F-J)
        # No need to add space after the last row; label_spacing and bottom_padding handle it

    max_len = 1000
    if final_image.width > max_len:
        scale = max_len / final_image.width
        new_size = (max_len, int(final_image.height * scale))
        try:  # Use updated resampling attribute
            final_image = final_image.resize(new_size, Image.Resampling.LANCZOS)
        except AttributeError:  # Fallback for older PIL
            final_image = final_image.resize(new_size, Image.LANCZOS)

    # --- Save the final image ---
    try:
        final_image.save(output_path, quality=95)  # Use high quality for saving
    except Exception as e:
        print(f"Error saving final image to {output_path}: {e}")
        raise

    # Return the label ('A' through 'J') of the correct answer
    return correct_answer_label


# --- Multiprocessing Task ---


def process_single_id(
    id_num, root_dir, image_dir, cell_size, grid_margin, relation_graph, valid_id_set
):
    """
    Processes a single ID: finds relatives, gathers images, creates 10-option puzzle.
    Returns a puzzle dict or None on failure.
    """
    # --- Start Debugging ---
    # print(f"DEBUG: Processing ID {id_num}")

    # 1. Find target ID's image folder
    target_folder_path = find_max_attempt(root_dir, id_num)
    if not target_folder_path:
        print(
            f"DEBUG: ID {id_num}: Cannot find attempt folder in {root_dir}. Skipping."
        )  # DEBUG PRINT
        return None

    # print(f"DEBUG: ID {id_num}: Found target folder: {target_folder_path}") # DEBUG PRINT

    # 2. Get target ID's images
    correct_images = get_images_from_dir(
        os.path.join(target_folder_path, "output_correct"), NUM_CORRECT_NEEDED
    )
    if correct_images is None:
        print(
            f"DEBUG: ID {id_num}: Failed to load correct images (expected {NUM_CORRECT_NEEDED}). Skipping."
        )  # DEBUG PRINT
        return None

    self_incorrect_images = get_images_from_dir(
        os.path.join(target_folder_path, "output_incorrect"), NUM_INCORRECT_SELF_NEEDED
    )
    if self_incorrect_images is None:
        print(
            f"DEBUG: ID {id_num}: Failed to load self incorrect images (expected {NUM_INCORRECT_SELF_NEEDED}). Skipping."
        )  # DEBUG PRINT
        return None

    # print(f"DEBUG: ID {id_num}: Loaded self images successfully.") # DEBUG PRINT

    # 3. Find two valid relatives
    relatives = find_two_valid_relatives(id_num, relation_graph, valid_id_set)
    # print(f"DEBUG: ID {id_num}: Result of find_two_valid_relatives: {relatives}") # DEBUG PRINT
    if len(relatives) != NUM_RELATIVES_NEEDED:
        print(
            f"DEBUG: ID {id_num}: Found {len(relatives)} valid relatives, need {NUM_RELATIVES_NEEDED}. Skipping."
        )  # DEBUG PRINT
        return None
    relative1_id, relative2_id = relatives
    # print(f"DEBUG: ID {id_num}: Using relatives: {relative1_id}, {relative2_id}") # DEBUG PRINT

    # 4. Get relatives' incorrect images
    relative1_incorrect = None
    rel1_folder_path = find_max_attempt(root_dir, relative1_id)
    if not rel1_folder_path:
        print(
            f"DEBUG: ID {id_num}: Cannot find attempt folder for relative {relative1_id}. Skipping."
        )  # DEBUG PRINT
        return None
    # print(f"DEBUG: ID {id_num}: Found folder for relative {relative1_id}: {rel1_folder_path}") # DEBUG PRINT
    relative1_incorrect = get_images_from_dir(
        os.path.join(rel1_folder_path, "output_incorrect"),
        NUM_INCORRECT_PER_RELATIVE,
    )
    if relative1_incorrect is None:
        print(
            f"DEBUG: ID {id_num}: Failed to load incorrect images for relative {relative1_id} (expected {NUM_INCORRECT_PER_RELATIVE}). Skipping."
        )  # DEBUG PRINT
        return None

    relative2_incorrect = None
    rel2_folder_path = find_max_attempt(root_dir, relative2_id)
    if not rel2_folder_path:
        print(
            f"DEBUG: ID {id_num}: Cannot find attempt folder for relative {relative2_id}. Skipping."
        )  # DEBUG PRINT
        return None
    # print(f"DEBUG: ID {id_num}: Found folder for relative {relative2_id}: {rel2_folder_path}") # DEBUG PRINT
    relative2_incorrect = get_images_from_dir(
        os.path.join(rel2_folder_path, "output_incorrect"),
        NUM_INCORRECT_PER_RELATIVE,
    )
    if relative2_incorrect is None:
        print(
            f"DEBUG: ID {id_num}: Failed to load incorrect images for relative {relative2_id} (expected {NUM_INCORRECT_PER_RELATIVE}). Skipping."
        )  # DEBUG PRINT
        return None

    # print(f"DEBUG: ID {id_num}: Loaded relative images successfully.") # DEBUG PRINT

    # 5. Combine all incorrect images (total 9)
    all_incorrect_images = (
        self_incorrect_images + relative1_incorrect + relative2_incorrect
    )

    # 6. Prepare for image creation
    question_images = correct_images[:4]
    correct_answer_image = correct_images[4]
    image_filename = f"image_{id_num}.png"
    output_image_path = os.path.join(image_dir, image_filename)
    relative_image_path = f"images/{image_filename}"
    # print(f"DEBUG: ID {id_num}: Prepared image paths. Output target: {output_image_path}") # DEBUG PRINT

    # 7. Create the puzzle image
    try:
        # print(f"DEBUG: ID {id_num}: Calling create_puzzle_image_10_options...") # DEBUG PRINT
        correct_answer_label = create_puzzle_image_10_options(
            question_images,
            correct_answer_image,
            all_incorrect_images,
            output_image_path,
            cell_size,
            grid_margin,
        )
        # print(f"DEBUG: ID {id_num}: create_puzzle_image_10_options successful. Label: {correct_answer_label}") # DEBUG PRINT
    except Exception as e:
        # This specific print should already be there from the original code
        print(f"Error creating puzzle image for ID {id_num}: {e}")
        return None

    # 8. Create JSON entry
    options_dict = {label: label for label in OPTION_LABELS}
    # print(f"DEBUG: ID {id_num}: Creating final JSON dictionary.") # DEBUG PRINT

    return {
        "id": int(id_num),
        "prompt": "From the four given options, select the most suitable one to fill in the question mark to present a certain regularity.",
        "options": options_dict,
        "image": relative_image_path,
        "correct_answer": correct_answer_label,
        "relatives_used": [
            int(relative1_id),
            int(relative2_id),
        ],
    }


# --- Orchestration ---


def create_puzzles_from_ids_with_relatives(
    id_list,
    relation_graph,  # Pre-built graph
    valid_id_set,  # Pre-built set
    root_dir="all_scripts",
    output_dir="output",
    cell_size=400,
    grid_margin=30,
    max_workers=16,
):
    """
    Creates 10-option puzzles using relatives for incorrect images.
    Uses multiprocessing plus a progress bar.
    """
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    puzzles = []
    # Prepare arguments for each process call
    args_list = [
        (
            id_num,
            root_dir,
            image_dir,
            cell_size,
            grid_margin,
            relation_graph,
            valid_id_set,
        )
        for id_num in id_list
    ]

    print(f"Starting processing for {len(id_list)} IDs using relationship logic...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_id, *args): args[0]
            for args in args_list  # args[0] is id_num
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing IDs", unit="id"
        ):
            id_num_processed = futures[future]
            try:
                result = future.result()  # Returns dict or None
                if result:
                    puzzles.append(result)
            except Exception as e:
                print(
                    f"Critical error processing ID {id_num_processed}: {e}"
                )  # Catch unexpected errors from the process

    return puzzles


# --- Main Execution ---


def main():
    parser = argparse.ArgumentParser(
        description="Generate 10-option puzzles using relatives and multiprocessing"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/step4/Dataset_check",
        help="Base output directory",
    )
    parser.add_argument(
        "--valid_ids",
        type=str,
        default="./data/step3/3.2_valid_style_1.json",
        help="JSON file with list of IDs to process {'ids': [...]}",
    )
    parser.add_argument(
        "--relations",
        type=str,
        default="./data/step2/2.3_rules.json",
        help="JSON file with parent-child relationship data",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./data/step3/3.1_all_scripts_style_1",
        help="Root directory containing ID_attempt folders",
    )
    parser.add_argument(
        "--cell_size", type=int, default=400, help="Size of each image cell in pixels"
    )  # Adjusted default for 10 options maybe?
    parser.add_argument(
        "--grid_margin",
        type=int,
        default=25,
        help="Margin around the puzzle grid",
    )
    parser.add_argument(
        "--workers", type=int, default=15, help="Number of worker processes"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load Valid IDs and convert to INTEGER Set
    try:
        with open(args.valid_ids, "r") as f:
            # Load raw list which might contain strings or ints
            raw_id_list = json.load(f).get("ids", [])
        if not raw_id_list:
            print(f"Error: No IDs found in {args.valid_ids}")
            return

        # --- Convert to integers and create set ---
        valid_id_list_int = []
        valid_id_set = set()
        for raw_id in raw_id_list:
            try:
                int_id = int(raw_id)  # Convert here
                valid_id_list_int.append(int_id)
                valid_id_set.add(int_id)
            except (ValueError, TypeError):
                print(
                    f"Warning: Could not convert ID '{raw_id}' to integer in {args.valid_ids}. Skipping it."
                )
        # --- End Conversion ---

        if not valid_id_set:
            print(f"Error: No valid integer IDs could be loaded from {args.valid_ids}")
            return

        # Keep the list for iteration order if needed later
        valid_id_list = valid_id_list_int
        print(f"Loaded {len(valid_id_set)} valid integer IDs.")

    except FileNotFoundError:
        print(f"Error: Valid IDs file not found: {args.valid_ids}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.valid_ids}")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading valid IDs: {e}")
        return

    # Load Relationship Data
    try:
        with open(args.relations, "r", encoding="utf-8") as f:  # Added encoding
            relations_list = json.load(f)
        if not isinstance(relations_list, list):
            print(f"Error: Relations file {args.relations} does not contain a list.")
            return
    except FileNotFoundError:
        print(f"Error: Relations file not found: {args.relations}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.relations}")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading relations: {e}")
        return

    # Preprocess relationships (this should work fine with integer IDs from JSON)
    relation_graph = build_relation_graph(
        relations_list
    )  # Make sure this returns dict()

    # Run the main puzzle creation process
    puzzles = create_puzzles_from_ids_with_relatives(
        valid_id_list,  # Pass the list of integer IDs
        relation_graph,
        valid_id_set,  # Pass the set of integer IDs
        args.root,
        args.output,
        args.cell_size,
        args.grid_margin,
        args.workers,
    )

    # Save the final JSON output
    out_json = os.path.join(args.output, "puzzles_relative.json")
    try:
        with open(out_json, "w", encoding="utf-8") as f:  # Added encoding
            json.dump(puzzles, f, indent=2)
        print(
            f"\nCreated {len(puzzles)} puzzles using relative logic and saved to {out_json}"
        )
        if len(puzzles) < len(valid_id_list):
            print(
                f"Note: {len(valid_id_list) - len(puzzles)} IDs were skipped due to errors (missing images, relatives, etc.). Check logs."
            )
    except Exception as e:
        print(f"Error saving output JSON to {out_json}: {e}")


if __name__ == "__main__":
    # Make sure build_relation_graph uses dict() before returning
    # Make sure the main() function correctly loads/converts valid_ids to int set
    main()
