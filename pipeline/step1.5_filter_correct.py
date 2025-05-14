"""
This script filters a dataset of puzzles by identifying items where the
predicted answer matches the correct answer. It uses two input JSON files
(puzzle results and merged questions) to generate a new, smaller dataset
containing only these correctly solved puzzles, optionally copying associated
images and re-indexing the puzzle IDs.
"""

import json
import os
import shutil
import argparse
from tqdm import tqdm
import sys


def filter_correct_predictions(
    puzzle_results_path: str,
    merged_questions_path: str,
    output_dir: str,
    copy_images: bool = True,
):
    """Filters puzzles based on correct predictions and creates a new dataset."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    images_output_dir = os.path.join(output_dir, "images")
    if copy_images and not os.path.exists(images_output_dir):
        os.makedirs(images_output_dir)
        print(f"Created image output directory: {images_output_dir}")

    try:
        with open(puzzle_results_path, "r", encoding="utf-8") as f:
            puzzle_results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Puzzle results file not found at '{puzzle_results_path}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(
            f"Error decoding JSON from puzzle results file '{puzzle_results_path}': {e}"
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error reading '{puzzle_results_path}': {e}")
        sys.exit(1)

    try:
        with open(merged_questions_path, "r", encoding="utf-8") as f:
            merged_questions = json.load(f)
    except FileNotFoundError:
        print(f"Error: Merged questions file not found at '{merged_questions_path}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(
            f"Error decoding JSON from merged questions file '{merged_questions_path}': {e}"
        )
        sys.exit(1)
    except Exception as e:
        print(f"Error reading '{merged_questions_path}': {e}")
        sys.exit(1)

    correct_ids = []
    for item in puzzle_results:
        if "answer" in item and "correct_answer" in item:
            prediction = str(item.get("answer", "")).strip().upper()
            correct = str(item.get("correct_answer", "")).strip().upper()
            item_id = item.get("id")

            if prediction == correct and item_id is not None:
                correct_ids.append(item_id)

    print(f"Found {len(correct_ids)} correctly predicted puzzles.")
    if not correct_ids:
        print("No correctly predicted puzzles found. Exiting.")
        sys.exit(0)

    id_mapping = {original_id: new_id for new_id, original_id in enumerate(correct_ids)}
    filtered_questions = []
    image_paths_to_copy = []
    id_to_question_map = {item.get("id"): item for item in merged_questions}

    for original_id in correct_ids:
        if original_id in id_to_question_map:
            question_item = id_to_question_map[original_id]
            new_id = id_mapping[original_id]

            new_question_item = question_item.copy()
            new_question_item["id"] = new_id

            original_image_path_in_json = question_item.get("image", "")
            if original_image_path_in_json:
                image_filename_original = os.path.basename(original_image_path_in_json)
                new_image_filename = (
                    f"image_{new_id}{os.path.splitext(image_filename_original)[1]}"
                )
                new_question_item["image"] = os.path.join(
                    "images", new_image_filename
                ).replace("\\", "/")

                image_paths_to_copy.append(
                    (original_image_path_in_json, new_image_filename)
                )

            filtered_questions.append(new_question_item)

    output_json_path = os.path.join(output_dir, "1.5_questions.json")
    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(filtered_questions, f, ensure_ascii=False, indent=2)
        print(f"Saved filtered questions to: {output_json_path}")
    except IOError as e:
        print(f"Error saving filtered questions to '{output_json_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while saving filtered questions: {e}")
        sys.exit(1)

    if copy_images:
        print("Starting to copy image files...")
        copied_count = 0
        source_base_dir = os.path.dirname(merged_questions_path)
        for original_relative_path, new_filename_only in tqdm(
            image_paths_to_copy, desc="Copying images"
        ):
            possible_source_paths = [
                original_relative_path,  # If it's an absolute path or relative to current execution
                os.path.join(
                    source_base_dir, original_relative_path
                ),  # Relative to the merged_questions_path
                os.path.join(
                    source_base_dir, "images", os.path.basename(original_relative_path)
                ),  # In an 'images' subdir relative to merged_questions_path
            ]

            found_source = False
            for source_path_attempt in possible_source_paths:
                if os.path.exists(source_path_attempt):
                    try:
                        destination_path = os.path.join(
                            images_output_dir, new_filename_only
                        )
                        shutil.copy(source_path_attempt, destination_path)
                        copied_count += 1
                        found_source = True
                        break
                    except Exception as e:
                        print(
                            f"Error copying image file from '{source_path_attempt}' to '{destination_path}': {e}"
                        )
                        break  # Stop trying for this image if copy fails
            if not found_source:
                print(
                    f"Warning: Image file not found for original path entry '{original_relative_path}' after checking multiple locations."
                )
        print(f"Finished copying images. {copied_count} images copied.")

    print("\nProcessing Summary:")
    print(f"- Total number of puzzles in source: {len(merged_questions)}")
    print(f"- Number of correctly predicted puzzles identified: {len(correct_ids)}")
    print(
        f"- Number of successfully filtered puzzles written: {len(filtered_questions)}"
    )
    if copy_images:
        print(
            f"- Image files for filtered puzzles processed. Target directory: {images_output_dir}"
        )
    print(f"- Filtered JSON saved to: {output_json_path}")


def main():
    """Parses command-line arguments and initiates the puzzle filtering process."""
    parser = argparse.ArgumentParser(
        description="Filters correctly predicted puzzles and creates a new dataset."
    )
    parser.add_argument(
        "--results",
        type=str,
        default="./data/step1/1.4_verify.json",
        help="Path to the p1.4_verify file.",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="./data/step1/1.3_raw_english.json",
        help="Path to the 1.3_raw_english.json file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/step1_filtered",
        help="Directory to save the output files.",
    )
    parser.add_argument(
        "--no-images", action="store_true", help="Do not copy image files."
    )

    args = parser.parse_args()

    filter_correct_predictions(
        args.results, args.questions, args.output, not args.no_images
    )


if __name__ == "__main__":
    main()
