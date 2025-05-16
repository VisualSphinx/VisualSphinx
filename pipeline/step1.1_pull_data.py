"""
This script downloads a dataset from Hugging Face, reformats its content,
and saves the data as a JSON file along with any associated images.
It is designed to be run from the command line with configurable arguments.
"""

import json
import os
import argparse
from datasets import load_dataset
from PIL import Image as PILImage
import traceback


def download_and_reformat_dataset(args):
    """
    Downloads data from Hugging Face, reformats it according to a specific
    structure, saves the textual data as a JSON file, and stores images
    in a designated subdirectory.

    Args:
        args: An argparse.Namespace object containing command-line arguments.
              Expected attributes include:
              - dataset_name (str): Name of the Hugging Face dataset.
              - split (str): Dataset split to download.
              - hf_token (str, optional): Hugging Face API token.
              - output_directory (str): Base directory for saving outputs.
              - output_json_filename (str): Filename for the output JSON.
              - images_subdir (str): Subdirectory for storing images.
    """
    print(f"Loading dataset '{args.dataset_name}', split '{args.split}'.")
    try:
        dataset = load_dataset(
            args.dataset_name,
            split=args.split,
            token=args.hf_token,
            # trust_remote_code=True # Uncomment if the dataset requires it
        )
    except Exception as e:
        print(
            f"Error loading dataset '{args.dataset_name}' with split '{args.split}': {e}"
        )
        traceback.print_exc()
        return

    output_data_list = []

    os.makedirs(args.output_directory, exist_ok=True)
    output_json_full_path = os.path.join(
        args.output_directory, args.output_json_filename
    )
    images_full_dir = os.path.join(args.output_directory, args.images_subdir)
    os.makedirs(images_full_dir, exist_ok=True)

    print(f"Processing {len(dataset)} items from the dataset...")
    for item_idx, hf_item in enumerate(dataset):
        if item_idx > 0 and item_idx % 100 == 0:
            print(f"Processed {item_idx} items...")

        output_item = {}
        try:
            item_id = hf_item.get("id")
            if item_id is None:
                print(
                    f"Warning: Skipping item at index {item_idx} due to missing 'id'."
                )
                continue

            output_item["id"] = item_id
            output_item["prompt"] = hf_item.get("english_prompt")

            options_str = hf_item.get("options")
            if options_str and isinstance(options_str, str):
                try:
                    output_item["options"] = json.loads(options_str)
                except json.JSONDecodeError:
                    print(
                        f"Warning: Could not parse options JSON string for id {item_id}: '{options_str[:100]}...'"
                    )
                    output_item["options"] = {}
            else:
                output_item["options"] = {}

            output_item["explanation"] = hf_item.get("chinese_explanation")
            output_item["correct_answer"] = hf_item.get("correct_answer")

            output_item["has_image"] = False
            output_item["image"] = None

            image_list_from_hf = hf_item.get("image")

            if (
                image_list_from_hf
                and isinstance(image_list_from_hf, list)
                and len(image_list_from_hf) > 0
            ):
                pil_image = image_list_from_hf[0]

                if pil_image and isinstance(pil_image, PILImage.Image):
                    image_filename = f"image_{item_id}.png"
                    image_save_path_full = os.path.join(images_full_dir, image_filename)

                    try:
                        if pil_image.mode == "RGBA" or pil_image.mode == "P":
                            pil_image = pil_image.convert("RGB")
                        pil_image.save(image_save_path_full)
                        output_item["has_image"] = True
                        output_item["image"] = os.path.join(
                            args.images_subdir, image_filename
                        ).replace("\\", "/")
                    except Exception as e:
                        print(
                            f"Error saving image for id {item_id} to {image_save_path_full}: {e}"
                        )
                        output_item["has_image"] = False
                        output_item["image"] = None
                elif pil_image:
                    print(
                        f"Warning: Item in image list for id {item_id} is not a PIL.Image object: {type(pil_image)}"
                    )

            target_keys = [
                "id",
                "prompt",
                "options",
                "has_image",
                "image",
                "explanation",
                "correct_answer",
            ]
            for key in target_keys:
                if key not in output_item:
                    if key == "has_image":
                        output_item[key] = False
                    elif key == "options":
                        output_item[key] = {}
                    elif key == "image":
                        output_item[key] = None
                    else:
                        output_item[key] = (
                            hf_item.get(key)
                            if key == "id"
                            else hf_item.get(
                                f"chinese_{key}"
                                if key in ["prompt", "explanation"]
                                else key
                            )
                        )

            output_data_list.append(output_item)

        except Exception as e:
            print(
                f"Error processing item with id {hf_item.get('id', f'UNKNOWN_AT_INDEX_{item_idx}')}: {e}"
            )
            traceback.print_exc()
            continue

    print(f"Saving JSON data to {output_json_full_path}...")
    try:
        with open(output_json_full_path, "w", encoding="utf-8") as f:
            json.dump(output_data_list, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved JSON data for {len(output_data_list)} items.")
        print(f"Images (if any) saved in: {images_full_dir}")
    except Exception as e:
        print(f"Error writing JSON to file '{output_json_full_path}': {e}")
        traceback.print_exc()


def main():
    """
    Parses command-line arguments and calls the dataset download
    and reformatting function.
    """
    parser = argparse.ArgumentParser(
        description="Download and reformat a Hugging Face dataset."
    )
    parser.add_argument(
        "--dataset_name",
        default="VisualSphinx/VisualSphinx-Seeds",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--split",
        default="all_seeds",
        help="Dataset split to download.",
    )
    parser.add_argument(
        "--output_directory",
        default="./data/step1",
        help="Output directory for JSON and images.",
    )
    parser.add_argument(
        "--output_json_filename",
        default="1.1_raw.json",
        help="Output JSON filename.",
    )
    parser.add_argument(
        "--images_subdir",
        default="images",
        help="Subdirectory for images.",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Optional Hugging Face API token.",
    )

    args = parser.parse_args()
    download_and_reformat_dataset(args)


if __name__ == "__main__":
    main()
