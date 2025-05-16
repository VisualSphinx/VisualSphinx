"""
Split a JSON dataset by entries where both tags are 'Others' versus the rest,
reassign new 0-based seed_ids, and record the ID mapping.
"""

import argparse
import copy
import json
import os


def parse_args():
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json", type=str, default="./data/step1_filtered/1.9_seeds.json"
    )
    parser.add_argument(
        "--others_output", type=str, default="./data/step2/2.1_others.json"
    )
    parser.add_argument(
        "--remaining_output", type=str, default="./data/step2/2.1_eight_classes.json"
    )
    parser.add_argument(
        "--mapping_output", type=str, default="./data/step2/2.1_map.json"
    )
    return parser.parse_args()


def split_json_by_others_tag(
    input_json_path: str,
    others_output_path: str,
    remaining_output_path: str,
    mapping_output_path: str,
) -> dict | None:
    """
    Load JSON from input_json_path, separate entries where both question_type
    and knowledge_point are 'Others', reassign seed_id values, and save three files:
    - entries with both tags 'Others'
    - remaining entries
    - mapping from original seed_id to new file and new_id
    Returns the ID mapping dictionary or None on error.
    """
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading input JSON: {e}")
        return None

    both_others_list = []
    remaining_items_list = []
    id_map = {}
    both_counter = 0
    remaining_counter = 0

    os.makedirs(os.path.dirname(others_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(remaining_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(mapping_output_path), exist_ok=True)

    for item in data:
        original_id = item.get("seed_id")
        if original_id is None:
            print(f"Warning: missing seed_id in item, skipping: {item}")
            continue
        item_copy = copy.deepcopy(item)
        qt = item.get("question_type", "")
        kp = item.get("knowledge_point", "")
        if qt == "Others" and kp == "Others":
            new_id = both_counter
            item_copy["seed_id"] = new_id
            both_others_list.append(item_copy)
            id_map[original_id] = {
                "file": os.path.basename(others_output_path),
                "new_id": new_id,
            }
            both_counter += 1
        else:
            new_id = remaining_counter
            item_copy["seed_id"] = new_id
            remaining_items_list.append(item_copy)
            id_map[original_id] = {
                "file": os.path.basename(remaining_output_path),
                "new_id": new_id,
            }
            remaining_counter += 1

    try:
        with open(others_output_path, "w", encoding="utf-8") as f:
            json.dump(both_others_list, f, ensure_ascii=False, indent=4)
        print(f"Created {both_counter} entries in {others_output_path}")
    except Exception as e:
        print(f"Error writing others_output: {e}")

    try:
        with open(remaining_output_path, "w", encoding="utf-8") as f:
            json.dump(remaining_items_list, f, ensure_ascii=False, indent=4)
        print(f"Created {remaining_counter} entries in {remaining_output_path}")
    except Exception as e:
        print(f"Error writing remaining_output: {e}")

    try:
        with open(mapping_output_path, "w", encoding="utf-8") as f:
            json.dump(id_map, f, ensure_ascii=False, indent=4)
        print(f"Saved ID mapping to {mapping_output_path}")
    except Exception as e:
        print(f"Error writing mapping_output: {e}")

    return id_map


def main(args):
    """
    Main entry point: run the split operation with provided arguments.
    """
    split_json_by_others_tag(
        args.input_json, args.others_output, args.remaining_output, args.mapping_output
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
