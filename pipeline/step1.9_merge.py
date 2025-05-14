"""
Merge reference JSON with source JSON tags, preserving all reference fields and
adding/updating question_type and knowledge_point from the source, then save merged output.
"""

import argparse
import json
import os


def parse_args():
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference_json",
        type=str,
        default="./data/step1_filtered/1.7_seed_rules.json",
    )
    parser.add_argument(
        "--source_json",
        type=str,
        default="./data/step1_filtered/1.8_rule_class.json",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="./data/step1_filtered/1.9_seeds.json",
    )
    return parser.parse_args()


def merge_json_data(
    reference_json_path: str, source_json_path: str, output_json_path: str
):
    """
    Load reference and source JSON files, merge entries by seed_id, and write to output.
    """
    try:
        with open(reference_json_path, "r", encoding="utf-8") as f:
            reference_data = json.load(f)
        with open(source_json_path, "r", encoding="utf-8") as f:
            source_data = json.load(f)

        source_id_map = {item["id"]: item for item in source_data}
        merged = []
        found_count = 0

        for ref_item in reference_data:
            ref_id = ref_item.get("seed_id")
            item = ref_item.copy()
            source_item = source_id_map.get(ref_id)
            if source_item:
                item["question_type"] = source_item.get("question_type", "")
                item["knowledge_point"] = source_item.get("knowledge_point", "")
                found_count += 1
            else:
                item["question_type"] = ""
                item["knowledge_point"] = ""
                print(f"Warning: ID {ref_id} not found in source JSON")
            merged.append(item)

        print(f"Processed {len(merged)} items, {found_count} matches found")

        out_dir = os.path.dirname(output_json_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        print(f"Saved merged data to {output_json_path}")
    except Exception as e:
        print(f"Error during merging: {e}")


def main(args):
    """
    Main entry point: merge JSON files using provided arguments.
    """
    merge_json_data(args.reference_json, args.source_json, args.output_json)


if __name__ == "__main__":
    args = parse_args()
    main(args)
