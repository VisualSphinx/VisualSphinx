"""
Extract specified fields from an input JSON file, rename them per a mapping, and save to an output JSON file.
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
        "--input_json",
        type=str,
        default="./data/step1_filtered/1.6_rules_raw.json",
    )
    parser.add_argument(
        "--output_json", type=str, default="./data/step1_filtered/1.7_seed_rules.json"
    )
    parser.add_argument(
        "--field_mapping",
        type=str,
        default='{"id": "seed_id", "key_points": "seed_rule"}',
    )
    return parser.parse_args()


def extract_and_rename_fields(
    input_json_path: str, output_json_path: str, field_mapping: dict
):
    """
    Load JSON from input_json_path, extract and rename fields according to field_mapping,
    and write the result to output_json_path.
    """
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            result = []
            for item in data:
                new_item = {}
                for old_field, new_field in field_mapping.items():
                    if old_field in item:
                        new_item[new_field] = item[old_field]
                result.append(new_item)
        else:
            result = {}
            for old_field, new_field in field_mapping.items():
                if old_field in data:
                    result[new_field] = data[old_field]

        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Saved processed data to {output_json_path}")
    except Exception as e:
        print(f"Error during processing: {e}")


def main(args):
    """
    Main entry point: parse the field mapping string and invoke the extractor.
    """
    mapping = {}
    if args.field_mapping:
        mapping = json.loads(args.field_mapping)
    extract_and_rename_fields(args.input_json, args.output_json, mapping)


if __name__ == "__main__":
    args = parse_args()
    main(args)
