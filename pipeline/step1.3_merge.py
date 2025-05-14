"""
This script merges translated 'explanation' fields from one JSON file into
another primary JSON file. It matches items based on their 'id' and
updates the 'explanation' in the primary data with the corresponding
translation.
"""

import json
import argparse
import sys


def replace_explanations(original_json_file, translations_json_file, output_json_file):
    """
    Replaces 'explanation' fields in the original JSON data with translations
    from a separate JSON file, based on matching 'id' fields.

    Args:
        original_json_file (str): Path to the original JSON file.
        translations_json_file (str): Path to the JSON file containing translations.
                                      Each item should have an "id" and a "translation" field.
        output_json_file (str): Path to save the modified JSON data.

    Returns:
        bool: True if successful, False otherwise.
    """
    # Read the original JSON file
    try:
        with open(original_json_file, "r", encoding="utf-8") as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Original JSON file not found at '{original_json_file}'")
        return False
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from original file '{original_json_file}'."
        )
        return False
    except Exception as e:
        print(f"Error reading original JSON file '{original_json_file}': {e}")
        return False

    # Read the translations JSON file
    try:
        with open(translations_json_file, "r", encoding="utf-8") as f:
            translations_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Translations JSON file not found at '{translations_json_file}'")
        return False
    except json.JSONDecodeError:
        print(
            f"Error: Could not decode JSON from translations file '{translations_json_file}'."
        )
        return False
    except Exception as e:
        print(f"Error reading translations JSON file '{translations_json_file}': {e}")
        return False

    # Create a map from ID to translation
    translations_map = {}
    for item in translations_data:
        if "id" in item and "translation" in item:
            translations_map[item["id"]] = item["translation"]

    if not translations_map:
        print(
            "No valid translations (with 'id' and 'translation' fields) found in the translations JSON."
        )
        # Decide if this is a fatal error or if we should proceed and write the original data
        # For now, let's consider it non-fatal but indicate no replacements will occur.
        # If it should be fatal, return False here.

    # Replace explanations in the original JSON
    replaced_count = 0
    missing_translation_for_id = 0
    no_explanation_field_count = 0

    for item in original_data:
        item_id = item.get("id")
        if item_id is not None:
            if item_id in translations_map:
                if "explanation" in item:
                    item["explanation"] = translations_map[item_id]
                    replaced_count += 1
                else:
                    no_explanation_field_count += 1
            else:
                missing_translation_for_id += 1
        # Items without an 'id' will be skipped for replacement

    if no_explanation_field_count > 0:
        print(
            f"Warning: {no_explanation_field_count} items in the original file had a translatable ID but no 'explanation' field."
        )
    if missing_translation_for_id > 0:
        print(
            f"Warning: {missing_translation_for_id} items in the original file had an ID for which no translation was found."
        )

    # Save the result to the output JSON file
    try:
        with open(output_json_file, "w", encoding="utf-8") as f:
            json.dump(original_data, f, ensure_ascii=False, indent=2)
        print(
            f"Completed! Replaced {replaced_count} explanations. Result saved to {output_json_file}"
        )
        return True
    except Exception as e:
        print(f"Error saving output JSON file '{output_json_file}': {e}")
        return False


def main():
    """
    Parses command-line arguments and calls the function to replace explanations.
    """
    parser = argparse.ArgumentParser(
        description="Replaces 'explanation' fields in a JSON file using a translations file."
    )
    parser.add_argument(
        "--original",
        type=str,
        default="./data/step1/1.1_raw.json",
        help="Path to the original JSON file.",
    )
    parser.add_argument(
        "--translations",
        type=str,
        default="./data/step1/1.2_claude_rewrite.json",
        help="Path to the JSON file containing translations.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/step1/1.3_raw_english.json",
        help="Path for the output JSON file with replaced explanations.",
    )

    args = parser.parse_args()

    if not replace_explanations(args.original, args.translations, args.output):
        sys.exit(1)  # Exit with an error code if replacement fails


if __name__ == "__main__":
    main()
