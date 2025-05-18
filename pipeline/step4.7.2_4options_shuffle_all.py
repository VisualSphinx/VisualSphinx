import json
import os

# --- Configuration ---
# Input file paths
json1_path = "./data/step4/Dataset_style_1_4options_shuffle/4.1_puzzles.json"
json2_path = "./data/step4/Dataset_style_1_4options/4.7_puzzles_all.json"
# Output file path
output_path = "./data/step4/Dataset_style_1_4options_shuffle/4.7_puzzles_all.json"

# List of fields to copy from JSON 2 items to JSON 1 items
keys_to_copy = [
    "explanation",
    "has_duplicate",
    "Reasonableness",
    "Readability",
    "accuracy",
]

# --- Main Logic ---
try:
    # 1. Load JSON 2 (details) and build lookup dictionary
    print(f"Loading details data source: {json2_path}")
    try:
        with open(json2_path, "r", encoding="utf-8") as f:
            list2 = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found {json2_path}")
        exit(1)  # Exit if crucial file is missing
    except json.JSONDecodeError:
        print(f"Error: File {json2_path} is not valid JSON format.")
        exit(1)

    if not isinstance(list2, list):
        print(f"Error: Content of {json2_path} should be a list.")
        exit(1)

    print("Building details lookup map based on 'id'...")
    details_lookup = {}
    skipped_count_j2 = 0
    duplicate_ids_j2 = 0
    for item2 in list2:
        if isinstance(item2, dict) and "id" in item2:
            try:
                # Assume ID in JSON 2 is numeric, use as integer key
                item_id_int = int(item2["id"])
                if item_id_int in details_lookup:
                    duplicate_ids_j2 += 1
                # If duplicates exist, the last one encountered will overwrite previous ones
                details_lookup[item_id_int] = item2
            except (ValueError, TypeError):
                print(
                    f"Warning: Found invalid or non-integer 'id' in {json2_path}: {item2.get('id')}. Skipping this item."
                )
                skipped_count_j2 += 1
        else:
            print(
                f"Warning: Found invalid item structure in {json2_path}: {item2}. Skipping this item."
            )
            skipped_count_j2 += 1

    if duplicate_ids_j2 > 0:
        print(
            f"Warning: Found {duplicate_ids_j2} duplicate IDs in {json2_path}. Using the last encountered data for each."
        )
    print(
        f"Lookup map built with {len(details_lookup)} entries. Skipped {skipped_count_j2} invalid items from {json2_path}."
    )

    # 2. Load JSON 1 (variations with "id": "xxxx_A")
    print(f"Loading variation data: {json1_path}")
    try:
        with open(json1_path, "r", encoding="utf-8") as f:
            list1 = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found {json1_path}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: File {json1_path} is not valid JSON format.")
        exit(1)

    if not isinstance(list1, list):
        print(f"Error: Content of {json1_path} should be a list.")
        exit(1)

    # 3. Iterate through JSON 1, find matches, and merge data
    print("Starting data merge...")
    merged_data = []
    processed_count = 0
    match_count = 0
    no_match_count = 0
    id_parse_error_count = 0

    for item1 in list1:
        processed_count += 1
        # Basic validation for item structure in list1
        if not isinstance(item1, dict) or "id" not in item1:
            print(
                f"Warning: Found invalid item structure in {json1_path}: {item1}. Skipping this item."
            )
            id_parse_error_count += 1
            continue

        original_id_str = str(item1["id"])  # Ensure it's a string
        base_id_int = None

        # Extract numeric prefix from the variation ID
        try:
            parts = original_id_str.split("_")
            if not parts or not parts[0]:  # Check if splitting worked and prefix exists
                raise ValueError("ID format incorrect, cannot extract prefix.")
            base_id_int = int(parts[0])  # Convert prefix to integer for lookup
        except (ValueError, TypeError, IndexError):
            print(
                f"Warning: Could not parse integer prefix from ID '{original_id_str}' in {json1_path}. Skipping merge for this item."
            )
            id_parse_error_count += 1
            merged_data.append(item1)  # Keep original item even if ID parsing fails
            continue

        # Look up the base ID in the details map
        if base_id_int in details_lookup:
            match_count += 1
            details_item = details_lookup[base_id_int]
            # Copy specified keys from details_item to item1
            for key in keys_to_copy:
                # Use .get() for safe access; if key doesn't exist in details_item, assign None
                item1[key] = details_item.get(key, None)
                # Optional: Warn if a specific key was missing in the source details
                # if key not in details_item:
                #    print(f"Info: Key '{key}' not found in details data for ID {base_id_int}.")
            merged_data.append(item1)  # Append the modified item1
        else:
            no_match_count += 1
            # Optional: Log which IDs didn't find a match
            # print(f"Info: No match found in {json2_path} for ID prefix {base_id_int} (from '{original_id_str}').")
            # Keep the original item1 without adding extra fields if no match found
            # Optionally add None for the keys that would have been copied
            # for key in keys_to_copy:
            #     if key not in item1: # Avoid overwriting if key somehow existed
            #         item1[key] = None
            merged_data.append(item1)

    print(
        f"Merging complete. Processed: {processed_count}. Matched and merged: {match_count}. No match found: {no_match_count}. Invalid ID/structure in JSON1: {id_parse_error_count}."
    )

    # 4. Save the merged result to a new JSON file
    print(f"Saving merged results to: {output_path}")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            # indent=2 makes the output file readable
            # ensure_ascii=False ensures non-ASCII characters (like Chinese) are written correctly
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error: Failed saving output file {output_path}: {e}")
        exit(1)

    print("Operation completed successfully!")

except Exception as e:
    # Catch any unexpected global errors
    print(f"An unexpected error occurred: {e}")
    exit(1)  # Indicate failure
