import json
import os

# --- Configuration ---
# Input file paths
json1_path = "./data/step4/Dataset_style_1_10options/4.1_puzzles.json"
json2_path = "./data/step4/Dataset_style_1_4options/4.7_puzzles_all.json"
# Output file path
output_path = "./data/step4/Dataset_style_1_10options/4.7_puzzles_all.json"

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
    id_error_count = 0  # Renamed counter

    for item1 in list1:
        processed_count += 1
        # Basic validation for item structure in list1
        if not isinstance(item1, dict) or "id" not in item1:
            print(
                f"Warning: Found invalid item structure in {json1_path}: {item1}. Skipping this item."
            )
            id_error_count += 1
            continue

        lookup_key_int = None
        # Directly use the 'id' field from JSON 1, ensuring it's an integer
        try:
            lookup_key_int = int(item1["id"])
        except (ValueError, TypeError, KeyError):
            print(
                f"Warning: Invalid or missing 'id' in {json1_path} for item: {item1}. Skipping merge for this item."
            )
            id_error_count += 1
            # Decide whether to keep item with errors or skip entirely.
            # Current logic keeps it but won't merge details. Let's ensure accuracy is None here too.
            if isinstance(item1, dict):  # Check again it's a dict before modifying
                item1["accuracy"] = None  # Set accuracy to None even on ID error
            merged_data.append(item1)
            continue

        # Look up the integer ID in the details map
        if lookup_key_int in details_lookup:
            match_count += 1
            details_item = details_lookup[lookup_key_int]
            # Copy specified keys from details_item to item1
            for key in keys_to_copy:
                # Use .get() for safe access; if key doesn't exist in details_item, assign None
                # Don't specifically handle 'accuracy' here anymore, as we overwrite it below
                item1[key] = details_item.get(key, None)
        else:
            no_match_count += 1
            # Item from list1 has no match in list2 details lookup
            # Optional: Ensure keys_to_copy (except accuracy) are present with None if they don't exist
            # for key in keys_to_copy:
            #     if key != "accuracy" and key not in item1:
            #          item1[key] = None
            pass  # Keep original item1 data

        # --- !!! KEY CHANGE HERE !!! ---
        # ALWAYS set 'accuracy' to None before appending, regardless of match status
        item1["accuracy"] = None
        # --- !!! END OF KEY CHANGE !!! ---

        merged_data.append(item1)  # Append the potentially modified item1

    print(
        f"Merging complete. Processed: {processed_count}. Matched and merged: {match_count}. No match found: {no_match_count}. Invalid ID/structure in JSON1: {id_error_count}."
    )  # Updated print message

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
