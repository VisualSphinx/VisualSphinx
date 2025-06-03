import argparse
import json
import os
import re
from collections import defaultdict
from math import ceil
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams

#######################################################################
#                           !!!  WARNING  !!!                         #
#                        USE YOUR MODEL ID HERE!!!!!!!!!              #
#######################################################################
MODEL_ID = "VisualSphinx/VisualSphinx-Difficulty-Tagging"

NUM_REPETITIONS = 8
DEFAULT_BATCH_SIZE = 512
INSTRUCTION_FOLLOWING = (
    r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."
)

# --- Helper Functions ---


def load_image(image_path: str) -> Image.Image:
    """Loads an image using PIL."""
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None  # Handle cases where image might be missing or corrupt


def format_options(options: dict) -> str:
    """Formats the options dictionary into a string."""
    return "\n".join([f"{key}: {value}" for key, value in options.items()])


def format_prompt(problem: str, options_text: str, instruction: str) -> str:

    # Define the specific placeholder and vision tags for Qwen2.5-VL
    image_placeholder = "<|image_pad|>"
    vision_tags = f"<|vision_start|>{image_placeholder}<|vision_end|>"

    # Construct the user message content:
    user_content = (
        f"{vision_tags}\n{problem}\n\nThe choices are:\n{options_text}\n\n{instruction}"
    )

    system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
    user_prompt = f"<|im_start|>user\n{user_content}<|im_end|>"
    assistant_prompt_start = "<|im_start|>assistant\n"

    # Combine all parts
    full_prompt = f"{system_prompt}\n{user_prompt}\n{assistant_prompt_start}"

    return full_prompt


def extract_boxed_answer(text: str) -> str | None:
    """Extracts the answer enclosed in \\boxed{}."""
    match = re.search(r"\\boxed{(.*?)}", text)
    if match:
        return match.group(1).strip()
    return None


# --- Main Script Logic ---


def main(args):
    # --- Setup ---
    json_path = Path(args.json_file)
    json_dir = json_path.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_path = output_dir / "raw_output.jsonl"
    accuracy_output_path = output_dir / "accuracy.json"

    # Determine tensor parallel size (number of GPUs)
    if args.tensor_parallel_size is None:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise ValueError("No GPUs detected. vLLM requires at least one GPU.")
        tensor_parallel_size = num_gpus
        print(
            f"Auto-detected {num_gpus} GPUs. Setting tensor_parallel_size={tensor_parallel_size}"
        )
    else:
        tensor_parallel_size = args.tensor_parallel_size
        print(f"Using specified tensor_parallel_size={tensor_parallel_size}")

    # Check if requested GPUs exceed available GPUs
    if tensor_parallel_size > torch.cuda.device_count():
        print(
            f"Warning: Requested tensor_parallel_size ({tensor_parallel_size}) > available GPUs ({torch.cuda.device_count()}). Using {torch.cuda.device_count()} GPUs."
        )
        tensor_parallel_size = torch.cuda.device_count()

    # --- Load Data ---
    print(f"Loading data from {json_path}...")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items.")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    done_counts = defaultdict(int)
    processed_request_count = 0  # Track total requests already completed
    if raw_output_path.exists():
        print(f"Found existing raw output file: {raw_output_path}. Loading progress...")
        try:
            with open(raw_output_path, "r", encoding="utf-8") as f_in:
                # Add tqdm here if the existing file might be large
                for line in tqdm(f_in, desc="Loading Progress"):
                    try:
                        result = json.loads(line)
                        item_id = result.get("id")
                        if item_id is not None:
                            done_counts[item_id] += 1
                            processed_request_count += 1
                    except json.JSONDecodeError:
                        # Keep existing warning style
                        print(
                            f"\nWarning: Skipping malformed line in existing raw output: {line.strip()}"
                        )
            print(
                f"Loaded progress for {len(done_counts)} unique IDs ({processed_request_count} requests completed)."
            )
        except Exception as e:
            print(
                f"Warning: Error reading existing raw output file: {e}. Will process all items."
            )
            done_counts = defaultdict(int)
            processed_request_count = 0

    print("Preparing inference requests (skipping completed items)...")
    all_requests = []  # Requests for THIS run
    skipped_items = 0  # Items skipped due to missing data/image
    items_needing_requests = []  # Items that need some/all repetitions

    # First pass: Filter items that don't need processing at all
    for item in data:  # Use original data list
        item_id = item.get("id")
        if item_id is None:
            skipped_items += 1
            continue

        # Check if already fully completed based on loaded counts
        if done_counts[item_id] >= NUM_REPETITIONS:
            continue  # Skip this item

        # Check other necessary fields (same checks as your original code)
        problem = item.get("prompt")
        options = item.get("options")
        image_relative_path = item.get("image")
        correct_answer = item.get("correct_answer")
        if not all([problem, options, image_relative_path, correct_answer]):
            skipped_items += 1
            continue
        image_path = os.path.join(json_dir, image_relative_path)
        if not os.path.exists(image_path):
            # Keep existing warning style
            print(
                f"\nWarning: Image not found for item {item_id} at {image_path}. Skipping."
            )
            skipped_items += 1
            continue

        items_needing_requests.append(item)  # This item needs processing

    if skipped_items > 0:
        print(f"Skipped {skipped_items} initial items due to missing data/images/ID.")
    print(
        f"{len(items_needing_requests)} unique items need processing or completion in this run."
    )

    # Second pass: Generate only the required number of requests for needed items
    # Use tqdm here as in your original code
    for item in tqdm(items_needing_requests, desc="Preparing Requests"):
        item_id = item["id"]  # We know ID exists here
        problem = item["prompt"]
        options = item["options"]
        image_relative_path = item["image"]
        correct_answer = item["correct_answer"]
        image_path = os.path.join(json_dir, image_relative_path)
        options_text = format_options(options)
        text_prompt = format_prompt(problem, options_text, INSTRUCTION_FOLLOWING)

        # Calculate how many more times this item needs to run
        needed_repetitions = NUM_REPETITIONS - done_counts[item_id]

        for _ in range(needed_repetitions):  # Generate only missing repetitions
            all_requests.append(
                {  # Structure remains the same
                    "id": item_id,
                    "text_prompt": text_prompt,
                    "image_path": str(image_path),
                    "correct_answer": correct_answer,
                    "original_problem": problem,
                    "original_options": options_text,
                }
            )

    print(f"Total inference requests to run in this session: {len(all_requests)}")

    # Handle case where everything is already done
    if not all_requests:
        print("All items appear to be completed based on the existing output file.")
        # Skip LLM initialization and inference if nothing to do
    else:
        # --- Initialize vLLM --- (保持不变, only run if needed)
        print(f"Initializing vLLM with model {MODEL_ID}...")
        try:
            llm = LLM(
                model=MODEL_ID,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=True,
                dtype="auto",
                enforce_eager=True,
            )
            print("vLLM Engine Initialized.")
        except Exception as e:
            print(f"Error initializing vLLM: {e}")
            return

        # --- Setup Sampling Parameters ---
        sampling_params = SamplingParams(temperature=1, max_tokens=1024, top_p=1.0)

        print(f"Running inference for {len(all_requests)} requests...")
        # results list is not strictly needed anymore for final accuracy
        # results_this_run = [] # You could collect current run results here if needed
        num_batches = ceil(len(all_requests) / args.batch_size)

        # --- CHANGE "w" to "a" below ---
        with open(raw_output_path, "a", encoding="utf-8") as raw_f:
            # tqdm total now reflects requests for THIS run
            for i in tqdm(
                range(0, len(all_requests), args.batch_size),
                desc="Batch Inference",
                total=num_batches,  # Total batches THIS run
            ):
                batch_requests_info = all_requests[i : i + args.batch_size]

                # Prepare batch inputs
                batch_inputs = []
                images_to_clear = []
                for idx, req in enumerate(
                    batch_requests_info
                ):  # Use idx from enumerate if needed for warnings
                    input_data = {"prompt": req["text_prompt"]}
                    pil_image = load_image(req["image_path"])
                    if pil_image:
                        input_data["multi_modal_data"] = {"image": pil_image}
                        images_to_clear.append(pil_image)
                    else:
                        # Keep existing warning
                        print(
                            f"\nWarning: Could not load image {req['image_path']} for request index {i+idx}. Prompt sent without image."
                        )
                    batch_inputs.append(input_data)

                # Run inference
                try:
                    outputs = llm.generate(
                        prompts=batch_inputs, sampling_params=sampling_params
                    )
                except Exception as e:
                    # Keep existing error handling
                    print(
                        f"\nError during vLLM generation for batch starting at index {i}: {e}"
                    )
                    del images_to_clear
                    continue  # Skip processing results for this failed batch

                # Process results for the batch
                for idx, output in enumerate(outputs):
                    request_info = batch_requests_info[idx]
                    generated_text = output.outputs[0].text
                    parsed_answer = extract_boxed_answer(generated_text)
                    is_correct = (
                        (parsed_answer == request_info["correct_answer"])
                        if parsed_answer is not None
                        else False
                    )

                    result_data = {  # Keep existing structure
                        "id": request_info["id"],
                        "prompt": request_info["original_problem"],
                        "options": request_info["original_options"],
                        "image_path": request_info["image_path"],
                        "full_prompt_sent": request_info["text_prompt"],
                        "generated_text": generated_text,
                        "parsed_answer": parsed_answer,
                        "correct_answer": request_info["correct_answer"],
                        "is_correct": is_correct,
                    }
                    # results_this_run.append(result_data) # Optional

                    # Write to raw output file immediately (now appends)
                    raw_f.write(json.dumps(result_data) + "\n")

                # Clear image cache for this batch
                del images_to_clear

        print("Inference completed for this session.")
        # End of the 'else' block for running inference

    print("Calculating final accuracy from full raw output file...")
    all_final_results = []  # List to hold all results from the file
    if not raw_output_path.exists():
        print("Error: Raw output file not found. Cannot calculate accuracy.")
        return  # Exit if file doesn't exist after potentially running

    try:
        line_count = 0
        # Count lines first for tqdm total
        with open(raw_output_path, "r", encoding="utf-8") as f_count:
            line_count = sum(1 for line in f_count)

        with open(raw_output_path, "r", encoding="utf-8") as f_in:
            # Add tqdm here for reading potentially large file
            for line in tqdm(f_in, total=line_count, desc="Reading Results"):
                try:
                    result = json.loads(line)
                    # Basic validation: ensure 'id' exists before appending
                    if result.get("id") is not None:
                        all_final_results.append(result)
                except json.JSONDecodeError:
                    print(
                        f"\nWarning: Skipping malformed line in final raw output: {line.strip()}"
                    )
    except Exception as e:
        print(f"Error reading final raw output file for accuracy calculation: {e}")
        # Decide if you want to proceed with potentially partial results or exit
        if not all_final_results:
            return  # Exit if list is still empty

    if not all_final_results:
        print("No valid results found in raw output file to calculate accuracy.")
        return

    # Calculate accuracy using all_final_results (logic is same as before, just input list changed)
    accuracy_by_id = defaultdict(lambda: {"correct": 0, "total": 0})
    for result in all_final_results:  # Use the full list from file
        item_id = result["id"]  # ID check already done above
        accuracy_by_id[item_id]["total"] += 1
        if result.get("is_correct"):  # Safe access
            accuracy_by_id[item_id]["correct"] += 1

    final_accuracies = {}
    Mismatched_totals = 0
    # Use tqdm here as well if many unique IDs
    for item_id, counts in tqdm(accuracy_by_id.items(), desc="Final Acc Calc"):
        # Your existing warning logic for mismatched totals:
        if counts["total"] != NUM_REPETITIONS:
            print(
                f"\nWarning: Item {item_id} has {counts['total']} results, expected {NUM_REPETITIONS}. Accuracy calculated on available results."
            )
            Mismatched_totals += 1
        accuracy = (counts["correct"] / counts["total"]) if counts["total"] > 0 else 0.0
        final_accuracies[item_id] = accuracy

    # Your existing print statements for summary:
    if Mismatched_totals > 0:
        print(f"Found {Mismatched_totals} items with unexpected number of results.")
    print(f"Calculated final accuracies for {len(final_accuracies)} unique IDs.")

    # --- Save Accuracy Results ---
    print(f"Saving accuracy results to {accuracy_output_path}...")
    try:
        with open(accuracy_output_path, "w", encoding="utf-8") as f:
            json.dump(final_accuracies, f, indent=2)
        print("Accuracy results saved.")
    except Exception as e:
        print(f"Error saving accuracy JSON: {e}")

    print("Script finished.")


# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch inference with vLLM for Qwen-VL model."
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="./data/Dataset_Synthetic_1_4/raw_data/questions_with_rank.json",
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output_1_4",
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=None,
        help="Number of GPUs to use (tensor parallel size). Defaults to all available GPUs.",
    )

    args = parser.parse_args()
    main(args)
