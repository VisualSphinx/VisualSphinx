import argparse
import json
import os
import re
from collections import defaultdict
from math import ceil
from pathlib import Path
import time  # Added for potential sleep
from typing import List, Dict

import torch
from PIL import Image
from tqdm import tqdm

# Import multiprocessing
import multiprocessing as mp

# --- Constants and Helpers (keep build_prompt, format_options, etc.) ---

#######################################################################
#                           !!!  WARNING  !!!                         #
#                        USE YOUR MODEL ID HERE!!!!!!!!!              #
#######################################################################
MODEL_ID = "VisualSphinx/VisualSphinx-Difficulty-Tagging"

NUM_REPETITIONS = 8
DEFAULT_BATCH_SIZE = 512  # This is PER GPU/WORKER
INSTRUCTION_FOLLOWING = (
    r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."
)


# --- (Keep load_image, format_options, format_prompt, extract_boxed_answer) ---
def load_image(
    image_path: str,
) -> Image.Image | None:  # Using Python 3.10+ Union syntax
    """Loads an image using PIL."""  # Keep existing comments
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:  # Keep simplified error handling
        return None


def format_options(options: dict) -> str:
    """Formats the options dictionary into a string."""  # Keep existing comments
    return "\n".join([f"{key}: {value}" for key, value in options.items()])


def format_prompt(problem: str, options_text: str, instruction: str) -> str:
    # Keep existing implementation
    image_placeholder = "<|image_pad|>"
    vision_tags = f"<|vision_start|>{image_placeholder}<|vision_end|>"
    user_content = (
        f"{vision_tags}\n{problem}\n\nThe choices are:\n{options_text}\n\n{instruction}"
    )
    system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
    user_prompt = f"<|im_start|>user\n{user_content}<|im_end|>"
    assistant_prompt_start = "<|im_start|>assistant\n"
    return f"{system_prompt}\n{user_prompt}\n{assistant_prompt_start}"


def extract_boxed_answer(text: str) -> str | None:  # Using Python 3.10+ Union syntax
    """Extracts the answer enclosed in \\boxed{}."""  # Keep existing comments
    match = re.search(r"\\boxed{(.*?)}", text)
    return match.group(1).strip() if match else None


# --- Worker Function (Keep existing function exactly as provided in the previous step) ---
def worker_inference(
    gpu_id: int, request_chunk: List[Dict], output_dir: Path, batch_size_per_worker: int
):
    """The function run by each data-parallel process."""
    # 1. Set GPU for this specific process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Worker writes to its own file in 'w' mode (overwriting previous content for this worker)
    worker_output_file = output_dir / f"raw_output_worker_{gpu_id}.jsonl"
    # Keep existing print statement
    # print(f"[Worker {gpu_id}] Started, using GPU {gpu_id}. Output: {worker_output_file}")

    # IMPORTANT: Delay vLLM import/init until after setting CUDA_VISIBLE_DEVICES
    from vllm import LLM, SamplingParams

    # Import tqdm for worker progress bar
    from tqdm import tqdm

    try:
        # 2. Initialize vLLM with tp_size=1 on the assigned GPU
        llm = LLM(
            model=MODEL_ID,
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype="auto",
            enforce_eager=True,
            disable_log_stats=True,  # Disable internal vLLM stats/progress
            # Keep existing commented-out line
            # Consider setting gpu_memory_utilization=0.9 or similar if needed
        )
        # Keep existing print statement
        # print(f"[Worker {gpu_id}] vLLM engine initialized on GPU {gpu_id}.")

        sampling_params = SamplingParams(temperature=1, max_tokens=1024, top_p=1.0)

        # 3. Run inference loop for the assigned chunk
        # --- Add tqdm wrapper for worker progress ---
        with open(worker_output_file, "a", encoding="utf-8") as raw_f, tqdm(
            total=len(request_chunk),
            desc=f"GPU {gpu_id}",
            position=gpu_id,
            leave=False,
        ) as pbar_worker:

            # Keep existing loop over batches:
            for i in range(0, len(request_chunk), batch_size_per_worker):
                batch_requests_info = request_chunk[i : i + batch_size_per_worker]

                # Keep existing input preparation:
                batch_inputs = []
                images_to_clear = []
                for req in batch_requests_info:
                    input_data = {"prompt": req["text_prompt"]}
                    pil_image = load_image(req["image_path"])
                    if pil_image:
                        input_data["multi_modal_data"] = {"image": pil_image}
                        images_to_clear.append(pil_image)
                    batch_inputs.append(input_data)

                try:
                    # Keep existing generate call:
                    outputs = llm.generate(
                        prompts=batch_inputs, sampling_params=sampling_params
                    )
                except Exception as e:
                    # Use tqdm's safe write for error messages inside worker loop
                    pbar_worker.write(
                        f"[Worker {gpu_id}] Error generating batch starting at {i}: {e}"
                    )
                    del images_to_clear
                    continue  # Skip to next batch

                # Keep existing result processing:
                items_processed_in_batch = 0
                for idx, output in enumerate(outputs):
                    # Protect against index out of bounds if outputs != batch_inputs
                    if i + idx >= len(request_chunk):
                        continue
                    request_info = request_chunk[
                        i + idx
                    ]  # Index into original chunk list

                    generated_text = output.outputs[0].text
                    parsed_answer = extract_boxed_answer(generated_text)
                    is_correct = (
                        (parsed_answer == request_info["correct_answer"])
                        if parsed_answer is not None
                        else False
                    )
                    result_data = {
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
                    # Write result immediately to the worker's file ('w' mode file)
                    raw_f.write(json.dumps(result_data) + "\n")
                    items_processed_in_batch += 1

                # Update the worker's progress bar
                pbar_worker.update(items_processed_in_batch)

                # Keep existing image cleanup:
                del images_to_clear

        # Keep existing print statement (optional):
        # print(f"[Worker {gpu_id}] Inference complete.")

    except Exception as e:
        # Keep existing error handling:
        print(f"[Worker {gpu_id}] FATAL error during initialization or processing: {e}")

    finally:
        # Keep existing print statement (optional):
        # print(f"[Worker {gpu_id}] Exiting.")
        pass


def main(args):
    # --- Setup (Paths etc. - Keep existing) ---
    json_path = Path(args.json_file)
    json_dir = json_path.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Main output file path (will be overwritten by merge)
    raw_output_path = output_dir / "raw_output.jsonl"
    accuracy_output_path = output_dir / "accuracy.json"

    # --- Determine number of GPUs (Keep existing) ---
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise ValueError("No GPUs detected.")
    print(f"Found {num_gpus} GPUs. Will launch {num_gpus} worker processes.")

    # --- Load Data (Keep existing) ---
    print(f"Loading data from {json_path}...")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items.")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    # === ADDED RESUME LOGIC: Step 1 - Read progress from PREVIOUS worker files ===
    done_counts = defaultdict(int)
    processed_request_count = 0
    print(
        f"Checking existing worker output files in {output_dir} for resume progress..."
    )
    # Find worker files potentially left over from the *last* run attempt
    worker_files_to_check = list(output_dir.glob("raw_output_worker_*.jsonl"))

    if worker_files_to_check:
        # Use tqdm to show progress reading potentially many small files
        for worker_file in tqdm(
            worker_files_to_check, desc="Loading Previous Progress"
        ):
            if not worker_file.is_file():
                continue  # Skip if it's somehow not a file
            try:
                with open(worker_file, "r", encoding="utf-8") as f_in:
                    for line in f_in:  # Read existing content
                        try:
                            result = json.loads(line)
                            item_id = result.get("id")
                            if item_id is not None:
                                # Count completions for each ID across all previous worker files
                                done_counts[item_id] += 1
                                processed_request_count += 1
                        except json.JSONDecodeError:
                            # Use tqdm.write for safe printing alongside progress bar
                            tqdm.write(
                                f"Warning: Skipping malformed line in {worker_file.name}: {line.strip()}"
                            )
            except Exception as e:
                # Use tqdm.write here too
                tqdm.write(
                    f"Warning: Error reading existing worker file {worker_file.name}: {e}"
                )
        print(
            f"Found {processed_request_count} completed requests for {len(done_counts)} unique IDs in previous worker files."
        )
    else:
        print("No existing worker output files found. Starting fresh.")

    # === MODIFIED RESUME LOGIC: Step 2 - Prepare ONLY needed requests for THIS run ===
    print("Preparing inference requests for this run (skipping completed items)...")
    all_requests = []  # Requests to be processed in THIS run session
    skipped_items = 0
    items_needing_requests = []  # Collect base items that need more repetitions

    # First pass: Filter based on done_counts from previous worker files
    # Keep existing tqdm bar for data preparation
    for item in tqdm(data, desc="Filtering Data"):
        item_id = item.get("id")
        if item_id is None:
            skipped_items += 1
            continue
        # Check if this item is already fully completed
        if done_counts[item_id] >= NUM_REPETITIONS:
            continue  # Skip this item

        # Check other necessary fields (same as original script)
        problem = item.get("prompt")
        options = item.get("options")
        image_relative_path = item.get("image")
        correct_answer = item.get("correct_answer")
        if not all([problem, options, image_relative_path, correct_answer]):
            skipped_items += 1
            continue
        image_path = os.path.join(json_dir, image_relative_path)
        if not os.path.exists(image_path):
            skipped_items += 1
            continue

        items_needing_requests.append(item)  # This item needs processing

    # Keep existing print statement
    if skipped_items > 0:
        print(f"Skipped {skipped_items} initial items (missing fields/image/ID).")
    print(
        f"{len(items_needing_requests)} unique items need processing or completion in this run."
    )

    # Second pass: Generate only the required repetitions for this run
    # Keep existing tqdm bar here
    for item in tqdm(items_needing_requests, desc="Preparing Requests"):
        item_id = item["id"]
        problem = item["prompt"]
        options = item["options"]
        image_relative_path = item["image"]
        correct_answer = item["correct_answer"]
        image_path = os.path.join(json_dir, image_relative_path)
        options_text = format_options(options)
        text_prompt = format_prompt(problem, options_text, INSTRUCTION_FOLLOWING)

        # Calculate remaining repetitions needed based on progress read earlier
        needed_repetitions = NUM_REPETITIONS - done_counts[item_id]

        for _ in range(
            needed_repetitions
        ):  # Generate only missing requests for this run
            all_requests.append(
                {  # Request structure is the same
                    "id": item_id,
                    "text_prompt": text_prompt,
                    "image_path": str(image_path),
                    "correct_answer": correct_answer,
                    "original_problem": problem,
                    "original_options": options_text,
                }
            )

    # Keep existing print statement
    print(
        f"Total inference requests generated for this run session: {len(all_requests)}"
    )

    # === Run Workers (Only if needed) ===
    if not all_requests:
        print("All required repetitions seem complete based on previous worker files.")
        # NOTE: Script will still proceed to merge (empty) worker files and calculate accuracy based on that.
    else:
        # --- Split requests for THIS run ---
        requests_per_gpu = ceil(len(all_requests) / num_gpus)
        request_chunks = [
            all_requests[i : i + requests_per_gpu]
            for i in range(0, len(all_requests), requests_per_gpu)
        ]
        print(f"Split this session's requests into {len(request_chunks)} chunks.")

        # --- Launch Worker Processes ---
        print("Launching worker processes...")
        ctx = mp.get_context("spawn")
        processes = []
        start_time = time.time()
        for gpu_id in range(num_gpus):
            if gpu_id >= len(request_chunks):
                break
            chunk_for_worker = request_chunks[gpu_id]
            p = ctx.Process(
                target=worker_inference,
                args=(gpu_id, chunk_for_worker, output_dir, args.batch_size),
            )
            processes.append(p)
            p.start()

        # --- Wait for Workers ---
        for i, p in enumerate(processes):
            p.join()
            # print(f"Worker process {i} finished.") # Reduced verbosity
        end_time = time.time()
        print(
            f"\nAll worker processes completed in {end_time - start_time:.2f} seconds."
        )  # Add newline for cleaner separation from tqdm bars

    # === Merge Results - KEEP EXISTING LOGIC ===
    # This section overwrites the main raw_output.jsonl with ONLY the results
    # from the worker files generated IN THIS RUN.
    print("Merging raw results from this run's worker files...")
    merged_results = []  # Results from THIS RUN ONLY
    # Ensure main raw output file is empty before merging (Original Behavior)
    if raw_output_path.exists():
        raw_output_path.unlink()  # Original Behavior: Overwrite main file

    # Merge files generated by workers *in this run* into the main file
    with open(raw_output_path, "w", encoding="utf-8") as main_raw_f:
        # Use tqdm for merging progress
        for gpu_id in tqdm(range(num_gpus), desc="Merging Worker Files"):
            worker_file = output_dir / f"raw_output_worker_{gpu_id}.jsonl"
            if worker_file.exists():
                # print(f"Merging results from {worker_file}...") # Reduce verbosity
                try:
                    with open(worker_file, "r", encoding="utf-8") as wf:
                        for line in wf:
                            try:
                                result = json.loads(line)
                                merged_results.append(
                                    result
                                )  # Collect for current run accuracy
                                main_raw_f.write(line)
                            except json.JSONDecodeError:
                                # Use tqdm.write for safety if needed during merge progress
                                tqdm.write(
                                    f"Warning: Skipping malformed line in {worker_file.name}: {line.strip()}"
                                )
                    # Keep optional delete logic commented out
                    # worker_file.unlink()
                except Exception as e:
                    tqdm.write(
                        f"Error reading worker file {worker_file.name} during merge: {e}"
                    )
            else:
                # This warning is expected if a worker failed or had no requests in this run
                tqdm.write(f"Warning: Worker output file {worker_file.name} not found.")

    # --- Calculate Accuracy - KEEP EXISTING LOGIC ---
    # Calculates accuracy based ONLY on merged_results from THIS RUN.
    print(f"Total results merged from this run: {len(merged_results)}")
    if len(all_requests) > 0 and len(merged_results) != len(
        all_requests
    ):  # Check only if requests were expected
        print(
            f"Warning: Number of merged results ({len(merged_results)}) does not match requests generated for this run ({len(all_requests)})."
        )

    print("Calculating accuracy (based on results from this run only)...")
    accuracy_by_id = defaultdict(lambda: {"correct": 0, "total": 0})
    # Use merged_results from the current run
    for result in merged_results:
        item_id = result.get("id")
        if item_id is None:
            continue
        accuracy_by_id[item_id]["total"] += 1
        if result.get("is_correct"):
            accuracy_by_id[item_id]["correct"] += 1

    final_accuracies = {}
    Mismatched_totals = 0
    # Keep existing tqdm bar
    for item_id, counts in tqdm(accuracy_by_id.items(), desc="Calculating Accuracy"):
        # This warning is less informative now, as it only checks counts from this run
        # if counts["total"] != NUM_REPETITIONS:
        #     print(f"\nWarning: Item {item_id} has {counts['total']} results in this run.")
        #     Mismatched_totals += 1
        accuracy = (counts["correct"] / counts["total"]) if counts["total"] > 0 else 0.0
        final_accuracies[item_id] = accuracy

    # if Mismatched_totals > 0: print(f"Found {Mismatched_totals} items with partial results in this run.")
    print(
        f"Calculated accuracies for {len(final_accuracies)} unique IDs processed in this run."
    )

    # --- Save Accuracy Results ---
    print(f"Saving accuracy results (for this run) to {accuracy_output_path}...")
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
        description="Run data-parallel inference with vLLM (resume checks worker files, overwrites main output)."
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="./data/step4/Dataset_style_1_4options/4.3_puzzles_filtered.json",
        help="Path to input JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/step4/Dataset_style_1_4options/4.5_output",
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Inference batch size PER WORKER/GPU.",
    )
    args = parser.parse_args()
    main(args)
