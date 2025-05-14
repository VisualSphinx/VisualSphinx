"""
This script translates Chinese text from a JSON file to English using the
Anthropic Claude API. It supports concurrent processing, API call retries
with exponential backoff, loading of a custom prompt template,
and saving intermediate results to resume progress. The API key is expected
to be in an 'api_config.py' file.
"""

import anthropic
import argparse
import json
import sys
import re
import threading
import time
from tqdm import tqdm
import concurrent.futures
import backoff
import random
from pathlib import Path

# Attempt to import API_KEY from api_config.py
from api_config import Anthropic_API_KEY


@backoff.on_exception(
    backoff.expo,
    Exception,  # Catches a broad range of exceptions for retrying
    max_tries=8,
    base=2,
    max_time=300,
    giveup=lambda e: not (
        isinstance(e, Exception) and "rate_limit_error" in str(e).lower()
    ),  # Give up if not a rate limit error
)
def translate_with_retry(client, model, prompt, temperature, max_tokens):
    """
    Sends a translation request to the Anthropic API with a retry mechanism.

    Args:
        client: The Anthropic API client.
        model (str): The model name to use for translation.
        prompt (str): The prompt content for the API.
        temperature (float): The temperature for generation.
        max_tokens (int): The maximum number of tokens for the response.

    Returns:
        The API response object.

    Raises:
        Exception: Propagates exceptions from the API call if retries fail,
                   or if it's a non-retryable error.
    """
    try:
        response = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response
    except Exception as e:
        print(f"Encountered API error: {str(e)}, retrying...")
        raise


def main():
    """
    Main function to parse arguments, load data, manage translation tasks
    concurrently, and save results.
    """
    parser = argparse.ArgumentParser(
        description="Translate Chinese to English using the Claude API."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./data/step1/1.1_raw.json",
        help="Input JSON file with Chinese explanations.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/step1/1.2_claude_rewrite.json",
        help="Output JSON file for translation results.",
    )
    parser.add_argument(
        "--prompt_file_path",
        type=str,
        default="Prompts/step1.2_rewrite.md",
        help="Path to the MD file containing the prompt template.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-haiku-20241022",
        help="Claude model to use.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Maximum number of concurrent worker threads.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--request_delay",
        type=float,
        default=0.5,
        help="Delay in seconds between requests.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="Save intermediate results after this many processed items.",
    )

    args = parser.parse_args()

    if not Anthropic_API_KEY:
        print("Error: ANTHROPIC_API_KEY is not correctly configured in api_config.py.")
        print("Please set your Anthropic API key in api_config.py.")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=Anthropic_API_KEY)

    prompt_template_path = Path(args.prompt_file_path)
    if not prompt_template_path.is_file():
        print(f"Error: Prompt file not found at {prompt_template_path}")
        sys.exit(1)
    try:
        prompt_template_content = prompt_template_path.read_text(encoding="utf-8")
        print(f"Loaded prompt template from {prompt_template_path}")
    except Exception as e:
        print(f"Error: Failed to load prompt file {prompt_template_path}: {str(e)}")
        sys.exit(1)

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from input file {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading input file {args.input}: {str(e)}")
        sys.exit(1)

    print(f"Found {len(data)} items to process.")
    print(f"Using {args.max_workers} worker threads for concurrent processing.")
    print(f"Delay between requests: {args.request_delay} seconds.")

    results = {}
    results_lock = threading.Lock()
    intermediate_output_path = Path(args.output)
    intermediate_output_file = (
        intermediate_output_path.parent
        / f"{intermediate_output_path.stem}_intermediate{intermediate_output_path.suffix}"
    )

    try:
        with open(intermediate_output_file, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
            for item in existing_results:
                item_id = item.get("id")
                if item_id is not None:
                    results[item_id] = item
            print(
                f"Loaded {len(results)} previously translated items from {intermediate_output_file}"
            )
    except FileNotFoundError:
        print("No existing intermediate results file found. Processing from scratch.")
    except Exception as e:
        print(f"Error loading intermediate results: {str(e)}. Processing from scratch.")

    processed_since_last_save = 0

    def save_intermediate_results_local():
        """Saves current results to the intermediate file."""
        nonlocal processed_since_last_save
        try:
            with results_lock:
                all_items_in_memory = list(results.values())
                all_items_in_memory_sorted = sorted(
                    [
                        item
                        for item in all_items_in_memory
                        if item.get("id") is not None
                    ],
                    key=lambda x: x.get("id", float("inf")),
                )
            with open(intermediate_output_file, "w", encoding="utf-8") as f:
                json.dump(all_items_in_memory_sorted, f, ensure_ascii=False, indent=2)
            print(
                f"Saved {len(all_items_in_memory_sorted)} intermediate results to {intermediate_output_file}"
            )
            processed_since_last_save = 0
        except Exception as e:
            print(f"Error saving intermediate results: {str(e)}")

    def translate_item_process(item_data, current_prompt_template):
        """
        Processes a single item for translation, including API call and result parsing.

        Args:
            item_data (dict): The item data containing text to be translated.
            current_prompt_template (str): The prompt template string.

        Returns:
            dict: The processed item with translation or error information.
        """
        nonlocal processed_since_last_save
        item_id = item_data.get("id")
        if item_id in results and (
            "translation" in results[item_id] or "error" in results[item_id]
        ):
            return results[item_id]  # Already processed

        try:
            chinese_explanation = item_data.get("explanation", "")
            prompt = current_prompt_template.format(
                chinese_explanation_placeholder=chinese_explanation
            )

            # Add a small random jitter to the delay
            time.sleep(args.request_delay + random.uniform(0, args.request_delay * 0.5))

            message = translate_with_retry(
                client=client,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            response_text = message.content[0].text
            pattern = r"<translated_explanation>\s*(.*?)\s*</translated_explanation>"
            match = re.search(pattern, response_text, re.DOTALL)
            translation = match.group(1).strip() if match else response_text.strip()
            if not match:
                print(
                    f"Warning: <translated_explanation> tag not found in output for item {item_id}. Using full response."
                )

            result = {"id": item_id, "translation": translation}
            with results_lock:
                results[item_id] = result
                processed_since_last_save += 1
                if processed_since_last_save >= args.save_interval:
                    save_intermediate_results_local()
            return result
        except Exception as e:
            error_message = f"Error processing item {item_id}: {str(e)}"
            print(error_message)
            error_result = {"id": item_id, "error": str(e)}
            with results_lock:
                results[item_id] = error_result
                processed_since_last_save += 1
                if processed_since_last_save >= args.save_interval:
                    save_intermediate_results_local()
            return error_result

    items_to_process = [
        item
        for item in data
        if not (
            item.get("id") in results
            and (
                "translation" in results[item.get("id")]
                or "error" in results[item.get("id")]
            )
        )
    ]
    if not items_to_process:
        print(
            "No new items to process. All items from input found in intermediate results."
        )
    else:
        print(f"Need to process {len(items_to_process)} new items.")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            future_to_item = {
                executor.submit(
                    translate_item_process, item, prompt_template_content
                ): item
                for item in items_to_process
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_item),
                total=len(items_to_process),
                desc="Processing translations",
            ):
                try:
                    future.result()
                except Exception as exc:
                    item = future_to_item[future]
                    item_id = item.get("id", "UnknownID")
                    print(
                        f"Task for item {item_id} generated an unhandled exception: {exc}"
                    )
                    if item_id not in results or "error" not in results.get(
                        item_id, {}
                    ):
                        with results_lock:
                            results[item_id] = {
                                "id": item_id,
                                "error": f"Top-level task error: {str(exc)}",
                            }
                            # Potentially trigger save here too if critical
                            # processed_since_last_save += 1
                            # if processed_since_last_save >= args.save_interval:
                            #     save_intermediate_results_local()

    # Final save of all results
    save_intermediate_results_local()  # Saves everything processed to intermediate
    # Then copy intermediate to final output name
    try:
        Path(intermediate_output_file).rename(args.output)
        print(f"Translation complete. Final results saved to {args.output}")
        if intermediate_output_file.exists() and str(intermediate_output_file) != str(
            Path(args.output)
        ):  # if rename failed or is on different filesystems
            # As a fallback or if rename isn't suitable cross-filesystem, copy then delete
            import shutil

            shutil.copy2(intermediate_output_file, args.output)
            # intermediate_output_file.unlink() # Optionally remove intermediate after successful copy
            print(
                f"Final results copied to {args.output}. Intermediate file retained or deleted based on implementation."
            )

    except Exception as e:
        print(
            f"Error finalizing output file: {str(e)}. Results are in {intermediate_output_file}"
        )

    # Final tally from the 'results' dictionary which should be comprehensive
    final_results_list = list(results.values())
    success_count = sum(
        1
        for item in final_results_list
        if "translation" in item and "error" not in item
    )
    error_count = sum(1 for item in final_results_list if "error" in item)

    print(
        f"Total items in memory (includes previously loaded and newly processed): {len(final_results_list)}"
    )
    print(
        f"Successfully translated: {success_count}, Failed translations: {error_count}"
    )


if __name__ == "__main__":
    main()
