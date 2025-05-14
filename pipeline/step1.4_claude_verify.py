# -*- coding: utf-8 -*-
"""
This script provides an automated solution for tackling logic puzzles,
leveraging the capabilities of the Anthropic Claude API. It processes
puzzle data, which can include images, by generating tailored prompts
via Jinja2 templates and then retrieves and records the AI's reasoning
and answers. The tool is designed for batch operations with command-line
configurability for ease of use.
"""
import anthropic
import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import random
import re
import threading
import time
from pathlib import Path
import sys  # For sys.exit

import backoff
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm
from api_config import Anthropic_API_KEY

MODEL_NAME = "claude-3-7-sonnet-20250219"  # Default model name
TEMPLATE_PATH = "Prompts/step1.4_verify.md"  # Default template path
REASON_REGEX = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
ANSWER_REGEX = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


class LogicPuzzleSolver:
    """
    Solves logic puzzles by generating prompts from data and images using
    Jinja2 templates, then querying the Anthropic Claude API.
    It supports batch processing, retries, and periodic saving.
    """

    def __init__(
        self,
        api_key: str,
        template_path: str,
        model: str = MODEL_NAME,
        max_workers: int = 4,
        request_delay: float = 1.0,
        max_tokens: int = 20000,
        temperature: float = 1.0,
    ):
        """
        Initializes the LogicPuzzleSolver.

        Args:
            api_key (str): The Anthropic API key.
            template_path (str): Path to the Jinja2 prompt template file.
            model (str): The name of the Claude model to use.
            max_workers (int): Maximum number of worker threads for batch processing.
            request_delay (float): Base delay in seconds between API requests.
            max_tokens (int): Maximum number of tokens for the API response.
            temperature (float): Temperature for the API response generation.
        """
        if not api_key:
            # This check is good, but main also handles missing API key.
            # Consider if __init__ should raise an error or if main's check is sufficient.
            # For robustness, raising ValueError here is fine.
            raise ValueError("API key must be provided for LogicPuzzleSolver.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.request_delay = request_delay
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.results = {}
        self.lock = threading.Lock()

        p_template_path = Path(template_path)
        if not p_template_path.is_file():
            raise FileNotFoundError(f"Prompt template file not found: {template_path}")
        env = Environment(loader=FileSystemLoader(p_template_path.parent))
        self.template = env.get_template(p_template_path.name)

    @staticmethod
    def encode_image(path: str) -> tuple[str, str]:
        """
        Encodes an image file to a base64 string and determines its MIME type.

        Args:
            path (str): The path to the image file.

        Returns:
            tuple[str, str]: A tuple containing the base64 encoded image string
                             and its MIME type.
        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        mime, _ = mimetypes.guess_type(path)
        if not mime:  # Fallback MIME type
            mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        with open(path, "rb") as f:
            b64_string = base64.b64encode(f.read()).decode("utf-8")
        return b64_string, mime

    def _build_prompt(self, item: dict, img_b64: str, mime: str) -> list[dict]:
        """
        Constructs the prompt for the API using a Jinja2 template,
        item data, and encoded image data. The template is expected to have an
        "" marker where the image representation will be implicitly handled
        by the API based on the surrounding <image> tags in the template.

        Args:
            item (dict): The puzzle item containing details like 'prompt', 'options', 'explanation'.
            img_b64 (str): The base64 encoded image string.
            mime (str): The MIME type of the image.

        Returns:
            list[dict]: A list of dictionaries representing the structured content for the API,
                        following the user's original data URI format for images.
        """
        options_block = "\n".join(
            f"{k}: {v}" for k, v in item.get("options", {}).items()
        )
        rendered_template = self.template.render(
            question=item.get("prompt", ""),
            options=options_block,
            hint="\n".join(item.get("explanation", [])) or "",
        )

        prefix, suffix = rendered_template.split("<!--SPLIT-->")

        return [
            {"type": "text", "text": prefix},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": img_b64,
                },
            },
            {"type": "text", "text": suffix},
        ]

    @backoff.on_exception(
        backoff.expo, Exception, max_tries=8, base=2, jitter=None, max_time=300
    )
    def analyze_puzzle(self, item: dict) -> dict:
        """
        Analyzes a single puzzle item.

        This involves encoding its image (if specified), building a prompt using the
        Jinja2 template, sending the request to the Anthropic API with retries,
        and parsing the reasoning and answer from the response.

        Args:
            item (dict): The puzzle item. Expected to have an 'id' and 'image' path.
                         Other fields like 'prompt', 'options', 'explanation',
                         'correct_answer' are used if present.

        Returns:
            dict: A dictionary containing the puzzle ID, correct answer (if provided
                  in input), extracted reasoning, extracted answer, or an error message
                  if processing failed.
        """
        try:
            if "image" not in item or not item["image"]:
                raise ValueError(
                    f"Missing 'image' field or image path is empty for item ID: {item.get('id')}"
                )
            img_b64, mime = self.encode_image(item["image"])
            content_payload = self._build_prompt(item, img_b64, mime)

            time.sleep(self.request_delay + random.uniform(0, 0.4))  # Small jitter
            api_response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": content_payload}],
            )

            response_text = "".join(
                c.text
                for c in api_response.content
                if hasattr(c, "text") and c.type == "text"
            )
            reasoning_match = REASON_REGEX.search(response_text)
            answer_match = ANSWER_REGEX.search(response_text)

            return {
                "id": item["id"],
                "correct_answer": item.get("correct_answer", ""),
                "reasoning": (
                    reasoning_match.group(1).strip() if reasoning_match else ""
                ),
                "answer": answer_match.group(1).strip() if answer_match else "",
                "raw_response": response_text,  # Optional: for debugging
            }
        except Exception as e:
            if "rate_limit_error" in str(e):
                raise
            return {"id": item.get("id", "UnknownID"), "error": str(e)}

    def _load(self, path: str):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                for rec in json.load(f):
                    if rec.get("id") is not None:
                        self.results[rec["id"]] = rec
            print(f"Resumed from {len(self.results)} cached items in '{path}'")
        except Exception as e:
            print(f"⚠  Failed to load previous results: {e}")

    def process_batch(self, data: list, save_interval: int, outfile: str):
        """
        Processes a list of puzzle items concurrently.

        Results are saved periodically to the specified output file.
        Items already processed (based on ID in internal results) are skipped.

        Args:
            data (list): A list of puzzle items (dictionaries) to process.
            save_interval (int): How often (in terms of number of processed items)
                                 to save intermediate results.
            outfile (str): Path to the output JSON file for saving results.
        """
        self._load(outfile)
        items_to_process = [d for d in data if d.get("id") not in self.results]
        if not items_to_process:
            print("No new items to process.")
            if (
                self.results
            ):  # Save if there were pre-loaded results that haven't been saved
                self._save(outfile)
            return

        print(f"Processing {len(items_to_process)} new items.")

        def worker(puzzle_item_data):
            # This function is called by the thread pool executor.
            result = self.analyze_puzzle(puzzle_item_data)
            need_save = False
            current_id = puzzle_item_data.get("id")
            if current_id is not None:
                with self.lock:
                    self.results[current_id] = result
                    need_save = len(self.results) % save_interval == 0
            else:
                print(f"Warning: Item without ID: {puzzle_item_data}")
            if need_save:
                self._save(outfile)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Using list to ensure all tasks are submitted and tqdm can track completion
            list(
                tqdm(
                    executor.map(worker, items_to_process),
                    total=len(items_to_process),
                    desc="Processing puzzles",
                )
            )

        self._save(outfile)  # Final save after batch completion

    def _save(self, path: str):
        """
        Saves the currently stored results (self.results) to a JSON file.
        The results are sorted by item ID before saving.

        Args:
            path (str): The path to the file where results should be saved.
        """
        with self.lock:
            # Filter out potential None ID items before sorting, though IDs should be present
            valid_results = [
                res for res in self.results.values() if res.get("id") is not None
            ]
            sorted_results = sorted(valid_results, key=lambda x: x["id"])
        try:
            # Ensure the output directory exists
            output_dir = Path(path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(sorted_results, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(sorted_results)} items -> {path}")
        except IOError as e:
            print(f"Error: Could not save results to {path}. Reason: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during save: {e}")


def main():
    """
    Command-line interface for the LogicPuzzleSolver.

    Parses arguments, loads puzzle data from an input JSON file,
    initializes the solver, and processes the puzzles.
    The API key can be provided via argument or environment variable.
    """
    parser = argparse.ArgumentParser(
        description="Solves puzzles using an AI model, Jinja2 templates, and image processing."
    )
    parser.add_argument(
        "--input",
        default="./data/step1/1.3_raw_english.json",
        help="Path to the input JSON file containing puzzle data.",
    )
    parser.add_argument(
        "--output",
        default="./data/step1/1.4_verify.json",
        help="Path to the output JSON file for results.",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="Anthropic API key. If not provided, attempts to use ANTHROPIC_API_KEY environment variable.",
    )
    parser.add_argument(
        "--template_path",
        default=TEMPLATE_PATH,
        help=f"Path to the Jinja2 prompt template file (default: {TEMPLATE_PATH}).",
    )
    parser.add_argument(
        "--model_name",
        default=MODEL_NAME,
        help=f"Name of the Claude model to use (default: {MODEL_NAME}).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of concurrent worker threads (default: 4).",
    )
    parser.add_argument(
        "--request_delay",
        type=float,
        default=2.0,
        help="Base delay in seconds between API requests (default: 1.0).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=20000,
        help="Maximum tokens for the API response (default: 20000).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for API response generation (default: 1.0).",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="Number of items to process before saving intermediate results.",
    )

    args = parser.parse_args()

    final_api_key = args.api_key or Anthropic_API_KEY
    if not final_api_key:
        print(
            "Error: Anthropic API key must be provided via --api_key argument or ANTHROPIC_API_KEY environment variable."
        )
        sys.exit(1)

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(
            f"Error: Could not decode JSON from input file '{args.input}'. Details: {e}"
        )
        sys.exit(1)
    except Exception as e:
        print(
            f"An unexpected error occurred while loading input file '{args.input}': {e}"
        )
        sys.exit(1)

    if not isinstance(input_data, list):
        if isinstance(input_data, dict):
            input_data = [input_data]
        else:
            print(
                f"Error: Input data from '{args.input}' must be a JSON list of puzzle objects or a single puzzle object."
            )
            sys.exit(1)
    if not input_data:
        print("Input data is empty. Nothing to process.")
        sys.exit(0)

    json_dir = Path(args.input).parent
    for item in input_data:
        img_raw = item.get("image", "")
        if img_raw and not os.path.isabs(img_raw):
            item["image"] = str(json_dir / img_raw)

    try:
        solver = LogicPuzzleSolver(
            api_key=final_api_key,
            template_path=args.template_path,
            model=args.model_name,
            max_workers=args.max_workers,
            request_delay=args.request_delay,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        solver.process_batch(input_data, args.save_interval, args.output)
        print("Processing complete.")
    except (
        FileNotFoundError
    ) as e:  # Specifically for template file not found in __init__
        print(f"Error initializing solver: {e}")
        sys.exit(1)
    except ValueError as e:  # Specifically for API key missing in __init__
        print(f"Error initializing solver: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
