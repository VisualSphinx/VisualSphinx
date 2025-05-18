"""
Generate and test correct/incorrect scripts for rules using the XAI API,
parallel execution, retries, and checkpointing.
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from itertools import cycle
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm
from openai import OpenAI
from api_config import XAI_API_KEYS


def parse_args():
    """
    Define and parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate and test scripts via XAI API."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./data/step2/2.8_rules_filtered.json",
        help="Path to input JSON file of rules.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./data/step3/3.1_style_1.jsonl",
        help="Path to output JSONL file for results.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://api.x.ai/v1",
        help="Base URL for the XAI API.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="grok-3-mini-beta",
        help="Model name to use for API calls.",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="high",
        help="Reasoning effort level for API calls.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=200,
        help="Maximum number of parallel workers.",
    )
    parser.add_argument(
        "--max_retries", type=int, default=8, help="Max retries for API call attempts."
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=4,
        help="Max attempts per rule before giving up.",
    )
    parser.add_argument(
        "--base_delay_s",
        type=float,
        default=1.0,
        help="Base delay seconds for exponential backoff.",
    )
    parser.add_argument(
        "--script_timeout",
        type=int,
        default=15,
        help="Timeout seconds for running generated scripts.",
    )
    parser.add_argument(
        "--wait_results", type=int, default=30, help="Seconds to wait for test results."
    )
    parser.add_argument(
        "--request_soft_timeout",
        type=int,
        default=1200,
        help="Soft timeout for API responses.",
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default="./Prompts",
        help="Directory containing prompt templates.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="step3.1_generation_style_1.md",
        help="Prompt template filename.",
    )
    parser.add_argument(
        "--folder_root",
        type=str,
        default="./data/step3/3.1_all_scripts_style_1",
        help="Root folder to write scripts and tests.",
    )
    return parser.parse_args()


def build_prompt(template, rule_content):
    """
    Render the prompt template with the given rule content.
    """
    return template.render(RULES="\n".join(rule_content))


def parse_scripts(text):
    """
    Extract correct_script and incorrect_script sections from the API response.
    """
    pat_corr = re.compile(r"<correct_script>(.*?)</correct_script>", re.DOTALL)
    pat_inc = re.compile(r"<incorrect_script>(.*?)</incorrect_script>", re.DOTALL)
    m1, m2 = pat_corr.search(text), pat_inc.search(text)
    if not (m1 and m2):
        raise ValueError("Missing script tags.")
    return {
        "correct_script": m1.group(1).strip(),
        "incorrect_script": m2.group(1).strip(),
    }


def write_folder(folder_root, item, attempt, scripts):
    """
    Create a folder for the item and attempt, write data and scripts there.
    """
    folder = Path(folder_root) / f"{item['id']}_{attempt}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "data.json").write_text(json.dumps(item, ensure_ascii=False, indent=2))
    (folder / "correct_script.py").write_text(scripts["correct_script"])
    (folder / "incorrect_script.py").write_text(scripts["incorrect_script"])
    bash = folder / "run_test.sh"
    bash.write_text(
        f"""#!/usr/bin/env bash
TIME_LIMIT={SCRIPT_TIMEOUT}
RESULT_FILE=\"test_result.txt\"

run_py() {{
  timeout --kill-after=2s $TIME_LIMIT python \"$1\" &>/dev/null
  return $?
}}

run_py correct_script.py
SC=$?
run_py incorrect_script.py
SI=$?

if [[ $SC -eq 0 && $SI -eq 0 ]]; then
  echo success >"$RESULT_FILE"
else
  if [[ $SC -eq 124 || $SI -eq 124 ]]; then
    echo fail - timeout >"$RESULT_FILE"
  else
    echo fail >"$RESULT_FILE"
  fi
fi
"""
    )
    bash.chmod(0o755)
    return folder


def append_jsonl(path, items):
    """
    Append JSON objects to a JSONL file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_bash_and_get_status(folder, wait_results, script_timeout):
    """
    Execute the bash script and retrieve its status result.
    """
    proc = subprocess.Popen(["bash", "run_test.sh"], cwd=folder)
    start = time.time()
    while proc.poll() is None and time.time() - start < wait_results:
        time.sleep(0.1)
    if proc.poll() is None:
        proc.kill()
        proc.wait()
    res_file = folder / "test_result.txt"
    end = time.time() + 2
    while not res_file.exists() and time.time() < end:
        time.sleep(0.1)
    return res_file.read_text().strip() if res_file.exists() else "fail - no result"


def handle_one(item, attempt, args, template):
    """
    Process one rule: call API, parse scripts, write files, run tests.
    Returns scripts dict on success, "soft_timeout" to skip, None to retry.
    """
    delay = args.base_delay_s
    for retry in range(1, args.max_retries + 1):
        try:
            with key_lock:
                api_key = next(key_cycle)
            client = OpenAI(api_key=api_key, base_url=args.base_url)
            with cf.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(
                    client.chat.completions.create,
                    model=args.model_name,
                    reasoning_effort=args.reasoning_effort,
                    messages=[
                        {
                            "role": "user",
                            "content": build_prompt(template, item["rule_content"]),
                        }
                    ],
                    temperature=0.7,
                )
                resp = fut.result(timeout=args.request_soft_timeout)
            scripts = parse_scripts(resp.choices[0].message.content)
            break
        except cf.TimeoutError:
            tqdm.write(f"[ID {item['id']}] attempt {attempt} API soft timeout, skip ID")
            return "soft_timeout"
        except Exception as e:
            tqdm.write(
                f"[ID {item['id']}] API error: {e} (retry {retry}/{args.max_retries})"
            )
            if retry == args.max_retries:
                return None
            time.sleep(delay)
            delay *= 2
    else:
        return None

    folder = write_folder(args.folder_root, item, attempt, scripts)
    status = run_bash_and_get_status(folder, args.wait_results, args.script_timeout)
    if status == "success":
        tqdm.write(f"[ID {item['id']}] attempt {attempt} bash success")
        return scripts
    tqdm.write(f"[ID {item['id']}] attempt {attempt} bash {status}, retry if allowed")
    return None


def main():
    args = parse_args()
    global SCRIPT_TIMEOUT, WAIT_RESULTS, REQUEST_SOFT_TIMEOUT, key_cycle, key_lock
    SCRIPT_TIMEOUT = args.script_timeout
    WAIT_RESULTS = args.wait_results
    REQUEST_SOFT_TIMEOUT = args.request_soft_timeout
    key_cycle, key_lock = cycle(XAI_API_KEYS), threading.Lock()

    env = Environment(loader=FileSystemLoader(args.prompt_dir))
    template = env.get_template(args.prompt_file)

    items = json.loads(Path(args.input_file).read_text())
    done_ids = (
        {json.loads(l)["id"] for l in Path(args.output_file).read_text().splitlines()}
        if Path(args.output_file).exists()
        else set()
    )
    pend = deque([it for it in items if it["id"] not in done_ids])
    total = len(pend)
    overall = tqdm(total=total, desc="Overall")
    attempt_map = {it["id"]: 1 for it in pend}

    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        running = {}
        while pend and len(running) < args.max_workers:
            it = pend.popleft()
            fut = pool.submit(handle_one, it, attempt_map[it["id"]], args, template)
            running[fut] = it

        while running:
            done, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                it = running.pop(fut)
                res = fut.result()
                if isinstance(res, dict):
                    it.update(res)
                    append_jsonl(args.output_file, [it])
                    overall.update(1)
                elif res == "soft_timeout":
                    overall.update(1)
                else:
                    attempt_map[it["id"]] += 1
                    if attempt_map[it["id"]] <= args.max_attempts:
                        pend.append(it)
                if pend:
                    nxt = pend.popleft()
                    f2 = pool.submit(
                        handle_one, nxt, attempt_map[nxt["id"]], args, template
                    )
                    running[f2] = nxt
    overall.close()
    print("\nAll done.")


if __name__ == "__main__":
    main()
