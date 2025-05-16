import concurrent.futures as cf
import concurrent
import json, os, re, subprocess, sys, threading, time
from collections import deque
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Union
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
from api_config import XAI_API_KEYS

BASE_URL, MODEL_NAME, REASONING_EFFORT = (
    "https://api.x.ai/v1",
    "grok-3-mini-beta",
    "high",
)
MAX_WORKERS = 2000
MAX_RETRIES, MAX_ATTEMPTS, BASE_DELAY_S = 8, 4, 1
SCRIPT_TIMEOUT, WAIT_RESULTS, REQUEST_SOFT_TIMEOUT = 15, 30, 1200
PROMPT_DIR, PROMPT_FILE = "./Prompts", "step3.1_generation_style_1.md"
FOLDER_ROOT = Path("./data/step3/all_scripts_style_1")

BASH_TEMPLATE = f"""#!/usr/bin/env bash
# Auto‑generated
TIME_LIMIT={SCRIPT_TIMEOUT}
RESULT_FILE=\"test_result.txt\"

run_py() {{
  local file=\"$1\"
  timeout --kill-after=2s $TIME_LIMIT python \"$file\" &>/dev/null
  return $?
}}

run_py correct_script.py
SC=$?
run_py incorrect_script.py
SI=$?

if [[ $SC -eq 0 && $SI -eq 0 ]]; then
  echo \"success\" >\"$RESULT_FILE\"
else
  if [[ $SC -eq 124 || $SI -eq 124 ]]; then
    echo \"fail - timeout\" >\"$RESULT_FILE\"
  else
    echo \"fail\" >\"$RESULT_FILE\"
  fi
fi
"""

# globals
env = Environment(loader=FileSystemLoader(PROMPT_DIR))
template = env.get_template(PROMPT_FILE)
PAT_CORRECT = re.compile(r"<correct_script>(.*?)</correct_script>", re.DOTALL)
PAT_INCORRECT = re.compile(r"<incorrect_script>(.*?)</incorrect_script>", re.DOTALL)
key_cycle, key_lock = cycle(XAI_API_KEYS), threading.Lock()


# helper functions
def build_prompt(rule_content: List[str]) -> str:
    return template.render(RULES="\n".join(rule_content))


def parse_scripts(text: str) -> Dict[str, str]:
    m1, m2 = PAT_CORRECT.search(text), PAT_INCORRECT.search(text)
    if not (m1 and m2):
        raise ValueError("Missing script tags.")
    return {
        "correct_script": m1.group(1).strip(),
        "incorrect_script": m2.group(1).strip(),
    }


# write scripts and data into folder
def write_folder(item: Dict, attempt: int, scripts: Dict[str, str]) -> Path:
    folder = FOLDER_ROOT / f"{item['id']}_{attempt}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "data.json").write_text(json.dumps(item, ensure_ascii=False, indent=2))
    (folder / "correct_script.py").write_text(scripts["correct_script"])
    (folder / "incorrect_script.py").write_text(scripts["incorrect_script"])
    bash_path = folder / "run_test.sh"
    bash_path.write_text(BASH_TEMPLATE)
    bash_path.chmod(0o755)
    return folder


# append successful items to JSONL
def append_jsonl(path: Path, items: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# improved bash runner: poll subprocess to avoid full WAIT_RESULTS delay
def run_bash_and_get_status(folder: Path) -> str:
    proc = subprocess.Popen(
        ["bash", "run_test.sh"],
        cwd=folder,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    start_time = time.time()
    while proc.poll() is None and time.time() - start_time < WAIT_RESULTS:
        time.sleep(0.1)
    if proc.poll() is None:
        proc.kill()
        proc.wait()
    # wait up to 1s for result file
    res_file = folder / "test_result.txt"
    deadline = time.time() + 2
    while not res_file.exists() and time.time() < deadline:
        time.sleep(0.1)
    return res_file.read_text().strip() if res_file.exists() else "fail - no result"


# handle_one: bail entire ID on soft timeout
# returns Dict on success, 'soft_timeout' to skip ID, None to retry attempt
def handle_one(item: Dict, attempt: int) -> Union[Dict[str, str], str, None]:
    delay = BASE_DELAY_S
    for retry in range(1, MAX_RETRIES + 1):
        try:
            with key_lock:
                api_key = next(key_cycle)
            client = OpenAI(api_key=api_key, base_url=BASE_URL)
            # wrap create in timeout executor
            with cf.ThreadPoolExecutor(max_workers=1) as exec:
                fut = exec.submit(
                    client.chat.completions.create,
                    model=MODEL_NAME,
                    reasoning_effort=REASONING_EFFORT,
                    messages=[
                        {"role": "user", "content": build_prompt(item["rule_content"])}
                    ],
                    temperature=0.7,
                )
                resp = fut.result(timeout=REQUEST_SOFT_TIMEOUT)
            scripts = parse_scripts(resp.choices[0].message.content)
            break
        except concurrent.futures.TimeoutError:
            tqdm.write(f"[ID {item['id']}] attempt {attempt} API soft timeout, bail ID")
            return "soft_timeout"
        except Exception as e:
            tqdm.write(
                f"[ID {item['id']}] attempt {attempt} API error: {e} (retry {retry}/{MAX_RETRIES})"
            )
            if retry == MAX_RETRIES:
                return None
            time.sleep(delay)
            delay *= 2
    else:
        return None

    folder = write_folder(item, attempt, scripts)
    status = run_bash_and_get_status(folder)
    if status == "success":
        tqdm.write(f"[ID {item['id']}] attempt {attempt} bash success")
        return scripts
    tqdm.write(f"[ID {item['id']}] attempt {attempt} bash {status}, bail this attempt")
    return None


# main orchestration
def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py INPUT.json OUTPUT.jsonl")
        sys.exit(1)
    inp, out = Path(sys.argv[1]), Path(sys.argv[2])

    items = json.loads(inp.read_text())
    done_ids = (
        {json.loads(l)["id"] for l in out.read_text().splitlines()}
        if out.exists()
        else set()
    )
    pend = deque([obj for obj in items if obj["id"] not in done_ids])

    total = len(pend)
    overall = tqdm(total=total, desc="Overall", ncols=80)
    attempt_map = {obj["id"]: 1 for obj in pend}

    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        running = {}
        while pend and len(running) < MAX_WORKERS:
            it = pend.popleft()
            fut = pool.submit(handle_one, it, attempt_map[it["id"]])
            running[fut] = it

        while running:
            done, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                item = running.pop(fut)
                try:
                    result = fut.result()
                    if isinstance(result, dict):
                        item.update(result)
                        append_jsonl(out, [item])
                        overall.update(1)
                    elif result == "soft_timeout":
                        tqdm.write(
                            f"[ID {item['id']}] attempt {attempt_map[item['id']]} soft timeout, skipping ID"
                        )
                        overall.update(1)
                    else:
                        attempt_map[item["id"]] += 1
                        tqdm.write(
                            f"[ID {item['id']}] scheduling attempt {attempt_map[item['id']]} next"
                        )
                        if attempt_map[item["id"]] <= MAX_ATTEMPTS:
                            pend.append(item)
                        else:
                            tqdm.write(
                                f"[ID {item['id']}] reached max attempts ({MAX_ATTEMPTS}), give up"
                            )
                except Exception as e:
                    attempt_map[item["id"]] += 1
                    tqdm.write(
                        f"[ID {item['id']}] fatal at attempt {attempt_map[item['id']]}: {e}"
                    )
                    if attempt_map[item["id"]] <= MAX_ATTEMPTS:
                        pend.append(item)
                    else:
                        tqdm.write(
                            f"[ID {item['id']}] reached max attempts ({MAX_ATTEMPTS}), give up"
                        )

                if pend:
                    nxt = pend.popleft()
                    fut2 = pool.submit(handle_one, nxt, attempt_map[nxt["id"]])
                    running[fut2] = nxt

    overall.close()
    print("\nAll done.")


if __name__ == "__main__":
    main()
