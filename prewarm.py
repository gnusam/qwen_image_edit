import json
import os
import shutil
import sys
import time
import urllib.request
import uuid

SERVER = "http://127.0.0.1:8188"
WORKFLOW = "/workflow/qwen_image_edit_1_1image.json"
INPUT_SRC = "/examples/input/test_input.png"
INPUT_DST = "/prewarm_input.png"


def wait_for_comfy(timeout_s: int = 180) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{SERVER}/", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def queue_prompt(prompt: dict) -> str:
    body = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(
        f"{SERVER}/prompt",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return resp["prompt_id"]


def wait_for_prompt(prompt_id: str, timeout_s: int = 600) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            h = json.loads(
                urllib.request.urlopen(
                    f"{SERVER}/history/{prompt_id}", timeout=5
                ).read()
            )
            if prompt_id in h and h[prompt_id].get("status", {}).get("completed"):
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> int:
    start = time.time()
    print("[prewarm] waiting for ComfyUI...", flush=True)
    if not wait_for_comfy():
        print("[prewarm] ComfyUI did not come up — skipping prewarm", flush=True)
        return 0

    try:
        shutil.copy(INPUT_SRC, INPUT_DST)
    except Exception as e:
        print(f"[prewarm] failed to stage input ({e}) — skipping", flush=True)
        return 0

    with open(WORKFLOW) as f:
        prompt = json.load(f)
    prompt["78"]["inputs"]["image"] = INPUT_DST
    if "111" in prompt:
        prompt["111"]["inputs"]["prompt"] = "warmup"

    try:
        pid = queue_prompt(prompt)
    except Exception as e:
        print(f"[prewarm] queue failed ({e}) — skipping", flush=True)
        return 0

    print(f"[prewarm] queued {pid} — running dummy inference", flush=True)
    ok = wait_for_prompt(pid)
    elapsed = int(time.time() - start)
    if ok:
        print(f"[prewarm] done in {elapsed}s — models now hot in VRAM", flush=True)
    else:
        print(f"[prewarm] timed out after {elapsed}s — continuing anyway", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
