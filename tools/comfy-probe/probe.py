"""Boot ComfyUI, POST each workflow to /prompt, print whether it validates.

We only care about the validation verdict, so the job never has to execute: a
400 from /prompt carries the node errors in its body, which is exactly what the
production handler was throwing away.

Probed, from the repo mounted at /repo:
  - every workflow/*.json, as is (LoadImage pointed at a real PNG);
  - the graphs of the masked refine, built by refine_mask.py the way the
    handler builds them: prompt A (the refine workflow pruned to its Qwen
    stage) and prompt B (the Lustify pass with its denoise mask).
"""
import glob
import json
import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib

PORT = 8188
BASE = f"http://127.0.0.1:{PORT}"
REPO = os.environ.get("REPO", "/repo")
INPUT = "/ComfyUI/input"
TASK = "/tmp/probe-task"


def png(width, height, gray=None):
    """Minimal valid PNG: RGB, or 8-bit grayscale when `gray` is given."""
    def chunk(tag, data):
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))
    ctype, px = (0, bytes([gray])) if gray is not None else (2, b"\x80\x80\x80")
    raw = b"".join(b"\x00" + px * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, ctype, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def wait_ready(timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{BASE}/system_stats", timeout=3).read()
            return True
        except Exception:
            time.sleep(2)
    return False


def graphs():
    """(name, prompt) for everything to probe."""
    sys.path.insert(0, REPO)
    import refine_mask

    # A real file for every LoadImage, so a missing input cannot masquerade as
    # the failure we are hunting. The stage-1 image and the mask sit outside
    # ComfyUI's input directory and are given by absolute path, as the handler
    # gives them.
    open(f"{INPUT}/probe.png", "wb").write(png(64, 64))
    os.makedirs(TASK, exist_ok=True)
    open(f"{TASK}/stage1.png", "wb").write(png(64, 64))
    open(f"{TASK}/face_mask.png", "wb").write(png(64, 64, gray=90))

    for path in sorted(glob.glob(f"{REPO}/workflow/*.json")):
        wf = json.load(open(path))
        for node in wf.values():
            if node.get("class_type") == "LoadImage":
                node["inputs"]["image"] = "probe.png"
            elif node.get("class_type") == "LoadImageMask":
                node["inputs"]["image"] = f"{TASK}/face_mask.png"
        yield os.path.basename(path), wf

    refine = json.load(open(f"{REPO}/workflow/qwen2511_lustify_refine_1image.json"))
    refine["78"]["inputs"]["image"] = "probe.png"
    yield "masked refine, prompt A (pruned)", refine_mask.prune_to_stage1(refine)
    yield "masked refine, prompt B", refine_mask.build_masked_refine_prompt(
        f"{TASK}/stage1.png", f"{TASK}/face_mask.png",
        {"refine_prompt": "probe", "refine_negative": "probe", "refine_denoise": 0.5})


def main():
    server = subprocess.Popen(
        [sys.executable, "main.py", "--cpu", "--port", str(PORT),
         "--listen", "127.0.0.1", "--disable-auto-launch"],
        cwd="/ComfyUI", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if not wait_ready():
        print("SERVER DID NOT START", flush=True)
        out = server.stdout.read() if server.stdout else ""
        print(out[-4000:], flush=True)
        return 2
    print(f"ComfyUI ready on {BASE}\n", flush=True)

    failures = 0
    for name, wf in graphs():
        body = json.dumps({"prompt": wf, "client_id": "probe"}).encode()
        req = urllib.request.Request(f"{BASE}/prompt", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
            print(f"[OK ] {name}  prompt_id={resp.get('prompt_id')}", flush=True)
        except urllib.error.HTTPError as e:
            failures += 1
            detail = e.read().decode("utf-8", "replace")
            print(f"[400] {name}  HTTP {e.code}", flush=True)
            print(json.dumps(json.loads(detail), indent=2)[:3000]
                  if detail.strip().startswith("{") else detail[:3000], flush=True)
        except Exception as e:
            failures += 1
            print(f"[ERR] {name}  {type(e).__name__}: {e}", flush=True)

    server.terminate()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
