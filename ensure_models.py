#!/usr/bin/env python3
"""Ensure every weight the workflows need is present, then expose each one
where ComfyUI and insightface expect it.

Weights used to be baked into the image (~40 GB of Hugging Face downloads at
build time). A cold build therefore took ~30 min — exactly RunPod Hub's limit
— so any layer-cache miss failed the release, and every worker cold start
pulled a 46 GB image. The image now ships code only; weights live on the
endpoint's network volume and are fetched once.

Store selection:
  /runpod-volume/models  when a network volume is mounted: shared by all
                         workers, persists across releases and cold starts.
  /ComfyUI/models        otherwise (Hub test pods, local runs): downloaded
                         onto the container disk on every start.

Concurrency: RunPod warns that concurrent writes to a volume corrupt data.
The first worker to mkdir the lock directory downloads; the others wait for
the files to appear. Each file is fetched to a `.part` sibling and renamed
into place only once its size matches the manifest, so a reader never sees
a truncated weight, and a downloader that dies mid-way (stale heartbeat) is
simply replaced by the next waiter.
"""

import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile

COMFY_MODELS = "/ComfyUI/models"
VOLUME_STORE = "/runpod-volume/models"
INSIGHTFACE_HOME = "/root/.insightface/models"

# (relative path in the store, URL, exact size in bytes)
# Sizes are checked after every download: a 401/403 page from HF is a few KB
# and would otherwise be renamed into place as a "model".
MANIFEST = [
    ("diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
     "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
     20533762817),
    ("loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
     "https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
     849608296),
    ("text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
     "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
     9384670680),
    ("vae/qwen_image_vae.safetensors",
     "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
     253806246),
    # Filename MUST match workflow node 200 (stage-2 refine).
    ("vae/sdxl_vae_fp16fix.safetensors",
     "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors",
     334641162),
    ("checkpoints/lustifySDXLNSFW_endgame.safetensors",
     "https://huggingface.co/xxxpo13/LUSTIFY_SDXL/resolve/main/lustifySDXLNSFW_endgame.safetensors",
     6938043264),
    ("insightface/inswapper_128.onnx",
     "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx",
     554253681),
    ("insightface/gfpgan_1.4.onnx",
     "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx",
     340299087),
]

# buffalo_l ships as a zip that insightface expects unpacked under
# ~/.insightface/models/buffalo_l/. Handled apart from the plain files.
BUFFALO_ZIP = ("insightface_home/buffalo_l.zip",
               "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
               288621354)
BUFFALO_DIR = "insightface_home/buffalo_l"

HEARTBEAT_EVERY = 20      # seconds
STALE_AFTER = 300         # a downloader silent this long is presumed dead
WAIT_POLL = 5


def log(msg):
    print(f"[ensure_models] {msg}", flush=True)


def pick_store():
    if os.path.isdir("/runpod-volume"):
        try:
            os.makedirs(VOLUME_STORE, exist_ok=True)
            probe = os.path.join(VOLUME_STORE, ".write_probe")
            with open(probe, "w") as fh:
                fh.write("ok")
            os.remove(probe)
            return VOLUME_STORE
        except OSError as e:
            log(f"network volume present but not writable ({e}), using container disk")
    return COMFY_MODELS


def complete(store, rel, size):
    p = os.path.join(store, rel)
    return os.path.isfile(p) and os.path.getsize(p) == size


def buffalo_complete(store):
    return os.path.isfile(os.path.join(store, BUFFALO_DIR, ".ok"))


def missing(store):
    todo = [m for m in MANIFEST if not complete(store, *m[0::2])]
    return todo, not buffalo_complete(store)


def download(store, rel, url, size):
    dest = os.path.join(store, rel)
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(1, 4):
        log(f"downloading {rel} ({size / 1e9:.2f} GB, attempt {attempt})")
        t0 = time.time()
        # -c resumes a .part left by a previous, dead downloader.
        rc = subprocess.call(["wget", "-q", "-c", "-O", part, url])
        got = os.path.getsize(part) if os.path.exists(part) else -1
        if rc == 0 and got == size:
            os.replace(part, dest)
            log(f"done {rel} in {time.time() - t0:.0f}s")
            return
        log(f"failed {rel}: wget rc={rc}, size {got} != {size}; retrying")
        if os.path.exists(part):
            os.remove(part)
        time.sleep(5)
    raise RuntimeError(f"could not download {rel}")


def install_buffalo(store):
    rel, url, size = BUFFALO_ZIP
    if not complete(store, rel, size):
        download(store, rel, url, size)
    final = os.path.join(store, BUFFALO_DIR)
    tmp = final + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(os.path.join(store, rel)) as zf:
        zf.extractall(tmp)
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    with open(os.path.join(final, ".ok"), "w") as fh:
        fh.write("ok")
    os.remove(os.path.join(store, rel))
    log("buffalo_l unpacked")


class Heartbeat(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path, self.stop = path, threading.Event()

    def run(self):
        while not self.stop.is_set():
            with open(self.path, "w") as fh:
                fh.write(str(time.time()))
            self.stop.wait(HEARTBEAT_EVERY)


def lock_is_stale(lock_dir):
    hb = os.path.join(lock_dir, "heartbeat")
    try:
        return time.time() - os.path.getmtime(hb) > STALE_AFTER
    except OSError:
        # Lock dir without a heartbeat yet: give its owner a grace period
        # measured from the dir's own mtime.
        try:
            return time.time() - os.path.getmtime(lock_dir) > STALE_AFTER
        except OSError:
            return True  # vanished: nothing to wait for


def populate(store):
    """Download whatever is missing, holding the store lock. Returns once
    every manifest entry is complete, whether we or another worker did it."""
    os.makedirs(store, exist_ok=True)
    lock_dir = os.path.join(store, ".lock")
    while True:
        todo, need_buffalo = missing(store)
        if not todo and not need_buffalo:
            return
        try:
            os.mkdir(lock_dir)
        except FileExistsError:
            if lock_is_stale(lock_dir):
                log("stale lock from a dead downloader, taking over")
                shutil.rmtree(lock_dir, ignore_errors=True)
                continue
            log(f"another worker is downloading ({len(todo)} file(s) left), waiting")
            time.sleep(WAIT_POLL)
            continue
        hb = Heartbeat(os.path.join(lock_dir, "heartbeat"))
        hb.start()
        try:
            for rel, url, size in todo:
                if not complete(store, rel, size):
                    download(store, rel, url, size)
            if need_buffalo:
                install_buffalo(store)
        finally:
            hb.stop.set()
            shutil.rmtree(lock_dir, ignore_errors=True)


def link(target, path):
    """Point `path` at `target`, replacing a stale/broken link or empty dir."""
    if os.path.islink(path):
        if os.readlink(path) == target:
            return
        os.remove(path)
    elif os.path.isdir(path) and not os.listdir(path):
        os.rmdir(path)
    elif os.path.exists(path):
        return  # a real file already there (e.g. baked by an older image)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.symlink(target, path)


def expose(store):
    if store != COMFY_MODELS:
        for rel, _, _ in MANIFEST:
            link(os.path.join(store, rel), os.path.join(COMFY_MODELS, rel))
    link(os.path.join(store, BUFFALO_DIR), os.path.join(INSIGHTFACE_HOME, "buffalo_l"))


def main():
    store = pick_store()
    log(f"store: {store}" + ("" if store == VOLUME_STORE else " (no network volume)"))
    t0 = time.time()
    populate(store)
    expose(store)
    todo, need_buffalo = missing(store)
    if todo or need_buffalo:
        log(f"ERROR: still missing {[m[0] for m in todo]} buffalo={need_buffalo}")
        sys.exit(1)
    log(f"all {len(MANIFEST) + 1} model(s) ready in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
