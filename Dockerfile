# Use specific version of nvidia cuda image
FROM wlsdml1114/multitalk-base:1.7 as runtime

# wget 설치 (URL 다운로드를 위해)
RUN apt-get update && apt-get install -y wget unzip && rm -rf /var/lib/apt/lists/*

RUN pip install -U "huggingface_hub[hf_transfer]"
RUN pip install runpod websocket-client librosa

# Set working directory
WORKDIR /

# Pinned, deliberately. These used to be bare clones of master, which made the
# image a function of the day it was built rather than of this repository. The
# v0.1.6 image (2026-06-02) and the v0.1.8 image (2026-07-30) therefore carried
# ComfyUI revisions ~100+ commits apart, and v0.1.8 came back with every job
# failing at `prompt_outputs_failed_validation` on an unchanged workflow. Bump
# these SHAs on purpose, in their own release, so an upstream break is
# attributable instead of arriving with an unrelated change.
ARG COMFYUI_SHA=33799c4a2ee286b5b6b8aac3c45c43245641fb47
ARG COMFYUI_MANAGER_SHA=2d373448bea60ed793a0fe39ec6ddda94f8cfde3
ARG KJNODES_SHA=a41e0d85b822ddc4bbd1256417aa2d58d39e94ce

RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd ComfyUI && \
    git checkout ${COMFYUI_SHA} && \
    pip install --no-cache-dir -r requirements.txt

RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    git checkout ${COMFYUI_MANAGER_SHA} && \
    pip install --no-cache-dir -r requirements.txt

# Disable ComfyUI-Manager network fetches at startup.
# Without this, every cold boot spends ~4min fetching the ComfyRegistry
# (139 pages from api.comfy.org) + 5 raw JSON caches from GitHub, which
# competes with the first job for network bandwidth.
# Writing to both legacy and new manager config paths for compatibility.
RUN mkdir -p /ComfyUI/user/default/ComfyUI-Manager /ComfyUI/user/__manager && \
    printf '[default]\nnetwork_mode = offline\n' > /ComfyUI/user/default/ComfyUI-Manager/config.ini && \
    printf '[default]\nnetwork_mode = offline\n' > /ComfyUI/user/__manager/config.ini

RUN cd /ComfyUI/custom_nodes/ && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && \
    git checkout ${KJNODES_SHA} && \
    pip install --no-cache-dir -r requirements.txt

# --- Models are NOT baked into the image any more. ---
# ~40 GB of weights (Qwen edit fp8, Qwen2.5-VL encoder, Lustify, VAEs,
# inswapper, GFPGAN, buffalo_l) used to be wget'ed here, which made a cold
# build ~30 min — RunPod Hub's hard limit — and every worker pull 46 GB.
# /ensure_models.py now fetches them at container start onto the endpoint's
# network volume (once, lock-protected) or the container disk when no volume
# is mounted, and symlinks them into the paths below. The manifest with URLs
# and exact sizes lives in that script.
RUN mkdir -p /ComfyUI/models/diffusion_models /ComfyUI/models/text_encoders \
             /ComfyUI/models/vae /ComfyUI/models/loras /ComfyUI/models/checkpoints \
             /ComfyUI/models/insightface /root/.insightface/models

# --- Face preservation (preserve_face): InsightFace inswapper — the engine that
# --- ReActor wraps, used directly so explicit/NSFW images are NOT blocked by
# --- ReActor's built-in NSFW filter. GFPGAN restore runs as ONNX on the same
# --- onnxruntime (the .pth release would drag in basicsr, which no longer
# --- imports against current torchvision). CUDA provider with CPU fallback.
RUN pip install --no-cache-dir insightface onnxruntime-gpu opencv-python-headless

COPY . .
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
