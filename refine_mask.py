"""Prompt-graph helpers of the masked refine (`refine_face_mode: "mask"`).

Kept apart from handler.py, which imports torch and exits without a GPU, so
that the graph surgery can be tested (and probed against ComfyUI) anywhere.

The masked refine runs as two successive ComfyUI prompts:
  A. the refine workflow pruned to its stage 1 (Qwen), whose only output is
     the stage-1 image (node 61);
  B. workflow/lustify_refine_masked.json: the Lustify pass over that image,
     denoising each pixel in proportion to a mask the handler builds from the
     faces found on the stage-1 image (face_preserve.build_face_denoise_mask).
"""

import json
import os

_WORKFLOW_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow")
MASKED_WORKFLOW = "lustify_refine_masked.json"

# Lustify stage of the refine workflow (200-206) and its SaveImage (60): what
# prompt A drops so that the stage-1 image is its only output.
_STAGE2_NODES = ("200", "201", "202", "203", "204", "205", "206", "60")

# Injection points of prompt B.
_NODE_STAGE1_LOAD = "207"   # LoadImage: the stage-1 image
_NODE_MASK_LOAD = "208"     # LoadImageMask: the denoise mask


def prune_to_stage1(prompt):
    """Drop the Lustify stage from the refine prompt, in place: the stage-1
    SaveImage (61) is left as its only output. Returns the prompt."""
    for node_id in _STAGE2_NODES:
        prompt.pop(node_id, None)
    # A leftover reference would make ComfyUI reject the whole graph.
    for node_id, node in prompt.items():
        for name, value in node.get("inputs", {}).items():
            if isinstance(value, list) and value and str(value[0]) in _STAGE2_NODES:
                raise RuntimeError(
                    f"node {node_id} input {name} still points at the pruned "
                    f"stage-2 node {value[0]}")
    return prompt


def apply_refine_overrides(prompt, job_input):
    """Job overrides of the Lustify stage (203 prompt, 204 negative, 205
    denoise). Shared by the paste refine, where the stage lives in the
    combined workflow, and by prompt B."""
    if job_input.get("refine_prompt") and "203" in prompt:
        prompt["203"]["inputs"]["text"] = job_input["refine_prompt"]
    if job_input.get("refine_negative") and "204" in prompt:
        prompt["204"]["inputs"]["text"] = job_input["refine_negative"]
    if "205" in prompt:
        try:
            prompt["205"]["inputs"]["denoise"] = float(
                job_input.get("refine_denoise", prompt["205"]["inputs"]["denoise"]))
        except (TypeError, ValueError):
            pass


def build_masked_refine_prompt(stage1_path, mask_path, job_input):
    """Prompt B for one job: the masked Lustify pass over `stage1_path`,
    denoised per pixel by `mask_path`. Node ids 200-206 and 60 are those of
    the combined workflow, so the final image is still read at node 60."""
    with open(os.path.join(_WORKFLOW_BASE, MASKED_WORKFLOW), "r") as f:
        prompt = json.load(f)
    apply_refine_overrides(prompt, job_input)
    prompt[_NODE_STAGE1_LOAD]["inputs"]["image"] = stage1_path
    prompt[_NODE_MASK_LOAD]["inputs"]["image"] = mask_path
    return prompt


def refine_face_params(job_input):
    """Validated `refine_face_*` job parameters: (mode, level, feather,
    min_px). An unknown mode or an unreadable number falls back to the
    default, a number out of range is clamped."""
    from face_preserve import (REFINE_FACE_FEATHER, REFINE_FACE_LEVEL,
                               REFINE_FACE_MIN_PX)

    def num(key, default, lo, hi):
        try:
            v = float(job_input.get(key, default))
        except (TypeError, ValueError):
            return default
        if v != v:   # NaN
            return default
        return min(max(v, lo), hi)

    mode = job_input.get("refine_face_mode", "paste")
    if mode not in ("paste", "mask"):
        mode = "paste"
    return (mode,
            num("refine_face_level", REFINE_FACE_LEVEL, 0.0, 1.0),
            num("refine_face_feather", REFINE_FACE_FEATHER, 0.05, 1.0),
            int(num("refine_face_min_px", REFINE_FACE_MIN_PX, 0, 512)))
