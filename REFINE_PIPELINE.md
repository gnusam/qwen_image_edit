# Two-stage NSFW refine pipeline (Qwen-2511 → Lustify)

Optional second stage that runs **after** Qwen-Image-Edit-2511 stages the scene:
re-encodes Qwen's output into an SDXL latent and refines it img2img with a
**Lustify V5** (SDXL NSFW) pass at **denoise 0.55** — pushing explicit detail
where Qwen is weak, while preserving pose / scene / framing.

Validated sweet spot (POC): denoise **0.50–0.60**. <0.40 leaves clothing/watermark
residue; >0.70 breaks composition.

## Activation (request)

Add `refine: true` (alias `nsfw_refine`) to the job input. Stage 1 still receives
`prompt` / `image_url` / `seed` / `lora_url` as usual (they target the Qwen stage).

```json
{
  "input": {
    "prompt": "full body shot, same person, same pose, same background",
    "image_url": "https://.../source.jpg",
    "refine": true,
    "refine_prompt": "nude, fully naked, photorealistic, natural skin texture, same pose",
    "refine_negative": "clothing, lingerie, cartoon, deformed, watermark",
    "refine_denoise": 0.55
  }
}
```

`refine_prompt` / `refine_negative` / `refine_denoise` are optional overrides of the
stage-2 (node 203/204/205) defaults baked in `workflow/qwen2511_lustify_refine_1image.json`.
Only the **1-image** variant exists today; multi-image falls back to the normal workflow.

## Required models on the endpoint

The combined workflow needs, in addition to the Qwen models already baked:
- `/ComfyUI/models/vae/sdxl_vae_fp16fix.safetensors` — public (madebyollin/sdxl-vae-fp16-fix). **Mandatory** (the SDXL embedded fp16 VAE NaNs → black images).
- `/ComfyUI/models/checkpoints/lustifySDXLNSFW_endgame.safetensors` — Lustify ENDGAME.

Both are baked by the Dockerfile from **public HF mirrors** — no Civitai token / age-gate:
- VAE: `madebyollin/sdxl-vae-fp16-fix`
- Lustify ENDGAME single-file: `xxxpo13/LUSTIFY_SDXL` → `lustifySDXLNSFW_endgame.safetensors`

`docker build .` (or a RunPod Hub rebuild on push) bakes everything; no build args needed.
Alternatives in the same repo: `lustifySDXLNSFW_endgameDMD2.safetensors` (few-step, change node 205
to ~6 steps / cfg ~1.5), or `_v40` / `_v30`. APEX V8 / GGWP V7 are Civitai-only (model 573152,
versions 2808677 / 2155386 — need a free token + age-verified account).
The filename **must match** workflow node 200. For lighter images / faster cold-start, drop the two
Dockerfile `wget` lines and mount the files via a RunPod network volume instead.

## Graph shape

`78 LoadImage → 93 scale(1MP) → [Qwen stage: 88/110/111/3/8] → 202 VAEEncode(SDXL VAE)
→ 205 KSampler(Lustify, dpmpp_2m_sde/karras, 28 steps, cfg 5.5, denoise 0.55)
→ 206 VAEDecode → 60 SaveImage (final)`.

Handler injection points (78 image, 111 prompt, 3 seed, 190 user-LoRA) are unchanged,
so existing callers keep working; `refine` only swaps the workflow file.

## Bench candidates (next)

Swap node 200's checkpoint to A/B the refiner: Lustify V5 vs Pony V6 XL (POC baseline)
vs Juggernaut XL. Stage-1 NSFW Qwen LoRA (QIE-SRD / META4) can be stacked via `lora_url`.
