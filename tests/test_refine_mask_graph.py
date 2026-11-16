"""Tests des graphes ComfyUI du refine a masque (refine_face_mode "mask").

Le prompt A est le workflow refine sans son etage Lustify : sa seule sortie est
l'image d'etape 1. Le prompt B est workflow/lustify_refine_masked.json : la
passe Lustify sur cette image, debruitee par le masque. Ces tests tiennent la
forme des deux graphes (aucune reference pendante, DifferentialDiffusion cable
sur le KSampler, overrides appliques) ; que ComfyUI les accepte, c'est le
travail de tools/comfy-probe/.

    python -m pytest tests/test_refine_mask_graph.py -q
"""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import refine_mask as rm


def _refine():
    with open(os.path.join(ROOT, "workflow", "qwen2511_lustify_refine_1image.json")) as f:
        return json.load(f)


def _refs(prompt):
    for nid, node in prompt.items():
        for name, v in node["inputs"].items():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                yield nid, name, v[0]


def test_prompt_a_sans_etage_lustify():
    a = rm.prune_to_stage1(_refine())
    assert not [n for n in a if n.startswith("2") and len(n) == 3]
    assert "60" not in a
    assert "61" in a and "8" in a and "3" in a          # l'etape 1 reste


def test_prompt_a_une_seule_sortie_et_aucune_reference_pendante():
    a = rm.prune_to_stage1(_refine())
    outputs = [n for n, node in a.items() if node["class_type"] == "SaveImage"]
    assert outputs == ["61"]
    for nid, name, target in _refs(a):
        assert target in a, f"{nid}.{name} -> {target}"


def test_prompt_a_garde_les_points_dinjection_du_handler():
    a = rm.prune_to_stage1(_refine())
    for node in ("78", "111", "3", "89", "66"):
        assert node in a


def test_elagage_refuse_une_reference_restante():
    p = _refine()
    p["8"]["inputs"]["vae"] = ["201", 0]        # un noeud 2xx encore utilise
    with pytest.raises(RuntimeError, match="pruned"):
        rm.prune_to_stage1(p)


def test_prompt_b_cablage():
    b = rm.build_masked_refine_prompt("/t/stage1.png", "/t/mask.png", {})
    for nid, name, target in _refs(b):
        assert target in b, f"{nid}.{name} -> {target}"
    assert b["207"]["class_type"] == "LoadImage"
    assert b["207"]["inputs"]["image"] == "/t/stage1.png"
    assert b["208"]["class_type"] == "LoadImageMask"
    assert b["208"]["inputs"]["image"] == "/t/mask.png"
    assert b["208"]["inputs"]["channel"] == "red"
    assert b["202"]["class_type"] == "VAEEncode"
    assert b["202"]["inputs"]["pixels"] == ["207", 0]
    assert b["202"]["inputs"]["vae"] == ["201", 0]
    assert b["209"]["class_type"] == "SetLatentNoiseMask"
    assert b["209"]["inputs"] == {"samples": ["202", 0], "mask": ["208", 0]}
    assert b["210"]["class_type"] == "DifferentialDiffusion"
    assert b["210"]["inputs"]["model"] == ["200", 0]
    assert b["210"]["inputs"]["strength"] == 1.0
    k = b["205"]["inputs"]
    assert k["model"] == ["210", 0] and k["latent_image"] == ["209", 0]
    assert b["206"]["inputs"]["samples"] == ["205", 0]
    assert b["60"]["class_type"] == "SaveImage"
    assert b["60"]["inputs"]["images"] == ["206", 0]


def test_prompt_b_meme_etage_lustify_que_le_refine():
    """Meme checkpoint, meme VAE, memes textes, memes reglages du sampler : seul
    le cablage change."""
    src, b = _refine(), rm.build_masked_refine_prompt("a", "b", {})
    for nid in ("200", "201", "203", "204"):
        assert b[nid] == src[nid]
    for key in ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"):
        assert b["205"]["inputs"][key] == src["205"]["inputs"][key]
    assert (b["205"]["inputs"]["steps"], b["205"]["inputs"]["cfg"],
            b["205"]["inputs"]["denoise"]) == (28, 5.5, 0.55)


def test_prompt_b_aucun_noeud_qwen():
    b = rm.build_masked_refine_prompt("a", "b", {})
    assert not {"3", "8", "37", "38", "39", "78", "89", "111", "61"} & set(b)


def test_prompt_b_overrides():
    b = rm.build_masked_refine_prompt("a", "b", {
        "refine_prompt": "pos", "refine_negative": "neg", "refine_denoise": "0.4"})
    assert b["203"]["inputs"]["text"] == "pos"
    assert b["204"]["inputs"]["text"] == "neg"
    assert b["205"]["inputs"]["denoise"] == 0.4


def test_prompt_b_denoise_illisible_garde_le_defaut():
    b = rm.build_masked_refine_prompt("a", "b", {"refine_denoise": "beaucoup"})
    assert b["205"]["inputs"]["denoise"] == 0.55


def test_overrides_du_refine_combine_inchanges():
    """Le mode paste passe par le meme helper : meme resultat qu'avant."""
    p = _refine()
    rm.apply_refine_overrides(p, {"refine_prompt": "pos", "refine_denoise": 0.7})
    assert p["203"]["inputs"]["text"] == "pos"
    assert p["204"]["inputs"]["text"] == _refine()["204"]["inputs"]["text"]
    assert p["205"]["inputs"]["denoise"] == 0.7


def test_parametres_par_defaut():
    assert rm.refine_face_params({}) == ("paste", 0.35, 0.25, 96)


def test_parametres_valides_et_bornes():
    assert rm.refine_face_params({
        "refine_face_mode": "mask", "refine_face_level": 5,
        "refine_face_feather": 0.0, "refine_face_min_px": 9999}) == ("mask", 1.0, 0.05, 512)
    assert rm.refine_face_params({
        "refine_face_level": -1, "refine_face_min_px": -3}) == ("paste", 0.0, 0.25, 0)


def test_parametres_illisibles_ou_mode_inconnu():
    assert rm.refine_face_params({
        "refine_face_mode": "sideways", "refine_face_level": "x",
        "refine_face_feather": None, "refine_face_min_px": float("nan")}) == (
            "paste", 0.35, 0.25, 96)
