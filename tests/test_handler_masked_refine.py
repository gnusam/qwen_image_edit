"""Le handler en mode refine_face_mode "mask", de bout en bout sans ComfyUI.

handler.py exige CUDA a l'import et parle a ComfyUI par HTTP et websocket : on
remplace torch, runpod et websocket par des faux, et get_images par un
ComfyUI minuscule qui enregistre les prompts recus. Ce que ces tests tiennent :
deux prompts dans l'ordre (A puis B), le masque ecrit et nomme dans B, les
statuts de refine_faces (la cle `mode` est la preuve de version sur les pods),
le recollage de repli seulement quand aucun visage n'a ete masque, et le mode
paste inchange.

    python -m pytest tests/test_handler_masked_refine.py -q
"""

import base64
import os
import sys
import types

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _png_b64(tone):
    a = np.full((256, 192, 3), tone, np.uint8)
    return base64.b64encode(cv2.imencode(".png", a)[1].tobytes()).decode()


@pytest.fixture
def h(monkeypatch, tmp_path):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    runpod = types.ModuleType("runpod")
    runpod.serverless = types.SimpleNamespace(
        start=lambda *a, **k: None, utils=types.SimpleNamespace(rp_upload=None))
    sv = types.ModuleType("runpod.serverless")
    sv_utils = types.ModuleType("runpod.serverless.utils")
    sv_utils.rp_upload = types.ModuleType("rp_upload")
    ws_mod = types.ModuleType("websocket")
    ws_mod.WebSocket = lambda: types.SimpleNamespace(
        connect=lambda url: None, close=lambda: None)
    for name, mod in (("torch", torch), ("runpod", runpod),
                      ("runpod.serverless", sv),
                      ("runpod.serverless.utils", sv_utils),
                      ("runpod.serverless.utils.rp_upload", sv_utils.rp_upload),
                      ("websocket", ws_mod)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "handler", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: None)
    import handler
    handler.calls = []

    def fake_get_images(ws, prompt):
        handler.calls.append(prompt)
        if "61" in prompt:                      # prompt A (ou refine combine)
            out = {"61": [_png_b64(40)]}
            if "60" in prompt:
                out["60"] = [_png_b64(200)]
            return out
        # prompt B : on relit les deux fichiers que le handler a poses
        for node in ("207", "208"):
            assert os.path.isfile(prompt[node]["inputs"]["image"])
        return {"60": [_png_b64(200)]}
    monkeypatch.setattr(handler, "get_images", fake_get_images)
    return handler


def _job(**extra):
    img = _png_b64(90)
    return {"input": {"image_base64": img, "prompt": "p", "refine": True, **extra}}


def _faces(h, monkeypatch, n_masked):
    status = {"mode": "mask", "masked": n_masked, "small": 0, "level": 0.35,
              "feather": 0.25, "min_px": 96,
              "faces": [{"w": 150, "h": 200, "det": 0.9}] * n_masked}
    import numpy as _np
    seen = {}

    def fake_build(stage_bytes, level, feather, min_px, source_path=None):
        seen.update(level=level, feather=feather, min_px=min_px,
                    source=os.path.basename(source_path or ""))
        return _np.ones((256, 192), _np.float32), dict(status)
    monkeypatch.setattr(h, "build_face_denoise_mask", fake_build)
    pasted = []

    def fake_paste(res, st):
        pasted.append((res, st))
        return res, {"kept": 1}
    monkeypatch.setattr(h, "keep_stage_faces", fake_paste)
    return seen, pasted


def test_deux_prompts_dans_lordre_et_statut_de_version(h, monkeypatch):
    seen, pasted = _faces(h, monkeypatch, n_masked=1)
    out = h.handler(_job(refine_face_mode="mask", refine_face_level=0.5,
                         refine_face_feather=0.4, refine_face_min_px=64,
                         refine_prompt="pos", refine_denoise=0.6))
    assert len(h.calls) == 2
    a, b = h.calls
    assert "61" in a and "60" not in a and "205" not in a     # A : etape 1 seule
    assert "207" in b and "210" in b and "3" not in b          # B : Lustify seul
    assert b["203"]["inputs"]["text"] == "pos"
    assert b["205"]["inputs"]["denoise"] == 0.6
    assert os.path.isabs(b["207"]["inputs"]["image"])
    assert seen == {"level": 0.5, "feather": 0.4, "min_px": 64,
                    "source": "input_image_1.jpg"}
    rf = out["refine_faces"]
    assert rf["mode"] == "mask" and rf["masked"] == 1
    assert rf["fallback_paste"] is False and pasted == []
    assert out["image"] == _png_b64(200)


def test_repli_paste_sans_visage_masque(h, monkeypatch):
    _, pasted = _faces(h, monkeypatch, n_masked=0)
    out = h.handler(_job(refine_face_mode="mask"))
    assert len(pasted) == 1
    rf = out["refine_faces"]
    assert rf["mode"] == "mask" and rf["masked"] == 0
    assert rf["fallback_paste"] is True and rf["paste"] == {"kept": 1}


def test_repli_coupe_par_refine_keep_face(h, monkeypatch):
    _, pasted = _faces(h, monkeypatch, n_masked=0)
    out = h.handler(_job(refine_face_mode="mask", refine_keep_face=False))
    assert pasted == []
    assert out["refine_faces"]["fallback_paste"] is False


def test_defauts_du_mode_mask(h, monkeypatch):
    seen, _ = _faces(h, monkeypatch, n_masked=1)
    h.handler(_job(refine_face_mode="mask"))
    assert seen == {"level": 0.35, "feather": 0.25, "min_px": 96,
                    "source": "input_image_1.jpg"}


def test_mode_paste_par_defaut_un_seul_prompt(h, monkeypatch):
    _, pasted = _faces(h, monkeypatch, n_masked=1)
    out = h.handler(_job())
    assert len(h.calls) == 1 and "205" in h.calls[0] and "210" not in h.calls[0]
    assert len(pasted) == 1
    assert out["refine_faces"] == {"kept": 1, "mode": "paste"}


def test_mode_inconnu_retombe_sur_paste(h, monkeypatch):
    _faces(h, monkeypatch, n_masked=1)
    out = h.handler(_job(refine_face_mode="sideways"))
    assert len(h.calls) == 1
    assert out["refine_faces"]["mode"] == "paste"


def test_mask_ignore_hors_refine_une_image(h, monkeypatch):
    """Sans refine, le mode n'a aucun effet : workflow normal, pas de statut."""
    _faces(h, monkeypatch, n_masked=1)
    job = _job(refine_face_mode="mask")
    job["input"]["refine"] = False
    h.calls.clear()

    def plain(ws, prompt):
        h.calls.append(prompt)
        return {"60": [_png_b64(200)]}
    monkeypatch.setattr(h, "get_images", plain)
    out = h.handler(job)
    assert len(h.calls) == 1 and "refine_faces" not in out


def test_lightning_par_defaut_et_statut_du_sampler(h, monkeypatch):
    """Sans option, l'etape 1 garde Lightning a 4 etapes cfg 1, et le dit."""
    _faces(h, monkeypatch, n_masked=1)
    out = h.handler(_job())
    p = h.calls[0]
    assert p["89"]["inputs"]["strength_model"] == 1
    assert (p["3"]["inputs"]["steps"], p["3"]["inputs"]["cfg"]) == (4, 1)
    assert out["sampler"] == {"lightning": True, "steps": 4, "cfg": 1}


def test_sans_lightning_reglage_officiel_40_etapes_cfg_4(h, monkeypatch):
    _faces(h, monkeypatch, n_masked=1)
    out = h.handler(_job(lightning=False))
    p = h.calls[0]
    assert p["89"]["inputs"]["strength_model"] == 0
    assert "89" in p and p["89"]["inputs"]["lora_name"]       # graphe intact pour la chaine LoRA
    assert (p["3"]["inputs"]["steps"], p["3"]["inputs"]["cfg"]) == (40, 4.0)
    assert out["sampler"] == {"lightning": False, "steps": 40, "cfg": 4.0}


def test_sans_lightning_les_reglages_explicites_priment(h, monkeypatch):
    _faces(h, monkeypatch, n_masked=1)
    out = h.handler(_job(lightning=False, sampler_steps=30, cfg=3))
    assert out["sampler"] == {"lightning": False, "steps": 30, "cfg": 3.0}


def test_sans_lightning_en_mode_mask_le_prompt_a_le_porte(h, monkeypatch):
    _faces(h, monkeypatch, n_masked=1)
    h.handler(_job(lightning=False, refine_face_mode="mask"))
    a = h.calls[0]
    assert a["89"]["inputs"]["strength_model"] == 0 and a["3"]["inputs"]["steps"] == 40
