"""Tests du masque de debruitage des visages (refine_face_mode "mask").

Ce que ces tests tiennent : le refine laisse le corps a plein debruitage (1.0),
garde les visages bas (`level`), laisse les trop petits visages intacts (0.0), et
ne plante jamais le job : sans visage ou en cas d'erreur, le masque est
entierement a 1.0 et le statut le dit (c'est lui qui declenche le recollage
de repli dans le handler).

Le detecteur est remplace par un faux, comme dans test_keep_stage_faces.py : ce
qui est teste ici, c'est la construction du masque, pas SCRFD.

    python -m pytest tests/test_face_denoise_mask.py -q
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import face_preserve as fp


class _FakeFace:
    def __init__(self, bbox, det_score=0.9):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.kps = np.zeros((5, 2), dtype=np.float32)
        self.det_score = det_score


class _FakeApp:
    def __init__(self, boxes):
        self.boxes = boxes

    def get(self, img):
        # Le vrai SCRFD ne retrouve pas un visage droit dans une vue tournee.
        # Le faux doit en faire autant, sinon la rotation-TTA produirait
        # quatre detections du meme visage : d'ou le marqueur en haut a
        # gauche, que seule l'orientation d'origine porte encore.
        if img[:4, :4, 1].mean() < 200:
            return []
        return [_FakeFace(b) for b in self.boxes]


_W, _H = 512, 768


def _png():
    a = np.full((_H, _W, 3), 90, np.uint8)
    a[:4, :4, 1] = 255          # marqueur d'orientation, voir _FakeApp.get
    return cv2.imencode(".png", a)[1].tobytes()


@pytest.fixture
def detect(monkeypatch):
    def run(boxes, **kw):
        monkeypatch.setattr(fp, "_load_face_models",
                            lambda: (_FakeApp(boxes), None))
        return fp.build_face_denoise_mask(_png(), **kw)
    return run


def test_corps_a_1_et_visage_au_niveau(detect):
    mask, status = detect([(180, 260, 330, 460)], level=0.35)   # 150 x 200 px
    assert mask.shape == (_H, _W) and mask.dtype == np.float32
    assert mask[700, 50] == 1.0                      # loin de tout visage
    assert mask[350, 255] == pytest.approx(0.35, abs=1e-3)   # centre du visage
    assert (status["mode"], status["masked"], status["small"]) == ("mask", 1, 0)
    assert status["faces"] == [{"w": 150, "h": 200, "det": 0.9, "sim_src": None}]
    assert "error" not in status


def test_le_niveau_vient_du_parametre(detect):
    mask, _ = detect([(180, 260, 330, 460)], level=0.6)
    assert mask[350, 255] == pytest.approx(0.6, abs=1e-3)


def test_petit_visage_a_zero(detect):
    """min(w, h) < min_px : aucun debruitage, a 12 px latents il brouillerait."""
    mask, status = detect([(200, 300, 270, 380)], level=0.35, min_px=96)  # 70 x 80
    assert mask[340, 235] == 0.0
    assert mask[700, 50] == 1.0
    assert (status["masked"], status["small"]) == (1, 1)


def test_le_seuil_de_petit_visage_est_un_parametre(detect):
    box = [(200, 300, 270, 380)]
    mask, status = detect(box, level=0.35, min_px=60)
    assert mask[340, 235] == pytest.approx(0.35, abs=1e-3)
    assert status["small"] == 0


def test_minimum_sur_recouvrement(detect):
    """Un grand visage et un petit qui le chevauche : la valeur basse gagne,
    le grand ne remonte pas le petit a son niveau."""
    mask, status = detect([(100, 200, 300, 450),        # 200 x 250, niveau 0.35
                           (240, 300, 320, 390)],       # 80 x 90, petit -> 0.0
                          level=0.35, min_px=96)
    assert status["masked"] == 2 and status["small"] == 1
    assert mask[345, 280] == 0.0                         # centre du petit
    assert mask[320, 150] == pytest.approx(0.35, abs=1e-3)   # grand seul
    assert mask.min() == 0.0 and mask.max() == 1.0


def test_bord_adouci(detect):
    """Le bord de l'ellipse descend progressivement : pas de marche, c'est la
    couture que le mode paste laissait voir."""
    mask, _ = detect([(180, 260, 330, 460)], level=0.0, feather=0.25)
    row = mask[360, 255:]                     # du centre vers la droite
    inter = row[(row > 0.05) & (row < 0.95)]
    assert len(inter) >= 8
    assert np.all(np.diff(row) >= -1e-6)      # monotone


def test_plus_de_flou_avec_un_feather_plus_grand(detect):
    def width(feather):
        mask, _ = detect([(180, 260, 330, 460)], level=0.0, feather=feather)
        row = mask[360]
        return int(((row > 0.05) & (row < 0.95)).sum())
    assert width(0.5) > width(0.1)


def test_aucun_visage_masque_tout_a_1(detect):
    mask, status = detect([])
    assert np.all(mask == 1.0)
    assert mask.shape == (_H, _W)
    assert (status["masked"], status["small"], status["faces"]) == (0, 0, [])
    assert "error" not in status


def test_detecteur_en_panne_masque_tout_a_1(monkeypatch):
    def boom():
        raise RuntimeError("model gone")
    monkeypatch.setattr(fp, "_load_face_models", boom)
    mask, status = fp.build_face_denoise_mask(_png())
    assert np.all(mask == 1.0) and mask.shape == (_H, _W)
    assert status["masked"] == 0
    assert "model gone" in status["error"]


def test_image_illisible_masque_tout_a_1():
    mask, status = fp.build_face_denoise_mask(b"not an image")
    assert np.all(mask == 1.0)
    assert status["error"] == "unreadable image"


def test_png_du_masque_est_relu_par_comfyui_en_canal_rouge(detect):
    """LoadImageMask lit le canal rouge : un PNG gris relu en couleur doit
    rendre la meme valeur (a la quantification 8 bits pres)."""
    mask, _ = detect([(180, 260, 330, 460)], level=0.35)
    png = fp.encode_mask_png(mask)
    back = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    red = back[:, :, 2].astype(np.float32) / 255.0   # BGR : rouge en dernier
    assert red.shape == mask.shape
    assert np.abs(red - mask).max() <= 0.5 / 255 + 1e-6


class _EmbFace(_FakeFace):
    def __init__(self, bbox, emb):
        super().__init__(bbox)
        self.normed_embedding = np.array(emb, np.float32)
        # Keypoints valides (le gabarit FFHQ lui-meme) : un visage de source
        # aux keypoints degeneres est ecarte, comme pour un donneur.
        self.kps = fp._FFHQ_512.copy() * 0.2 + 10


class _EmbApp:
    """Etape 1 (teinte 90) : un visage d'embedding e0 ; source (teinte 150) :
    les visages de `src_faces`."""

    def __init__(self, src_faces):
        self.src_faces = src_faces

    def get(self, img):
        if img[:4, :4, 1].mean() < 200:
            return []
        if int(round(img[:, :, 0].mean())) == 90:
            return [_EmbFace((180, 260, 330, 460), [1.0, 0.0])]
        # Des boites distinctes : la TTA fusionne celles qui se recouvrent.
        return [_EmbFace((10 + 200 * i, 10, 120 + 200 * i, 140), e)
                for i, e in enumerate(self.src_faces)]


def _source(tmp_path):
    a = np.full((_H, _W, 3), 150, np.uint8)
    a[:4, :4, 1] = 255
    path = str(tmp_path / "src.png")
    cv2.imwrite(path, a)
    return path


def test_similarite_a_la_source_meilleure_correspondance(monkeypatch, tmp_path):
    """Plusieurs visages dans la source : le meilleur compte. C'est une mesure,
    le masque ne bouge pas."""
    app = _EmbApp([[0.0, 1.0], [0.6, 0.8]])
    monkeypatch.setattr(fp, "_load_face_models", lambda: (app, None))
    mask, status = fp.build_face_denoise_mask(_png(), source_path=_source(tmp_path))
    assert status["faces"][0]["sim_src"] == pytest.approx(0.6, abs=1e-3)
    ref, _ = fp.build_face_denoise_mask(_png())
    assert np.array_equal(mask, ref)


def test_source_sans_visage_ou_absente_sim_none(monkeypatch, tmp_path):
    app = _EmbApp([])
    monkeypatch.setattr(fp, "_load_face_models", lambda: (app, None))
    _, status = fp.build_face_denoise_mask(_png(), source_path=_source(tmp_path))
    assert status["faces"][0]["sim_src"] is None
    _, status = fp.build_face_denoise_mask(_png(), source_path="/nope.png")
    assert status["faces"][0]["sim_src"] is None
    _, status = fp.build_face_denoise_mask(_png())
    assert status["faces"][0]["sim_src"] is None


def test_sim_src_non_mesurable_sur_keypoints_degeneres(monkeypatch, tmp_path):
    """Un visage garde pour sa seule boite (keypoints a zero) a un embedding
    non aligne, proche de 0 : ce n'est pas une identite perdue, c'est une
    mesure impossible."""
    class _Degenerate(_EmbApp):
        def get(self, img):
            faces = super().get(img)
            if int(round(img[:, :, 0].mean())) == 90:
                for f in faces:
                    f.kps = np.zeros((5, 2), np.float32)
            return faces
    app = _Degenerate([[1.0, 0.0]])
    monkeypatch.setattr(fp, "_load_face_models", lambda: (app, None))
    mask, status = fp.build_face_denoise_mask(_png(), source_path=_source(tmp_path))
    assert status["masked"] == 1 and status["faces"][0]["sim_src"] is None
    assert mask[350, 255] == pytest.approx(0.35, abs=1e-3)   # le masque, lui, tient
