"""Tests du recollage des visages d'etape 1 par-dessus la passe Lustify.

Ce que ces tests tiennent : le collage ne doit jamais etre abandonne parce que
l'image d'etape 1 est molle. Les deux images partagent leur geometrie, donc une
boite trouvee dans le rendu raffine vaut pour l'etape 1, et c'est ce repli qui
recupere la moitie des visages signales (mesure du 2026-09-28 : 77 rendus sans
aucun collage, 43 % signales, contre 6 % quand le collage a eu lieu).

Le detecteur est remplace par un faux : ce qui est teste ici, c'est la decision
(quelle image interroger, quelles boites retenir), pas SCRFD.

    python -m pytest tests/test_keep_stage_faces.py -q
"""

import base64
import sys
import os

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
    """Rend des visages selon la moyenne de l'image : chaque image de test a sa
    propre teinte, ce qui suffit a savoir laquelle le code est en train
    d'interroger."""

    def __init__(self, by_tone):
        self.by_tone = by_tone      # {valeur du canal bleu: [bbox, ...]}
        self.seen = []

    def get(self, img):
        tone = int(round(img[:, :, 0].mean()))
        # Le vrai SCRFD ne retrouve pas un visage droit dans une vue tournee.
        # Le faux doit en faire autant, sinon la rotation-TTA produirait
        # quatre detections du meme visage : d'ou le marqueur en haut a
        # gauche, que seule l'orientation d'origine porte encore (une rotation
        # de 180 degres conserve les dimensions, elle, donc ne suffit pas).
        if img[:4, :4, 1].mean() < 200:
            return []
        self.seen.append(tone)
        return [_FakeFace(b) for b in self.by_tone.get(tone, [])]


_W, _H = 512, 768
_TONE_STAGE, _TONE_REFINE = 40, 200


def _img(tone):
    a = np.zeros((_H, _W, 3), np.uint8)
    a[:, :, 0] = tone
    a[:4, :4, 1] = 255          # marqueur d'orientation, voir _FakeApp.get
    return a


def _b64(img):
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


@pytest.fixture
def fake_app(monkeypatch):
    def install(by_tone):
        app = _FakeApp(by_tone)
        monkeypatch.setattr(fp, "_load_face_models", lambda: (app, None))
        return app
    return install


def _run(app_boxes, fake_app):
    app = fake_app(app_boxes)
    out_b64, status = fp.keep_stage_faces(_b64(_img(_TONE_REFINE)),
                                          _b64(_img(_TONE_STAGE)))
    out = cv2.imdecode(np.frombuffer(base64.b64decode(out_b64), np.uint8),
                       cv2.IMREAD_COLOR)
    return out, status, app


def test_visage_trouve_sur_letape_1(fake_app):
    """Le cas nominal ne change pas : la boite vient de l'etape 1."""
    out, status, _ = _run({_TONE_STAGE: [(180, 260, 330, 460)]}, fake_app)
    assert status["kept"] == 1
    assert (status["from_stage1"], status["from_refine"]) == (1, 0)
    assert "error" not in status
    # Le centre du visage porte la teinte de l'etape 1, le bord celle du rendu.
    assert out[360, 255, 0] < 100
    assert out[10, 10, 0] > 150


def test_repli_sur_le_rendu_quand_letape_1_ne_donne_rien(fake_app):
    """Le cas qui fait 43 % de signalements aujourd'hui : l'etape 1 est molle,
    le detecteur n'y voit rien, mais le rendu porte un visage evident."""
    out, status, _ = _run({_TONE_REFINE: [(180, 260, 330, 460)]}, fake_app)
    assert status["kept"] == 1
    assert (status["from_stage1"], status["from_refine"]) == (0, 1)
    assert "error" not in status
    assert out[360, 255, 0] < 100      # les pixels d'etape 1 sont bien colles


def test_union_des_deux_images(fake_app):
    """Deuxieme visage vu seulement dans le rendu : il est recolle aussi,
    au lieu de rester a Lustify."""
    _, status, _ = _run({_TONE_STAGE: [(60, 100, 160, 220)],
                         _TONE_REFINE: [(60, 100, 160, 220),     # le meme
                                        (300, 400, 420, 560)]},  # un autre
                        fake_app)
    assert status["kept"] == 2
    assert (status["from_stage1"], status["from_refine"]) == (1, 1)


def test_aucun_visage_nulle_part(fake_app):
    """Rien a recoller : le rendu ressort intact, et l'erreur le dit."""
    src = _b64(_img(_TONE_REFINE))
    app = fake_app({})
    out_b64, status = fp.keep_stage_faces(src, _b64(_img(_TONE_STAGE)))
    assert status["kept"] == 0
    assert status["error"] == "no face in either image"
    assert out_b64 == src


def test_le_rendu_est_interroge_meme_quand_letape_1_repond(fake_app):
    """L'union coute une detection de plus : elle doit bien avoir lieu."""
    _, _, app = _run({_TONE_STAGE: [(180, 260, 330, 460)]}, fake_app)
    assert _TONE_REFINE in app.seen


def test_keypoints_degeneres_acceptes_pour_le_collage():
    """Meme plancher de confiance que le swap — la mesure ne justifie pas de
    le baisser — mais des keypoints incoherents ne disqualifient plus la
    boite : le collage ne s'aligne sur rien, il copie des pixels."""
    class _App:
        def get(self, img):
            if img[:4, :4, 1].mean() < 200:
                return []
            f = _FakeFace((180, 260, 330, 460), det_score=0.80)
            f.kps = np.zeros((5, 2), dtype=np.float32)   # degeneres
            return [f]
    app = _App()
    assert len(fp._detect_faces(app, _img(_TONE_STAGE), require_keypoints=False)) == 1
    assert len(fp._detect_faces(app, _img(_TONE_STAGE))) == 0


def test_le_plancher_de_confiance_reste_celui_du_swap():
    """Une detection a 0.35 (genou, main) n'est pas recollee non plus."""
    class _App:
        def get(self, img):
            if img[:4, :4, 1].mean() < 200:
                return []
            return [_FakeFace((180, 260, 330, 460), det_score=0.35)]
    assert fp._detect_faces(_App(), _img(_TONE_STAGE), require_keypoints=False) == []
