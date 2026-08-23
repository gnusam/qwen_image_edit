"""Face preservation: swap the original subject's face(s) onto the generated
result, then restore each swapped region with GFPGAN.

v2 pipeline — fixes the three defects of the v1 largest-vs-largest swap:
  1. Donor faces come from ALL source images, clustered by identity (ArcFace
     cosine similarity); each person gets a quality-weighted centroid
     embedding, instead of whatever face happened to be biggest in
     image_paths[0].
  2. Result faces are matched to donors by embedding similarity (greedy
     one-to-one assignment), not by bounding-box area — identities no longer
     land on the wrong head in multi-person images.
  3. Each swapped region is re-rendered through GFPGAN 1.4: inswapper outputs
     a 128x128 face, which reads soft/waxy (mouths and teeth especially) on
     any face rendered larger than that.

InsightFace inswapper is the engine ReActor wraps; we call it directly so
explicit/NSFW results are NOT blocked by ReActor's built-in NSFW filter.
GFPGAN runs as ONNX on the onnxruntime already required by insightface — the
.pth release would drag in basicsr, which no longer imports against current
torchvision. All models lazy-load once per worker. Fail-safe throughout: any
problem returns the best image produced so far, with the error in the status
dict instead of a broken job.
"""

import base64
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_INSWAPPER_PATH = "/ComfyUI/models/insightface/inswapper_128.onnx"
_GFPGAN_PATH = "/ComfyUI/models/insightface/gfpgan_1.4.onnx"

# Same-identity threshold when clustering donor faces across source photos.
# buffalo_l ArcFace cosine: same person across photos ~0.5-0.8, different
# people ~0.0-0.3.
_CLUSTER_SIM = 0.45

# Share of the GFPGAN output blended over the swapped region. 1.0 looks
# slightly plastic; keeping a trace of the inswapper texture reads more
# photographic (facefusion ships 0.8 for this same model).
_RESTORE_BLEND = 0.85

# FFHQ 5-point alignment template GFPGAN was trained on, scaled to the
# model's 512x512 input.
_FFHQ_512 = np.array([
    [0.37691676, 0.46864664],
    [0.62285697, 0.46912813],
    [0.50123859, 0.61331904],
    [0.39308822, 0.72541100],
    [0.61150205, 0.72490465],
], dtype=np.float32) * 512.0

_face_app = None
_face_swapper = None
_face_restorer = None  # onnxruntime session; False once a load has failed


def _load_face_models():
    global _face_app, _face_swapper
    if _face_swapper is None:
        import insightface
        from insightface.app import FaceAnalysis
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _face_app = FaceAnalysis(name="buffalo_l", providers=providers)
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
        _face_swapper = insightface.model_zoo.get_model(_INSWAPPER_PATH,
                                                        providers=providers)
        logger.info("🧠 Face models loaded (buffalo_l + inswapper_128)")
    return _face_app, _face_swapper


def _load_restorer():
    global _face_restorer
    if _face_restorer is None:
        try:
            import onnxruntime
            _face_restorer = onnxruntime.InferenceSession(
                _GFPGAN_PATH,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            logger.info("🧠 Face restorer loaded (gfpgan_1.4)")
        except Exception as e:
            logger.error(f"face restorer unavailable, swaps stay raw: {e}")
            _face_restorer = False
    return _face_restorer or None


def _area(face):
    return float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))


def _quality(face):
    # Detection confidence weighted by size: a big frontal crop carries a
    # cleaner identity embedding than a small or half-occluded one.
    return float(face.det_score) * float(np.sqrt(_area(face)))


def _collect_donors(app, source_paths):
    """Detect faces in every source image and cluster them by identity.
    Returns one representative Face per person, its embedding replaced by the
    cluster centroid (inswapper only reads the embedding, not the pixels)."""
    clusters = []  # {"sum": vec, "rep": Face, "rep_q": float}
    for path in source_paths:
        img = cv2.imread(path)
        if img is None:
            logger.warning("preserve_face: unreadable source %s, skipping", path)
            continue
        for face in app.get(img):
            emb = face.normed_embedding
            best, best_sim = None, _CLUSTER_SIM
            for c in clusters:
                centroid = c["sum"] / np.linalg.norm(c["sum"])
                sim = float(np.dot(emb, centroid))
                if sim >= best_sim:
                    best, best_sim = c, sim
            if best is None:
                clusters.append({"sum": emb.copy(), "rep": face,
                                 "rep_q": _quality(face)})
            else:
                best["sum"] += emb
                if _quality(face) > best["rep_q"]:
                    best["rep"], best["rep_q"] = face, _quality(face)
    donors = []
    for c in clusters:
        rep = c["rep"]
        rep.embedding = c["sum"] / np.linalg.norm(c["sum"])
        donors.append(rep)
    return donors


def _match_faces(targets, donors):
    """Greedy one-to-one assignment of result faces to donor identities by
    embedding similarity. Leftover result faces stay untouched."""
    pairs = []
    for ti, t in enumerate(targets):
        for di, d in enumerate(donors):
            sim = float(np.dot(t.normed_embedding, d.normed_embedding))
            pairs.append((sim, ti, di))
    pairs.sort(reverse=True)
    used_t, used_d, matched = set(), set(), []
    for sim, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        matched.append((targets[ti], donors[di], sim))
    return matched


def _restore_region(img, kps, session):
    """Re-render one face region through GFPGAN and feather it back in."""
    matrix, _ = cv2.estimateAffinePartial2D(kps, _FFHQ_512, method=cv2.RANSAC,
                                            ransacReprojThreshold=100)
    if matrix is None:
        return img
    crop = cv2.warpAffine(img, matrix, (512, 512),
                          borderMode=cv2.BORDER_REPLICATE)
    inp = crop[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
    inp = (inp - 0.5) / 0.5
    inp = np.expand_dims(inp.transpose(2, 0, 1), 0)
    out = session.run(None, {session.get_inputs()[0].name: inp})[0][0]
    out = np.clip(out, -1.0, 1.0)
    out = (out.transpose(1, 2, 0) + 1.0) / 2.0 * 255.0
    restored = out.round().astype(np.uint8)[:, :, ::-1]  # RGB -> BGR
    restored = cv2.addWeighted(restored, _RESTORE_BLEND,
                               crop, 1.0 - _RESTORE_BLEND, 0)
    # Feathered box mask so the paste-back has no visible seam.
    blur_amount = int(512 * 0.5 * 0.3)
    blur_area = max(blur_amount // 2, 1)
    mask = np.ones((512, 512), np.float32)
    mask[:blur_area, :] = 0
    mask[-blur_area:, :] = 0
    mask[:, :blur_area] = 0
    mask[:, -blur_area:] = 0
    mask = cv2.GaussianBlur(mask, (0, 0), blur_amount * 0.25)
    inverse = cv2.invertAffineTransform(matrix)
    h, w = img.shape[:2]
    paste = cv2.warpAffine(restored, inverse, (w, h),
                           borderMode=cv2.BORDER_REPLICATE)
    paste_mask = np.clip(cv2.warpAffine(mask, inverse, (w, h)), 0.0, 1.0)
    paste_mask = paste_mask[:, :, None]
    return (paste_mask * paste.astype(np.float32)
            + (1.0 - paste_mask) * img.astype(np.float32)).astype(np.uint8)


def apply_face_preservation(result_b64, source_paths):
    """Swap each original face onto its matching face in the generated result,
    then restore the swapped regions. Returns (b64, status_dict)."""
    status = {"applied": False, "swapped": 0, "restored": 0,
              "donors": 0, "targets": 0}
    try:
        app, swapper = _load_face_models()
        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(result_b64), np.uint8),
            cv2.IMREAD_COLOR)
        if img is None:
            status["error"] = "unreadable result image"
            logger.warning("preserve_face: unreadable result image, skipping")
            return result_b64, status
        donors = _collect_donors(app, source_paths)
        targets = app.get(img)
        status["donors"], status["targets"] = len(donors), len(targets)
        if not donors or not targets:
            status["error"] = ("no face detected (donors=%d targets=%d)"
                               % (len(donors), len(targets)))
            logger.warning("preserve_face: %s, skipping", status["error"])
            return result_b64, status
        restorer = _load_restorer()
        for target, donor, sim in _match_faces(targets, donors):
            img = swapper.get(img, target, donor, paste_back=True)
            status["swapped"] += 1
            if restorer is not None:
                try:
                    img = _restore_region(img, target.kps, restorer)
                    status["restored"] += 1
                except Exception as e:
                    logger.error(f"preserve_face: restore failed on one face: {e}")
            logger.info("preserve_face: swapped face (sim=%.2f)", sim)
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            status["error"] = "png encode failed"
            return result_b64, status
        status["applied"] = status["swapped"] > 0
        logger.info("✅ preserve_face: %d face(s) swapped, %d restored",
                    status["swapped"], status["restored"])
        return base64.b64encode(buf.tobytes()).decode("utf-8"), status
    except Exception as e:
        logger.error(f"preserve_face failed, returning original result: {e}")
        status["error"] = str(e)[:300]
        return result_b64, status
