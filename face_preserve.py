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

# Detections below this confidence are dropped: SCRFD false positives (a knee,
# a hand) score ~0.3-0.5, real faces in their best orientation ~0.65-0.95.
# Swapping onto a non-face is how "monstrous" artifacts happen — skipping is
# always the safer failure.
_MIN_DET_SCORE = 0.55

# Mean 512-scale reprojection error of the 5 keypoints onto the FFHQ template
# above which the GFPGAN pass is skipped: the crop would be misaligned and the
# "restore" would sharpen garbage. The swap itself is kept. Calibration:
# frontal/moderate poses fit at ~5-20, while fully degenerate keypoints
# (collinear) still reach ~47 because the similarity fit can shrink its scale,
# so the usable gate range is narrow.
_MAX_RESTORE_RESIDUAL = 30.0

# Validity gate for a DETECTION (donor or target), same residual metric.
# SCRFD's det_score does not tell orientations apart: on an upright face the
# 180-degree view can score HIGHER than the 0-degree view (measured 0.89 vs
# 0.83) while its keypoints come back inverted (eyes on the chin). Such a
# candidate poisons everything downstream — an inverted donor turns every
# swap of the batch into a smeared "new face". Measured on production
# images: correct keypoints fit the FFHQ template at 4-21, inverted or
# collinear ones at 32-85. Candidates above the gate are discarded before
# the per-face dedup, so the best VALID orientation wins.
_MAX_KPS_RESIDUAL = 25.0

# Minimum donor/target ArcFace cosine for a swap to happen at all. This gate
# only has to catch garbage: with sane keypoints on both sides the same
# person measures 0.5-0.9, and even a render where Qwen drifted the identity
# (exactly when the swap matters most) still measures 0.31-0.35, while a
# garbage embedding or a face already destroyed measures -0.01..0.18.
# Below this the swap can only destroy the face Qwen already rendered.
_MIN_SWAP_SIM = 0.25

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


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-6)


def _kps_residual(kps):
    """Similarity-fit the 5 keypoints onto the FFHQ template. Returns the
    2x3 affine (or None) and the mean 512-scale reprojection error."""
    matrix, _ = cv2.estimateAffinePartial2D(kps, _FFHQ_512, method=cv2.RANSAC,
                                            ransacReprojThreshold=100)
    if matrix is None:
        return None, float("inf")
    proj = kps @ matrix[:, :2].T + matrix[:, 2]
    return matrix, float(np.mean(np.linalg.norm(proj - _FFHQ_512, axis=1)))


def _map_back(face, rot, w, h):
    """Map a face detected on a rotated view back to original-image
    coordinates (w, h = ORIGINAL image size). Keypoint semantics (left eye,
    right eye, ...) are preserved, so downstream similarity alignment
    naturally includes the rotation."""
    if rot is None:
        return
    def back(pts):
        u, v = pts[:, 0], pts[:, 1]
        if rot == cv2.ROTATE_90_CLOCKWISE:
            return np.stack([v, h - 1 - u], axis=1)
        if rot == cv2.ROTATE_180:
            return np.stack([w - 1 - u, h - 1 - v], axis=1)
        return np.stack([w - 1 - v, u], axis=1)  # ROTATE_90_COUNTERCLOCKWISE
    face.kps = back(face.kps).astype(np.float32)
    x1, y1, x2, y2 = face.bbox[:4]
    c = back(np.array([[x1, y1], [x2, y2]], np.float32))
    face.bbox = np.array([c[:, 0].min(), c[:, 1].min(),
                          c[:, 0].max(), c[:, 1].max()], np.float32)


def _detect_faces(app, img):
    """Rotation-TTA detection. SCRFD is not rotation-invariant: on a heavily
    rotated face (a person lying down reads as upside-down) it either misses
    the face or lands the keypoints inverted (eyes on the chin), which poisons
    everything downstream — alignment, embedding, swap, restore. Detect on all
    four 90-degree orientations, map results back, keep the best-scored
    detection per face."""
    h, w = img.shape[:2]
    found = []
    for rot, name in ((None, 0), (cv2.ROTATE_90_CLOCKWISE, 90),
                      (cv2.ROTATE_180, 180),
                      (cv2.ROTATE_90_COUNTERCLOCKWISE, 270)):
        view = img if rot is None else cv2.rotate(img, rot)
        for face in app.get(view):
            if float(face.det_score) < _MIN_DET_SCORE:
                continue
            _map_back(face, rot, w, h)
            _, residual = _kps_residual(face.kps)
            face.rot, face.residual = name, residual
            if residual > _MAX_KPS_RESIDUAL:
                logger.debug("preserve_face: dropped rot=%d score=%.2f "
                             "residual=%.0f (inverted/degenerate keypoints)",
                             name, float(face.det_score), residual)
                continue
            found.append(face)
    # Among valid candidates prefer confidence, but let a cleaner keypoint
    # fit break near-ties: two orientations of the same face often score
    # within 0.05 of each other.
    found.sort(key=lambda f: float(f.det_score) - f.residual / 100.0,
               reverse=True)
    kept = []
    for f in found:
        if all(_iou(f.bbox, k.bbox) < 0.4 for k in kept):
            kept.append(f)
    return kept


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
        for face in _detect_faces(app, img):
            _set_pos(face, img)
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


def _set_pos(face, img):
    """Face centre in image-relative coordinates (0..1), so a source and a
    render of different sizes can be compared."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = face.bbox[:4]
    face.pos = (float(x1 + x2) / 2.0 / w, float(y1 + y2) / 2.0 / h)


def _match_faces(targets, donors, by_position=False):
    """Greedy one-to-one assignment of result faces to donor identities.
    Leftover result faces stay untouched. Returns (target, donor, sim).

    By embedding similarity when identities can be read. When they cannot —
    every pair under _MIN_SWAP_SIM, which is what a two-stage render does to
    faces — similarity is noise and the pairing is a coin toss: with two
    people it swapped their faces half the time. The render keeps the
    source's composition, so `by_position` pairs faces by where they are
    (nearest image-relative centre) instead. Only meaningful with a single
    source image; the caller decides. Returns (matches, "identity"|"position")."""
    sims = {}
    for ti, t in enumerate(targets):
        for di, d in enumerate(donors):
            sims[ti, di] = float(np.dot(t.normed_embedding, d.normed_embedding))
    identity_readable = any(v >= _MIN_SWAP_SIM for v in sims.values())
    ambiguous = len(targets) > 1 or len(donors) > 1
    use_position = by_position and ambiguous and not identity_readable
    if use_position:
        def score(ti, di):
            (tx, ty), (dx, dy) = targets[ti].pos, donors[di].pos
            return -((tx - dx) ** 2 + (ty - dy) ** 2)
        logger.info("preserve_face: identities unreadable (best sim %.2f), "
                    "pairing %d target(s) and %d donor(s) by position",
                    max(sims.values()), len(targets), len(donors))
    else:
        def score(ti, di):
            return sims[ti, di]
    pairs = sorted(((score(ti, di), ti, di) for (ti, di) in sims), reverse=True)
    used_t, used_d, matched = set(), set(), []
    for _, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        matched.append((targets[ti], donors[di], sims[ti, di]))
    return matched, ("position" if use_position else "identity")


def _restore_region(img, kps, session):
    """Re-render one face region through GFPGAN and feather it back in.
    Returns (image, restored_flag)."""
    matrix, residual = _kps_residual(kps)
    if matrix is None:
        return img, False
    if residual > _MAX_RESTORE_RESIDUAL:
        logger.warning("preserve_face: restore skipped, alignment residual "
                       "%.0f > %.0f (extreme pose)", residual,
                       _MAX_RESTORE_RESIDUAL)
        return img, False
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
    out = (paste_mask * paste.astype(np.float32)
           + (1.0 - paste_mask) * img.astype(np.float32)).astype(np.uint8)
    return out, True


def keep_stage_faces(result_b64, stage_b64):
    """Paste the faces of the stage-1 (Qwen) image back over the refined one.

    The refine pass is an SDXL img2img at denoise 0.55 over the whole frame:
    it redraws every face, and the identity similarity to the source drops to
    about 0 — at which point no swap can be trusted to find who is who. Qwen's
    faces still carry the identity (0.3-0.9 measured), so the refine keeps the
    body and the faces come back from stage 1, feathered to hide the seam.
    Both images share their geometry: the refine encodes the stage-1 decode.
    Returns (b64, status)."""
    status = {"kept": 0}
    try:
        app, _ = _load_face_models()
        dec = lambda b: cv2.imdecode(np.frombuffer(base64.b64decode(b), np.uint8),
                                     cv2.IMREAD_COLOR)
        res, st = dec(result_b64), dec(stage_b64)
        if res is None or st is None:
            status["error"] = "unreadable image"
            return result_b64, status
        h, w = res.shape[:2]
        if st.shape[:2] != (h, w):
            st = cv2.resize(st, (w, h), interpolation=cv2.INTER_AREA)
        faces = _detect_faces(app, st)
        if not faces:
            status["error"] = "no face in stage-1 image"
            return result_b64, status
        mask = np.zeros((h, w), np.float32)
        smallest = float("inf")
        for f in faces:
            x1, y1, x2, y2 = [float(v) for v in f.bbox[:4]]
            fw, fh = x2 - x1, y2 - y1
            # Ellipse a bit larger than the detector box: forehead and chin
            # sit outside SCRFD's box, and a seam across them shows.
            centre = (int((x1 + x2) / 2), int((y1 + y2) / 2 - 0.05 * fh))
            axes = (max(1, int(0.62 * fw)), max(1, int(0.78 * fh)))
            cv2.ellipse(mask, centre, axes, 0, 0, 360, 1.0, -1)
            smallest = min(smallest, fw, fh)
            status["kept"] += 1
        # Feather sized on the smallest face, so a small one keeps its shape.
        k = max(3, int(0.25 * smallest) | 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)[:, :, None]
        out = (st.astype(np.float32) * mask + res.astype(np.float32) * (1.0 - mask))
        ok, buf = cv2.imencode(".png", np.clip(out, 0, 255).astype(np.uint8))
        if not ok:
            status["error"] = "png encode failed"
            return result_b64, status
        logger.info("keep_stage_faces: %d stage-1 face(s) pasted over the refine",
                    status["kept"])
        return base64.b64encode(buf.tobytes()).decode("utf-8"), status
    except Exception as e:
        logger.error(f"keep_stage_faces failed, keeping the refined faces: {e}")
        status["error"] = str(e)[:300]
        return result_b64, status


def apply_face_preservation(result_b64, source_paths, sim_min=None):
    """Swap each original face onto its matching face in the generated result,
    then restore the swapped regions. Returns (b64, status_dict).

    `sim_min` overrides the identity floor for this job. The default (None ->
    _MIN_SWAP_SIM) is what the one-stage Qwen workflow relies on and must stay
    as it is. A two-stage job (Qwen staged, then an SDXL img2img pass) repaints
    the face hard enough that the source-to-render similarity lands under that
    floor every time, so it passes a lower one — and keeps the untouched render
    alongside, because a forced swap is exactly the one the user must arbitrate.
    """
    floor = _MIN_SWAP_SIM if sim_min is None else float(sim_min)
    status = {"applied": False, "swapped": 0, "restored": 0,
              "skipped": 0, "donors": 0, "targets": 0, "sim_min": floor}
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
        targets = _detect_faces(app, img)
        for t in targets:
            _set_pos(t, img)
        status["donors"], status["targets"] = len(donors), len(targets)
        if not donors or not targets:
            status["error"] = ("no face detected (donors=%d targets=%d)"
                               % (len(donors), len(targets)))
            logger.warning("preserve_face: %s, skipping", status["error"])
            return result_b64, status
        restorer = _load_restorer()
        for d in donors:
            logger.info("preserve_face: donor rot=%d score=%.2f residual=%.0f",
                        d.rot, float(d.det_score), d.residual)
        # Position is a fallback for unreadable identities, and only holds
        # when every donor comes from the same frame as the render's layout.
        matched, status["paired_by"] = _match_faces(
            targets, donors, by_position=len(source_paths) == 1)
        for target, donor, sim in matched:
            if sim < floor:
                status["skipped"] += 1
                logger.warning("preserve_face: swap skipped, sim=%.2f < %.2f "
                               "(target rot=%d score=%.2f residual=%.0f)",
                               sim, floor, target.rot,
                               float(target.det_score), target.residual)
                continue
            img = swapper.get(img, target, donor, paste_back=True)
            status["swapped"] += 1
            restored = False
            if restorer is not None:
                try:
                    img, restored = _restore_region(img, target.kps, restorer)
                except Exception as e:
                    logger.error(f"preserve_face: restore failed on one face: {e}")
            status["restored"] += int(restored)
            status["sim"] = round(float(sim), 3)
            logger.info("preserve_face: swapped face (sim=%.2f, target rot=%d "
                        "score=%.2f residual=%.0f, restored=%s)", sim,
                        target.rot, float(target.det_score), target.residual,
                        restored)
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
