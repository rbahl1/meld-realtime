"""vision expert: yunet faces, five-point alignment, the opencv zoo facial
expression model (mobilefacenet, progressive teacher, raf-db) and a mouth
motion signal per face track.

per frame (bgr, any size):
- yunet (cv2.FaceDetectorYN) on a copy whose long side is DETECT_LONG px,
  score >= score_threshold, at most MAX_FACES faces kept by area.
- each face is warped to 112x112 by the similarity that maps its five
  landmarks onto the template of the zoo's facial_fer_model.py (least squares,
  no reflection), then the fer graph gives 7 logits in the zoo order
  (angry, disgust, fearful, happy, neutral, sad, surprised) and the 512-d
  input of its last gemm (added as a graph output when the net is built).
  both graphs run in opencv dnn.
- tracks: greedy iou matching against the faces of the previous frame
  (iou >= TRACK_IOU), a new id otherwise. a track that misses a frame ends.
- motion: on the aligned gray face, the mean absolute difference to the same
  track's previous frame, over the mouth box (MOUTH_ROWS x MOUTH_COLS) and over
  the whole face, both on a 0..1 scale. nan on a track's first frame.

posterior_meld(logits) gives the softmax in MELD order (meld_rt.labels.EMOTIONS).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from meld_rt import labels

FER_CLASSES = ["angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
FER_TO_MELD = {"angry": "anger", "disgust": "disgust", "fearful": "fear", "happy": "joy",
               "neutral": "neutral", "sad": "sadness", "surprised": "surprise"}
# index into the fer logits for each MELD class, MELD order
MELD_FROM_FER = [FER_CLASSES.index(next(k for k, v in FER_TO_MELD.items() if v == e)) for e in labels.EMOTIONS]

FER_INPUT = "data"
FER_LOGITS = "label"
FER_EMBED = "481"          # flatten output feeding the last gemm (fc.weight is 7 x 512)
EMBED_DIM = 512
ALIGN_SIZE = 112
# facial_fer_model.py, FaceAlignment._std_points: eyes, nose tip, mouth corners in 112x112
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)
MOUTH_ROWS = slice(78, 106)
MOUTH_COLS = slice(28, 84)

DETECT_LONG = 480
MAX_FACES = 6
TRACK_IOU = 0.3


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def posterior_meld(fer_logits) -> np.ndarray:
    """softmax of the fer logits, reordered to MELD order (7,) or (n, 7)."""
    p = softmax(np.asarray(fer_logits, dtype=np.float64))
    return p[..., MELD_FROM_FER]


def similarity_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2x3 matrix of the similarity (scale, rotation, translation, no reflection)
    that maps src onto dst with the least squared error."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    n = len(src)
    a = np.zeros((2 * n, 4))
    b = np.zeros(2 * n)
    # [x' y'] = [[s c, -s s], [s s, s c]] [x y] + [tx ty]  with  p = s cos, q = s sin
    a[0::2] = np.column_stack([src[:, 0], -src[:, 1], np.ones(n), np.zeros(n)])
    a[1::2] = np.column_stack([src[:, 1], src[:, 0], np.zeros(n), np.ones(n)])
    b[0::2] = dst[:, 0]
    b[1::2] = dst[:, 1]
    (p, q, tx, ty), *_ = np.linalg.lstsq(a, b, rcond=None)
    return np.array([[p, -q, tx], [q, p, ty]], dtype=np.float64)


def iou(a, b) -> float:
    """boxes as (x, y, w, h)."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Face:
    """one face in one frame. box and landmarks in pixels of the input frame."""
    track: int
    box: np.ndarray            # (4,) x, y, w, h
    landmarks: np.ndarray      # (5, 2)
    score: float
    logits: np.ndarray         # (7,) fer order
    embed: np.ndarray | None   # (512,) float32
    mouth_motion: float        # nan on the track's first frame
    face_motion: float
    gray: np.ndarray = field(repr=False, default=None)  # aligned gray face, kept for the next frame's motion

    @property
    def area(self) -> float:
        return float(self.box[2] * self.box[3])


class VisionExpert:
    """process(frame_bgr) -> list[Face] for one stream; call reset() between clips or streams."""

    def __init__(self, yunet_path: Path, fer_path: Path, threads: int | None = 1, score_threshold: float = 0.6,
                 nms_threshold: float = 0.3, embed: bool = True):
        import cv2
        import onnx

        self.cv2 = cv2
        self.yunet_path = Path(yunet_path)
        self.fer_path = Path(fer_path)
        self.embed = bool(embed)
        if threads:
            cv2.setNumThreads(int(threads))
        self._det = cv2.FaceDetectorYN.create(str(self.yunet_path), "", (320, 320), float(score_threshold),
                                              float(nms_threshold), 50)
        self._det_size: tuple[int, int] | None = None
        # the fer graph runs in opencv dnn: 6.9 ms per face on one thread against
        # 17 ms in onnxruntime, which wraps every prelu in layout reorders. the zoo
        # graph lists its weights as graph inputs, so they are dropped from the inputs,
        # and the embedding is added as a second output (same values as onnxruntime
        # within 1e-6, tests/test_vision.py).
        model = onnx.load(str(self.fer_path))
        inits = {i.name for i in model.graph.initializer}
        keep = [i for i in model.graph.input if i.name not in inits]
        del model.graph.input[:]
        model.graph.input.extend(keep)
        outputs = [FER_LOGITS]
        if self.embed:
            if FER_EMBED not in {o.name for o in model.graph.output}:
                model.graph.output.append(onnx.helper.make_tensor_value_info(FER_EMBED, onnx.TensorProto.FLOAT,
                                                                             [1, EMBED_DIM]))
            outputs.append(FER_EMBED)
        self._outputs = outputs
        self.graph_bytes = model.SerializeToString()
        self.net = cv2.dnn.readNetFromONNX(np.frombuffer(self.graph_bytes, dtype=np.uint8))
        self._prev: list[Face] = []
        self._next_track = 0

    def reset(self) -> None:
        self._prev = []
        self._next_track = 0

    # detection ----------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """yunet rows (n, 15) in input-frame pixels: x, y, w, h, 10 landmark coords, score."""
        cv2 = self.cv2
        h, w = frame_bgr.shape[:2]
        s = min(1.0, DETECT_LONG / float(max(h, w)))
        dw, dh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        small = frame_bgr if (dw, dh) == (w, h) else cv2.resize(frame_bgr, (dw, dh), interpolation=cv2.INTER_AREA)
        if self._det_size != (dw, dh):
            self._det.setInputSize((dw, dh))
            self._det_size = (dw, dh)
        _, faces = self._det.detect(small)
        if faces is None or len(faces) == 0:
            return np.zeros((0, 15), dtype=np.float64)
        faces = np.asarray(faces, dtype=np.float64)
        faces[:, :14] /= s
        order = np.argsort(-(faces[:, 2] * faces[:, 3]))[:MAX_FACES]
        return faces[order]

    # recognition --------------------------------------------------------
    def align(self, frame_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        m = similarity_from_points(landmarks, TEMPLATE)
        return self.cv2.warpAffine(frame_bgr, m, (ALIGN_SIZE, ALIGN_SIZE), flags=self.cv2.INTER_LINEAR,
                                   borderMode=self.cv2.BORDER_CONSTANT, borderValue=0)

    def recognize(self, aligned_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """fer logits (7,) and the 512-d embedding (or None) for one aligned face."""
        rgb = aligned_bgr[:, :, ::-1].astype(np.float32) / 255.0
        x = ((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None]
        self.net.setInput(np.ascontiguousarray(x, dtype=np.float32), FER_INPUT)
        out = self.net.forward(self._outputs)
        logits = out[0].reshape(-1).astype(np.float32)
        emb = out[1].reshape(-1).astype(np.float32) if self.embed else None
        return logits, emb

    # one frame ----------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> list[Face]:
        cv2 = self.cv2
        rows = self.detect(frame_bgr)
        faces: list[Face] = []
        used: set[int] = set()
        for r in rows:
            box = r[:4].copy()
            lm = r[4:14].reshape(5, 2).copy()
            aligned = self.align(frame_bgr, lm)
            logits, emb = self.recognize(aligned)
            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            best, best_iou = None, TRACK_IOU
            for j, p in enumerate(self._prev):
                if j in used:
                    continue
                v = iou(box, p.box)
                if v >= best_iou:
                    best, best_iou = j, v
            if best is None:
                track = self._next_track
                self._next_track += 1
                mm = fm = float("nan")
            else:
                used.add(best)
                prev = self._prev[best]
                track = prev.track
                d = np.abs(gray - prev.gray)
                mm = float(d[MOUTH_ROWS, MOUTH_COLS].mean())
                fm = float(d.mean())
            faces.append(Face(track, box, lm, float(r[14]), logits, emb, mm, fm, gray))
        self._prev = faces
        return faces
