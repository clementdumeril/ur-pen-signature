"""Signature strokes: load, fit to the paper, resample, and compare a drawing with the reference."""

import json
from pathlib import Path

import numpy as np


def load_strokes(path: str | Path) -> list[np.ndarray]:
    """Strokes saved by scripts/capture_signature.py, in screen pixels (y pointing down)."""
    data = json.loads(Path(path).read_text())
    return [np.asarray(s, float) * (1, -1) for s in data["strokes"] if len(s) >= 2]


def fit_to_box(strokes: list[np.ndarray], width: float, center) -> list[np.ndarray]:
    """Scale the signature to `width` meters and place it on the table, centered on `center` (x, y),
    readable by someone standing in front of the robot: writing right is +y, writing up is -x."""
    pts = np.concatenate(strokes)
    lo, hi = pts.min(0), pts.max(0)
    scale = width / max(hi[0] - lo[0], 1e-9)
    mid = (lo + hi) / 2
    c = np.asarray(center, float)
    return [np.column_stack([-(s[:, 1] - mid[1]), s[:, 0] - mid[0]]) * scale + c for s in strokes]


def to_page(points: np.ndarray, center) -> np.ndarray:
    """Table (x, y) back to page coordinates (right, up), for plotting."""
    p = np.asarray(points, float) - np.asarray(center, float)
    return np.column_stack([p[:, 1], -p[:, 0]])


def resample(stroke: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    if arc[-1] < 1e-9:
        return stroke[:1]
    t = np.linspace(0.0, arc[-1], max(int(np.ceil(arc[-1] / step)), 1) + 1)
    return np.column_stack([np.interp(t, arc, stroke[:, 0]), np.interp(t, arc, stroke[:, 1])])


def distance_to_strokes(points: np.ndarray, strokes: list[np.ndarray]) -> np.ndarray:
    """Distance from each point to the nearest segment of the strokes."""
    a = np.concatenate([s[:-1] for s in strokes])
    b = np.concatenate([s[1:] for s in strokes])
    ab = b - a
    t = np.clip(np.einsum("nmk,mk->nm", points[:, None] - a, ab) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    closest = a + t[..., None] * ab
    return np.linalg.norm(points[:, None] - closest, axis=2).min(1)


def compare(drawn: np.ndarray, reference: list[np.ndarray], tol: float = 0.0015) -> dict:
    """Accuracy: how far the ink lies from the reference. Coverage: how much of the reference got ink."""
    if len(drawn) == 0:
        return {"ink_points": 0, "mean_error_mm": None, "p95_error_mm": None, "coverage": 0.0}
    err = distance_to_strokes(drawn, reference)
    ref_pts = np.concatenate([resample(s, 0.001) for s in reference])
    gap = np.linalg.norm(ref_pts[:, None] - drawn[None], axis=2).min(1)
    return {
        "ink_points": int(len(drawn)),
        "mean_error_mm": float(err.mean() * 1e3),
        "p95_error_mm": float(np.percentile(err, 95) * 1e3),
        "coverage": float((gap < tol).mean()),
    }
