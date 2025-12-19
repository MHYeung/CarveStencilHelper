from __future__ import annotations
import numpy as np
import cv2
import trimesh

def mesh_silhouette_mask(mesh: trimesh.Trimesh, view: str, img_size: int = 1024, pad: float = 0.08) -> np.ndarray:
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)

    if view == "front":
        X, Y = V[:, 0], V[:, 1]
    elif view == "right":
        X, Y = V[:, 2], V[:, 1]
    elif view == "top":
        X, Y = V[:, 0], V[:, 2]
    else:
        raise ValueError("view must be front/right/top")

    xmin, xmax = float(X.min()), float(X.max())
    ymin, ymax = float(Y.min()), float(Y.max())
    w = max(xmax - xmin, 1e-6)
    h = max(ymax - ymin, 1e-6)

    xmin -= pad * w; xmax += pad * w
    ymin -= pad * h; ymax += pad * h
    w = xmax - xmin; h = ymax - ymin

    sx = (img_size - 1) / w
    sy = (img_size - 1) / h

    xpix = ((X - xmin) * sx).astype(np.float32)
    ypix = ((Y - ymin) * sy).astype(np.float32)

    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    for tri in F:
        pts = np.stack([xpix[tri], ypix[tri]], axis=1).astype(np.int32)
        cv2.fillConvexPoly(mask, pts, 255)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return mask
