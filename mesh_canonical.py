from __future__ import annotations
import numpy as np
import trimesh

def canonicalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    V = np.asarray(m.vertices).astype(np.float32)

    Vc = V - V.mean(axis=0, keepdims=True)
    C = np.cov(Vc.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    R = eigvecs[:, np.argsort(eigvals)[::-1]]

    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    m.vertices = Vc @ R
    return m
