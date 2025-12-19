from __future__ import annotations

import os
import re
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import cv2
from PIL import Image
import svgwrite

# Optional background removal (recommended)
try:
    from rembg import remove as rembg_remove
    REMBG_OK = True
except Exception:
    REMBG_OK = False


# -----------------------------
# Data models
# -----------------------------
@dataclass
class BlockDims:
    L: float  # length in mm
    W: float  # width in mm
    H: float  # height in mm


@dataclass
class PolylineResult:
    points: list[tuple[float, float]]             # normalized (0..1)
    bbox: tuple[float, float, float, float]       # (minx, miny, maxx, maxy)


@dataclass
class JobResult:
    job_id: str
    out_dir: str
    files: dict


# -----------------------------
# Helpers
# -----------------------------
def _resample_closed_xy(xy: np.ndarray, n: int) -> np.ndarray:
    """
    Resample a closed polyline to exactly n points using arc-length interpolation.
    xy: (N,2) float32
    """
    if len(xy) < 3:
        return xy
    # ensure closed
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0]])

    d = np.sqrt(((xy[1:] - xy[:-1]) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 1e-6:
        return xy[:n]

    t = np.linspace(0, s[-1], n, endpoint=False)
    x = np.interp(t, s, xy[:, 0])
    y = np.interp(t, s, xy[:, 1])
    return np.stack([x, y], axis=1)


def _chaikin_smooth_closed(xy: np.ndarray, iters: int = 2) -> np.ndarray:
    """
    Chaikin corner-cutting smoothing for a closed polyline.
    """
    if len(xy) < 4:
        return xy
    # ensure closed
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0]])

    for _ in range(iters):
        new_pts = []
        for i in range(len(xy) - 1):
            p = xy[i]
            q = xy[i + 1]
            new_pts.append(0.75 * p + 0.25 * q)
            new_pts.append(0.25 * p + 0.75 * q)
        xy = np.vstack([np.array(new_pts), new_pts[0]])  # re-close
    return xy


def _auto_resize(img: Image.Image, max_side: int = 1400) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    if s <= max_side:
        return img
    scale = max_side / s
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem).strip("_")
    return stem or "image"


def keep_best_component(mask: np.ndarray, anchor_xy=(0.5, 0.75), sigma=0.35) -> np.ndarray:
    """
    Pick the connected component most likely to be the subject.
    anchor_xy: normalized expected subject location.
    """
    import math

    m = (mask > 127).astype(np.uint8)
    H, W = m.shape[:2]
    ax, ay = anchor_xy[0] * W, anchor_xy[1] * H

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return (m * 255).astype(np.uint8)

    best_i = None
    best_score = -1.0

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        area_norm = area / float(W * H)
        dist = math.hypot(cx - ax, cy - ay) / float(max(W, H))

        touches_border = (x <= 1) or (y <= 1) or (x + w >= W - 2) or (y + h >= H - 2)
        border_pen = 0.35 if touches_border else 1.0

        score = area_norm * math.exp(-(dist * dist) / (2 * sigma * sigma)) * border_pen
        if score > best_score:
            best_score = score
            best_i = i

    out = np.zeros((H, W), dtype=np.uint8)
    out[labels == best_i] = 255
    return out


def normalize_subject_mask(mask: np.ndarray, anchor_xy=(0.5, 0.75)) -> np.ndarray:
    """
    Ensure subject is white(255) on black(0), clean noise, and keep best component.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = (mask > 127).astype(np.uint8) * 255

    # If background is white, invert
    if mask.mean() > 127:
        mask = 255 - mask

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    mask = keep_best_component(mask, anchor_xy=anchor_xy, sigma=0.35)
    return mask


# -----------------------------
# Segmentation
# -----------------------------
def grabcut_mask(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    h, w = rgb.shape[:2]

    rect = (int(0.1 * w), int(0.1 * h), int(0.8 * w), int(0.8 * h))

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    cv2.grabCut(rgb, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    out = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    return out


def remove_background_to_mask(pil_img: Image.Image) -> np.ndarray:
    """
    Returns binary-ish mask uint8.
    Uses rembg when available; otherwise GrabCut fallback.
    """

    if REMBG_OK:
        rgba = rembg_remove(pil_img.convert("RGBA"))
        rgba = Image.fromarray(np.array(rgba))
        alpha = np.array(rgba.split()[-1])
        return (alpha > 8).astype(np.uint8) * 255

    # fallback
    return grabcut_mask(pil_img)


# -----------------------------
# Outline extraction (robust)
# -----------------------------
def mask_to_polyline(mask: np.ndarray, detail: float = 0.015) -> PolylineResult:
    """
    Higher-quality outline:
    - find dense contour (CHAIN_APPROX_NONE)
    - resample to many points (based on detail)
    - optional smoothing to remove jaggies
    """
    h, w = mask.shape[:2]

    # smooth mask edges a bit to reduce staircase artifacts
    mask2 = cv2.medianBlur(mask, 5)

    cnts, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("No contour found. Try tighter ROI or better mask.")

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 300:
        raise ValueError("Contour too small. Try tighter ROI or higher-res image.")

    pts = c[:, 0, :].astype(np.float32)  # (N,2) in pixels

    # Map "detail" (0.05..0.005) -> target points (~250..1600)
    detail = float(detail)
    detail = max(0.005, min(detail, 0.05))
    t = (0.05 - detail) / 0.045  # 0..1 (more detailed -> 1)
    target_n = int(250 + t * 1350)  # 250..1600

    pts_rs = _resample_closed_xy(pts, target_n)
    pts_sm = _chaikin_smooth_closed(pts_rs, iters=2 if t > 0.3 else 1)

    # normalize to 0..1
    nx = pts_sm[:, 0] / float(w)
    ny = pts_sm[:, 1] / float(h)
    out_pts = list(zip(nx.tolist(), ny.tolist()))

    minx, maxx = float(nx.min()), float(nx.max())
    miny, maxy = float(ny.min()), float(ny.max())
    bbox = (minx, miny, maxx, maxy)

    return PolylineResult(points=out_pts, bbox=bbox)

def fit_polyline_to_face(poly: PolylineResult, face_w_mm: float, face_h_mm: float, margin_mm: float):
    minx, miny, maxx, maxy = poly.bbox
    src_w = max(maxx - minx, 1e-6)
    src_h = max(maxy - miny, 1e-6)

    tgt_w = face_w_mm - 2 * margin_mm
    tgt_h = face_h_mm - 2 * margin_mm
    if tgt_w <= 1 or tgt_h <= 1:
        raise ValueError("Margin is too large for the selected face size.")

    s = min(tgt_w / src_w, tgt_h / src_h)
    extra_x = (tgt_w - src_w * s) / 2
    extra_y = (tgt_h - src_h * s) / 2

    out = []
    for (x, y) in poly.points:
        xn = (x - minx) * s
        yn = (y - miny) * s
        out.append((margin_mm + extra_x + xn, margin_mm + extra_y + yn))
    return out


def build_face_svg(path: str, face_w_mm: float, face_h_mm: float, poly_mm_pts, label: str):
    dwg = svgwrite.Drawing(path, size=(f"{face_w_mm}mm", f"{face_h_mm}mm"))
    dwg.viewbox(0, 0, face_w_mm, face_h_mm)  # IMPORTANT for preview correctness

    dwg.add(dwg.rect((0, 0), (face_w_mm, face_h_mm),
                     fill="white", stroke="black", stroke_width=0.6))

    # centerlines
    dwg.add(dwg.line((face_w_mm / 2, 0), (face_w_mm / 2, face_h_mm), stroke="gray", stroke_width=0.2))
    dwg.add(dwg.line((0, face_h_mm / 2), (face_w_mm, face_h_mm / 2), stroke="gray", stroke_width=0.2))

    # registration marks
    def cross(x, y, s=4):
        dwg.add(dwg.line((x - s, y), (x + s, y), stroke="black", stroke_width=0.4))
        dwg.add(dwg.line((x, y - s), (x, y + s), stroke="black", stroke_width=0.4))

    pad = 8
    cross(pad, pad)
    cross(face_w_mm - pad, pad)
    cross(pad, face_h_mm - pad)
    cross(face_w_mm - pad, face_h_mm - pad)

    if not poly_mm_pts or len(poly_mm_pts) < 10:
        raise ValueError("Outline too simplified. Increase detail or select ROI tighter.")

    dwg.add(dwg.polyline(poly_mm_pts + [poly_mm_pts[0]], fill="none", stroke="black", stroke_width=0.9))

    # scale bar
    bar = 50
    x0, y0 = 10, face_h_mm - 12
    dwg.add(dwg.line((x0, y0), (x0 + bar, y0), stroke="black", stroke_width=1.2))
    dwg.add(dwg.text("50mm", insert=(x0, y0 - 3), font_size=6))

    dwg.add(dwg.text(label, insert=(10, 14), font_size=8))
    dwg.save()


# -----------------------------
# Job wrapper (GUI uses this)
# -----------------------------
def generate_job(
    image_path: str,
    block: BlockDims,
    margin_mm: float,
    detail: float,
    include_top: bool,
    out_root: str = "jobs",
    roi_xywh: Optional[Tuple[int, int, int, int]] = None,   # ROI in original image pixels
    mode: str = "2d",  # "2d" or "tripo3d"
) -> JobResult:
    os.makedirs(out_root, exist_ok=True)

    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _safe_stem(image_path)
    folder_name = f"{dt}_{stem}"

    out_dir = os.path.join(out_root, folder_name)
    # Avoid collision if user runs twice within the same second
    if os.path.exists(out_dir):
        out_dir = os.path.join(out_root, f"{folder_name}_{uuid.uuid4().hex[:4]}")
    os.makedirs(out_dir, exist_ok=True)

    img_full = Image.open(image_path)

    # Apply ROI crop if provided
    if roi_xywh is not None:
        x, y, w, h = roi_xywh
        x = max(0, x); y = max(0, y)
        w = max(1, w); h = max(1, h)
        img = img_full.crop((x, y, x + w, y + h))
        # Save ROI preview for debugging
        img.save(os.path.join(out_dir, "debug_roi.png"))
    else:
        img = img_full

    img = _auto_resize(img)
    img.save(os.path.join(out_dir, "debug_roi_resized.png"))
    # Ensure debug_roi.png always exists (for GUI display consistency)
    if not os.path.exists(os.path.join(out_dir, "debug_roi.png")):
        img.save(os.path.join(out_dir, "debug_roi.png"))

    mask = remove_background_to_mask(img)
    mask = normalize_subject_mask(mask, anchor_xy=(0.5, 0.75))
    cv2.imwrite(os.path.join(out_dir, "debug_mask.png"), mask)
    mask_2d = mask  # keep a photo-aligned mask for FRONT + cutout alpha

    left_mask = None
    back_mask = None
    front_mask = None
    right_mask = None
    top_mask = None

    if mode.lower() == "tripo3d":
        # Use TripoSR mesh -> orthographic silhouettes -> masks
        # (Requires: trimesh + your TripoSR folder working)
        from recon_triposr import TripoSRReconstructor
        from mesh_canonical import canonicalize_mesh
        from mesh_views import mesh_silhouette_mask

        recon_dir = os.path.join(out_dir, "recon")
        os.makedirs(recon_dir, exist_ok=True)

        # TripoSR works best on a cutout. Use the resized ROI image here.
        cutout_path = os.path.join(out_dir, "cutout.png")
        rgba = img.convert("RGBA")
        mask_pil = Image.fromarray(mask)  # or your bg-removal mask BEFORE normalize
        rgba.putalpha(mask_pil)
        rgba.save(cutout_path)

        recon = TripoSRReconstructor("third_party/TripoSR")
        rr = recon.reconstruct(cutout_path, recon_dir)

        mesh = canonicalize_mesh(rr.mesh)
        mesh_path = os.path.join(out_dir, "debug_mesh.obj")
        mesh.export(mesh_path)

        # HYBRID:
        # FRONT stays 2D (aligned with the photo)
        front_mask = mask_2d

        # other faces from 3D silhouettes
        right_mask = mesh_silhouette_mask(mesh, "right", img_size=2048)
        left_mask  = mesh_silhouette_mask(mesh, "left",  img_size=2048)
        back_mask  = mesh_silhouette_mask(mesh, "back",  img_size=2048)

        top_mask = mesh_silhouette_mask(mesh, "top", img_size=2048) if include_top else None

        cv2.imwrite(os.path.join(out_dir, "debug_front_mask.png"), front_mask)
        cv2.imwrite(os.path.join(out_dir, "debug_right_mask.png"), right_mask)
        cv2.imwrite(os.path.join(out_dir, "debug_left_mask.png"), left_mask)
        cv2.imwrite(os.path.join(out_dir, "debug_back_mask.png"), back_mask)
        if include_top and top_mask is not None:
            cv2.imwrite(os.path.join(out_dir, "debug_top_mask.png"), top_mask)

    else:
        # Existing 2D approach: one silhouette reused
        front_mask = mask
        right_mask = mask
        left_mask = None
        back_mask = None
        top_mask = mask_2d if include_top else None

    # Now extract polylines from whichever masks were selected
    front_poly = mask_to_polyline(front_mask, detail=detail)
    right_poly = mask_to_polyline(right_mask, detail=detail)
    top_poly = mask_to_polyline(top_mask, detail=detail) if include_top and top_mask is not None else None

    left_poly = mask_to_polyline(left_mask, detail=detail) if left_mask is not None else None
    back_poly = mask_to_polyline(back_mask, detail=detail) if back_mask is not None else None

    # Overlays (front overlay on the same resized ROI image)
    overlay_path = os.path.join(out_dir, "debug_overlay_front.png")
    save_polyline_overlay(img, front_poly, overlay_path)

    # Optional: overlays for right/top (draw on their masks for debugging)
    save_polyline_overlay(Image.fromarray(cv2.cvtColor(right_mask, cv2.COLOR_GRAY2RGB)),
                        right_poly,
                        os.path.join(out_dir, "debug_overlay_right.png"))

    if include_top and top_poly is not None:
        save_polyline_overlay(Image.fromarray(cv2.cvtColor(top_mask, cv2.COLOR_GRAY2RGB)),
                            top_poly,
                            os.path.join(out_dir, "debug_overlay_top.png"))

    left_svg = None
    back_svg = None

    # Face templates
    front_svg = os.path.join(out_dir, "face_front.svg")
    right_svg = os.path.join(out_dir, "face_right.svg")

    if left_poly is not None:
        left_svg = os.path.join(out_dir, "face_left.svg")
        left_pts = fit_polyline_to_face(left_poly, block.W, block.H, margin_mm)
        build_face_svg(left_svg, block.W, block.H, left_pts, f"LEFT (W×H) {block.W}×{block.H}mm")

    if back_poly is not None:
        back_svg = os.path.join(out_dir, "face_back.svg")
        back_pts = fit_polyline_to_face(back_poly, block.L, block.H, margin_mm)
        build_face_svg(back_svg, block.L, block.H, back_pts, f"BACK (L×H) {block.L}×{block.H}mm")

    top_svg: Optional[str] = None

    front_pts = fit_polyline_to_face(front_poly, block.L, block.H, margin_mm)
    build_face_svg(front_svg, block.L, block.H, front_pts, f"FRONT (L×H) {block.L}×{block.H}mm")

    right_pts = fit_polyline_to_face(right_poly, block.W, block.H, margin_mm)
    build_face_svg(right_svg, block.W, block.H, right_pts, f"RIGHT (W×H) {block.W}×{block.H}mm")

    if include_top and top_poly is not None:
        top_svg = os.path.join(out_dir, "face_top.svg")
        top_pts = fit_polyline_to_face(top_poly, block.L, block.W, margin_mm)
        build_face_svg(top_svg, block.L, block.W, top_pts, f"TOP (L×W) {block.L}×{block.W}mm")

    # Net layout
    from utils_net import build_net_layout_svg
    used_faces = {"FRONT", "RIGHT"} | ({"TOP"} if include_top else set())
    net_svg = os.path.join(out_dir, "net_layout.svg")
    outlines = {
        "FRONT": front_pts,
        "RIGHT": right_pts,
    }
    
    if left_poly is not None:
        used_faces.add("LEFT")
        outlines["LEFT"] = left_pts

    if back_poly is not None:
        used_faces.add("BACK")
        outlines["BACK"] = back_pts

    if include_top and top_svg is not None:
        used_faces.add("TOP")
        outlines["TOP"] = top_pts

    print(f"[pipeline] mode={mode} out_dir={out_dir}")
    build_net_layout_svg(net_svg, block.L, block.W, block.H, used_faces=used_faces, outlines=outlines)

    # Guide
    from guide import GuideMeta, generate_guide_md
    largest = "FRONT" if (block.L * block.H) >= (block.W * block.H) else "RIGHT"

    guide_md_path = os.path.join(out_dir, "carving_guide.md")
    guide_md = generate_guide_md(
        GuideMeta(
            L=block.L, W=block.W, H=block.H,
            faces=["FRONT", "RIGHT"] + (["TOP"] if include_top else []),
            largest_face=largest,
            allowance_mm=margin_mm,
            skill="beginner",
            tools=["knife", "sandpaper"],
        )
    )
    with open(guide_md_path, "w", encoding="utf-8") as f:
        f.write(guide_md)

    # Zip
    zip_path = os.path.join(out_dir, f"{folder_name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(front_svg, "face_front.svg")
        z.write(right_svg, "face_right.svg")
        if top_svg:
            z.write(top_svg, "face_top.svg")
        z.write(net_svg, "net_layout.svg")
        z.write(guide_md_path, "carving_guide.md")
        # include debug artifacts (handy while developing)
        for name in [
            "debug_overlay_front.png", "debug_overlay_right.png", "debug_overlay_top.png",
            "debug_front_mask.png", "debug_right_mask.png", "debug_top_mask.png",
            "debug_mesh.obj"]:
            p = os.path.join(out_dir, name)
            if os.path.exists(p):
                z.write(p, name)

    return JobResult(
        job_id=folder_name,
        out_dir=out_dir,
        files={
            "front_svg": front_svg,
            "right_svg": right_svg,
            "top_svg": top_svg,
            "net_svg": net_svg,
            "guide_md": guide_md_path,
            "zip": zip_path,
            "debug_overlay": overlay_path,
            "debug_mask": os.path.join(out_dir, "debug_mask.png"),
            "debug_roi": os.path.join(out_dir, "debug_roi.png"),
        },
    )

def save_polyline_overlay(pil_img: Image.Image,
                          poly: "PolylineResult",
                          out_path: str,
                          color_bgr=(0, 0, 255),
                          thickness: int = 3) -> None:
    """
    Draw the polyline (normalized 0..1) on top of the image and save as PNG.
    pil_img must be the SAME image used to create the mask (same size).
    """
    img_rgb = np.array(pil_img.convert("RGB"))
    h, w = img_rgb.shape[:2]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    pts = np.array([[int(x * w), int(y * h)] for (x, y) in poly.points], dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(img_bgr, [pts], isClosed=True, color=color_bgr, thickness=thickness, lineType=cv2.LINE_AA)

    # optional: draw bbox too
    minx, miny, maxx, maxy = poly.bbox
    x1, y1, x2, y2 = int(minx*w), int(miny*h), int(maxx*w), int(maxy*h)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.imwrite(out_path, img_bgr)
