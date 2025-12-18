import svgwrite

def build_net_layout_svg(path: str, L: float, W: float, H: float,
                         used_faces: set[str],
                         outlines: dict | None = None):
    """
    Net layout with optional outline polylines drawn on each face.
    outlines: {"FRONT": [(xmm, ymm), ...], "RIGHT": [...], "TOP": [...]}
              points are in mm in that face coordinate system (0..face_w, 0..face_h)
    """
    gap = 8.0
    total_w = L + 2 * (W + gap)
    total_h = W + 2 * (H + gap)

    dwg = svgwrite.Drawing(path, size=(f"{total_w}mm", f"{total_h}mm"))
    dwg.viewbox(0, 0, total_w, total_h)

    dwg.add(dwg.rect((0, 0), (total_w, total_h), fill="white"))

    cx = W + gap
    cy = H + gap

    faces = {
        "TOP":   ((cx, cy),         (L, W)),
        "FRONT": ((cx, cy + W + gap),(L, H)),
        "BACK":  ((cx, cy - H - gap),(L, H)),
        "LEFT":  ((cx - W - gap, cy),(W, H)),
        "RIGHT": ((cx + L + gap, cy),(W, H)),
    }

    def font_mm(face_w, face_h):
        return f"{max(3.5, min(face_w, face_h) * 0.12)}mm"

    def draw_face(name: str, pos, size):
        x, y = pos
        fw, fh = size
        fill = "white" if name in used_faces else "#f0f0f0"
        dwg.add(dwg.rect((x, y), (fw, fh), fill=fill, stroke="black", stroke_width=0.6))

        # label inside the box
        dwg.add(dwg.text(name, insert=(x + 4, y + 8),
                         font_size=font_mm(fw, fh), fill="black"))

        # draw outline if available
        if outlines and name in outlines and outlines[name]:
            pts = [(x + px, y + py) for (px, py) in outlines[name]]
            dwg.add(dwg.polyline(pts + [pts[0]], fill="none", stroke="black", stroke_width=0.6))

    for n, (pos, size) in faces.items():
        draw_face(n, pos, size)

    # small header (not huge)
    dwg.add(dwg.text("Net layout — stick templates to matching faces",
                     insert=(3, 6), font_size="4mm", fill="black"))

    dwg.save()
