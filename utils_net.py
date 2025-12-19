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
    header_h = 12.0  # reserved space for header so it won't overlap faces

    total_w = L + 2 * (W + gap)
    total_h = (W + 2 * (H + gap)) + header_h

    dwg = svgwrite.Drawing(path, size=(f"{total_w}mm", f"{total_h}mm"))
    dwg.viewbox(0, 0, total_w, total_h)

    dwg.add(dwg.rect((0, 0), (total_w, total_h), fill="white"))

    # Header (unitless font size behaves better in Qt preview)
    dwg.add(dwg.text("Net layout — stick templates to matching faces",
                     insert=(3, 8), font_size=6, fill="black"))

    cx = W + gap
    cy = (H + gap) + header_h

    faces = {
        "TOP":   ((cx, cy),          (L, W)),
        "FRONT": ((cx, cy + W + gap),(L, H)),
        "BACK":  ((cx, cy - H - gap),(L, H)),
        "LEFT":  ((cx - W - gap, cy),(W, H)),
        "RIGHT": ((cx + L + gap, cy),(W, H)),
    }

    def label_font(fw, fh):
        # unitless; scales nicely with the viewBox in most renderers
        return max(7.0, min(fw, fh) * 0.18)

    def draw_face(name: str, pos, size):
        x, y = pos
        fw, fh = size

        fill = "white" if name in used_faces else "#f0f0f0"
        dwg.add(dwg.rect((x, y), (fw, fh), fill=fill, stroke="black", stroke_width=0.6))

        fs = label_font(fw, fh)
        # put label near top-left but with safe padding
        dwg.add(dwg.text(name, insert=(x + 3, y + fs + 2),
                         font_size=fs, fill="black"))

        # outline
        if outlines and name in outlines and outlines[name]:
            pts = [(x + px, y + py) for (px, py) in outlines[name]]
            if len(pts) >= 2:
                dwg.add(dwg.polyline(pts + [pts[0]],
                                     fill="none", stroke="black", stroke_width=0.8))

    for n, (pos, size) in faces.items():
        draw_face(n, pos, size)

    dwg.save()
