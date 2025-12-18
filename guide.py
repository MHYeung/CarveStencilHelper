from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GuideMeta:
    L: float
    W: float
    H: float
    faces: list[str]
    largest_face: str
    allowance_mm: float
    skill: str
    tools: list[str]


def generate_guide_md(meta: GuideMeta) -> str:
    tools = ", ".join(meta.tools) if meta.tools else "knife + sandpaper"
    return f"""# Carving Guide ({meta.skill.title()})

**Tools:** {tools}  
**Wood block:** {meta.L} × {meta.W} × {meta.H} mm (L×W×H)  
**Templates included:** {", ".join(meta.faces)}

---

## 1) Template placement & alignment
1. Print at **100% scale** (do not use “Fit to page”).
2. Cut around each template roughly and stick it onto the correct face.
3. Use the **centerlines** + **corner registration crosses** to keep the template square.
4. Mark “TOP” direction on the wood so you don’t accidentally rotate a face.

## 2) Which surface to start from (and why)
Start from **{meta.largest_face}** first.  
It usually defines the biggest silhouette, so it sets the overall shape early and helps prevent over-carving on the other face.

## 3) Roughing phase (big shape first)
1. If you have a saw: rough-saw outside the outline, leaving about **{meta.allowance_mm} mm** extra.
2. With a knife, remove corners gradually: **square → chamfer → octagon → round**.
3. Stay **outside the printed line** at first (keep that allowance).

### Checkpoint A (after roughing)
- From the **FRONT** view: silhouette matches the front template.
- From the **RIGHT** view: thickness/profile matches the right template.
- You still have a small allowance remaining.

## 4) Lock thickness using the adjacent face
1. Rotate the block and carve a little to match the second face.
2. Alternate: carve a bit on one face → check the other face → repeat.

## 5) Refining phase (approach the line)
1. Reduce the allowance slowly.
2. Make small “stop cuts” at sharp corners before removing surrounding material.
3. Keep checking both templates so you don’t “win” one view and ruin the other.

### Checkpoint B (after refining)
- The outline sits **on the line** (or just a hair outside).
- Both views still match without twisting/leaning.

## 6) Detail phase (only after the form is correct)
- Now add small features (eyes, grooves, textures).
- Don’t add details too early — you’ll often carve them away while correcting the form.

## 7) Safety & common mistakes
- Carve **away from your holding hand**; use small controlled cuts.
- Common mistake: finishing one face completely, then destroying it while shaping the next.
  - Fix: **alternate faces frequently**.

## 8) Finish
- Light sanding (e.g., 240 → 400 grit).
- Optional oil/wax for a smooth finish.
"""
