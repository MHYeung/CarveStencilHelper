from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import subprocess, sys
import trimesh

@dataclass
class ReconResult:
    mesh: trimesh.Trimesh
    mesh_path: Path

class TripoSRReconstructor:
    def __init__(self, repo_dir: str = "third_party/TripoSR"):
        self.repo = Path(repo_dir).resolve()
        self.run_py = self.repo / "run.py"
        if not self.run_py.exists():
            raise FileNotFoundError(f"TripoSR run.py not found: {self.run_py}")

    def reconstruct(self, input_image_path: str, out_dir: str) -> ReconResult:
        out_dir = Path(out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(self.run_py),
            str(Path(input_image_path).resolve()),
            "--output-dir", str(out_dir),
        ]
        subprocess.run(cmd, cwd=str(self.repo), check=True)

        # Find a mesh file in out_dir/**/
        candidates = []
        for ext in ("*.obj", "*.glb", "*.ply", "*.stl"):
            candidates += list(out_dir.rglob(ext))
        if not candidates:
            raise RuntimeError(f"No mesh produced in {out_dir}")

        mesh_path = max(candidates, key=lambda p: p.stat().st_mtime)
        mesh = trimesh.load(mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(mesh.dump())
        return ReconResult(mesh=mesh, mesh_path=mesh_path)
