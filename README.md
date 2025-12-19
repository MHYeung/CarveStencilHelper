# CarveStencil Helper

A small helper application for generating carving guides and processing images for CNC or laser carving workflows.

## Overview

This repository contains a lightweight GUI and processing pipeline to produce carving guides and save job outputs into the `jobs/` folder.

## Files

- `gui_app.py` — GUI application entrypoint
- `guide.py` — carving guide generation utilities
- `pipeline.py` — processing pipeline and orchestration
- `utils_net.py` — network and miscellaneous utilities
- `jobs/` — job output folders and generated artifacts

## Requirements

- Python 3.10 or newer
- Optional: Git for version control

## Installation

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1    # PowerShell
```

2. Install dependencies (if you maintain a `requirements.txt`):

```powershell
pip install -r requirements.txt
```

Alternatively, install packages you need as you add features.

## Running the app

Run the GUI from the project root:

```powershell
python gui_app.py
```

The GUI will guide you through loading images and exporting carving guides. Outputs are written to the `jobs/` directory with timestamped subfolders.

## Usage notes

- Place source images in a known folder and select them from the GUI.
- The `jobs/` folder stores generated guides and logs. Don't delete it if you want reproducibility.
- If you add heavy dependencies (OpenCV, Pillow, numpy), add them to `requirements.txt`.

## Development

- Run the GUI locally while iterating on `guide.py` or `pipeline.py`.
- Consider adding unit tests and a `requirements.txt` to lock dependencies.

## Adding a `requirements.txt`

If you'd like, I can generate a `requirements.txt` from your virtual environment. This helps others reproduce your environment.

## Contributing

- Open issues or PRs in your repo.
- Document new dependencies and add usage examples where helpful.

## License

Add a `LICENSE` file or a short license block here.

## Contact

For questions, add your contact or project maintainer info here.
