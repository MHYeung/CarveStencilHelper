# CarveStencil Helper

Simple helper project for creating carving guides and processing images.

## Project files

- `gui_app.py` — GUI application entrypoint
- `guide.py` — carving guide generation utilities
- `pipeline.py` — processing pipeline
- `utils_net.py` — network / helper utilities
- `jobs/` — output and job folders created by the app

## Requirements

- Python 3.10+
- Recommended: create and use a virtual environment

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt  # if you add one
```

## Run

From the project root run:

```bash
python gui_app.py
```

## Usage

Open the GUI and follow the prompts to create carving guides and save job outputs in the `jobs/` folder.

## Contributing

Feel free to open issues or add PRs. Add a `requirements.txt` if your changes add dependencies.

## License

Add a license file or text here when ready.
