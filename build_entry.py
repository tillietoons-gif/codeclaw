"""Frozen-entry shim for PyInstaller.

PyInstaller runs the entry script as a top-level module, which breaks the
relative imports inside `codeclaw.cli` (e.g. `from . import __version__`).
This shim forces the `codeclaw` package to be importable on `sys.path` and
then calls its `main()` function.

Build with:
    pyinstaller --noconfirm codeclaw.spec

Or, if you change the entry:
    pyinstaller --noconfirm --onefile --name codeclaw --add-data \
        "src/codeclaw/prompts:codeclaw/prompts" \
        --hidden-import codeclaw --paths src build_entry.py
"""
from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> None:
    """Ensure `codeclaw` is importable when frozen or run from source."""
    here = Path(__file__).resolve().parent
    src = here / "src"
    if src.is_dir():
        sys.path.insert(0, str(src))
    # PyInstaller unpacks _MEIPASS at runtime; nothing else to do.
    # When running as a normal script (`python build_entry.py`), the
    # editable install / PYTHONPATH is responsible for providing `codeclaw`.


def main() -> int:
    _bootstrap()
    # Import after sys.path is set up.
    from codeclaw.cli import main as codeclaw_main

    return codeclaw_main()


if __name__ == "__main__":
    sys.exit(main())
