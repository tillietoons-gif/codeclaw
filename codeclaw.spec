# PyInstaller spec for CodeClaw.
#
# Build with:
#   pyinstaller codeclaw.spec
# or, from the project root (cleaner):
#   pyinstaller --noconfirm codeclaw.spec
#
# Output: dist/codeclaw   (single-file executable, no Python required on target)

# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files
from pathlib import Path

block_cipher = None

# Bundle the prompts/ markdown file(s) that the agent reads at runtime.
datas = collect_data_files("codeclaw", includes=["prompts/*.md"])

# CodeClaw imports tool classes by name in tools/__init__.py, so the modules
# must be importable after frozen. PyInstaller's static analysis misses a few
# dynamic paths; list them explicitly to be safe.
hiddenimports = [
    "codeclaw",
    "codeclaw.cli",
    "codeclaw.agent",
    "codeclaw.config",
    "codeclaw.memory",
    "codeclaw.ollama",
    "codeclaw.tools",
    "codeclaw.tools.base",
    "codeclaw.tools.filesystem",
    "codeclaw.tools.git",
    "codeclaw.tools.search",
    "codeclaw.tools.shell",
    # third-party (defensive — pyinstaller usually catches these, but rich/prompt
    # import lazily inside functions, which static analysis can miss).
    "rich",
    "rich.console",
    "rich.prompt",
    "rich.markdown",
    "rich.syntax",
    "httpx",
    "httpcore",
    "pydantic",
]

a = Analysis(
    ["build_entry.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Trim a few heavy modules we definitely don't use, to shrink the
        # binary a bit. Safe to remove from the list if something breaks.
        "tkinter",
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "pytest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="codeclaw",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
