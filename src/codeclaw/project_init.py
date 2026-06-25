"""Auto-discover project conventions for /init."""
from __future__ import annotations

from pathlib import Path


def discover_agents_md(project_dir: str | Path) -> str:
    root = Path(project_dir).resolve()
    lines = ["# CodeClaw Project Notes", ""]
    sections: list[str] = []

    if (root / "pyproject.toml").exists():
        sections.append(_python_section(root))
    if (root / "package.json").exists():
        sections.append(_node_section(root))
    if (root / "Cargo.toml").exists():
        sections.append(_rust_section(root))
    if (root / "go.mod").exists():
        sections.append(_go_section(root))
    if (root / "Makefile").exists():
        sections.append(_makefile_section(root))

    if sections:
        lines.extend(sections)
    else:
        lines.extend([
            "## Commands",
            "- Describe build, test, and lint commands here.",
            "",
            "## Conventions",
            "- Add project conventions and safety notes here.",
        ])
    lines.extend([
        "",
        "## Safety",
        "- Destructive shell commands require explicit approval.",
        "- Never commit secrets or credentials.",
    ])
    return "\n".join(lines) + "\n"


def _python_section(root: Path) -> str:
    test = "python -m pytest -q"
    lint = "ruff check ." if (root / "pyproject.toml").exists() else ""
    install = 'pip install -e ".[dev]"' if (root / "pyproject.toml").exists() else "pip install -r requirements.txt"
    body = [
        "## Python",
        f"- Install: `{install}`",
        f"- Test: `{test}`",
    ]
    if lint:
        body.append(f"- Lint: `{lint}`")
    return "\n".join(body)


def _node_section(root: Path) -> str:
    return "\n".join([
        "## Node.js",
        "- Install: `npm install`",
        "- Test: `npm test`",
        "- Lint: `npm run lint` (if defined)",
    ])


def _rust_section(root: Path) -> str:
    return "\n".join([
        "## Rust",
        "- Build: `cargo build`",
        "- Test: `cargo test`",
        "- Lint: `cargo clippy`",
    ])


def _go_section(root: Path) -> str:
    return "\n".join([
        "## Go",
        "- Test: `go test ./...`",
        "- Build: `go build ./...`",
    ])


def _makefile_section(root: Path) -> str:
    text = (root / "Makefile").read_text(encoding="utf-8", errors="replace")
    targets = []
    for name in ("test", "lint", "build", "check"):
        if f"{name}:" in text:
            targets.append(f"- `{name}`: `make {name}`")
    if not targets:
        return "## Makefile\n- See Makefile for available targets."
    return "## Makefile\n" + "\n".join(targets)
