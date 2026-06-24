# CodeClaw

An autonomous, Claude-Code-style software-engineering agent that runs entirely
on your machine, powered by a local [Ollama](https://ollama.com) model.

CodeClaw inspects your codebase, plans changes, edits files, runs shell
commands, and uses git — all driven by an LLM loop with tool calling.
Destructive actions are gated behind explicit human approval by default.

## Why

- **Local & private.** Your code and your prompts never leave the box.
- **Capable coder model.** Default is `qwen2.5-coder:32b` (32K context, native
  tool-calling, Q4_K_M). Any other Ollama model with the `tools` capability
  works.
- **Inspect-first.** The agent reads the repo before changing anything.
- **Safe by default.** `write_file`, `edit_file`, `exec`, and `git_commit`
  require confirmation. There is a `--auto-approve` flag for trusted
  environments; there is no flag to disable the safety net entirely.

## Install

```bash
cd /workspace/ollamacode
python3 -m venv .venv
source .venv/bin/activate        # bash / zsh
# . .venv/bin/activate           # POSIX sh (dash)
pip install -e ".[dev]"
cp .env.example .env              # tweak as needed
```

If `source` isn't available, use the dot form (`. .venv/bin/activate`).
If the venv's `activate` script is missing, copy it from the system
template and substitute the real path:

```bash
cp /usr/lib/python3.11/venv/scripts/common/activate .venv/bin/activate
sed -i "s|__VENV_DIR__|$(pwd)/.venv|g; s|__VENV_BIN_NAME__|bin|g" .venv/bin/activate
```

(You can also skip activation entirely and invoke the binaries directly:
`.venv/bin/codeclaw`, `.venv/bin/pytest`, etc.)

Make sure Ollama is running and the configured model is pulled:

```bash
ollama serve &
ollama pull qwen2.5-coder:32b
```

## Quick start

```bash
# Open the interactive console UI. Pick a model, then type objectives.
codeclaw

# Sanity-check the connection and installed models.
codeclaw check

# One-shot: give the agent an objective and walk away.
codeclaw "add a Makefile that runs pytest"
```

## Tools

| Tool          | Read/Write | Approval required |
|---------------|-----------:|:-----------------:|
| `read_file`   | read       | no                |
| `list_dir`    | read       | no                |
| `grep`        | read       | no                |
| `git_status`  | read       | no                |
| `git_diff`    | read       | no                |
| `git_log`     | read       | no                |
| `write_file`  | write      | **yes**           |
| `edit_file`   | write      | **yes**           |
| `exec`        | side-effect| **yes**           |
| `git_commit`  | write      | **yes**           |

## How it works

```
           ┌─────────────────────────┐
user ──▶   │   system + AGENTS.md    │
objective  │   + MEMORY.md           │
           │   + tool schemas        │
           └────────────┬────────────┘
                        ▼
                ┌───────────────┐
                │  Ollama chat  │◀────────┐
                │   (qwen...)   │         │
                └───────┬───────┘         │
                        │                 │
              tool_calls?                 │
                │          │              │
               yes         no             │
                ▼          ▼              │
        ┌──────────┐   final answer       │
        │ approval │───────────▶  done     │
        │   gate   │                      │
        └────┬─────┘                      │
             ▼                            │
       ┌───────────┐    tool result       │
       │  execute  │──────────────────────┘
       │   tool    │
       └───────────┘
```

Each turn the model is sent the full conversation; the agent loop trims the
middle when the prompt approaches the model's context window. The loop
terminates when the model returns a plain text reply (interpreted as a
final report) or when `CODECLAW_MAX_STEPS` is hit.

## Project memory

Drop `AGENTS.md` and/or `MEMORY.md` in your project root and CodeClaw will
load them into the system context at the start of every run. Use them for
project conventions, prior decisions, or "things to remember."

## Configuration

All settings come from environment variables (or `.env`):

| Variable                    | Default                    | Purpose                                  |
|-----------------------------|----------------------------|------------------------------------------|
| `OLLAMA_HOST`               | `http://127.0.0.1:11434`   | Ollama server URL                        |
| `CODECLAW_MODEL`            | `qwen2.5-coder:32b`        | Model name (must support `tools`)        |
| `CODECLAW_MAX_STEPS`        | `40`                       | Hard cap on agent steps                  |
| `CODECLAW_CONTEXT_TOKENS`   | `24000`                    | Sliding-window budget for prompt         |
| `CODECLAW_TEMPERATURE`      | `0.2`                      | Sampling temperature                     |
| `CODECLAW_PROJECT_DIR`      | `.`                        | Working directory for tools              |

CLI flags override env vars: `--model`, `--select-model`, `--project-dir`,
`--max-steps`, `--temperature`, `--auto-approve`, `--non-interactive`.

## Testing

```bash
pytest -q
```

The test suite runs without Ollama. The agent tests use a scripted
fake client; the HTTP tests use `httpx.MockTransport`.

## Development

```bash
ruff check src tests
```

## License

MIT.

## Standalone binary (no Python required on target)

If you just want to drop `codeclaw` onto a machine without installing Python,
build a single-file executable with [PyInstaller](https://pyinstaller.org/):

```bash
# One-time setup (Linux/macOS):
cd /workspace/ollamacode
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"      # installs pyinstaller via the [dev] extra

# Build:
pyinstaller --noconfirm codeclaw.spec

# Result:
ls -lh dist/codeclaw         # single ~30 MB executable
```

Run it on the target machine (any modern Linux x86_64, macOS 12+, or
Windows 10+):

```bash
./codeclaw --version
./codeclaw --check
./codeclaw "add a Makefile that runs pytest"
```

### Cross-platform builds

PyInstaller produces a binary for the **host platform** — to ship to another
OS, build on that OS (or use a CI matrix). A minimal GitHub Actions snippet:

```yaml
jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: pyinstaller --noconfirm codeclaw.spec
      - uses: actions/upload-artifact@v4
        with:
          name: codeclaw-${{ matrix.os }}
          path: dist/codeclaw*
```

### Files involved

| File              | Purpose                                                      |
|-------------------|--------------------------------------------------------------|
| `codeclaw.spec`   | PyInstaller build recipe (entry, datas, hidden imports).     |
| `build_entry.py`  | Shim that bootstraps the `codeclaw` package for the frozen binary. |
| `dist/codeclaw`   | Output binary. Rename / rebrand freely.                       |

### Notes & caveats

- **First run is slow.** The binary unpacks itself into a temp dir; expect
  ~1–2 s startup overhead. Subsequent invocations are fast because the OS
  caches the temp dir.
- **Antivirus false positives.** Some AV engines flag PyInstaller binaries
  as suspicious. Sign the binary (`codesign` on macOS, signtool on Windows)
  for production releases.
- **Static analysis limits.** If a runtime `ImportError` appears for a
  module, add it to `hiddenimports` in `codeclaw.spec` and rebuild.
- **Source files only.** `.env`, `AGENTS.md`, `MEMORY.md` are read from the
  **current working directory** at runtime — they are not bundled. The
  binary stays self-contained otherwise.
