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

### Windows (PowerShell)

```powershell
cd C:\path\to\codeclaw
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
ollama pull qwen2.5-coder:14b
codeclaw check
codeclaw
```

Install a global launcher: `codeclaw install` (adds `%LOCALAPPDATA%\CodeClaw\bin` to PATH).

### Linux / macOS

```bash
cd codeclaw
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

To install a small `~/.local/bin/codeclaw` launcher from the current Python
environment:

```bash
codeclaw install
```

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

Inside the interactive console, commands start with `/`:

```text
/help            show available slash commands
/status          show current model, project, approval mode, and session
/init            create AGENTS.md and .codeclaw/settings.json defaults
/config          show project configuration defaults
/set KEY VALUE   set defaults like model, host, max_steps, temperature
/plan            toggle read-only planning mode
/compact         compact saved session context
/todo            show the current session task list
/sessions        list saved sessions in .codeclaw/sessions
/current         show current session details
/resume ID       resume a saved session
/memory          show loaded AGENTS.md and MEMORY.md context
/hooks           show configured project lifecycle hooks
/hook-example    write example hook templates
/tools           list available tools
/permissions     show approval rules for tools
/diff            show current git diff summary
/changes         show git status and diff summary
/checkpoint NAME save a local project snapshot
/checkpoints     list saved local snapshots
/restore ID      restore a saved local snapshot
/models          choose from installed Ollama models
/model qwen3:14b switch directly to a model
/reset           start a fresh saved session
/quit            exit
```

Resume the latest saved session from the current project:

```bash
codeclaw continue
```

Saved sessions store the **full conversation history** (not just summaries), so
`/resume` and `continue` pick up where you left off. Use `/compact` to shrink
older turn metadata when sessions grow large.

Run logs are written to `.codeclaw/runs/<session-id>/*.jsonl` for debugging.

### OpenAI-compatible backend

Point CodeClaw at any OpenAI-compatible server (vLLM, LM Studio, etc.):

```bash
export CODECLAW_BACKEND=openai
export CODECLAW_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export CODECLAW_MODEL=your-model
codeclaw
```

### Exec policy

Configure allow/deny rules in `.codeclaw/settings.json`:

```json
{
  "exec": {
    "deny_patterns": ["rm -rf", "sudo"],
    "allow_without_prompt": ["pytest", "ruff check"]
  }
}
```

### Custom plugin tools

Drop Python files in `.codeclaw/tools/` that define `Tool` subclasses. They are
loaded automatically at startup.

Project hooks can be configured in `.codeclaw/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      { "type": "command", "command": "python scripts/check_tool.py" }
    ],
    "UserPromptSubmit": [
      { "type": "command", "command": "python scripts/check_prompt.py" }
    ]
  }
}
```

Hook commands run from the project directory and receive JSON on stdin. A
non-zero `PreToolUse` exit blocks the tool call; a non-zero
`UserPromptSubmit` exit blocks the prompt.

The interactive prompt uses a multiline editor when `prompt-toolkit` is
installed. Approval prompts support allow once (`y`), allow always for that
tool (`a`), and deny (`n`).

## Tools

| Tool           | Read/Write | Approval required |
|----------------|-----------:|:-----------------:|
| `read_file`    | read       | no                |
| `list_dir`     | read       | no                |
| `glob`         | read       | no                |
| `grep`         | read       | no                |
| `update_todo`  | read       | no                |
| `git_status`   | read       | no                |
| `git_diff`     | read       | no                |
| `git_log`      | read       | no                |
| `write_file`   | write      | **yes**           |
| `edit_file`    | write      | **yes**           |
| `apply_patch`  | write      | **yes**           |
| `exec`         | side-effect| **yes**\*         |
| `run_tests`    | side-effect| **yes**           |
| `git_branch`   | write      | **yes**           |
| `git_stash`    | write      | **yes**           |
| `git_commit`   | write      | **yes**           |
| `web_fetch`    | network    | **yes**           |

\* `exec` can be allowlisted in `.codeclaw/settings.json` under `exec.allow_without_prompt`.

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
| `CODECLAW_BACKEND`          | `ollama`                   | `ollama` or `openai` (compatible API)    |
| `OLLAMA_HOST`               | `http://127.0.0.1:11434`   | Ollama server URL                        |
| `CODECLAW_OPENAI_BASE_URL`  | `http://127.0.0.1:8000/v1` | OpenAI-compatible base URL               |
| `CODECLAW_OPENAI_API_KEY`   | (empty)                    | API key for OpenAI-compatible backend    |
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
cd codeclaw
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