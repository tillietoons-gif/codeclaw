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
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env       # tweak as needed
```

Make sure Ollama is running and the configured model is pulled:

```bash
ollama serve &
ollama pull qwen2.5-coder:32b
```

## Quick start

```bash
# Sanity-check the connection and the model.
codeclaw --check

# One-shot: give the agent an objective and walk away.
codeclaw "add a pyproject.toml that depends on httpx and pytest, plus a tests/ folder with one passing test"

# Interactive REPL
codeclaw repl
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

The `exec` tool refuses to run anything matching a configurable
`CODECLAW_DANGEROUS_PATTERNS` list (defaults cover `rm -rf`, `sudo`,
`mkfs`, `dd`, force-pushes, etc.) without explicit approval.

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
| `CODECLAW_DANGEROUS_PATTERNS` | (see file)               | Comma-separated patterns to gate         |
| `CODECLAW_PROJECT_DIR`      | `.`                        | Working directory for tools              |

CLI flags override env vars: `--model`, `--project-dir`, `--max-steps`,
`--temperature`, `--auto-approve`, `--non-interactive`.

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
