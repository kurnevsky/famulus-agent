# fa

A minimal terminal coding agent in Rust, in the spirit of [pi](https://pi.dev):
one streaming transcript, one input box, four tools (`read`, `write`, `edit`,
`bash`). Built on [rig](https://crates.io/crates/rig-core) for the LLM loop and
[ratatui](https://ratatui.rs) + [ratatui-textarea](https://crates.io/crates/ratatui-textarea)
for the interface. Talks to any OpenAI-compatible chat completions endpoint,
or to Google Gemini.

## Usage

```sh
cargo build --release

# Local server (llama.cpp, vLLM, Ollama, LM Studio, ...)
./target/release/fa --base-url http://localhost:8080/v1 --model qwen2.5-coder

# Hosted OpenAI-compatible API
OPENAI_BASE_URL=https://api.example.com/v1 OPENAI_API_KEY=sk-... ./target/release/fa -m gpt-5.2

# Google Gemini
GEMINI_API_KEY=... ./target/release/fa --provider gemini -m gemini-2.5-pro
```

| Flag | Env | Default | Meaning |
|------|-----|---------|---------|
| `--provider` | `FA_PROVIDER` | `openai` | `openai` (chat completions and compatible servers) or `gemini` |
| `--base-url` | `OPENAI_BASE_URL` | provider default | Endpoint root, e.g. `http://localhost:8080/v1` |
| `--api-key` | `FA_API_KEY` | `OPENAI_API_KEY` / `GEMINI_API_KEY`, else `none` | API key (any value for servers without auth) |
| `-m, --model` | `FA_MODEL` | required | Model name |
| `--system-prompt` | `FA_SYSTEM_PROMPT` | built-in | Replace the system prompt |
| `--max-turns` | | `50` | Max model calls per user message |
| `--context-window` | `FA_CONTEXT_WINDOW` | `128000` | Model context size in tokens |
| `--reserve-tokens` | | `16384` | Compact when fewer tokens than this remain |
| `--keep-recent-tokens` | | `20000` | Recent tokens kept verbatim when compacting |
| `--no-compaction` | | | Disable automatic compaction |
| `--no-vision` | `FA_NO_VISION` | | Model cannot take images: `read` omits image data and says so |
| `-c, --continue` | | | Continue the most recent session |
| `-r, --resume` | | | Pick a session to resume |
| `--session` | | | Resume a session by file path or id prefix |
| `--no-session` | | | Do not save this session |
| `--sessions-dir` | `FA_SESSIONS_DIR` | `~/.local/share/fa/sessions` | Where session files live (global) |
| `--scrollbar` | `FA_SCROLLBAR` | `auto` | Transcript scrollbar: `auto` (while scrolling), `always`, `hidden` |

The agent works in the current directory. If `AGENTS.md` (or `CLAUDE.md`)
exists there, it is appended to the system prompt.

## Syntax highlighting

Fenced code blocks are highlighted with [tree-sitter](https://tree-sitter.github.io).
The fence's info string picks the grammar, by name or by the extension people
write instead (`rs`, `py`, `yml`, `c++`, `sh`, …), and a language with no
grammar — or a fence with no info string at all — keeps the single colour code
blocks had before. Injections are followed too, so the `<script>` in an HTML
block is highlighted as JavaScript.

Every grammar is a C parser compiled into the binary, so each language is its
own cargo feature. The default build has all of them:

```sh
# Just the ones you care about
cargo build --release --no-default-features --features lang-rust,lang-python

# No tree-sitter at all
cargo build --release --no-default-features
```

`lang-bash`, `lang-c`, `lang-cpp`, `lang-css`, `lang-go`, `lang-haskell`,
`lang-html`, `lang-java`, `lang-javascript`, `lang-json`, `lang-python`,
`lang-rust`, `lang-scala`, `lang-toml`, `lang-typescript` (TypeScript and TSX),
`lang-yaml`.

## Sessions

Like pi, every conversation is saved as an append-only JSONL file and can be
resumed later. Unlike pi, all sessions live in one global directory
(`~/.local/share/fa/sessions` by default, respecting `XDG_DATA_HOME`),
so the picker shows each session's working directory. The file is created on
the first message, so empty sessions leave nothing behind. Records are the
header, each transcript message, compaction checkpoints (summary plus the kept
tail), and renames. Aborted turns keep the prompt and any partial answer.

- `src/session.rs` – store, session file, listing, and replay.

## Continuing

`/continue` runs the model again without adding a user message: the last
message of the history is re-sent as the prompt of the request, so the model
sees exactly the conversation it already had. Use it to pick the loop back up
after `Esc`, after a `/compact`, or on a session resumed with `-c`. An
unanswered user message gets answered, and a half-written answer is continued
from where it stopped. Nothing is written to the session twice: the resumed
message is dropped from the turn's result.

## Compaction

After each turn the agent compares the provider-reported context size of the
last request with `context_window - reserve_tokens`. When it is exceeded (or
on `/compact`), the older part of the history is rendered as a transcript and
summarized by the same model into pi's structured checkpoint format (goal,
progress, decisions, next steps, critical context). The summary replaces those
messages as a single user message; the most recent turns, about
`keep_recent_tokens` worth, stay verbatim, and tool call/result pairs are never
split. A later compaction updates the existing summary instead of nesting it.
The footer shows the current context usage as `ctx N%`.

## Keys

| Key | Action |
|-----|--------|
| `Enter` | Send (queued if a run is in progress) |
| `/` | Command popup: type to fuzzy-filter, `↑`/`↓` move, `Tab`/`Enter` complete, `Esc` dismiss |
| `Alt+Enter`, `Ctrl+J`, `Shift+Enter`* | Newline |
| `Esc` | Abort the current run |
| `PageUp` / `PageDown`, mouse wheel | Scroll transcript |
| `Ctrl+C` | Abort if running, otherwise quit |
| `Ctrl+D` | Quit (empty input) |
| `/compact` | Summarize older history now |
| `/continue` | Run the model again with no new message |
| `/new` | Start a new session |
| `/resume` | Pick a saved session to resume |
| `/name <name>` | Name the current session |
| `/session` | Show session id, file, and stats |
| `/quit` | Quit |

\* in terminals that support the kitty keyboard protocol.

Mouse capture is enabled for wheel scrolling, so selecting text with the
mouse usually requires holding `Shift`.

## Layout

- `src/main.rs` – CLI flags, terminal setup.
- `src/agent.rs` – builds the rig agent (OpenAI completions client, tools,
  system prompt) and runs one streaming turn per user message, forwarding
  `AgentEvent`s to the UI over a channel. Tool calls and results are reported
  through an `AgentHook` so they stay ordered and carry an error flag. On the
  OpenAI path a `CompletionModel` wrapper moves images out of tool results into
  a follow-up user message, since the chat completions API only accepts text
  in tool messages (the same workaround pi's provider uses). Gemini accepts
  images inside function responses, so it uses rig's model directly.
- `src/compaction.rs` – context compaction: trigger rule, cut point, transcript
  serialization, pi's summarization prompts.
- `src/tools.rs` – the four tools, mirroring pi's descriptions, truncation
  limits (2000 lines / 50 KB), continuation notes, and error messages. Paths
  are resolved like pi (`~`, leading `@`, `file://`, Unicode spaces; reads also
  try NFD and curly-apostrophe name variants). `read` returns images (jpg,
  png, gif, webp, bmp) as attachments and decodes text leniently. `bash` follows pi's executor: stdout and stderr interleaved,
  live output streamed to the UI, the last 2000 lines / 50 KB kept with the
  full output spilled to a temp file, the whole process group killed on
  timeout or abort, and pi's exit-code and timeout messages. `edit` is a port of
  pi's edit engine (`src/edit.rs`): BOM and CRLF preserved, exact match first
  and then pi's fuzzy normalization (trailing whitespace, smart quotes, dashes,
  Unicode spaces) with untouched lines kept verbatim, pi's error messages, and
  a pi-style numbered diff shown in the UI instead of the result text.
- `src/images.rs` – image preparation for `read`, following pi: magic-byte
  detection, conversion of gif/webp/bmp to PNG, and resizing to fit 2000x2000
  pixels and 4.5 MB of base64 (PNG first, then JPEG at decreasing quality).
- `src/highlight.rs` – tree-sitter syntax highlighting for code blocks: the
  grammar registry (one cargo feature per language), the capture-name theme,
  and the per-line spans the markdown renderer draws. Grammars ship their own
  highlight queries and are used as they come, except Haskell, whose query is
  written for neovim's pattern precedence and needs one of our own.
- `src/ui.rs` – ratatui app: transcript, input, footer.

## Testing

`src/agent.rs` has an end-to-end test that runs against a mock server when
`FA_TEST_BASE_URL` is set; without it the test is skipped.

```sh
FA_TEST_BASE_URL=http://127.0.0.1:8123/v1 cargo test
```
