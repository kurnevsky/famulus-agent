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
| `--mcp-config` | `FA_MCP_CONFIG` | XDG search path | Read MCP servers from this file instead |
| `--no-mcp` | | | Start no MCP servers this session |
| `--tools` | `FA_TOOLS` | all of them | Offer the model only these tools, by name |
| `--no-tools` | `FA_NO_TOOLS` | | Keep these tools from the model, by name |

The agent works in the current directory. If `AGENTS.md` (or `CLAUDE.md`)
exists there, it is appended to the system prompt.

## Which tools

`--tools` and `--no-tools` name what the model is offered, out of the four
built-in ones and whatever the MCP servers brought — one list, since the model
is offered them as one:

```sh
# Read-only: it can look and answer, and change nothing
fa -m gpt-5.2 --no-tools write,edit,bash

# Only these, whatever else is there
fa -m gpt-5.2 --tools read,weather
```

An allow-list is the first word and a deny-list the last, so a tool named in
both is refused — the narrower intent wins, which is the safe way round for a
list whose point is usually to keep something away from the model. Saying
nothing is everything, which is not the same as allowing everything by name: a
tool that arrives later is kept by the one and not by the other. A name nothing
answers to is said in the transcript, since a typo in a list like this is a
tool quietly left in or out.

Tools are all registered either way; the list is what each request advertises,
and rig refuses a call to anything left out of it. The system prompt follows:
what it says about `bash`, `edit` and `write` is dropped when they are not
this session's, because a model told about a tool and then refused it tries
anyway and reports being refused, instead of using what it does have. A
`--system-prompt` of your own is left alone.

## MCP

Tools from [MCP](https://modelcontextprotocol.io) servers sit beside the four
built-in ones, and the transcript draws them the same way — rig speaks the
protocol, through `rmcp`, so a server's tool is a tool like any other.

Both halves of such a call are read as the JSON they are: the arguments on the
call's own line, highlighted the way a command is, and an answer with a shape
laid out a field to a line rather than left as the one long line it arrived as.
Only for these tools — what `bash` printed is the command's own to lay out, and
a file `read` holds should be seen the way it is written, so neither is
re-indented for being valid JSON.

Servers are declared in `mcp.toml`, a table each, under the name its tools will
be called by:

```toml
# ~/.config/fa/mcp.toml

[fetch]
command = "uvx mcp-server-fetch"

[files]
command = "mcp-server-files --root ~/src"
env.TOKEN = "…"

[docs]
url = "https://example.com/mcp"
headers.Authorization = "Bearer …"
timeout = 60
except = ["delete_page"]
```

A server is a `command` to run, spoken to over its own stdin and stdout, or a
`url` to call, spoken to over streamable HTTP.

| Key | Meaning |
|-----|---------|
| `command` | A line of shell, run the way the `bash` tool runs one — quoting, `~` and `$HOME` all work |
| `env` | Added to the environment that command inherits |
| `url` | Endpoint of a server that speaks streamable HTTP |
| `headers` | Sent with every request to it, which is where a token goes |
| `timeout` | Seconds one of this server's tools may take, `0` to wait forever (default 300) |
| `tools` | Take only these of the tools it offers |
| `except` | Take everything but these |

`tools` and `except` are the same idea as `--tools` and `--no-tools`, for one
server: a server with thirty tools can be cut to the two worth having without
naming every tool of every other server.

The file is fa's own, so it reads the way the rest of fa does — a command is
the line you would type, not an argv — and a key it does not know is an error
naming the line it is on, not something to read past: a misspelled `comand`
that went quietly would be a server that never came up for no stated reason.
There is no `disabled`; the file has comments.

Where it lives is the XDG search path and nothing else — `$XDG_CONFIG_HOME/fa/mcp.toml`
(`~/.config/fa/mcp.toml` by default), then each of `$XDG_CONFIG_DIRS`, with the
nearer file winning the names two of them share. There is no dotfile in a home
directory and none beside the project: a file in the working directory would be
a file whose name depends on where fa was started from.

Servers come up before the terminal does, and what happened is the first thing
the transcript says: which server offered how many tools, and what went wrong
with the rest. The footer keeps a count of both — `2 mcp, 14 tools` — since a
session with servers is a session with more than the four tools fa was built
with. A session with no servers says nothing about MCP anywhere.

A server that fails, or takes more than 20 seconds to say what it offers, is a
note rather than a failure — the session still has its own four tools, which
beats refusing to start. A tool named like one of those four is
left alone: the model is told about `read`, `write`, `edit` and `bash` in the
system prompt, and cannot say which of two it meant.

MCP is a feature, on by default:

```sh
# No MCP support at all
cargo build --release --no-default-features --features languages
```

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
tail), the moves `/tree` makes, and renames. A run only hands back its
transcript when it reaches the end, so an aborted one keeps its turns as they
happen: the prompt, every call and result it got through, and the half-written
answer it was on. The files its tools changed stay changed either way, and the
conversation should not be the only thing that forgets. Calls it was stopped in the middle of are
answered as interrupted — matched by the id each result carries, so every one
of them is answered however many were in flight, since a call left hanging is
a transcript no provider will take back. Because entries name their parent, one file holds every
branch the conversation took, not only the one it is on.

- `src/session.rs` – store, session file, listing, and replay. Tool results
  travel in a message of their own, after the one that asked for them, so
  rebuilding the transcript pairs each with the call it answers rather than
  taking the history as it comes — a turn that ran two commands at once would
  otherwise read as both commands and then both outputs. A message
  record also notes which of the tool calls it answers came back an error:
  nothing in a tool result says so, and without it a reopened session would
  draw every command as though it had worked.

## Continuing

`/continue` runs the model again without adding a user message: the last
message of the history is re-sent as the prompt of the request, so the model
sees exactly the conversation it already had. Use it to pick the loop back up
after `Esc`, after a `/compact`, or on a session resumed with `-c`. An
unanswered user message gets answered, and a half-written answer is continued
from where it stopped. Nothing is written to the session twice: the resumed
message is dropped from the turn's result.

## Typing while it works

`Enter` during a run queues the message rather than dropping it, and the run
stops for it at its next turn: the tool calls it was in the middle of are
answered, and everything it got through is kept, exactly as an abort keeps it
— then the waiting message is sent over that history as a run of its own. So
the model reads it at the first point it could have, not after it has finished
answering a question you have already moved past. A run is never cut before
its first model call, where it has done nothing yet.

A message waiting its turn is drawn at the bottom of the transcript, under
whatever the run is still saying, because the bottom is where it will be sent
from. `Alt+↑` takes the last one back out of the queue and into the input box,
to be fixed and sent again — from an empty box only, where it cannot land on
top of something half-typed. `Esc` stops the run and what was waiting behind
it — it stays in the transcript as `Not sent`, since it was typed.

## Going back

`/tree` — or `Esc` twice on an empty input box, which is pi's shortcut and its
default action — lists every point the conversation can move to and takes you
to the one you pick. Not just your own prompts: an answer, or a tool result
in the middle of a turn, so you can go back to between two tool calls and carry
on from there with `/continue`. Following pi:

- Pick a **prompt** and it is taken back out of the history and returned to the
  input box, to be edited and asked again.
- Pick **anything else** and it is kept as the conversation's new end; the
  input box is left alone.

The one thing that is never a point is an assistant turn that called tools.
Stopping there would leave a call with no result behind it, which is a
transcript no provider will accept — the tool results that answer it are the
point just after, and the message before it the point just before.

Nothing is deleted. Going back moves where the conversation ends; what it said
down the path you left stays as a branch of its own, and the list shows it —
indented under the point the two ways part, with the path you are on first. So
you can go back, try something else, and later walk into the answer you
abandoned:

```
╭ Tree — ↑↓ PgUp/PgDn select · Enter go there · Esc cancel ──────────────────╮
│  ❯ what does main.rs do?                                       0 messages  │
│  ⚙ read                                                        3 messages  │
│›   Actually it prints hi and exits 0.                                here  │
│    It prints hi.                                               4 messages  │
```

`/fork` is the same list narrowed to your prompts, and it branches into a file
instead of within one: the conversation up to that point is carried into a
**new session**, the prompt goes back in the input box, and the session you
came from is left on disk exactly as it was, still its own thing to resume. The
fork's header names its parent. Only the one path is copied — the branches
beside it stay with the session being left, and the fork starts as a straight
line. Use `/tree` to take this conversation a different way, `/fork` to start
another one beside it.

There is no branch summary — going back is a plain move, with nothing
summarized and nothing lost.

The session file stays append-only and is a tree rather than a list: every
message record names itself and its parent, going back writes the entry the
conversation moved to, and replaying the file rebuilds the same branches.

## Watching a tool call being written

A tool call appears as the model writes it, rather than all at once when it is
run: the arguments stream in a few characters at a time, and the line grows
with them under a cursor.

```
⚙ bash cargo te▌
⚙ bash cargo test --all --r▌
⚙ bash cargo test --all --release -- --nocapture
```

`write` and `edit` carry a file's worth of text in their arguments, which is
the slow part of such a call, so that text is shown arriving too — under the
line rather than on it, and for an edit marked the way the diff it becomes
will mark it. A tool offers the model what it acts on first (`path`, or
`command`), since arguments tend to be written in the order they are offered
and a file's name is worth having before its contents:

```
⚙ write new.rs              ⚙ edit existing.rs
  │ fn main() {               -     println!("old");
  │     let x = 1;            +     println!("new");
  │     println!▌             +     done();▌
```

Everything in the transcript is bound to its tool call by the call's own id,
not by where it happens to sit: the line a command's live output goes under,
the result that replaces it, and the pairing rebuilt on reload. Tools run one
at a time (rig's `tool_concurrency` defaults to 1 and fa leaves it there), but
were that raised, two commands in flight would still each keep their own
output.

Every tool shows a preview of what it has to say rather than all of it, the
way a thinking block does, with a line saying how much is folded away;
`Ctrl+O` swaps between the preview and the whole thing, including for a call
still being written, so a long file can be watched arriving in full or kept to
its last lines. A command keeps the
end of its output and a file the start of its contents, since that is the end
that matters in each.

`edit` shows what it changed as a diff. `write` shows the file it wrote, as the
file it is: highlighted by the language its name gives, with none of a diff's
pluses and none of its green, because a write did not change lines, it put them
there. Either way what the tool did is on screen, where "Successfully wrote to
it" would only repeat the line above. The write costs the session nothing to
show: the content is already in the call that asked for it, so the transcript
keeps the sentence and the screen draws the file.

A finished tool's output carries a stripe down its side saying how it went —
green when it worked, red when it did not — in place of the `│` gutter it has
while it is still running, which is not yet a verdict. The same for every
tool, so a glance down the transcript reads as pass or fail without the text
being recoloured, which the text has its own uses for. The stripe is a
background rather than coloured text, and one cell wide: that is where the
terminal's own red and green are right, saturated enough to read at a glance
and carrying no text to be legible against, so it needs no colour of fa's own
and follows whatever theme the terminal is wearing. A command's own line is
highlighted as bash, by the same tree-sitter grammar the code blocks use.

All of it reads the same on a session reopened later, which takes the session
remembering two things a transcript does not carry: whether each tool result
was an error, and the diff an `edit` produced.

The line itself shows only what the finished line leads with — the command, or
the path — so it and the entry it becomes read the same and nothing moves when
the call starts running. Until the arguments parse they are read straight out
of the half-written JSON; after that they are read as arguments, which is what
keeps an edit's two halves together once a serializer has sorted its keys. A
call the model never finished writing leaves nothing behind.

## Compaction

After each turn the agent compares the provider-reported context size of the
last request with `context_window - reserve_tokens`. When it is exceeded (or
on `/compact`), the older part of the history is rendered as a transcript and
summarized by the same model into pi's structured checkpoint format (goal,
progress, decisions, next steps, critical context). The summary replaces those
messages as a single user message; the most recent turns, about
`keep_recent_tokens` worth, stay verbatim as entries of their own, so `/tree`
can still go back to any of them; tool call/result pairs are never split. A
later compaction updates the existing summary instead of nesting it. The footer
shows the current context usage as `ctx N%`.

## Keys

| Key | Action |
|-----|--------|
| `Enter` | Send (queued if a run is in progress) |
| `Alt+↑` | Take the last queued message back for editing (empty input) |
| `/` | Command popup: type to fuzzy-filter, `↑`/`↓` move, `Tab`/`Enter` complete, `Esc` dismiss |
| `Alt+Enter`, `Ctrl+J`, `Shift+Enter`* | Newline |
| `Esc` | Abort the current run |
| `Esc` `Esc` | Open `/tree` (empty input, within half a second) |
| `PageUp` / `PageDown`, mouse wheel | Scroll transcript |
| `Ctrl+C` | Abort if running, otherwise quit |
| `Ctrl+D` | Quit (empty input) |
| `Ctrl+T` | Thinking in full, or only its last lines |
| `Ctrl+O` | Tool output in full, or only its preview |
| `/compact` | Summarize older history now |
| `/continue` | Run the model again with no new message |
| `/new` | Start a new session |
| `/resume` | Pick a saved session to resume |
| `/tree` | Move to another point in this session, on any branch |
| `/fork` | Branch a new session from an earlier prompt |
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
  `AgentEvent`s to the UI over a channel. Tool-call argument fragments are
  accumulated per call and sent whole, so the UI has nothing to reassemble.
  Tool calls and results are reported
  through an `AgentHook`, which carries the call id and an error flag.
  Nothing in rig hands a tool its own call id and the hook that knows it runs
  before the body rather than around it, so for `bash` the hook rewrites the
  arguments to carry it — the one channel between the two. It is stripped from
  the schema, so the model is never asked for it and never sends it. On the
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
  png, gif, webp, bmp) as attachments and decodes text leniently. `bash`
  follows pi's executor: stdout and stderr interleaved, live output streamed
  to the UI, the last 2000 lines / 50 KB kept with the full output spilled to
  a temp file, the whole process group killed on timeout or abort, and pi's
  exit-code and timeout messages. The two streams share one pipe rather than
  getting one each, so the order they are read in is the order the command
  wrote them: two pipes keep no record of which came first, and a command
  that says something on each would be reported whichever way round they
  happened to be polled. `edit` is a port of pi's edit engine
  (`src/edit.rs`): BOM and CRLF preserved, exact match first
  and then pi's fuzzy normalization (trailing whitespace, smart quotes, dashes,
  Unicode spaces) with untouched lines kept verbatim, pi's error messages, and
  a pi-style numbered diff shown in the UI instead of the result text.
- `src/images.rs` – image preparation for `read`, following pi: magic-byte
  detection, conversion of gif/webp/bmp to PNG, and resizing to fit 2000x2000
  pixels and 4.5 MB of base64 (PNG first, then JPEG at decreasing quality).
- `src/highlight.rs` – tree-sitter syntax highlighting for code blocks, a
  command's own line, and the JSON either half of an MCP call is: the
  grammar registry (one cargo feature per language), the capture-name theme,
  and the per-line spans the markdown renderer draws. Grammars ship their own
  highlight queries and are used as they come, except Haskell, whose query is
  written for neovim's pattern precedence and needs one of our own.
- `src/mcp.rs` – finding MCP servers and starting them: the XDG search for
  `mcp.toml`, the table-per-server file it reads, and handing each
  server's tools to the agent. Rig speaks the protocol; this only decides who
  to speak to. Holding the result is what keeps the servers running, so it
  lives as long as the program does.
- `src/ui.rs` – ratatui app: transcript, input, footer.
- `tests/e2e.rs` – the binary driven through tmux against a mock provider: a
  call written token by token and its output landing under it, pass and fail
  as the stripe beside each, an aborted run keeping its work and carrying on
  from it, a reopened session reading exactly as it did before, a session held
  to some of its tools, a message typed
  mid-run waiting at the bottom and going at the next turn, going back into a
  branch the conversation left, forking into a session of its own, a compacted
  conversation reaching the model as its summary, and a tool from a real MCP
  server being offered, called and drawn like any other.

## Testing

```sh
cargo test
```

`tests/e2e.rs` runs the real binary under tmux against a scripted mock of an
OpenAI-compatible server, types at it, and reads the screen back with
`capture-pane`. tmux does the terminal emulation, so what comes back is what a
person would have seen — wrapping, overwriting, colours and all, which is
where most of this program's behaviour lives and none of which a unit test can
reach. The mock decides which turn to play from the request rather than
counting, so a resumed session picks up where the last one left off. Without
tmux installed these skip rather than fail.

`src/agent.rs` also has an end-to-end test against a mock server when
`FA_TEST_BASE_URL` is set; without it the test is skipped.

```sh
FA_TEST_BASE_URL=http://127.0.0.1:8123/v1 cargo test
```
