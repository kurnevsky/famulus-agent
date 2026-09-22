# famulus-agent

A minimal terminal coding agent in Rust, in the spirit of [pi](https://pi.dev):
one streaming transcript, one input box, five tools (`read`, `write`, `edit`,
`bash`, `ask`). The crate is `famulus-agent`; the binary it installs — and the
name used for it throughout this file — is `fa`.
Built on [rig](https://crates.io/crates/rig-core) for the LLM loop and
[ratatui](https://ratatui.rs) + [ratatui-textarea](https://crates.io/crates/ratatui-textarea)
for the interface. Talks to any OpenAI-compatible chat completions endpoint,
and to seventeen providers by name — Anthropic, Gemini, OpenRouter, Ollama and
the rest of the table below.

## Usage

```sh
cargo build --release

# Local server (llama.cpp, vLLM, LM Studio, ...)
./target/release/fa --base-url http://localhost:8080/v1 --model qwen2.5-coder

# Hosted OpenAI-compatible API
FA_BASE_URL=https://api.example.com/v1 OPENAI_API_KEY=sk-... ./target/release/fa -m gpt-5.2

# OpenRouter
OPENROUTER_API_KEY=sk-or-... ./target/release/fa --provider openrouter -m anthropic/claude-sonnet-4.5

# Ollama, on http://localhost:11434 unless --base-url says otherwise
./target/release/fa --provider ollama -m qwen2.5-coder

# Google Gemini
GEMINI_API_KEY=... ./target/release/fa --provider gemini -m gemini-2.5-pro

# Anthropic
ANTHROPIC_API_KEY=sk-ant-... ./target/release/fa --provider anthropic -m claude-sonnet-4-5
```

`--provider` picks which API to speak, and with it the endpoint and the key
variable it falls back to:

| `--provider` | Default endpoint | Key from |
|---|---|---|
| `openai` (default) | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `anthropic` | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` |
| `gemini` | Google's generateContent API | `GEMINI_API_KEY` |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `ollama` | `http://localhost:11434` | `OLLAMA_API_KEY`, or none |
| `llamafile` | `http://localhost:8080` | none |
| `cohere` | `https://api.cohere.ai` | `COHERE_API_KEY` |
| `deepseek` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |
| `doubleword` | `https://api.doubleword.ai/v1` | `DOUBLEWORD_API_KEY` |
| `groq` | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |
| `hyperbolic` | `https://api.hyperbolic.xyz` | `HYPERBOLIC_API_KEY` |
| `mira` | `https://api.mira.network` | `MIRA_API_KEY` |
| `mistral` | `https://api.mistral.ai` | `MISTRAL_API_KEY` |
| `perplexity` | `https://api.perplexity.ai` | `PERPLEXITY_API_KEY` |
| `together` | `https://api.together.xyz` | `TOGETHER_API_KEY` |
| `venice` | `https://api.venice.ai/api/v1` | `VENICE_API_KEY` |
| `xai` | `https://api.x.ai` | `XAI_API_KEY` |

Most of these are OpenAI-compatible and reachable through `--provider openai
--base-url ...` as well; naming one buys the default endpoint, the right key
variable, and whatever rig does differently on that wire. Anthropic will not
take a request that names no `max_tokens`, so it is given what the named model
allows — or 2048 for a model rig does not recognise, which is when
`--max-tokens` is worth setting.

| Flag | Env | Default | Meaning |
|------|-----|---------|---------|
| `--provider` | `FA_PROVIDER` | `openai` | API flavour to speak; see the table above |
| `--base-url` | `FA_BASE_URL` | provider default | Endpoint root, e.g. `http://localhost:8080/v1` |
| `--api-key` | `FA_API_KEY` | the provider's own variable, else `none` | API key (any value for servers without auth; Ollama and llamafile need none) |
| `-m, --model` | `FA_MODEL` | required | Model name; `/model` changes it later |
| `--system-prompt` | `FA_SYSTEM_PROMPT` | built-in | Replace the system prompt |
| `--max-tokens` | `FA_MAX_TOKENS` | the provider's own | Cap on what one answer may come to, in tokens |
| `--context-window` | `FA_CONTEXT_WINDOW` | what the provider reports once `/model` has been opened, else `128000` | Model context size in tokens; given here it stands whatever model is chosen |
| `--reserve-tokens` | | `16384` | Compact when fewer tokens than this remain |
| `--keep-recent-tokens` | | `20000` | Recent tokens kept verbatim when compacting |
| `--no-compaction` | | | Disable automatic compaction |
| `--no-turn-summary` | | | One summarization request, never a second for the start of a split turn |
| `--no-vision` | `FA_NO_VISION` | | Model cannot take images: `read` omits image data and says so |
| `-c, --continue` | | | Continue the most recent session |
| `-r, --resume` | | | Pick a session to resume |
| `--session` | | | Resume a session by file path or id prefix |
| `--no-session` | | | Do not save this session |
| `--sessions-dir` | `FA_SESSIONS_DIR` | `~/.local/share/fa/sessions` | Where session files live (global) |
| `--scrollbar` | `FA_SCROLLBAR` | `auto` | Transcript scrollbar: `auto` (while scrolling), `always`, `hidden` |
| `--no-bell` | `FA_NO_BELL` | | Do not ring the terminal when `ask` puts a question up |
| `--mcp-config` | `FA_MCP_CONFIG` | XDG search path | Read MCP servers from this file instead |
| `--no-mcp` | | | Start no MCP servers this session |
| `--tools` | `FA_TOOLS` | all of them | Offer the model only these tools, by name |
| `--no-tools` | `FA_NO_TOOLS` | | Keep these tools from the model, by name |

The agent works in the current directory. If `AGENTS.md` (or `CLAUDE.md`)
exists there, it is appended to the system prompt.

## Which model

`-m` names what a session starts on, and `/model` changes it whenever you
like: the conversation carries on against the new one, since what a session
holds is messages rather than a connection. Nothing is asked of the provider
until `/model` asks it — a session that never opens the picker never lists
anything — and the list is fetched afresh each time it does, so a model the
provider has gained since fa started is simply there.

The list is typed at rather than scrolled through, which is what four hundred
models on OpenRouter need: letters narrow it to what they fuzzily match
(`snt` finds `claude-sonnet-5`), `Backspace` widens it again, and the arrows
steer what is left. The letters that matched are picked out in each row the
way the `/` popup picks out its own, so a fuzzy match reads as the match it
was. The filter is drawn in the list's own title, since a list being filtered
is a list of that.

`/model <id>` names one outright, listed or not: the list is what a provider
admits to, not the whole of what it answers to, and a local server that lists
nothing useful is still reachable by name.

The list is also where the context window comes from, when the provider
reports one. Only four of them do — Gemini, Groq, Mistral and OpenRouter —
and a model whose window nobody reports gets the 128000 fallback, which is
what `--context-window` is for. A figure given there stands whatever model is
chosen afterwards, so it is the answer to a provider that reports nothing, or
reports the wrong thing.

Since the list is only fetched when `/model` asks for it, a window reported
for the model already in use arrives the first time the picker is opened —
even if you press Esc straight back out of it, and whether or not you picked
anything. Until then it is the fallback: too small a figure only compacts
sooner than it had to, and too large a one is caught by the run itself, which
stops when the window fills, makes room, and picks itself back up. A provider
that takes the request and never answers is given twenty seconds before fa
stops listening.

Seven providers cannot be asked at all (Cohere, Doubleword, Hyperbolic,
llamafile, Perplexity, Together, xAI): rig speaks no listing endpoint for
them, so `/model` on one of those says so without a request going out, and
`/model <id>` is the way to change model there.

## Which tools

`--tools` and `--no-tools` name what the model is offered, out of the five
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
what it says about `bash`, `edit`, `write` and `ask` is dropped when they are
not this session's, because a model told about a tool and then refused it tries
anyway and reports being refused, instead of using what it does have. A
`--system-prompt` of your own is left alone.

## Asking you

Four of the tools do something to the project. The fifth, `ask`, does something
to you: it stops and puts a question on the screen, and the run waits on the
answer. It is a port of [`rpiv-ask-user-question`](https://github.com/juicesharp/rpiv-mono/tree/main/packages/rpiv-ask-user-question),
the pi extension: the same questions, the same dialog, and the same envelope
read back to the model word for word. The tool answers to `ask` here rather
than to `ask_user_question`, which is the one thing a prompt written for the
extension has to be told.

```
╭ The model is asking ─────────────────────────────────────────────────────────╮
│ ←  ■ Cache   □ Tests   ✓ Submit  →                                           │
│                                                                              │
│Which tests?                                                                  │
│                                                                              │
│› 1. [✔] Unit                                                                 │
│         the functions on their own                                           │
│  2. [ ] Integration                                                          │
│         the pieces together                                                  │
│  3. [ ] Type something.                                                      │
│  Next                                                                        │
│                                                                              │
│Enter to select · ↑/↓ to navigate · Space to toggle · Tab to switch question… │
╰──────────────────────────────────────────────────────────────────────────────╯
```

One call carries up to four questions, each with two to four options and a line
under each saying what it means. More than one and they are tabs: `Tab` and
`Shift+Tab` (or `←`/`→`) move between them, a box in the strip fills as each is
answered, and the last tab reviews the answers and names anything still blank.
Submitting with a question left blank is allowed — a partial answer beats a
dismissed dialog — and the questions left out simply say nothing to the model.

Every question ends with a `Type something.` row, so the options are never a
cage: walk onto it and it takes the keyboard, `Alt+Enter` breaks a line,
`Ctrl+U` clears it, and the arrows walk the draft before they walk the list
again. What you type stays in the row while you look at the other options, and
is kept per question.

A multi-select question has boxes instead of a choice. `Space` ticks the row
under the cursor and so does `Enter`, which makes ticking a list cost nothing;
the question is committed from the `Next` row at the bottom. Ticking a box
answers the question straight away, so the tab strip keeps up.

`Esc` walks away from the whole questionnaire, and so does `Ctrl+C`, which also
stops the run it belonged to. Either way the model is told
`User declined to answer questions` — one signal for "they did not answer",
rather than one per way of not answering. A run aborted while the dialog is up
takes the dialog with it.

A question going up rings the terminal once — a plain `BEL`, so what it turns
into is whatever your terminal has already been told to do with one: a sound, a
flash of the window, a badge, or nothing. A run is usually left to get on with
its work, and this is the one thing in fa that goes nowhere until somebody comes
back to it. `--no-bell` (or `FA_NO_BELL`) keeps it quiet.

`--no-tools ask` takes it away altogether, for a session that should get on
with it rather than stop to ask — the system prompt then says nothing about
asking either.

## MCP

Tools from [MCP](https://modelcontextprotocol.io) servers sit beside the five
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
headers-command.Authorization = "echo Bearer $(pass show work/mcp)"
timeout = 60
except = ["delete_page"]
```

A server is a `command` to run, spoken to over its own stdin and stdout, or a
`url` to call, spoken to over streamable HTTP.

| Key | Meaning |
|-----|---------|
| `command` | A line of shell, run the way the `bash` tool runs one — quoting, `~` and `$HOME` all work |
| `env` | Added to the environment that command inherits |
| `env-command` | The same, each value the output of a line of shell rather than written out |
| `url` | Endpoint of a server that speaks streamable HTTP |
| `headers` | Sent with every request to it, which is where a token goes |
| `headers-command` | The same, each value the output of a line of shell rather than written out |
| `timeout` | Seconds one of this server's tools may take, `0` to wait forever (default 300) |
| `tools` | Take only these of the tools it offers |
| `except` | Take everything but these |

`tools` and `except` are the same idea as `--tools` and `--no-tools`, for one
server: a server with thirty tools can be cut to the two worth having without
naming every tool of every other server.

A token is better kept out of the file it is used from, so `env-command` and
`headers-command` take the line of shell that produces the value instead of the
value — `pass show …`, `gh auth token`, `op read …`, `echo Bearer $(…)` to put
a word in front of it. Each runs once, when its server starts, and what it
printed is the value, without the newline it was printed with. A name given
both a value and a command is refused rather than resolved by some rule about
which wins.

Such a command has ten seconds and no terminal: its input is closed, so an
agent that wants a passphrase must be one that can ask elsewhere or one that is
already unlocked — a pinentry in this terminal would draw over the session and
wait for an answer nobody could give it. One that fails is a note naming the
value, the command and the first line of what it said for itself, never what it
printed. A value is fetched once per session and held, so a token that expires
mid-session is a session to restart.

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
session with servers is a session with more than the five tools fa was built
with. A session with no servers says nothing about MCP anywhere.

A server that fails, or takes more than 20 seconds to say what it offers, is a
note rather than a failure — the session still has its own five tools, which
beats refusing to start. A tool named like one of those five is
left alone: the model is told about `read`, `write`, `edit`, `bash` and `ask`
in the system prompt, and cannot say which of two it meant.

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

`lang-bash`, `lang-c`, `lang-c-sharp`, `lang-cmake`, `lang-cpp`, `lang-css`,
`lang-dart`, `lang-diff`, `lang-elixir`, `lang-erlang`, `lang-fortran`,
`lang-gleam`, `lang-go`, `lang-haskell`, `lang-html`, `lang-ini`, `lang-java`,
`lang-javascript`, `lang-jsdoc`, `lang-json`, `lang-kotlin`, `lang-lua`,
`lang-make`, `lang-nix`, `lang-ocaml` (implementation and interface),
`lang-php`, `lang-powershell`, `lang-python`, `lang-r`, `lang-regex`,
`lang-ruby`, `lang-rust`, `lang-scala`, `lang-sql`, `lang-swift`, `lang-toml`,
`lang-typescript` (TypeScript and TSX), `lang-xml`, `lang-yaml`, `lang-zig`.

`lang-jsdoc` and `lang-regex` are grammars nobody writes a fence for:
JavaScript injects them into its doc comments and its regex literals, so they
only add colour to blocks that were already highlighted.

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

The picker is typed at the way the model list is: letters narrow it to the
sessions whose titles they fuzzily match, `Backspace` widens it again, the
arrows steer what is left, and the matched letters are picked out in each row.
A title is the session's name where it has one and its first message where it
does not — which is what a conversation is remembered by, and a directory
worked in for a month has more sessions than rows. The query is drawn in the
list's own title, and `Enter` takes the row the filter left under the cursor.

A session is deleted from the picker: `Delete` on a row asks, and a second
`Delete` removes the file. Since that is the only copy of the conversation, it
is asked about first, and any other key answers no. The session on screen is
not one of them — it is still writing to its file — so deleting it means
starting another with `/new` first.

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

## Prompt history

`↑` walks back through the prompts already sent, `↓` walks forward again, and
past the newest it hands back whatever was in the box when the walk began.
Like pi, there is no history file of its own: the prompts come from the
session, so one resumed with `-c` or `/resume` brings its own back, a new one
starts empty, and `/tree` or `/fork` rebuilds the list to match the
conversation now on screen. Slash commands are recalled too as long as the
session lasts, though the session itself never sees them.

Both keys stay the cursor's while it has a line to move to, so a prompt of
several lines can still be edited: `↑` walks only from the first line of the
box and from the start of it — a first press goes there, a second walks back —
and `↓` only from the last. Editing a recalled prompt ends the walk and keeps
what it handed back; sending it puts it at the front, where `↑` finds it next.

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

## Images

An image a tool answered with is drawn where it was read rather than described
as `[image]`. Every cell is `▄`: the upper pixel is its background and the
lower one its foreground, so a cell carries one pixel across and two down —
which is the shape of a terminal cell, and what keeps the picture's own
proportions. The colours are the terminal's own 24-bit ones, and a transparent
pixel is left without a colour at all, so an icon with no background of its own
sits on whatever the terminal is wearing.

```
⚙ read diagram.png
    Read image file [image/png]
   ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
   ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
```

The picture is drawn at the width the transcript has, which is as much detail
as a terminal can hold, and folded at sixteen lines like every other block of a
tool's output — so `Ctrl+O` shows the rest of it rather than a larger copy of
it, and nothing already on screen moves when it does. Scaling it to fit a
preview instead would cost the detail everywhere to save the scrolling in one
place. It is never enlarged: a 16x16 icon is the eight lines it is, not blown
up to the width of the transcript. Eighty lines is the ceiling on a drawn one,
which binds only for the very tall and narrow.

A drawn image is kept by its bytes and the width it was drawn at, the way
rendered markdown is, because scaling one is more work than a frame has. The
fold is not part of that: unfolding shows more of the same drawing rather than
making another.

That is what the transcript shows; what the model gets is the image itself. So
`--no-vision` is a session with neither — `read` sends no image, and there is
nothing left to draw.

### Attaching one

An image goes into a prompt as `@path`, which may be relative to the working
directory, a full path, or `~/...`. A path with a space in it goes in quotes:
`@"~/my shots/a.png"`. Dropping a file on the terminal pastes its path, and a
lone path to an image is written down as a token by itself, so the gesture
works without typing anything.

```
╭────────────────────────────────────────────────────────╮
│ why does @shot.png render the sidebar twice?           │
╰─ ▣ shot.png 1280×800 ──────────────────────────────────╯
```

The bottom border says what the tokens found, as they are typed — the size it
will be sent at, or `⚠ shot.png not found` for one that resolved to nothing. It
costs the transcript no room and is gone again the moment the tokens are. A
half-typed token opens a popup of the images and directories it could mean, the
same list the `/` commands use; `Tab` takes a row, and a directory is carried
on into rather than attached. Once the token names a real image the popup
closes, so `Enter` sends rather than being taken by the list.

The token stays in the text that is sent. It is not the image — that travels as
a content part of its own, behind a note naming it, the same shape `read`
answers in — but the sentence has to read as one, the token says which image is
which when there are several, and the session stores only what was sent. That
last one is why: a prompt handed back by `/tree`, or walked back to with `Up`,
brings its attachments with it because they are written in it.

What could not be attached is said in the transcript rather than passed over,
since a prompt that reads like it carries an image and does not is worse than
one that says so. Under `--no-vision` that is every attachment.

## Compaction

Every request is weighed against `context_window - reserve_tokens`, both as it
comes back and before the next one goes out. What a request cost is the
provider's own count, which is the truth about everything up to the answer it
gave — and says nothing about what a tool has returned since, which is how a
single `read` can fill the window between one call and the next. So the
weighing before a request is that count plus a chars/4 estimate of only the
messages added after it, and nothing the provider has already counted is ever
guessed at. A provider that reports no usage leaves the estimate standing for
the whole conversation, which is the one case where all of it is a guess.

Whichever weighing finds the window full, the run stops at its next turn
boundary, which is the only place a run can be cut without leaving a tool call
unanswered: what it got through is kept the way an abort keeps it.

Then the older part of the history is rendered as a transcript and summarized
by the same model into pi's structured checkpoint format (goal, progress,
decisions, next steps, critical context). The summary replaces those messages
as a single user message; the most recent turns, about `keep_recent_tokens`
worth, stay verbatim as entries of their own, so `/tree` can still go back to
any of them; tool call/result pairs are never split. Where that cut falls is
chars/4 and nothing else — it needs a size per message, and no provider
reports one. A later compaction updates the existing summary instead of
nesting it.

The tail starts wherever the budget runs out, at the nearest message the
model can carry on from: a user turn, or a turn of its own. So a turn may be
split — a run long enough to fill the window on its own, which is the run
compaction is for, keeps the budget it was promised rather than the little
that happens to come after its last user message. Never a tool result,
though, which belongs to the call above it: the cut moves on past the pair
rather than landing between them.

When the budget runs out with nothing after it to keep from — one tool result
worth more than the whole of it — the tail starts at the last such message
instead: the call the model made and what came back. Keeping nothing would
hand it a summary and no work, which is the state compaction is there to
rescue it from.

When the cut does fall inside a turn, the beginning of that turn is
summarized a second time, by pi's prompt for it: where the checkpoint
describes a conversation that is over, this one is written for the half of
the turn still on screen — *"This is the PREFIX of a turn that was too large
to keep. The SUFFIX (recent work) is retained"*, and asks for the original
request, how far it got, and what the kept half needs to be read by. The two
come back as one message:

```
## Goal
…the checkpoint…

---

**Turn Context (split turn):**

## Original Request
…
```

A turn kept whole has no beginning left over, so that is one request as
before; and when the split turn is all there was to summarize, the
checkpoint is the previous summary, or pi's `No prior history.` when there
is none. `--no-turn-summary` turns the second request off altogether: what
the turn was for then goes into the checkpoint with everything else.

pi also appends the list of files read and written to every summary, which
this does not: what the summarizer says about them is all there is.

With the room made, the run picks itself back up where it stopped — unless
something was typed while it ran, which goes first, as it would have anyway.
A context full of a single turn is where it ends instead: there is nothing
left to summarize, and carrying on would only fill the window again and ask
for the same summary, so it says so and leaves the next move to `/continue`.
`/compact` does all of this on demand, without waiting for the window to
fill.

The footer shows the current context usage as `ctx N%` — dim while there is
room, yellow past 70%, red past 90%, which on a session that compacts means
the window is filling with nothing being done about it. Beside it is what the
conversation has cost, as `N↑ M↓`: a running total of every call's tokens, so
it passes the size of the window early and keeps going. Both move as each
call comes back rather than when the run ends, so a run of twenty turns
counts twenty times — and a run that ends in an error or an abort still says
what it spent.

## Keys

| Key | Action |
|-----|--------|
| `Enter` | Send (queued if a run is in progress) |
| `Alt+↑` | Take the last queued message back for editing (empty input) |
| `↑` / `↓` | Walk back through the prompts already sent, and forward again — from the first/last line of the input |
| `/` | Command popup: type to fuzzy-filter, `↑`/`↓` move, `Tab`/`Enter` complete, `Esc` dismiss |
| `@` | Attach an image: `@path`, `@/full/path`, `@~/shot.png`, `@"with a space.png"` — same popup, `Tab` completes, a directory is carried on into |
| `Alt+Enter`, `Ctrl+J`, `Shift+Enter`* | Newline |
| Paste | Goes in whole, newlines and all — a pasted snippet is not sent at its first line break. A lone path to an image becomes an `@` token |
| `Esc` | Abort the current run |
| `Esc` `Esc` | Open `/tree` (empty input, within half a second) |
| `PageUp` / `PageDown`, mouse wheel | Scroll transcript |
| Drag the scrollbar | Scroll transcript — clicking its track jumps there, and keeps hold |
| Drag over the transcript | Select and copy — letting go copies what was covered, dragging along the top or bottom row scrolls |
| `Ctrl+C` | Abort if running, otherwise quit |
| `Ctrl+D` | Quit (empty input) |
| `Ctrl+T` | Thinking in full, or only its last lines |
| `Ctrl+O` | Tool output in full, or only its preview |
| `↑` / `↓`, `Enter`, `Space`, `Tab`, `Esc` | Answer what `ask` put on the screen — see [Asking you](#asking-you) |
| `/compact` | Summarize older history now |
| `/continue` | Run the model again with no new message |
| `/model` | Pick the model from what the provider offers, asked for afresh each time — type to filter, or `/model <id>` to name one outright |
| `/new` | Start a new session |
| `/resume` | Pick a saved session to resume — type to filter, `Delete` removes the selected one, a second `Delete` confirms |
| `/tree` | Move to another point in this session, on any branch |
| `/fork` | Branch a new session from an earlier prompt |
| `/name <name>` | Name the current session |
| `/session` | Show session id, file, and stats |
| `/quit` | Quit |

\* in terminals that support the kitty keyboard protocol.

Mouse capture is enabled, so the terminal's own selection usually needs
`Shift` held. Dragging without it selects here instead: the cells the drag
covers are drawn reversed while the button is down, and letting go copies them
to the terminal's clipboard — the primary selection and the clipboard proper
both, so a middle click and a `Ctrl+V` paste the same thing. What is copied is
what is on screen, line by line as it is drawn, without the blanks each line
ends in. The highlight is the drag rather than a state of its own: it goes
when the button does, so nothing is left on screen to be dismissed, and
nothing has to notice when the lines under it move.

The copy travels as an OSC 52 escape, which is what makes it work over ssh:
the text lands on the clipboard of the terminal being looked at rather than of
the machine `fa` runs on. Terminals that ship the escape turned off need it
turned on — tmux wants `set -g set-clipboard on`, xterm `allowWindowOps` — and
one that does not read it at all ignores the copy silently, since nothing
comes back to say otherwise.

## Layout

- `src/main.rs` – CLI flags, terminal setup.
- `src/agent.rs` – builds the rig agent (the provider's client, tools,
  system prompt), asks the provider what models it has, and rebuilds the model
  handle when `/model` picks another — the tools and the preamble outlive it,
  which is what lets a session change model without starting again. It runs
  one streaming turn per user message, forwarding
  `AgentEvent`s to the UI over a channel. Tool-call argument fragments are
  accumulated per call and sent whole, so the UI has nothing to reassemble.
  Tool calls and results are reported
  through an `AgentHook`, which carries the call id and an error flag. The
  same hook reports what each model call cost as it comes back, and is where
  a run learns to stop at a turn boundary — for a message typed while it ran,
  or for a context window that has no room left in it.
  Nothing in rig hands a tool its own call id and the hook that knows it runs
  before the body rather than around it, so for `bash` the hook rewrites the
  arguments to carry it — the one channel between the two. It is stripped from
  the schema, so the model is never asked for it and never sends it. On the
  chat-completions paths images are moved out of tool results into a follow-up
  user message, since those accept nothing but text in a tool message (the same
  workaround pi's provider uses). Gemini takes an image inside a function
  response and Anthropic inside a tool result, so those two are sent them where
  they are.
- `src/compaction.rs` – context compaction: trigger rule, cut point, transcript
  serialization, pi's summarization prompts.
- `src/ask.rs` – the `ask` tool's questionnaire: what a question may be, the
  validation the model is held to, the dialog's state machine and how it draws.
  All of it testable without a model or a terminal, which is what most of the
  file's tests do.
- `src/tools.rs` – the five tools, mirroring pi's descriptions, truncation
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
  `mcp.toml`, the table-per-server file it reads, the values it runs a command
  for rather than holding, and handing each server's tools to the agent. Rig
  speaks the protocol; this only decides who to speak to. Holding the result is what keeps the servers running, so it
  lives as long as the program does.
- `src/ui.rs` – ratatui app: transcript, input, footer. The transcript is
  wrapped to the width here rather than by the `Paragraph` that draws it, so a
  row on screen is a line of a list: what the mouse points at can be named,
  which is what a selection is made of and cut from.
- `src/clipboard.rs` – copying as an OSC 52 escape handed to the terminal,
  rather than as a call on the machine this runs on — which is what makes it
  work over ssh.
- `tests/e2e.rs` – the binary driven through tmux against a mock provider: a
  call written token by token and its output landing under it, pass and fail
  as the stripe beside each, an aborted run keeping its work and carrying on
  from it, a reopened session reading exactly as it did before, a session held
  to some of its tools, a message typed
  mid-run waiting at the bottom and going at the next turn, `↑` walking back
  through the prompts of a session and of the one that resumes it, going back into a
  branch the conversation left, forking into a session of its own, a compacted
  conversation reaching the model as its summary, what a call cost being
  counted while the run it belongs to is still going, a run that fills the
  window compacting and picking itself back up, a window with nothing left to
  compact stopping instead of doing it forever, a tool from a real MCP
  server being offered, called and drawn like any other, and a drag over the
  transcript selecting what it covered and copying it — read back out of
  tmux's own clipboard, which is where the OSC 52 escape lands.

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
counting, so a resumed session picks up where the last one left off. Each test
gets a tmux server of its own rather than the one you are working in: they set
server options and read the clipboard back, and two tests sharing either would
be reading each other's. Without tmux installed these skip rather than fail.

`src/agent.rs` also has end-to-end tests against a mock server, gated on an
environment variable each: `FA_TEST_BASE_URL` for the OpenAI-compatible ones
(streaming, tool calls, compaction, image input) and `FA_TEST_GEMINI_BASE_URL`
for the Gemini image one. Unset, each test prints why and returns, so
`cargo test` passes without them. The server is not part of the repository —
point these at one you are running, answering as the provider would; the model
is sent as `mock`.

```sh
FA_TEST_BASE_URL=http://127.0.0.1:8123/v1 cargo test
```
