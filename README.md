# Pebble

Pebble is a two-package agent library with a small command-line front end.
`pebble-agent` is the provider-neutral agent loop. `pebble-coding-agent`
builds a coding agent on top of it. `pebble-cli` runs that coding agent from a
terminal.

It owns the turn loop between a model and a machine: it calls the model,
streams what comes back, runs the tools the model asks for, keeps the history
those calls have to stay paired in, summarizes that history before it outgrows
the context window, and publishes an event for everything it does. It does not
own transport, storage, credentials, or process isolation. Those belong to the
application that embeds it.

## How the pieces sit

Four layers, each with one job.

At the bottom is [lithos-llm](https://github.com/lithoscomputer/lithos-llm):
the `Client`, the provider adapters and codecs, the catalog of models, and the
retry middleware. It knows how to talk to a provider.

Above it is `pebble-agent`: model turns, generic tool execution, conversation
history, steering, follow-up, cancellation, and a small lifecycle event stream.
It accepts a model service and tools from its caller. It knows how to run an
agent, but it knows nothing about coding.

Above that is `pebble-coding-agent`: coding profiles, filesystem and shell
tools, memory, skills, context compaction policy, subagents, and the durable
coding event stream. Its primary type is `CodingAgent`.

On top is the application — a CLI, a server, a workflow engine, the example in
this repository. It builds the `Client`, supplies the `Environment` the tools
act through, subscribes to events or installs a sink for them, decides what the
agent is allowed to do, and drives the input. It knows what the agent is for.

Dependencies point down. `pebble-coding-agent` depends on `pebble-agent`, which
depends on the `lithos-llm` runtime without enabling provider features. The
coding layer also uses `lithos-llm` directly for model selection and
coding-specific summary calls. Neither package builds a client or opens a
socket.

## A coding agent

```rust,no_run
use std::error::Error;
use std::sync::Arc;

use lithos_llm::Client;
use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::ConventionalCredentials;
use lithos_llm::middleware::{RetryMiddleware, RetryPolicy};
use pebble_coding_agent::events::RetryEventObserver;
use pebble_coding_agent::environment::LocalEnvironment;
use pebble_coding_agent::{CodingAgent, ShutdownReason};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let policy = RetryPolicy::exponential().max_attempts(4);
    let client = Client::builder()
        .catalog(Catalog::builder().with_builtin().build()?)
        .credentials(ConventionalCredentials::new())
        .middleware(RetryMiddleware::new(policy).observer(RetryEventObserver))
        .build()?
        .client;

    let mut agent = CodingAgent::builder(
        client,
        Arc::new(LocalEnvironment::new("/path/to/work")),
    )
        .model("claude-sonnet-5")
        .build()
        .await?;

    let mut events = agent.subscribe();
    let report = agent.prompt("fix the failing test").await;
    let outcome = report.result?;
    agent.shutdown(ShutdownReason::Completed).await?;

    println!("{:?} — first event: {:?}", outcome.text, events.try_recv());
    Ok(())
}
```

`CodingAgentBuilder::build().await` returns a ready coding agent. Resource loading
and system-prompt construction happen inside the build. There is no separate
initialization step to remember. `prompt` returns a `PromptReport` with usage
and timing on success and failure. Its `result` contains either a
`PromptOutput` with the final message and text, or the prompt error.
Accounting covers accepted main-model responses in this session, including
queued follow-ups, and the summary call of each compaction the prompt
performed. It excludes subagents and model calls made inside tools. Usage is
lithos-llm's `Usage`: the five token buckets and, when the catalog or the
provider priced every response, a `Cost` with its source. The cost is `None`
rather than a subtotal when a response that used tokens had no price. A
dropped prompt future cannot return a report. The report also names every file
the prompt wrote or edited (`files_touched`, sorted, the children's work
included, deletions left out), the one it touched last, and the
`provider/model` route it ended on. Its `compactions` list each compaction the
prompt performed, in order, as a `CompactionAccount` with the facts the
`Message::Compaction` turn records and the summary call's usage: a breakdown of
the report's usage, not an addition to it, and a manual `compact` between
prompts is on no report.

`continue_prompt` continues a prompt the history left unfinished, without new
input: the history ends with the prompt itself or with tool results the model
has not answered, as it does when a model call failed for good after a tool ran.
An agent resumed from that record, on the recorded model or on another one
through `ResumeMode::UseModel`, is asked again on the history as it stands and
repeats no tool effect. A history that ends with the model's own answer has
nothing to continue, and the report says so.

`CodingAgentBuilder::mcp_servers`, behind the `mcp` feature, names the MCP
servers whose tools the agent gets: a child process spoken to over its
standard streams, a server reached over HTTP (streamable HTTP or the older
SSE transport), or a server launched in the environment and reached through
the environment's route to its port: a `PortRoutes` the application implements
over its own sandbox and hands over with `port_routes`, so pebble shares no
sandbox crate with the application. The servers start while the agent is
built; every tool they advertise is registered under `mcp__{server}__{tool}`
with `ToolSource::Mcp`, so middleware, history, output policy, cancellation
and events treat it as any other tool. Each server's outcome is on the stream
as `McpServerReady` or `McpServerFailed`, each carrying how long the server
took to reach that outcome in `startup_ms`, and a server that fails is
skipped.
A result the server marks `isError` is the tool's error text; a transport
failure, a timeout or a cancellation is a failed call with a reason. A server
whose connection closes during the session is reported once as
`McpServerDisconnected`, and every later call to its tools fails. The
servers close when the agent shuts down. See `pebble_coding_agent::mcp`.

`CodingAgentBuilder::fallback_routes` names the routes a prompt continues on
when its model fails for a reason another route might not share (lithos-llm's
`failover_eligible`: authentication, access, not-found and quota failures,
rate limits, server and network errors, timeouts, stream decoding, and a
refusal from the content filter). Each `FallbackRoute` is a selector and the
request controls that route takes. On such a failure the agent takes its own
record, closes the failed session, resumes the record on the next route with
`ResumeMode::UseModel`, requeues the steering and follow-ups the failed session
still held, and continues the prompt as `continue_prompt` would, so no tool
effect repeats. Control handles and `subscribe` receivers taken before the
prompt keep working; the stream carries `RouteFailover` from the new route;
accounting spans both. Pebble executes the list it is given and nothing more:
a route's own fallbacks are not consulted, a route that cannot be built ends
the prompt with `Error::FallbackRoute`, and once the list is spent a model
error ends the prompt as it would without one. `remaining_fallback_routes`
says what is left, for a successor built from an export.

`CodingAgentOptions::with_memory_discovery` and `with_skill_discovery` name
a convention instead of a list. `MemoryDiscovery::from_git_root()` reads the
profile's own instruction files (`AgentProfileKind::memory_filenames`:
`AGENTS.md` and `CLAUDE.md` for the Claude harnesses, `AGENTS.md` and
`.codex/instructions.md` for the Codex ones, `AGENTS.md` and `GEMINI.md` for
Gemini, `AGENTS.md` alone for Kimi) in every directory from the repository
root down to the working directory, root first; `from_root` and
`working_directory` are the other two starts. `SkillDiscovery` lists the
directories to search in precedence order, anchored at the working directory
or the git root, and a directory the application `require`s is reported as
`MissingDirectory` on `SkillsDiscovered` when it is not there. Pebble does the
git probe, the walk, and the checks through the `Environment`; explicit
`with_memory_files` and `with_skill_dirs` still work and come first.

`pebble_coding_agent::projection::SessionProjection` folds one session
tree's events into what a view or an accountant needs: the root's usage as
one `Usage` (tokens, and cost where every answer was priced), each
descendant's and the model it ran,
the context window, which tools ran and how often they failed, how many model
calls were retried, the route and every failover that moved it, the MCP
servers and whether they were called, skills, todo lists, the children and
how they ended, compactions with their summary calls' usage, and the files
touched. It is
serializable, so a view resumes from a stored value, and it keeps a
`PromptDelta` for the prompt in progress, because a retained session spans
stages: the root's spend, each descendant's, the child lifecycle counts,
retries, compactions, and files touched, all restarted at each prompt. It
reports counts; pricing them from a catalog stays with the application. The
`pebble` command's closing summary is this projection, read out.

`pebble_coding_agent::steering::SteeringBus` is one control plane for many
sessions: sessions attach at a key of the application's choosing (a stage, a
node) and detach; a steer or an interrupt reaches every attached session; a
steer that arrives with none attached waits on the bus, up to a cap, and
drains into the next attachment; and a hold keeps an attached session's
natural completion open while a party is paired with it. Sessions implement
`SteerableSession`; `CodingAgentControlHandle` does natively. Every operation
returns what it did, so the application records buffered and dropped steers
in its own vocabulary and order.

`CodingAgent::export_for_reuse(reason)` closes the agent and returns the
export a successor continues from, its cursor already past the close, so an
application no longer takes an export, shuts down, and advances the cursor by
hand. `SessionRecord::resume_after(log_head)` is the rule for a record and an
event log saved separately: the cursor moves up to the log's head, never back.

The crate root contains the normal coding-agent path and the environment
contract. The environment a session acts through is in
`pebble_coding_agent::environment`. Durable event types are in
`pebble_coding_agent::events`. Tool contracts, built-in tools, and the standalone
tool runner are in `pebble_coding_agent::tools`. Durable session state is in
`pebble_coding_agent::state`. Optional application services are in
`pebble_coding_agent::extensions`. Subagent configuration is in
`pebble_coding_agent::subagents`. `ProjectMemory` is the loader an agent reads
its memory files with — explicit paths, one 32 KB budget, duplicate text
skipped, the file that crosses the budget cut and marked — so an application
that puts the same instructions into a model call of its own loads them by the
same rules. The internal runtime is not public.

Take a `CodingAgentControlHandle` before calling `prompt` when another task
must steer, follow up, cancel compaction, abort a prompt, close the agent, or
wait for idle while the prompt holds the mutable agent borrow. `abort` ends the
active prompt and leaves the agent reusable. `close` is permanent. The task
that owns the agent then calls `shutdown` to publish the terminal event and
join owned tasks.

The control handle also exposes queued input. `pending_input` clones steering
and follow-ups in queue order. `take_pending_input` removes and returns them.
An application can use this to restore unsent text to an editor after it aborts
a prompt.

`CodingInput` and `SteeringMessage::from_content` accept ordered
`lithos_llm::types::ContentPart` values. Pebble keeps those parts in history,
stored records, queued input, and input events. This includes images, audio,
and documents. The stable `text` event member remains available to text-only
renderers. `InputSource` distinguishes a direct prompt, a follow-up,
agent-generated input, and input from another application integration.

Use `CodingAgent::observe` when a view needs a complete starting point and
live updates. It returns a `CodingAgentSnapshot` and an event receiver at one
committed event cursor. Apply the snapshot first. Discard queued events at or
below `snapshot.committed_event_seq()`. Then apply later events in sequence.
The snapshot includes history, pending input, state, route, profile, memory,
skills, tools, and the latest context-window measurement.

Automatic context compaction remains policy-driven. An application can also
call `CodingAgent::compact(CompactionOptions)` while the agent is idle. The
operation returns a structured `CompactionOutcome`. A completed summary is a
dedicated `Message::Compaction` turn with its reason, counts, and usage.
Each started compaction ends with a completed, failed, or cancelled event.
Pass a token to `compact_with_cancellation`, or use
`CodingAgentControlHandle::cancel_compaction` from another task.

`crates/pebble-coding-agent/examples/coding_agent.rs` is the same thing at full
size. It renders the event stream, steers one prompt while it works, interrupts
the next, and reports what the agent used. Run it with `mise run dev`.

## The command line

Run `pebble` in a terminal for an interactive coding session:

```sh
pebble
pebble --model gpt-5.6 --cwd ../service
pebble --continue
pebble --resume <session-id>
```

The conversation stays in normal terminal scrollback. The prompt accepts
multiple lines, paste, history, and completion while the agent works. Enter
queues steering during a turn; Alt+Enter queues a follow-up. Escape cancels
active work and restores pending input. `/help` lists commands and shortcuts.
Sessions save locally and can resume after exit. See the
[interactive guide](docs/interactive.md) for commands, settings, and recovery.

On first use, Pebble offers provider setup and model selection. `/login` saves
an API key through masked input, and `/model` lists configured models. From the
shell, use `pebble auth login <provider>` and `pebble auth status`. The TUI and
`exec` share `~/.pebble/settings.json`, `models.toml`, and `auth.json`, relocated
by `PEBBLE_HOME`. See [models and credentials](docs/models-and-credentials.md)
for precedence, custom providers, and credential storage.

Interactive mode asks before each tool call outside the selected permission
level. Approval starts on **Deny**. `--no-approvals` hides those tools instead.
The permission level is an application policy, not an operating-system sandbox.

`pebble exec` runs one prompt to completion without a person in the loop: the
prompt goes in, the tools the model asks for run, and the final answer comes
out on standard output. Events are rendered to standard error as they happen,
and the exit status says how the prompt ended: `0` for an answer, `1` for a
failure, `130` when the prompt was interrupted or ran out of time.

```sh
pebble exec "add a --dry-run flag to bin/deploy"
pebble exec --model gpt-5.6 --cwd ../service --permission full "run the tests and fix what fails"
echo "summarize what this repository does" | pebble exec --quiet
pebble exec --json "..."   # one JSON event per line on standard error
pebble exec --verbose "..." # tool arguments and results, and the model's reasoning
```

The permission flag chooses what the agent may do without asking:
`read-only`, `read-write` (the default), or `full`, which enables commands.
In `pebble exec`, there is no approval path, so a tool the level does not allow is hidden from
the model and refused if called anyway. Credentials come from explicit sources,
the provider's usual environment variables, or saved API keys. Environment
variables override saved keys. `exec` uses `--model`, then the saved default,
then `claude-sonnet-5`, and never opens a setup prompt.
`PEBBLE_<PROVIDER>_BASE_URL`, such as `PEBBLE_OPENAI_BASE_URL` or
`PEBBLE_MOONSHOT_BASE_URL`, points a built-in provider at a compatible
endpoint instead: a proxy, a self-hosted model, or a test double. `--subagents`
lets the agent spawn children for independent work. `--tool-results` prints
each tool call's arguments in full and what the call answered;
`--transcript` prints the reasoning behind each turn when the model reports
it, and the text of a turn that did not stream; `--verbose` is both. The
command is the smallest application pebble ships, and its source is a worked
example of what an embedding application supplies.

The command line is a library first. `pebble-cli-core` holds the `exec`
session (`session::run_prompt`: the events rendered as they happen, the answer
on standard output, the summary after it, the agent shut down for the reason
the prompt ended with, and `session::run_prompt_with` for a front end with a
verbose mode), its `render::Renderer` with its `RenderOptions` and closing
`Summary`, the `approval::TerminalApproval` prompt for tools the permission
level does not allow outright, and the interactive session, credential store,
and settings whole. The `pebble` binary is an argument parser over it. Another program that
builds its own agent, with its own client, environment, tools, and settings,
runs it through the same session and renderer instead of writing them again;
fabro's `fabro exec` does.

`TerminalApproval` keeps Pebble's three permission levels. Choosing "yes"
allows only the current call. Choosing "always" grants the least level that
allows the tool: `read-write` for native writes, `full` for shell, web, and
unrecognized tools, including MCP. The prompt states that scope before asking.
The grant applies to every tool allowed by that level for the lifetime of the
approval-service instance, including sessions that share it; a fresh instance
starts at its configured level. Noninteractive approval refuses calls outside
the configured level. Pebble's own `exec` uses a fixed permission level, and
its interactive UI uses separate per-call approvals.

## What an application has to supply

**A client, with retry middleware.** Pebble takes a built `lithos_llm::Client`
and cannot add middleware to one, so the application installs the retry
middleware when it builds the client, with Pebble's `RetryEventObserver` on it.
The observer is what puts the client's own retries onto the agent's event
stream. Without it, an agent still runs correctly and does not report retries.
`CodingAgentOptions::turn_replay` controls the separate replay that happens
after a response stream opens. Its default waits one second before the first
replay, doubles the wait each time, caps a wait at sixty seconds, and jitters
each wait. The two policies can share settings, but they have different
ownership and do not have to match.

An answer that stops at the model's output limit is not replayed: the same
request would stop at the same place. Pebble keeps the answer as far as it got,
emits an `output_limit` warning, and asks the model once, in a user turn the
event stream attributes to the agent, to continue from where it stopped. A
second cut in the same prompt is reported and left as it stands. A tool call
the limit cut short never reaches pebble: `lithos-llm` drops it and reports a
`truncated_tool_call` warning, which pebble passes on as a `Warning` event.

`CodingAgentOptions::with_max_tool_rounds` bounds how many tool rounds one
prompt may run, where a round is one model turn that asks for tools plus the
execution of those calls. Without it there is no limit. Once the rounds are
spent the model is still asked for its answer; a turn that asks for tools
instead ends the prompt with `Error::ToolRoundsExhausted`, its calls recorded
as `Cancelled` without running and a `ToolRoundsExhausted` event on the stream.
The agent stays open and the next prompt gets a fresh budget. A workflow that
runs an agent to reach a decision uses this to cap the cost of one decision and
to fall back to a default when the agent does not reach one.

`CodingAgentOptions::with_max_turns` bounds model responses instead. A prompt
that has used its turns and would ask the model again ends with
`Error::Interrupted(InterruptReason::TurnLimit)` at the boundary a wall-clock
timeout uses: every tool call the last turn made has its result. The two
budgets count different things and may be set together; whichever is reached
first ends the prompt, with its own error.

Applications use `lithos-llm` directly to build clients and use its public
model types. The Pebble packages do not re-export their dependencies.

**Credentials.** They belong to the client, and lithos-llm resolves them per
call — `ConventionalCredentials::new()` reads the usual variables
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and the rest), and an application with
a vault implements lithos-llm's `CredentialProvider` instead. The agent libraries
never see a key. `pebble-cli` supplies its own credential provider for saved keys
and environment sources.

**An `Environment`.** Every tool acts through this one seam: reading and
writing files, listing a directory, searching by content or by name, and
running a command as Bash source. `LocalEnvironment` does that on this machine.
An application working in a container, a VM, or a remote workspace implements
the trait over that instead, and nothing else in the crate changes.
`pebble_coding_agent::test_support::MockEnvironment` stands in for a machine in tests,
behind the `test-util` feature.

An implementation over another machine does not write pebble's rules again.
`pebble_coding_agent::environment::support` holds the pieces of the contract
an adapter would otherwise hand-write: `validate_glob` refuses the patterns
pebble refuses, in pebble's words; `tree_order` sorts a listing the way
`list_directory` promises; `OutputCaptureBuffer` and `capture_stats` keep and
count bounded process output; and `classify_exec_error` names the error kind a
command that failed to run reports. `LocalEnvironment` uses the same helpers,
so an adapter built on them passes the `EnvironmentContract` checks they cover.

One such adapter ships with the crate, behind the `sandbox-driver` feature:
`pebble_coding_agent::sandbox_driver::SandboxEnvironment` is the trait over a
[sandbox-driver](https://github.com/lithoscomputer/sandbox-driver) handle, for
an application whose sandboxes (a host directory, a container, a remote
workspace) come from that driver. It resolves paths against the working
directory the application names, runs commands under `SandboxExec`'s policy
(a stop grace, text-only output), answers pebble's MCP port routes from the
driver's preview URLs when `mcp` is on too, and renders a driver failure for a
log with `display_for_log`. The sandbox stays the application's to start and
dispose of, and the places that render process output take the application's
`Redactor`. `sandbox_driver::test_support::MockSandbox`, behind
`sandbox-driver-test-util`, is the adapter over the driver's scripted doubles
for an application's tests.

**Somewhere for the events to go, if they matter.**
`CodingAgent::subscribe` hands out a bounded broadcast receiver, which is
lossy for a reader that falls behind: right for a terminal, wrong for a ledger.
The stream ends with the agent rather than with the agent value: once
`CodingAgent::shutdown` has returned, a reader looping until
`RecvError::Closed` finishes, so a renderer task can be joined before the
agent is dropped. An application that must see every event installs a
`pebble_coding_agent::events::EventSink` instead.

One root agent and all its subagents write directly to one ordered stream. Each
`CodingAgentEvent` names that stream with `stream_id`, names its producer with
`session_id`, and has a contiguous `seq` within the stream. The stream identity
and numbering continue when the root session resumes. A sink receives every
event, including streaming deltas, in sequence before a live subscriber sees
it. Treat `(event.stream_id(), event.seq)` as the idempotency key because a
process can stop after storage commits an event but before its latest session
record is saved. If the log and record do not share one transaction, read the
log's highest sequence and call `SessionRecord::advance_event_cursor` before
resume. This prevents the resumed stream from reusing an already committed
position.

The producer queue is bounded by `event_capacity`. A full queue, a refused sink
write, or a write longer than `event_sink_timeout` stops the stream, cancels
active work, closes the session tree, and returns an `event_stream` error. It
never lets the agent continue with a partial ledger. Building an agent and
finishing a prompt both wait for their events to reach the sink. Use
`CodingAgent::flush_events` for an explicit durability boundary and
`CodingAgent::committed_event_seq` to read its committed cursor.

**Tool middleware, if the agent should not do everything.** Pebble installs no
application policy. With no middleware, every registered tool is exposed and
every valid call runs. Add each layer with `CodingAgentBuilder::tool_middleware`.
The first added layer is outermost.

Permissions use the same middleware path as logging, metering, or other tool
behavior. `PermissionMiddleware` filters discovery and checks every call again.
The call check sees the validated arguments. Several permission layers compose
by narrowing access.

```rust
# use pebble_coding_agent::CodingAgentBuilder;
use pebble_coding_agent::tools::PermissionLevel;

# fn configure(builder: CodingAgentBuilder) -> CodingAgentBuilder {
builder.permission_level(PermissionLevel::ReadWrite)
# }
```

`CodingAgentBuilder::permission_level` installs the built-in permission policy and records its level together. The last call selects the level, regardless of where `.options(...)` appears. Subagents inherit the policy.

`pebble_agent::SessionScope`, re-exported by `pebble_coding_agent`, is the shared
identity for a session tree. `TurnContext::session()`, `ToolCallRequest::session()`,
and both tool contexts' `session()` methods expose it. A
`ToolPermissionPolicy::permission(&self, session, tool)` can use the root,
immediate parent, and depth in both discovery and invocation. The same middleware
instance can serve many concurrent roots and their descendants.

Generic agents and standalone tool runners get fresh root identities by default.
Use their builder's `session(scope)` to supply a scope. A coding agent's scope is
set at creation or restored from its record; spawned children derive theirs from
the parent. Applications use `agent.session()` to read it.

For a custom policy or approval flow, install `PermissionMiddleware` directly. `CodingAgentOptions::with_recorded_permission_level` records metadata only; it does not enforce permissions. Add a `ToolApprovalService` with `PermissionMiddleware::with_approval` when calls that are not auto-approved should remain visible and ask for approval. Without an approval service, those tools are hidden and direct attempts are denied.

Optional seams follow the same rule. Pebble installs no implementation by
default and advertises no tool without one: an `extensions::HumanInputProvider`
(no provider, no question tool), an `extensions::SearchProvider` (no provider,
no `web_search`), an `extensions::Redactor` for process output and failed tool
call messages, and a
`subagents::SubagentOptions` for subagents. A child is given a task, not the
project briefing: it loads no memory files and discovers no skills unless the
options ask for them with `with_inherited_memory` and `with_inherited_skills`,
which give a child the parent's configured memory files and skill directories.

**Search providers.** Search is the one seam pebble ships implementations
for, because the engines behind it are public HTTP APIs that every embedder
would otherwise write the same way. With the `search-providers` feature, off
by default, `pebble_coding_agent::search::providers::Brave` and
`search::providers::Venice` implement `SearchProvider` over the Brave Search
API and Venice's search endpoint. Each is built with `new(api_key, client)`
from an API key and a `reqwest::Client` the application owns, so proxies, TLS,
and timeouts stay the application's; `search::providers::default_client()`
builds a client for an application with none. Which engine a session gets,
from whichever key the application holds, is still the application's decision,
made by handing the provider to `CodingAgentBuilder::search_provider`. Without
the feature pebble depends on no HTTP client of its own.

## Native embedding extensions

`RegisteredTool::function` and `RegisteredTool::new` keep their string-returning
API. Use `RegisteredTool::rich_function` or `RegisteredTool::new_rich` when a tool
returns multiple `ContentPart` values. Its `ToolOutput` can include images and
other provider-neutral content. `with_details` and `with_artifact` attach
observer-only data. Middleware and the durable `ToolCallCompleted.metadata`
field receive that data. Model requests and conversation records contain only
the model-facing content. Applications keep completion events to restore their
rich tool views. `ToolError::with_metadata` supports the same data on failures.

The coding layer bounds mixed content as well as text. It omits oversized media
parts instead of cutting media payloads. Metadata is limited to the smaller of
64 KiB and one quarter of the serialized output budget. Large details belong in
an artifact. Existing serialized completion events without metadata still load.

**Bounded output.** Pebble retains bounded head-and-tail previews and discards
the omitted bytes. Truncation does not make a successful tool call fail.
Retaining complete tool output is an explicit anti-goal: Pebble does not store
full output or provide a retrieval tool for discarded bytes. Environment
adapters must continue draining process output after the capture limit is
reached, while retaining only the bounded preview and byte counts;
`environment::support::OutputCaptureBuffer` is that capture, and
`environment::support::capture_stats` the counts for a driver that bounds
output itself.

**Context and compaction policy.** Install `extensions::ContextPolicy` with
`CodingAgentBuilder::context_policy` to prepare the messages for each model
request. The hook runs after compaction and tool discovery, receives the
session identity and visible tools, and returns an optional replacement view.
It does not rewrite committed history or replace the profile's system prompt.
Pebble rejects unpaired tool calls and results before sending the request.

Install `extensions::CompactionPolicy` with `compaction_policy` to supply summary
generation for both automatic and manual compaction. The hook receives the
selected history, recent turns, and the default summary request. It returns
summary text and its usage, with the cost optional. Pebble owns the safe cut, summary limits,
history replacement, and terminal events. Failed, empty, or cancelled summaries
leave history intact. Existing compaction rules still clear stale usage
estimates and non-replayable provider data in retained turns.

These hooks and the output store are inherited by subagents. They are runtime
services, so install them again when resuming a record. They support native
library embedding; no transport adapter is required.

## The generic agent API

Use `pebble-agent` directly when the application supplies its own tools and
does not need coding profiles or environment policy:

```rust,no_run
use pebble_agent::{Agent, Tool};
use serde_json::json;

# async fn example(client: lithos_llm::Client) -> Result<(), Box<dyn std::error::Error>> {
let inspect = Tool::function(
    "inspect",
    "Inspect a named value",
    json!({"type": "object"}),
    |_context, arguments| async move {
        Ok(format!("inspected {}", arguments["name"]).into())
    },
)?;

let mut agent = Agent::builder(client, "provider/model")
    .system_prompt("Use tools when they help.")
    .tools([inspect])
    .build()?;

agent.follow_up("Now explain the result");
let outcome = agent.prompt("Inspect the parser").await?;
println!("{}", outcome.text());
# Ok(())
# }
```

The public verbs are `prompt`, `steer`, `follow_up`, `abort`, and
`wait_for_idle`. `Agent::snapshot` returns an owned immutable view instead of
exposing mutable agent state. Its control handle can also close the agent and
clone or drain pending steering and follow-ups.

The coding layer has the same concise path for application tools:

```rust
use pebble_coding_agent::tools::RegisteredTool;
use serde_json::json;

let inspect = RegisteredTool::function(
    "inspect",
    "Inspect an application value",
    json!({"type": "object"}),
    |_context, arguments| async move {
        Ok(format!("inspected {}", arguments["name"]))
    },
);

# let _ = inspect;
```

Pass it to `CodingAgentBuilder::tools`. Pebble records its source as
`ToolSource::Application`. The call then uses the same middleware, event,
output, and cancellation path as a built-in tool.

Duplicate tool names or identities are errors. To replace a registered tool,
use `.replace_tool("read_file", replacement)` on the coding agent builder.
The replacement keeps the original identity and the name the profile exposes
to the model. Its schema and executor come from `replacement`.
`CodingToolSet::with_tool` is also fallible and has a matching `replace_tool`
method for intentional replacements.

## Which harness a session runs

Pebble never guesses. The model selector resolves through the client's catalog,
and the resolved entry's `metadata.agent.profile` names one of the six
harnesses pebble ships — Claude, Claude 5, Gemini CLI, OpenAI, Codex, Kimi
Code. That profile decides the system prompt, the starting tools, and the
vocabulary those tools are named in, because a model trained inside a coding
harness expects that harness back. A model whose catalog entry names no profile
is a build error, not a session that runs with the wrong prompt.

## Stability

The serialized form of `pebble_coding_agent::events::CodingAgentEvent` and
`pebble_coding_agent::events::CodingEvent` is public API. Event evolution is
additive: new variants and new optional fields. Consumers should ignore members
they do not know and tolerate variants they do not know.

`pebble_coding_agent::state::SessionRecord` is also public API. Its format
version changes when a stored shape changes. Version 5 stores each assistant
and compaction turn's `usage` as lithos-llm's `Usage`, its token counts under
`tokens` and its `cost` when priced; version 4 stored the counts bare, with a
`cost_usd_micros` beside them on compaction turns. Every version since 4
stores a required `scope` with the session ID, root ID, immediate parent ID,
and depth. This build requires version 5: `SessionRecord::is_supported` is
false for any other version and resuming one fails with
`CodingAgentBuildError::UnsupportedRecord`, so an older record is refused
rather than read back as nothing used, and no ancestry is inferred from it.
Warm exports preserve the same scope.

Ignoring an unknown member is free; tolerating an unknown *variant* is the
reader's own work, because a variant a build has never heard of fails the whole
envelope with it. The crate documentation shows the pattern: read the envelope
with the event held as raw JSON, then parse the payload on its own.

## Setup

Pebble depends on `lithos-llm` and `sandbox-driver` as git dependencies on
exactly `branch = "main"` (not a `rev` or an omitted ref, which Cargo treats as
a different source). Update them with `cargo update -p <crate>`; an
application's own `Cargo.lock` decides what it ships. The repositories are
public, so Cargo fetches them over HTTPS with no credentials.

Install the locked tools and prepare the repository:

```sh
mise trust
mise install --locked --jobs=1
mise run setup
```

Use the repository tasks for development and verification:

```sh
mise run dev     # run the coding-agent example (needs a provider key)
mise run exec -- "write hello.txt"   # run `pebble exec` (needs a provider key)
mise run test    # the test suite, which needs neither key nor network
mise run check   # the complete routine gate
```

See [DEVELOPING.md](DEVELOPING.md) for the complete development workflow.

Routine and nightly checks run on macOS arm64, Linux x86_64, and Linux arm64.
GitHub cannot run them yet, because the runners have no key for the private
`lithos-llm` repository. See [DEVELOPING.md](DEVELOPING.md).
