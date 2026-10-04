//! The public memory loader agrees with the agent's own.
//!
//! An application that assembles a plain model call of its own — a workflow
//! engine's prompt node — loads the project's instructions through
//! `ProjectMemory`. It must see exactly what a coding agent given the same
//! paths puts in its system prompt: the same files kept and skipped, the same
//! measurements, the same cut.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use lithos_llm::types::Role;
use pebble_coding_agent::events::{CodingAgentEvent, CodingEvent, EventSink, EventSinkError};
use pebble_coding_agent::test_support::{
    MockEnvironment, ScriptedCall, ScriptedProvider, client_from, message_text, text_response,
};
use pebble_coding_agent::{CodingAgent, CodingAgentOptions, ProjectMemory, ShutdownReason};
use tokio_util::sync::CancellationToken;

/// Keeps every event the agent records.
#[derive(Default)]
struct Recorded(Mutex<Vec<CodingEvent>>);

#[async_trait]
impl EventSink for Recorded {
    async fn record(&self, event: &CodingAgentEvent) -> Result<(), EventSinkError> {
        self.0
            .lock()
            .expect("the sink lock is healthy")
            .push(event.event.clone());
        Ok(())
    }
}

/// A repository with every case the loader decides, and the paths an
/// application would name for it: a file that fits, a copy of it under
/// another name, an empty file, a missing one, a file that crosses the budget,
/// and one past it.
fn fixture() -> (MockEnvironment, Vec<String>) {
    let shared = "Run the tests before you commit.\n";
    let oversized = "b".repeat(ProjectMemory::BUDGET_BYTES);
    let files = [
        ("/home/test/AGENTS.md", shared),
        ("/home/test/CLAUDE.md", shared),
        ("/home/test/EMPTY.md", ""),
        ("/home/test/.codex/instructions.md", oversized.as_str()),
        ("/home/test/LATE.md", "never reached"),
    ];
    let env = MockEnvironment {
        files: files
            .iter()
            .map(|(path, content)| ((*path).to_owned(), (*content).to_owned()))
            .collect::<HashMap<_, _>>(),
        ..MockEnvironment::linux()
    };
    let paths = [
        "/home/test/AGENTS.md",
        "/home/test/CLAUDE.md",
        "/home/test/EMPTY.md",
        "/home/test/MISSING.md",
        "/home/test/.codex/instructions.md",
        "/home/test/LATE.md",
    ]
    .map(str::to_owned)
    .to_vec();
    (env, paths)
}

#[tokio::test]
async fn the_public_loader_matches_what_the_agent_loads() {
    let (env, paths) = fixture();
    let env = Arc::new(env);

    let memory = ProjectMemory::load(env.as_ref(), &paths, &CancellationToken::new())
        .await
        .expect("the load succeeds");

    // The loader's decisions on the fixture: the copy, the empty file, the
    // missing file, and the file past the budget leave no document; the file
    // that crossed the budget is cut to what remained.
    let documents = memory.documents();
    assert_eq!(
        documents
            .iter()
            .map(|document| (document.path(), document.truncated()))
            .collect::<Vec<_>>(),
        [
            ("/home/test/AGENTS.md", false),
            ("/home/test/.codex/instructions.md", true),
        ]
    );
    assert_eq!(memory.loaded_bytes(), ProjectMemory::BUDGET_BYTES);
    assert_eq!(
        memory.text(),
        format!("{}\n\n{}", documents[0].content(), documents[1].content())
    );

    // The agent, given the same paths, loads the same files and puts the same
    // text in its system prompt.
    let recorded = Arc::new(Recorded::default());
    let (client, provider) = client_from(ScriptedProvider::new(vec![ScriptedCall::response(
        text_response("done"),
    )]));
    let mut agent = CodingAgent::builder(client, env)
        .model("test/model")
        .options(CodingAgentOptions::default().with_memory_files(paths.clone()))
        .event_sink(Arc::clone(&recorded) as Arc<dyn EventSink>)
        .build()
        .await
        .expect("the coding agent builds");
    assert_eq!(agent.snapshot().memory(), memory.summaries().as_slice());

    agent
        .prompt("hello")
        .await
        .result
        .expect("the prompt succeeds");
    let requests = provider.requests();
    let system = requests
        .first()
        .expect("the round was requested")
        .messages()
        .first()
        .expect("the request is not empty")
        .clone();
    assert_eq!(system.role(), Role::System);
    let system_prompt = message_text(&system);
    assert!(
        system_prompt.ends_with(&format!("\n\n{}", memory.text())),
        "the system prompt ends with the loaded memory: {}",
        &system_prompt[system_prompt.len().saturating_sub(200)..]
    );
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");

    let events = recorded.0.lock().expect("the sink lock is healthy");
    let loaded = events
        .iter()
        .find_map(|event| match event {
            CodingEvent::MemoryLoaded {
                files,
                total_loaded_bytes,
                budget_bytes,
                ..
            } => Some((files.clone(), *total_loaded_bytes, *budget_bytes)),
            _ => None,
        })
        .expect("the agent reported its memory");
    assert_eq!(
        loaded,
        (
            memory.summaries(),
            memory.loaded_bytes(),
            ProjectMemory::BUDGET_BYTES
        )
    );
}

#[tokio::test]
async fn nothing_to_load_gives_the_agent_and_the_loader_nothing() {
    let env = Arc::new(MockEnvironment::linux());
    let paths = vec!["/home/test/AGENTS.md".to_owned()];

    let memory = ProjectMemory::load(env.as_ref(), &paths, &CancellationToken::new())
        .await
        .expect("the load succeeds");
    let (client, _provider) = client_from(ScriptedProvider::new(vec![ScriptedCall::response(
        text_response("done"),
    )]));
    let mut agent = CodingAgent::builder(client, env)
        .model("test/model")
        .options(CodingAgentOptions::default().with_memory_files(paths))
        .build()
        .await
        .expect("the coding agent builds");

    assert!(memory.is_empty());
    assert_eq!(memory.text(), "");
    assert!(agent.snapshot().memory().is_empty());
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("the agent shuts down");
}

#[tokio::test]
async fn an_absent_candidate_probes_existence_instead_of_failing_the_read() {
    let (env, paths) = fixture();
    let env = Arc::new(env);

    let memory = ProjectMemory::load(env.as_ref(), &paths, &CancellationToken::new())
        .await
        .expect("the load succeeds");
    assert!(
        memory
            .documents()
            .iter()
            .all(|document| document.path() != "/home/test/MISSING.md"),
        "the absent file never becomes a document"
    );

    // The regression this pins (fabro-5c45): a blind read of the absent
    // candidate makes the sandbox driver surface the miss as a failed
    // operation. The loader must probe existence instead, so no read is
    // issued for the missing path at all.
    let attempts = env.read_attempts.lock().expect("read attempts lock");
    assert!(
        !attempts.contains(&"/home/test/MISSING.md".to_owned()),
        "no read issued for the absent candidate, got {attempts:?}"
    );
    assert!(
        attempts.contains(&"/home/test/AGENTS.md".to_owned()),
        "present candidates are still read"
    );
}
