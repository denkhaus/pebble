//! What a session hands its own tools.
//!
//! The built-in tools are given things a session owns rather than a profile:
//! what strips secrets out of what they publish, where a `web_search` goes,
//! and the skills a `use_skill` call loads from — configured on the builder,
//! with the skills discovered by `initialize` — plus the summarizer a
//! `web_fetch` tool captures at construction. Each reaches a tool only
//! through the whole chain, so they are tested through a real prompt rather
//! than by reading a field back.
//!
//! The two `apply_patch` tests here are the pair fabro ran through its own
//! executor: a custom tool call is free-form text rather than JSON, and it
//! travels the same round as every other call, error included.

use std::borrow::Cow;
use std::collections::HashMap;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex, PoisonError};

use async_trait::async_trait;

use super::*;
use crate::environment::{Environment, ExecResult};
use crate::error::ErrorKind;
use crate::redact::Redactor;
use crate::runtime::testing::{TestProfile, builder};
use crate::search::seam::{SearchError, SearchProvider, SearchRequest, SearchResult};
use crate::subagent::ChildObserver;
use crate::test_support::{
    MockEnvironment, MutableMockEnvironment, ScriptedCompletion, ScriptedProvider, client_from,
    custom_tool_call_response, scripted_client,
};
use crate::tools::{
    WebFetchSummarizer, make_apply_patch_tool, make_shell_tool, make_web_fetch_tool,
};
use crate::types::{CommandTermination, ExecOutputTail, InputSource, SkillActivationSource};

const SECRET: &str = "AKIAYRWQG5EJLPZLBYNP";

/// A redactor that knows exactly one secret.
struct DropKey;

impl Redactor for DropKey {
    fn redact<'a>(&self, text: &'a str) -> Cow<'a, str> {
        Cow::Owned(text.replace(SECRET, "[REDACTED]"))
    }
}

/// An environment whose one command writes `SECRET` to standard error.
fn leaking_environment() -> Arc<MockEnvironment> {
    Arc::new(MockEnvironment {
        exec_result: ExecResult {
            stdout:      "out".to_owned(),
            stderr:      format!("boom key={SECRET}"),
            exit_code:   Some(0),
            termination: CommandTermination::Exited,
            duration_ms: 3,
        },
        ..MockEnvironment::default()
    })
}

/// The output tail of the one process event in `events`.
fn process_tail(events: &[CodingEvent]) -> ExecOutputTail {
    events
        .iter()
        .find_map(|event| match event {
            CodingEvent::ToolProcessCompleted {
                exec_output_tail, ..
            } => exec_output_tail.clone(),
            _ => None,
        })
        .expect("the shell tool published a process event with a tail")
}

#[tokio::test]
async fn a_sessions_redactor_reaches_the_process_tail_its_shell_tool_publishes() {
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response(
            "shell",
            "call_1",
            json!({"command": "printf 'boom' >&2"}),
        )),
        ScriptedCall::response(text_response("done")),
    ])
    .tools([make_shell_tool()])
    .environment(leaking_environment())
    .redacting(Arc::new(DropKey))
    .build();
    let mut events = session.subscribe();

    let output = session.prompt("run it").await.expect("the prompt succeeds");
    let published = settled(&mut session, &mut events).await;

    assert_eq!(output.as_deref(), Some("done"));
    let stderr = process_tail(&published)
        .stderr
        .expect("a standard error tail");
    assert_eq!(stderr, "boom key=[REDACTED]");
}

/// Without a redactor the tail is what the process wrote, which is a decision
/// rather than an oversight: pebble ships no secret detector and will not
/// pretend to.
#[tokio::test]
async fn a_session_without_a_redactor_publishes_what_the_process_wrote() {
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response(
            "shell",
            "call_1",
            json!({"command": "printf 'boom' >&2"}),
        )),
        ScriptedCall::response(text_response("done")),
    ])
    .tools([make_shell_tool()])
    .environment(leaking_environment())
    .build();
    let mut events = session.subscribe();

    session.prompt("run it").await.expect("the prompt succeeds");
    let published = settled(&mut session, &mut events).await;

    let stderr = process_tail(&published)
        .stderr
        .expect("a standard error tail");
    assert_eq!(stderr, format!("boom key={SECRET}"));
}

#[tokio::test]
async fn the_fetch_tools_summarizer_answers_a_web_fetch_prompt() {
    let environment = Arc::new(MockEnvironment {
        exec_result: ExecResult {
            stdout:      "<html><body><p>Rust is a language.</p></body></html>".to_owned(),
            stderr:      String::new(),
            exit_code:   Some(0),
            termination: CommandTermination::Exited,
            duration_ms: 7,
        },
        ..MockEnvironment::default()
    });
    let (client, provider) = client_from(
        ScriptedProvider::new(vec![
            ScriptedCall::response(tool_call_response(
                "web_fetch",
                "call_1",
                json!({"url": "https://example.com", "prompt": "What is Rust?"}),
            )),
            ScriptedCall::response(text_response("done")),
        ])
        .completing(vec![ScriptedCompletion::response(text_response(
            "Rust is a language.",
        ))]),
    );
    let summarizer = Arc::new(WebFetchSummarizer::new(client.clone(), "test/model"));
    let mut session = builder(client)
        .with_profile(TestProfile::with_tools(vec![make_web_fetch_tool(Some(
            summarizer,
        ))]))
        .environment(environment as Arc<dyn Environment>)
        .build()
        .expect("the session builds");

    session
        .prompt("read that page")
        .await
        .expect("the prompt succeeds");

    let results = tool_results(&session, 2);
    assert_eq!(results.len(), 1);
    assert_eq!(result_text(&results[0]), "Rust is a language.");
    assert_eq!(
        provider.completion_count(),
        1,
        "the summarizing call is the session's only non-streaming one"
    );
}

#[tokio::test]
async fn a_fetch_tool_without_a_summarizer_returns_the_page_instead() {
    let environment = Arc::new(MockEnvironment {
        exec_result: ExecResult {
            stdout:      "<html><body><p>Rust is a language.</p></body></html>".to_owned(),
            stderr:      String::new(),
            exit_code:   Some(0),
            termination: CommandTermination::Exited,
            duration_ms: 7,
        },
        ..MockEnvironment::default()
    });
    let (mut session, provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response(
            "web_fetch",
            "call_1",
            json!({"url": "https://example.com", "prompt": "What is Rust?"}),
        )),
        ScriptedCall::response(text_response("done")),
    ])
    .tools([make_web_fetch_tool(None)])
    .environment(environment)
    .build();

    session
        .prompt("read that page")
        .await
        .expect("the prompt succeeds");

    let results = tool_results(&session, 2);
    assert_eq!(
        result_text(&results[0]),
        "[Note: prompt summarization unavailable, returning full content]\n\nRust is a language."
    );
    assert_eq!(provider.completion_count(), 0, "no model was asked");
}

/// The environment variables a session resolves per call reach a command the
/// model asked for, which is how a prompt-scoped credential gets to a tool
/// without living in the session's own environment.
#[tokio::test]
async fn a_sessions_tool_environment_reaches_the_command_a_tool_runs() {
    let environment = Arc::new(MockEnvironment::default());
    let (client, _provider) = scripted_client(vec![
        ScriptedCall::response(tool_call_response(
            "shell",
            "call_1",
            json!({"command": "echo $GITHUB_TOKEN"}),
        )),
        ScriptedCall::response(text_response("done")),
    ]);
    let mut session = builder(client)
        .with_profile(TestProfile::with_tools(vec![make_shell_tool()]))
        .environment(Arc::clone(&environment) as Arc<dyn Environment>)
        .tool_env(HashMap::from([(
            "GITHUB_TOKEN".to_owned(),
            "t1".to_owned(),
        )]))
        .build()
        .expect("the session builds");

    session.prompt("run it").await.expect("the prompt succeeds");

    assert_eq!(
        *environment
            .captured_command
            .lock()
            .expect("captured_command lock is not poisoned"),
        Some("echo $GITHUB_TOKEN".to_owned())
    );
    assert_eq!(
        *environment
            .captured_env_vars
            .lock()
            .expect("captured_env_vars lock is not poisoned"),
        Some(HashMap::from([(
            "GITHUB_TOKEN".to_owned(),
            "t1".to_owned()
        )]))
    );
}

// --- Skills: the one tool a session registers for itself ---

/// A mock environment holding the skill files `globs` will find.
fn environment_with_skills(files: Vec<(&str, &str)>, globs: Vec<&str>) -> Arc<MockEnvironment> {
    Arc::new(MockEnvironment {
        files: files
            .into_iter()
            .map(|(path, content)| (path.to_owned(), content.to_owned()))
            .collect::<HashMap<_, _>>(),
        glob_results: globs.into_iter().map(ToOwned::to_owned).collect(),
        ..MockEnvironment::linux()
    })
}

/// A session whose one discovered skill is `commit`, answering `calls`.
fn session_with_a_skill(calls: Vec<ScriptedCall>) -> (CodingRuntime, Arc<ScriptedProvider>) {
    TestSession::new(calls)
        .environment(environment_with_skills(
            vec![(
                "/skills/commit/SKILL.md",
                "---\nname: commit\ndescription: Make a commit\n---\nRun commit.",
            )],
            vec!["/skills/commit/SKILL.md"],
        ))
        .options(CodingAgentOptions {
            skill_dirs: vec!["/skills".to_owned()],
            ..CodingAgentOptions::default()
        })
        .build()
}

/// Every `SkillActivated` in `events`, as `(name, source)` pairs.
fn activations(events: &[CodingEvent]) -> Vec<(String, SkillActivationSource)> {
    events
        .iter()
        .filter_map(|event| match event {
            CodingEvent::SkillActivated { skill_name, source } => {
                Some((skill_name.clone(), *source))
            }
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn a_session_that_discovered_skills_advertises_the_tool_that_loads_one() {
    let (mut session, _provider) = session_with_a_skill(answers("ok"));

    assert!(
        !session
            .registered_tools()
            .iter()
            .any(|tool| tool.definition.name == "use_skill"),
        "nothing is advertised before the directories are read"
    );
    session.initialize().await.expect("initialization succeeds");

    let skill_tool = session
        .registered_tools()
        .into_iter()
        .find(|tool| tool.definition.name == "use_skill")
        .expect("the skill tool is registered");
    assert_eq!(skill_tool.source, ToolSource::Skill);
}

#[tokio::test]
async fn a_session_with_no_skills_advertises_no_way_to_load_one() {
    let (mut session, _provider) = TestSession::new(answers("ok")).build();

    session.initialize().await.expect("initialization succeeds");

    assert!(
        !session
            .registered_tools()
            .iter()
            .any(|tool| tool.definition.name == "use_skill")
    );
}

#[tokio::test]
async fn an_unknown_slash_skill_is_an_invalid_input_error() {
    let (mut session, provider) = session_with_a_skill(answers("unused"));
    session.initialize().await.expect("initialization succeeds");

    let error = session
        .prompt("/missing do this")
        .await
        .expect_err("the input names no discovered skill");

    assert!(matches!(error, Error::SkillExpansion(_)), "{error:?}");
    let data = ErrorData::from(&error);
    assert_eq!(data.kind, ErrorKind::InvalidInput);
    assert_eq!(
        data.message,
        "expanding a skill reference: Unknown skill: /missing"
    );
    assert_eq!(
        provider.call_count(),
        0,
        "invalid input makes no model call"
    );
}

#[tokio::test]
async fn an_unknown_slash_skill_in_a_follow_up_keeps_its_typed_error() {
    let (mut session, _provider) = session_with_a_skill(answers("first"));
    session.initialize().await.expect("initialization succeeds");
    session.follow_up("/missing do this");

    let error = session
        .prompt("start")
        .await
        .expect_err("the follow-up names no discovered skill");

    assert!(matches!(error, Error::SkillExpansion(_)), "{error:?}");
    assert_eq!(ErrorData::from(&error).kind, ErrorKind::InvalidInput);
}

#[tokio::test]
async fn typed_input_with_two_slash_references_is_refused() {
    let (mut session, provider) = session_with_a_skill(answers("unused"));
    session.initialize().await.expect("initialization succeeds");

    let error = session
        .prompt("cd /workspace using /tmp storage")
        .await
        .expect_err("two references are ambiguous typed input");

    assert!(matches!(error, Error::SkillExpansion(_)), "{error:?}");
    assert_eq!(
        ErrorData::from(&error).message,
        "expanding a skill reference: Only one skill reference per input is allowed"
    );
    assert_eq!(
        provider.call_count(),
        0,
        "invalid input makes no model call"
    );
}

#[tokio::test]
async fn synthesized_input_passes_slash_tokens_through_unchanged() {
    let (mut session, provider) = session_with_a_skill(answers("done"));
    session.initialize().await.expect("initialization succeeds");

    let output = session
        .prompt(
            CodingInput::text("cd /workspace using /tmp storage").with_source(InputSource::Agent),
        )
        .await
        .expect("synthesized input is not skill-expanded");

    assert_eq!(output.as_deref(), Some("done"));
    assert_eq!(provider.call_count(), 1, "the turn ran one model call");
}

#[tokio::test]
async fn a_synthesized_queued_turn_passes_slash_tokens_through() {
    let (mut session, _provider) = session_with_a_skill(answers("first"));
    session.initialize().await.expect("initialization succeeds");
    session.follow_up_unattributed("cd /workspace using /tmp storage");

    let output = session
        .prompt("start")
        .await
        .expect("a queued synthesized turn is not skill-expanded");

    assert_eq!(
        output.as_deref(),
        Some("first"),
        "the scripted provider answers forever, so the turn completed"
    );
}

#[tokio::test]
async fn use_skill_tool_success_emits_skill_activated_with_tool_source() {
    let (mut session, _provider) = session_with_a_skill(vec![
        ScriptedCall::response(tool_call_response(
            "use_skill",
            "call_1",
            json!({"skill_name": "commit"}),
        )),
        ScriptedCall::response(text_response("done")),
    ]);
    session.initialize().await.expect("initialization succeeds");
    let mut events = session.subscribe();

    session
        .prompt("please commit")
        .await
        .expect("the prompt succeeds");

    let results = tool_results(&session, 2);
    assert_eq!(result_text(&results[0]), "Run commit.");
    let published = settled(&mut session, &mut events).await;
    assert_eq!(activations(&published), vec![(
        "commit".to_owned(),
        SkillActivationSource::Tool
    )]);
}

#[tokio::test]
async fn use_skill_tool_failed_lookup_does_not_emit_activation() {
    let (mut session, _provider) = session_with_a_skill(vec![
        ScriptedCall::response(tool_call_response(
            "use_skill",
            "call_1",
            json!({"skill_name": "nope"}),
        )),
        ScriptedCall::response(text_response("done")),
    ]);
    session.initialize().await.expect("initialization succeeds");
    let mut events = session.subscribe();

    session
        .prompt("please commit")
        .await
        .expect("the prompt succeeds");

    let results = tool_results(&session, 2);
    assert!(results[0].is_error, "the call failed");
    assert_eq!(result_text(&results[0]), "Unknown skill: nope");
    let published = settled(&mut session, &mut events).await;
    assert!(
        activations(&published).is_empty(),
        "a lookup that found nothing activated nothing: {published:?}"
    );
}

// --- apply_patch: a custom tool call, through the whole round ---

/// The files a patched project starts with.
fn project() -> Arc<MutableMockEnvironment> {
    Arc::new(MutableMockEnvironment::new(HashMap::from([
        (
            "src/app.py".to_owned(),
            "def greet(name):\n    return f\"Hi, {name}\"\n\ndef farewell(name):\n    return \
             f\"Bye, {name}\"\n"
                .to_owned(),
        ),
        (
            "src/obsolete.py".to_owned(),
            "def old():\n    pass\n".to_owned(),
        ),
    ])))
}

#[tokio::test]
async fn a_custom_patch_call_reaches_the_tool_and_changes_the_files() {
    let patch = "\
*** Begin Patch
*** Add File: src/created.py
+def created():
+    return \"created\"
*** Update File: src/app.py
@@ def greet(name):
-    return f\"Hi, {name}\"
+    return f\"Hello, {name}!\"
@@ def farewell(name):
-    return f\"Bye, {name}\"
+    return f\"Goodbye, {name}!\"
*** Delete File: src/obsolete.py
*** End Patch";
    let environment = project();
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(custom_tool_call_response("apply_patch", "call_1", patch)),
        ScriptedCall::response(text_response("Done! Updated greet and farewell functions.")),
    ])
    .tools([make_apply_patch_tool()])
    .environment(Arc::clone(&environment) as Arc<dyn Environment>)
    .build();

    let output = session
        .prompt("Update the greeting functions")
        .await
        .expect("the prompt succeeds");

    assert_eq!(
        output.as_deref(),
        Some("Done! Updated greet and farewell functions.")
    );
    let results = tool_results(&session, 2);
    assert!(!results[0].is_error, "the patch applied");
    assert_eq!(
        result_text(&results[0]),
        "Success. Updated the following files:\nA src/created.py\nM src/app.py\nD \
         src/obsolete.py\n"
    );
    let updated = environment
        .read_file_text("src/app.py")
        .await
        .expect("the patched file is readable");
    assert!(updated.contains("Hello, {name}!"), "{updated}");
    assert!(updated.contains("Goodbye, {name}!"), "{updated}");
    assert!(
        environment.read_file_text("src/obsolete.py").await.is_err(),
        "the deleted file is gone"
    );
    assert_eq!(
        environment
            .read_file_text("src/created.py")
            .await
            .expect("the added file is readable"),
        "def created():\n    return \"created\"\n"
    );
}

/// A patch that does not match is the model's to repair, so the failure goes
/// back into the conversation as the call's result rather than ending the
/// prompt.
#[tokio::test]
async fn failed_custom_tool_call_returns_codex_style_error_to_session_history() {
    let patch = "\
*** Begin Patch
*** Update File: src/app.py
@@ def missing():
-    return 1
+    return 2
*** End Patch";
    let environment = Arc::new(MutableMockEnvironment::new(HashMap::from([(
        "src/app.py".to_owned(),
        "def present():\n    return 1\n".to_owned(),
    )])));
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(custom_tool_call_response("apply_patch", "call_1", patch)),
        ScriptedCall::response(text_response("I will correct the patch.")),
    ])
    .tools([make_apply_patch_tool()])
    .environment(environment as Arc<dyn Environment>)
    .build();

    session
        .prompt("Patch a missing function")
        .await
        .expect("the prompt succeeds");

    let results = tool_results(&session, 2);
    assert_eq!(results.len(), 1);
    assert!(results[0].is_error);
    assert_eq!(
        result_text(&results[0]),
        "Failed to find context 'def missing():' in src/app.py"
    );
}

// --- web_search: the tool a session advertises only when it can search ---

/// A provider that answers with one result and records what it was asked.
#[derive(Default)]
struct RecordingSearch {
    requests: Mutex<Vec<SearchRequest>>,
}

impl RecordingSearch {
    fn requests(&self) -> Vec<SearchRequest> {
        self.requests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

#[async_trait]
impl SearchProvider for RecordingSearch {
    async fn search(&self, request: SearchRequest) -> StdResult<Vec<SearchResult>, SearchError> {
        self.requests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(request);
        Ok(vec![SearchResult::new(
            "Rust Lang",
            "https://rust-lang.org",
            "A systems language",
        )])
    }
}

#[tokio::test]
async fn a_session_with_no_search_provider_advertises_no_search_tool() {
    let (client, _provider) = scripted_client(answers("ok"));
    let session = builder(client).build().expect("the session builds");

    assert!(
        !session
            .registered_tools()
            .iter()
            .any(|tool| tool.definition.name == "web_search"),
        "nothing to search through is nothing to advertise"
    );
}

#[tokio::test]
async fn a_configured_search_provider_reaches_the_registered_tool() {
    let provider = Arc::new(RecordingSearch::default());
    let (client, _scripted) = scripted_client(vec![
        ScriptedCall::response(tool_call_response(
            "web_search",
            "call_1",
            json!({"query": "rust", "max_results": 3}),
        )),
        ScriptedCall::response(text_response("done")),
    ]);
    let mut session = builder(client)
        .search_provider(Arc::clone(&provider) as Arc<dyn SearchProvider>)
        .build()
        .expect("the session builds");

    assert!(
        session
            .registered_tools()
            .iter()
            .any(|tool| tool.definition.name == "web_search"),
        "a session that can search says so"
    );
    session
        .prompt("look it up")
        .await
        .expect("the prompt succeeds");

    let results = tool_results(&session, 2);
    assert_eq!(
        result_text(&results[0]),
        "1. Rust Lang\n   https://rust-lang.org\n   A systems language\n\n"
    );
    assert_eq!(provider.requests(), vec![SearchRequest::new("rust", 3)]);
}

/// A child researches the task its parent gave it, so it searches through the
/// same provider.
#[tokio::test]
async fn a_child_inherits_its_parents_search_provider() {
    let searched: Arc<Mutex<Vec<bool>>> = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::clone(&searched);
    let observer: ChildObserver = Arc::new(move |child: &CodingRuntime| {
        recorder
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(
                child
                    .registered_tools()
                    .iter()
                    .any(|tool| tool.definition.name == "web_search"),
            );
    });
    let (parent, _provider) = TestSession::new(answers("ok"))
        .searching_with(Arc::new(RecordingSearch::default()))
        .observe_children(observer)
        .build();
    let supervisor = parent
        .subagent_supervisor()
        .expect("the parent was given a factory")
        .clone();

    supervisor
        .spawn(parent.session(), "research".to_owned())
        .expect("the spawn succeeds");

    assert_eq!(
        *searched.lock().unwrap_or_else(PoisonError::into_inner),
        vec![true],
        "the child searches through the same provider"
    );
}
