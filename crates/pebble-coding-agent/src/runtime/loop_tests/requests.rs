//! What the session asks the model for.
//!
//! One request per round, built from the system prompt, the conversation, and
//! the tools its middleware leaves exposed. These tests read the requests
//! the scripted provider captured, which is the only place the session's own
//! decisions about a round are visible.

use lithos_llm::types::{ContentPart, Message as LlmMessage, ReasoningEffort, Request, Role};
use pebble_agent::{
    ToolCallNext, ToolCallRequest, ToolDescriptor, ToolMiddleware, ToolOutcome, ToolSystemError,
};
use serde_json::{Value, json};

use super::super::testing::wait_for_event;
use super::*;
use crate::SessionScope;
use crate::task_reminder::TASK_REMINDER_TEXT;
use crate::test_support::message_text;
use crate::tool::{
    ApprovalDecision, PermissionMiddleware, ToolApprovalService, ToolPermission,
    ToolPermissionPolicy,
};

/// A policy that answers from a table and denies whatever it does not know.
struct NamedToolPermissionPolicy {
    decisions: Vec<(&'static str, ToolPermission)>,
}

impl NamedToolPermissionPolicy {
    fn installed(decisions: Vec<(&'static str, ToolPermission)>) -> Arc<dyn ToolPermissionPolicy> {
        Arc::new(Self { decisions })
    }
}

impl ToolPermissionPolicy for NamedToolPermissionPolicy {
    fn permission(&self, _session: &SessionScope, tool: &ToolDescriptor) -> ToolPermission {
        self.decisions
            .iter()
            .find_map(|(id, permission)| (id == &tool.id().as_str()).then(|| permission.clone()))
            .unwrap_or_else(|| ToolPermission::Deny {
                reason: "not allowed by the test policy".to_owned(),
            })
    }
}

struct FixedApproval(ApprovalDecision);

#[async_trait::async_trait]
impl ToolApprovalService for FixedApproval {
    async fn approve(
        &self,
        _request: &ToolCallRequest,
    ) -> StdResult<ApprovalDecision, ToolSystemError> {
        Ok(self.0.clone())
    }
}

struct RecordingApproval {
    captured: Arc<Mutex<Option<(String, Value)>>>,
}

#[async_trait::async_trait]
impl ToolApprovalService for RecordingApproval {
    async fn approve(
        &self,
        request: &ToolCallRequest,
    ) -> StdResult<ApprovalDecision, ToolSystemError> {
        *self.captured.lock().unwrap_or_else(PoisonError::into_inner) = Some((
            request.call().name.clone(),
            request
                .call()
                .input
                .to_value()
                .expect("valid fixture arguments"),
        ));
        Ok(ApprovalDecision::Allow)
    }
}

fn permission_middleware(
    decisions: Vec<(&'static str, ToolPermission)>,
    approval: Option<Arc<dyn ToolApprovalService>>,
) -> Arc<dyn ToolMiddleware> {
    let middleware = PermissionMiddleware::new(NamedToolPermissionPolicy::installed(decisions));
    Arc::new(match approval {
        Some(approval) => middleware.with_approval(approval),
        None => middleware,
    })
}

/// The names of the tools the request exposed.
fn tool_names(request: &Request) -> Vec<&str> {
    request
        .tools()
        .iter()
        .map(|tool| tool.name.as_str())
        .collect()
}

#[tokio::test]
async fn the_system_prompt_carries_the_application_instructions() {
    let (mut session, provider) = TestSession::new(answers("captured"))
        .options(CodingAgentOptions {
            user_instructions: Some("Always use TDD".to_owned()),
            ..CodingAgentOptions::default()
        })
        .build();
    session.initialize().await.expect("initialization succeeds");

    session.prompt("test").await.expect("the prompt succeeds");

    let requests = provider.requests();
    let request = requests.first().expect("the round was requested");
    let system = request
        .messages()
        .first()
        .expect("the request is not empty");
    assert_eq!(system.role(), Role::System);
    assert!(
        message_text(system).contains("Always use TDD"),
        "the instructions reach the model: {}",
        message_text(system)
    );
}

#[tokio::test]
async fn a_session_with_no_prompt_sends_no_system_message() {
    // Deliberately not initialized, so the system prompt is still empty.
    let (mut session, provider) = TestSession::answering(answers("captured"));

    session.prompt("test").await.expect("the prompt succeeds");

    let requests = provider.requests();
    let request = requests.first().expect("the round was requested");
    assert!(
        request
            .messages()
            .iter()
            .all(|message| message.role() != Role::System),
        "an empty prompt is left out rather than sent empty"
    );
    assert_eq!(
        request.messages().first().map(LlmMessage::role),
        Some(Role::User),
        "the input is the first thing the model reads"
    );
}

/// fabro-c98c: the openai-wire error-flag wrap must ride the REQUEST, not
/// the first agent build — a tool result that fails mid-session reaches the
/// next request rendered, or the openai wire drops the flag and the model
/// reads the failure as success (the mimo tool-loop incident class: 176
/// identical failing calls the model could not see). The test catalog's
/// `openai-tools` row names no profile of its own, so it inherits the
/// provider's `openai` — the same profile kind zai resolves to, which is
/// what the production loop lane runs under.
#[tokio::test]
async fn a_mid_session_failed_tool_result_is_wrapped_in_the_next_request() {
    let (mut session, provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response("fail_tool", "call_1", json!({}))),
        ScriptedCall::response(text_response("recovered")),
    ])
    .model("test/openai-tools")
    .tools([failing_tool()])
    .build();

    session
        .prompt("Use fail tool")
        .await
        .expect("the prompt succeeds");

    let requests = provider.requests();
    let second = requests.get(1).expect("the round after the failure");
    let tool_message = second
        .messages()
        .iter()
        .rev()
        .find(|message| message.role() == Role::Tool)
        .expect("the failed call's result rides the next request");
    let ContentPart::ToolResult(result) = tool_message
        .content()
        .iter()
        .find_map(|part| match part {
            part @ ContentPart::ToolResult(_) => Some(part),
            _ => None,
        })
        .expect("the tool message carries the result")
    else {
        unreachable!("the part is a tool result");
    };
    let ContentPart::Text { text } = result
        .content
        .iter()
        .find_map(|part| match part {
            part @ ContentPart::Text { .. } => Some(part),
            _ => None,
        })
        .expect("the failure is text")
    else {
        unreachable!("the part is text");
    };
    assert!(
        text.starts_with("[TOOL ERROR]"),
        "the error flag is rendered into the content: {text}"
    );
    assert!(
        !result.is_error,
        "the marker replaces the flag (fabro-c98c): the chat wire reports \
         unsupported_control for a retained flag, and the marker already \
         carries the failure"
    );
}

#[tokio::test]
async fn every_registered_tool_is_exposed_when_no_policy_says_otherwise() {
    let (mut session, provider) = TestSession::new(answers("captured"))
        .tools([noop_tool("read_file"), noop_tool("write_file")])
        .build();

    session.prompt("test").await.expect("the prompt succeeds");

    let requests = provider.requests();
    let request = requests.first().expect("the round was requested");
    let mut names = tool_names(request);
    names.sort_unstable();
    assert_eq!(names, ["read_file", "write_file"]);
}

#[tokio::test]
async fn a_denied_tool_is_never_advertised() {
    let (mut session, provider) = TestSession::new(answers("captured"))
        .tools([noop_tool("read_file"), noop_tool("write_file")])
        .tool_middleware(permission_middleware(
            vec![
                ("read_file", ToolPermission::Allow),
                ("write_file", ToolPermission::Deny {
                    reason: "writes are disabled".to_owned(),
                }),
            ],
            None,
        ))
        .build();

    session.prompt("test").await.expect("the prompt succeeds");

    let requests = provider.requests();
    let request = requests.first().expect("the round was requested");
    assert_eq!(tool_names(request), ["read_file"]);
}

#[tokio::test]
async fn an_approval_required_tool_is_advertised_when_approval_is_available() {
    let (mut session, provider) = TestSession::new(answers("captured"))
        .tools([noop_tool("read_file"), noop_tool("shell")])
        .tool_middleware(permission_middleware(
            vec![
                ("read_file", ToolPermission::Allow),
                ("shell", ToolPermission::RequireApproval),
            ],
            Some(Arc::new(FixedApproval(ApprovalDecision::Allow))),
        ))
        .build();

    session.prompt("test").await.expect("the prompt succeeds");

    let requests = provider.requests();
    let request = requests.first().expect("the round was requested");
    let mut names = tool_names(request);
    names.sort_unstable();
    assert_eq!(names, ["read_file", "shell"]);
}

#[tokio::test]
async fn ten_unused_assistant_turns_bring_back_the_task_reminder() {
    let (mut session, provider) = TestSession::new(answers("done"))
        .tools([noop_tool("TaskCreate"), noop_tool("TaskUpdate")])
        .build();

    for index in 0..=10 {
        session
            .prompt(&format!("turn {index}"))
            .await
            .expect("the prompt succeeds");
    }

    let requests = provider.requests();
    let last = requests.last().expect("the last round was requested");
    assert!(
        last.messages().iter().any(|message| {
            message.role() == Role::System && message_text(message) == TASK_REMINDER_TEXT
        }),
        "the eleventh round reminds the model of its task tools"
    );
}

#[tokio::test]
async fn hidden_task_tools_do_not_trigger_a_task_reminder() {
    let (mut session, provider) = TestSession::new(answers("done"))
        .tools([noop_tool("TaskCreate"), noop_tool("TaskUpdate")])
        .tool_middleware(permission_middleware(
            vec![
                ("TaskCreate", ToolPermission::Deny {
                    reason: "tasks are disabled".to_owned(),
                }),
                ("TaskUpdate", ToolPermission::Deny {
                    reason: "tasks are disabled".to_owned(),
                }),
            ],
            None,
        ))
        .build();

    for index in 0..=10 {
        session
            .prompt(&format!("turn {index}"))
            .await
            .expect("the prompt succeeds");
    }

    let requests = provider.requests();
    let last = requests.last().expect("the last round was requested");
    assert!(
        last.messages().iter().all(|message| {
            message.role() != Role::System || message_text(message) != TASK_REMINDER_TEXT
        }),
        "hidden task tools are not mentioned to the model"
    );
}

#[tokio::test]
async fn the_reasoning_effort_a_session_is_given_reaches_the_request() {
    // A model whose catalog row says it can be asked to think harder.
    let (mut session, provider) = TestSession::new(answers("captured"))
        .model("test/thinking")
        .build();
    session.set_reasoning_effort(Some(ReasoningEffort::High));

    session.prompt("test").await.expect("the prompt succeeds");

    let requests = provider.requests();
    let request = requests.first().expect("the round was requested");
    assert_eq!(request.reasoning_effort(), Some(ReasoningEffort::High));
}

// --- Approving a call ---

/// The tool round every approval test runs.
fn approval_calls() -> Vec<ScriptedCall> {
    vec![
        ScriptedCall::response(tool_call_response(
            "echo",
            "call_1",
            json!({"text": "hello"}),
        )),
        ScriptedCall::response(text_response("Done")),
    ]
}

#[tokio::test]
async fn a_refused_call_answers_the_model_with_the_reason() {
    let (mut session, _provider) = TestSession::new(approval_calls())
        .tools([echo_tool()])
        .tool_middleware(permission_middleware(
            vec![("echo", ToolPermission::RequireApproval)],
            Some(Arc::new(FixedApproval(ApprovalDecision::Deny {
                reason: "denied by policy".to_owned(),
            }))),
        ))
        .build();
    let mut events = session.subscribe();

    session
        .prompt("Use echo")
        .await
        .expect("the prompt succeeds");

    assert_eq!(session.state(), CodingAgentState::Idle);
    assert_eq!(session.history().turns().len(), 4);
    let results = tool_results(&session, 2);
    assert!(results[0].is_error);
    assert!(
        result_text(&results[0]).contains("denied by policy"),
        "the model reads why the call was refused"
    );

    let published = settled(&mut session, &mut events).await;
    let completions: Vec<&CodingEvent> = published
        .iter()
        .filter(|event| matches!(event, CodingEvent::ToolCallCompleted { .. }))
        .collect();
    assert_eq!(completions.len(), 1);
    assert!(matches!(completions[0], CodingEvent::ToolCallCompleted {
        is_error: true,
        ..
    }));
}

#[tokio::test]
async fn an_approved_call_runs() {
    let (mut session, _provider) = TestSession::new(approval_calls())
        .tools([echo_tool()])
        .tool_middleware(permission_middleware(
            vec![("echo", ToolPermission::RequireApproval)],
            Some(Arc::new(FixedApproval(ApprovalDecision::Allow))),
        ))
        .build();

    session
        .prompt("Use echo")
        .await
        .expect("the prompt succeeds");

    let results = tool_results(&session, 2);
    assert!(!results[0].is_error);
    assert_eq!(result_text(&results[0]), "echo: hello");
}

#[tokio::test]
async fn the_approval_service_sees_the_call_the_model_asked_for() {
    let captured: Arc<Mutex<Option<(String, Value)>>> = Arc::new(Mutex::new(None));
    let recorder = Arc::clone(&captured);
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response(
            "echo",
            "call_1",
            json!({"text": "world"}),
        )),
        ScriptedCall::response(text_response("Done")),
    ])
    .tools([echo_tool()])
    .tool_middleware(permission_middleware(
        vec![("echo", ToolPermission::RequireApproval)],
        Some(Arc::new(RecordingApproval { captured: recorder })),
    ))
    .build();

    session
        .prompt("Use echo")
        .await
        .expect("the prompt succeeds");

    let seen = captured.lock().unwrap_or_else(PoisonError::into_inner);
    let (name, arguments) = seen.as_ref().expect("the approval service was called");
    assert_eq!(name, "echo");
    assert_eq!(arguments, &json!({"text": "world"}));
}

struct RecordingToolMiddleware {
    entries: Arc<Mutex<Vec<String>>>,
}

#[async_trait::async_trait]
impl ToolMiddleware for RecordingToolMiddleware {
    async fn call(
        &self,
        request: ToolCallRequest,
        next: ToolCallNext<'_>,
    ) -> StdResult<ToolOutcome, ToolSystemError> {
        let name = request.call().name.clone();
        let id = request.call().id.clone();
        self.entries
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(format!("before {name}"));
        let outcome = next.run(request).await?;
        let status = match &outcome {
            ToolOutcome::Success { .. } => "success",
            ToolOutcome::Failure { .. } => "failure",
            _ => "unknown",
        };
        self.entries
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(format!("after {name} {id} {status}"));
        Ok(outcome)
    }
}

#[tokio::test]
async fn tool_middleware_brackets_the_call() {
    let entries = Arc::new(Mutex::new(Vec::new()));
    let (mut session, _provider) = TestSession::new(approval_calls())
        .tools([echo_tool()])
        .tool_middleware(Arc::new(RecordingToolMiddleware {
            entries: Arc::clone(&entries),
        }))
        .build();

    session
        .prompt("Use echo")
        .await
        .expect("the prompt succeeds");

    let recorded = entries.lock().unwrap_or_else(PoisonError::into_inner);
    assert_eq!(recorded.as_slice(), [
        "before echo",
        "after echo call_1 success"
    ]);
}

#[tokio::test]
async fn a_session_with_no_middleware_runs_the_call() {
    let (mut session, _provider) = TestSession::new(approval_calls())
        .tools([echo_tool()])
        .build();

    session
        .prompt("Use echo")
        .await
        .expect("the prompt succeeds");

    let results = tool_results(&session, 2);
    assert!(!results[0].is_error);
    assert_eq!(result_text(&results[0]), "echo: hello");
}

// --- Watching a round from outside ---

#[tokio::test]
async fn a_subscriber_sees_the_round_while_it_runs() {
    let (mut session, _provider) = TestSession::new(vec![
        ScriptedCall::response(tool_call_response("echo", "call_1", json!({"text": "hi"}))),
        ScriptedCall::response(text_response("Done")),
    ])
    .tools([echo_tool()])
    .build();
    let mut events = session.subscribe();
    let watcher = tokio::spawn(async move {
        wait_for_event(&mut events, |event| {
            matches!(event, CodingEvent::ToolCallStarted { tool_name, .. } if tool_name == "echo")
        })
        .await;
        wait_for_event(&mut events, |event| {
            matches!(event, CodingEvent::ToolCallCompleted { .. })
        })
        .await;
    });

    session
        .prompt("Use echo")
        .await
        .expect("the prompt succeeds");

    watcher.await.expect("the watcher saw the whole call");
}
