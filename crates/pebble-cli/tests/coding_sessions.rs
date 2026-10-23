//! Public Rust API scenarios with real tools and the OpenAI HTTP codec.
//!
//! These live beside the CLI tests to reuse its concrete adapter and twin
//! dependencies. Only the provider's responses are scripted.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::{fs, mem};

use async_trait::async_trait;
use axum::body::{Body, to_bytes};
use axum::extract::Request;
use axum::middleware::{Next, from_fn};
use lithos_llm::Client;
use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::{Credentials, SecretValue, StaticCredentials};
use lithos_llm::middleware::RetryPolicy;
use pebble_coding_agent::environment::LocalEnvironment;
use pebble_coding_agent::events::{
    CodingAgentEvent, CodingEvent, EventSink, EventSinkError, LlmRetryPhase,
};
use pebble_coding_agent::state::Message;
use pebble_coding_agent::{CodingAgent, CodingAgentOptions, ShutdownReason};
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::task::{JoinError, JoinHandle};
use tokio::time::timeout;
use twin_openai::config::Config;

const PATIENCE: Duration = Duration::from_secs(20);
const MODEL: &str = "openai/gpt-5.6-sol";

struct Twin {
    url:       String,
    requests:  Arc<Mutex<Vec<Value>>>,
    task:      JoinHandle<()>,
    _fixtures: TempDir,
}

impl Twin {
    async fn start(scenarios: Vec<Value>) -> Self {
        let fixtures = tempfile::tempdir().expect("twin fixtures");
        let path = fixtures.path().join("scenarios.json");
        fs::write(
            &path,
            serde_json::to_vec(&json!({"scenarios": scenarios})).expect("serialize scenarios"),
        )
        .expect("write scenarios");
        let mut config = Config::from_lookup(&|_| None).expect("twin config");
        config.scenarios_path = Some(path);
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&requests);
        let app = twin_openai::build_app_with_config(config)
            .expect("twin app")
            .layer(from_fn(move |request: Request, next: Next| {
                let captured = Arc::clone(&captured);
                async move {
                    let (parts, body) = request.into_parts();
                    let bytes = to_bytes(body, 4 * 1024 * 1024)
                        .await
                        .expect("bounded HTTP request");
                    captured
                        .lock()
                        .unwrap()
                        .push(serde_json::from_slice(&bytes).expect("JSON request"));
                    next.run(Request::from_parts(parts, Body::from(bytes)))
                        .await
                }
            }));
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind twin");
        let url = format!("http://{}/v1", listener.local_addr().expect("twin address"));
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("serve twin");
        });
        Self {
            url,
            requests,
            task,
            _fixtures: fixtures,
        }
    }

    async fn agent(
        &self,
        workspace: &Path,
        options: CodingAgentOptions,
        events: &Arc<EventLog>,
    ) -> CodingAgent {
        let catalog = Catalog::builder()
            .with_builtin()
            .toml_layer(
                "local twin",
                &format!(
                    "schema_version = 1\n[providers.openai]\nbase_url = {:?}\n",
                    self.url
                ),
            )
            .expect("override provider URL")
            .build()
            .expect("model catalog");
        let client = Client::builder()
            .catalog(catalog)
            .credentials(StaticCredentials::new().with(
                "openai",
                Credentials::bearer(SecretValue::new("coding-session")),
            ))
            .build()
            .expect("HTTP client")
            .client;
        let environment = LocalEnvironment::new(workspace);
        environment.prepare().await.expect("prepare local tools");
        CodingAgent::builder(client, Arc::new(environment))
            .model(MODEL)
            .options(options.with_wall_clock_timeout(PATIENCE))
            .event_sink(Arc::clone(events) as Arc<dyn EventSink>)
            .build()
            .await
            .expect("build coding agent")
    }

    fn requests(&self) -> Vec<Value> {
        self.requests.lock().unwrap().clone()
    }

    async fn stop(&mut self) {
        self.task.abort();
        assert!(
            (&mut self.task)
                .await
                .as_ref()
                .is_err_and(JoinError::is_cancelled)
        );
    }
}

impl Drop for Twin {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[derive(Default)]
struct EventLog(Mutex<Vec<CodingAgentEvent>>);

#[async_trait]
impl EventSink for EventLog {
    async fn record(&self, event: &CodingAgentEvent) -> Result<(), EventSinkError> {
        self.0.lock().unwrap().push(event.clone());
        Ok(())
    }
}

impl EventLog {
    fn take(&self) -> Vec<CodingAgentEvent> {
        mem::take(&mut *self.0.lock().unwrap())
    }
}

fn scenario(step: usize, input: &str, script: Value) -> Value {
    let mut scenario = json!({
        "scenario_id": format!("coding-session/{step}"),
        "namespace": "coding-session",
        "matcher": {"endpoint": "responses", "input_contains": input},
    });
    scenario["script"] = script;
    scenario
}

fn answer(text: &str) -> Value {
    json!({"kind": "success", "usage": {"input_tokens": 10, "output_tokens": 5}, "response_text": text})
}

fn shell(id: &str, command: &str) -> Value {
    let mut script = answer("Running the command.");
    script["tool_calls"] =
        json!([{"id": id, "name": "shell_command", "arguments": {"command": command}}]);
    script
}

fn tool_output<'a>(request: &'a Value, id: &str) -> &'a str {
    let outputs: Vec<_> = request["input"]
        .as_array()
        .expect("Responses input array")
        .iter()
        .filter(|item| item["type"] == "function_call_output" && item["call_id"] == id)
        .collect();
    assert_eq!(outputs.len(), 1, "exactly one result for {id}");
    outputs[0]["output"].as_str().expect("text tool output")
}

#[tokio::test]
async fn oversized_output_is_bounded_on_the_wire_and_in_events() {
    let workspace = tempfile::tempdir().unwrap();
    // Exercise the environment's 1 MiB capture cap, UTF-8 boundaries, and
    // JSON escaping, with a marker that must disappear from every consumer.
    let padding = "é\"\t\n".repeat(250_000);
    let payload = format!("OUTPUT_HEAD\n{padding}DISCARDED_MIDDLE\n{padding}OUTPUT_TAIL\n");
    fs::write(workspace.path().join("output.txt"), &payload).unwrap();
    let mut twin = Twin::start(vec![
        scenario(1, "Read the output", shell("read-output", "cat output.txt")),
        scenario(2, "OUTPUT_TAIL", answer("Output received.")),
    ])
    .await;
    let events = Arc::new(EventLog::default());
    let mut agent = twin
        .agent(
            workspace.path(),
            CodingAgentOptions::default()
                .with_context_compaction(false)
                .with_tool_output_retention_bytes(4_096)
                .with_tool_output_serialized_bytes(6_144)
                .with_tool_output_limit("shell_command", 1_024),
            &events,
        )
        .await;
    let report = timeout(PATIENCE, agent.prompt("Read the output"))
        .await
        .unwrap();
    assert_eq!(
        report.result.unwrap().text.as_deref(),
        Some("Output received.")
    );
    agent.shutdown(ShutdownReason::Completed).await.unwrap();
    assert!(
        !serde_json::to_string(&agent.to_record())
            .unwrap()
            .contains("DISCARDED_MIDDLE")
    );

    let requests = twin.requests();
    assert_eq!(requests.len(), 2);
    let output = tool_output(&requests[1], "read-output");
    assert!(output.chars().count() <= 1_024, "history character budget");
    assert!(
        serde_json::to_vec(output).unwrap().len() <= 6_144,
        "serialized output budget"
    );
    assert!(output.contains("OUTPUT_HEAD"));
    assert!(output.contains("OUTPUT_TAIL"));
    assert!(output.contains("truncated"));
    assert!(
        !serde_json::to_string(&requests)
            .unwrap()
            .contains("DISCARDED_MIDDLE")
    );

    let published = events.take();
    assert_eq!(
        published
            .iter()
            .filter(|event| matches!(event.event, CodingEvent::ToolProcessCompleted { .. }))
            .count(),
        1
    );
    assert!(
        !serde_json::to_string(&published)
            .unwrap()
            .contains("DISCARDED_MIDDLE")
    );
    let completions: Vec<_> = published
        .iter()
        .filter_map(|event| match &event.event {
            CodingEvent::ToolCallCompleted {
                output,
                is_error,
                output_bytes_observed,
                output_bytes_retained,
                output_bytes_omitted,
                ..
            } => {
                assert!(!is_error);
                assert!(output.as_str().unwrap().len() <= 4_096);
                assert!(serde_json::to_vec(output).unwrap().len() <= 6_144);
                assert!(*output_bytes_observed >= payload.len());
                assert!(*output_bytes_retained <= 4_096);
                assert!(*output_bytes_omitted > 0);
                assert_eq!(
                    *output_bytes_observed,
                    output_bytes_retained + output_bytes_omitted
                );
                Some(())
            }
            CodingEvent::ToolProcessCompleted {
                output_bytes_observed,
                output_bytes_retained,
                output_bytes_omitted,
                ..
            } => {
                assert_eq!(*output_bytes_observed, payload.len());
                assert!(*output_bytes_retained <= 1024 * 1024);
                assert!(*output_bytes_omitted > 0);
                assert_eq!(
                    *output_bytes_observed,
                    output_bytes_retained + output_bytes_omitted
                );
                None
            }
            _ => None,
        })
        .collect();
    assert_eq!(completions.len(), 1);
    twin.stop().await;
}

#[tokio::test]
async fn a_broken_stream_after_an_edit_does_not_execute_the_edit_twice() {
    let workspace = tempfile::tempdir().unwrap();
    let mut broken = answer("Checking the edit.");
    broken["close_after_chunks"] = json!(3);
    let mut twin = Twin::start(vec![
        scenario(
            1,
            "Implement task.sh",
            shell("edit", "echo 'echo FIXED' >> task.sh; bash task.sh"),
        ),
        scenario(2, "FIXED", broken),
        scenario(
            3,
            "FIXED",
            shell(
                "check",
                "test \"$(bash task.sh)\" = FIXED && echo CHECK_PASSED",
            ),
        ),
        scenario(
            4,
            "CHECK_PASSED",
            answer("Implemented and checked task.sh."),
        ),
    ])
    .await;
    let events = Arc::new(EventLog::default());
    let mut agent = twin
        .agent(
            workspace.path(),
            CodingAgentOptions::default()
                .with_context_compaction(false)
                .with_turn_replay(
                    RetryPolicy::exponential()
                        .max_attempts(2)
                        .initial_delay(Duration::from_millis(1))
                        .jitter(false),
                ),
            &events,
        )
        .await;
    let report = timeout(PATIENCE, agent.prompt("Implement task.sh"))
        .await
        .unwrap();
    assert_eq!(
        report.result.unwrap().text.as_deref(),
        Some("Implemented and checked task.sh.")
    );
    assert_eq!(report.usage.tokens.input, 30);
    assert_eq!(report.usage.tokens.output, 15);
    agent.shutdown(ShutdownReason::Completed).await.unwrap();

    assert_eq!(
        fs::read_to_string(workspace.path().join("task.sh")).unwrap(),
        "echo FIXED\n"
    );
    let requests = twin.requests();
    assert_eq!(
        requests.len(),
        4,
        "edit, interrupted response, replay, final answer"
    );
    assert_eq!(
        requests[1]["input"], requests[2]["input"],
        "replay keeps the committed edit and its result"
    );
    assert!(tool_output(&requests[2], "edit").contains("FIXED"));
    assert!(tool_output(&requests[3], "check").contains("CHECK_PASSED"));
    let published = events.take();
    let completions: Vec<_> = published
        .iter()
        .filter_map(|event| match &event.event {
            CodingEvent::ToolCallCompleted {
                tool_call_id,
                is_error,
                ..
            } => {
                assert!(!is_error);
                Some(tool_call_id.as_str())
            }
            _ => None,
        })
        .collect();
    assert_eq!(completions, ["edit", "check"]);
    assert_eq!(
        published
            .iter()
            .filter(|event| matches!(event.event, CodingEvent::LlmRetry {
                phase: LlmRetryPhase::Consume,
                ..
            }))
            .count(),
        1
    );
    twin.stop().await;
}

#[tokio::test]
async fn a_coding_task_survives_compaction_between_failing_and_passing_tests() {
    const PROMPT: &str = "Fix clamp.sh and rerun bash test_clamp.sh. Do not change the tests.";
    const TESTS: &str = include_str!("cmd/exec_fixes_failing_tests.in/test_clamp.sh");
    const SUMMARY: &str = "COMPACTED_TASK: Fix clamp.sh without changing test_clamp.sh. The test below lower bound failed: expected 0, got -2. Read clamp.sh, clamp values to both bounds, then rerun bash test_clamp.sh.";
    let workspace = tempfile::tempdir().unwrap();
    fs::write(
        workspace.path().join("clamp.sh"),
        include_str!("cmd/exec_fixes_failing_tests.in/clamp.sh"),
    )
    .unwrap();
    fs::write(workspace.path().join("test_clamp.sh"), TESTS).unwrap();

    let mut read = shell("read", "cat clamp.sh");
    // Cross the pinned model's 80% threshold after the failed test. Pebble
    // must compact while this new tool call still awaits its result.
    read["usage"]["input_tokens"] = json!(900_000);
    let mut summary = scenario(3, "FAIL: below lower bound", answer(SUMMARY));
    summary["matcher"]["stream"] = json!(false);
    let mut patch = answer("Applying the fix.");
    patch["tool_calls"] = json!([{
        "id": "patch",
        "name": "apply_patch",
        "kind": "custom",
        "arguments": "*** Begin Patch\n*** Update File: clamp.sh\n@@\n     local value=$1 lower=$2 upper=$3\n+    if (( value < lower )); then\n+        value=$lower\n+    elif (( value > upper )); then\n+        value=$upper\n+    fi\n     echo \"$value\"\n*** End Patch"
    }]);
    let mut twin = Twin::start(vec![
        scenario(1, PROMPT, shell("test-before", "bash test_clamp.sh")),
        scenario(2, "FAIL: below lower bound", read),
        summary,
        scenario(4, "local value=$1 lower=$2 upper=$3", patch),
        scenario(
            5,
            "Success. Updated the following files",
            shell("test-after", "bash test_clamp.sh"),
        ),
        scenario(
            6,
            "PASS: all 4 clamp cases",
            answer("Fixed clamp.sh; all four tests pass."),
        ),
    ])
    .await;
    let events = Arc::new(EventLog::default());
    let mut agent = twin
        .agent(
            workspace.path(),
            CodingAgentOptions::default()
                .with_context_compaction(true)
                .with_compaction_preserve_turns(1),
            &events,
        )
        .await;
    let report = timeout(PATIENCE, agent.prompt(PROMPT)).await.unwrap();
    assert_eq!(
        report.result.unwrap().text.as_deref(),
        Some("Fixed clamp.sh; all four tests pass.")
    );
    assert_eq!(
        report.usage.tokens.input, 900_050,
        "summary usage is included"
    );
    assert_eq!(report.usage.tokens.output, 30);
    agent.shutdown(ShutdownReason::Completed).await.unwrap();
    let summary_cost: u64 = agent
        .history()
        .turns()
        .iter()
        .filter_map(|message| match message {
            Message::Compaction { usage, .. } => usage.cost.map(|cost| cost.usd_micros),
            _ => None,
        })
        .sum();
    assert!(summary_cost > 0);
    assert_eq!(
        fs::read_to_string(workspace.path().join("clamp.sh")).unwrap(),
        include_str!("cmd/exec_fixes_failing_tests.out/clamp.sh")
    );
    assert_eq!(
        fs::read_to_string(workspace.path().join("test_clamp.sh")).unwrap(),
        TESTS
    );

    let requests = twin.requests();
    assert_eq!(requests.len(), 6);
    assert_eq!(
        requests[2]["stream"], false,
        "compaction uses a separate completion request"
    );
    let summary_input = serde_json::to_string(&requests[2]["input"]).unwrap();
    assert!(summary_input.contains(PROMPT));
    assert!(summary_input.contains("FAIL: below lower bound (expected 0, got -2)"));
    let resumed_input = serde_json::to_string(&requests[3]["input"]).unwrap();
    assert!(
        resumed_input.contains(SUMMARY),
        "the next model call sees the summary"
    );
    assert!(
        resumed_input.contains(PROMPT),
        "the original task and constraint survive"
    );
    assert!(tool_output(&requests[3], "read").contains("local value=$1 lower=$2 upper=$3"));
    assert!(
        !requests[3]["input"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["call_id"] == "test-before"),
        "older tool turns were actually compacted"
    );
    assert!(tool_output(&requests[5], "test-after").contains("PASS: all 4 clamp cases"));
    for request in &requests {
        assert_tool_pairs(request);
    }

    let published = events.take();
    let coding_cost: u64 = published
        .iter()
        .filter_map(|event| match &event.event {
            CodingEvent::AssistantMessage { usage, .. } => usage.cost.map(|cost| cost.usd_micros),
            _ => None,
        })
        .sum();
    assert_eq!(
        report.usage.cost.map(|cost| cost.usd_micros),
        Some(coding_cost + summary_cost),
        "every answer and the summary call were priced, so the report's cost is their sum"
    );
    let progress: Vec<_> = published
        .iter()
        .filter_map(|event| match &event.event {
            CodingEvent::ToolCallCompleted {
                tool_call_id,
                is_error,
                ..
            } => {
                assert_eq!(*is_error, tool_call_id == "test-before");
                Some(tool_call_id.as_str())
            }
            CodingEvent::CompactionStarted { .. } => Some("compaction-start"),
            CodingEvent::CompactionCompleted {
                original_turn_count,
                preserved_turn_count,
                ..
            } => {
                assert!(preserved_turn_count < original_turn_count);
                Some("compaction-end")
            }
            CodingEvent::CompactionFailed { .. } => panic!("compaction must succeed"),
            _ => None,
        })
        .collect();
    assert_eq!(progress, [
        "test-before",
        "compaction-start",
        "compaction-end",
        "read",
        "patch",
        "test-after"
    ]);
    twin.stop().await;
}

/// Every call in a model request has exactly one corresponding result, in
/// model order, including the custom apply_patch protocol.
fn assert_tool_pairs(request: &Value) {
    let input = request["input"].as_array().expect("Responses input array");
    let ids = |kinds: &[&str]| -> Vec<&str> {
        input
            .iter()
            .filter(|item| {
                item["type"]
                    .as_str()
                    .is_some_and(|kind| kinds.contains(&kind))
            })
            .map(|item| item["call_id"].as_str().expect("tool call id"))
            .collect()
    };
    let calls = ids(&["function_call", "custom_tool_call"]);
    let results = ids(&["function_call_output", "custom_tool_call_output"]);
    assert_eq!(
        calls, results,
        "tool protocol remains paired across compaction"
    );
    for (index, id) in calls.iter().enumerate() {
        assert!(!calls[..index].contains(id), "tool call {id} occurs once");
    }
}

#[cfg(unix)]
mod cancellation {
    use std::io::ErrorKind;

    use pebble_coding_agent::events::CommandTermination;
    use pebble_coding_agent::{Error, InterruptReason};
    use rustix::io::Errno;
    use rustix::process::{Pid, Signal, kill_process_group, test_kill_process};
    use tokio::fs::read_to_string;
    use tokio::time::sleep;
    use tokio_util::sync::CancellationToken;

    use super::*;

    /// Clean up the fixture if an assertion fails while its processes run.
    struct ProcessGroup(Option<Pid>);

    impl Drop for ProcessGroup {
        fn drop(&mut self) {
            if let Some(pid) = self.0 {
                let _ = kill_process_group(pid, Signal::KILL);
            }
        }
    }

    #[tokio::test]
    async fn cancelling_a_prompt_reaps_its_shell_and_child_and_allows_another_prompt() {
        let workspace = tempfile::tempdir().unwrap();
        // The parent reaps its child on TERM. Signalling only the parent
        // would leave it waiting for the child and fail the cleanup checks.
        fs::write(
            workspace.path().join("worker.sh"),
            r#"
trap 'wait "$worker"; exit 0' TERM
sleep 60 &
worker=$!
echo "$$ $worker" > processes.tmp
mv processes.tmp processes
wait "$worker"
"#,
        )
        .unwrap();
        let mut twin = Twin::start(vec![
            scenario(1, "Run the worker", shell("worker", "bash worker.sh")),
            scenario(
                2,
                "Continue after cancellation",
                answer("Ready for more work."),
            ),
        ])
        .await;
        let events = Arc::new(EventLog::default());
        let mut agent = twin
            .agent(
                workspace.path(),
                CodingAgentOptions::default().with_context_compaction(false),
                &events,
            )
            .await;
        let cancel = CancellationToken::new();
        let cancel_when_running = async {
            let text = loop {
                match read_to_string(workspace.path().join("processes")).await {
                    Ok(text) => break text,
                    Err(error) if error.kind() == ErrorKind::NotFound => {
                        sleep(Duration::from_millis(10)).await;
                    }
                    Err(error) => panic!("reading process readiness: {error}"),
                }
            };
            let pids: Vec<_> = text
                .split_whitespace()
                .map(|pid| Pid::from_raw(pid.parse().unwrap()).unwrap())
                .collect();
            assert_eq!(pids.len(), 2);
            let group = ProcessGroup(Some(pids[0]));
            for pid in &pids {
                assert!(
                    test_kill_process(*pid).is_ok(),
                    "process {pid:?} is running before cancellation"
                );
            }
            cancel.cancel();
            (group, pids)
        };
        let (report, (mut group, pids)) = timeout(PATIENCE, async {
            tokio::join!(
                agent.prompt_with_cancellation("Run the worker", &cancel),
                cancel_when_running
            )
        })
        .await
        .expect("cancellation finishes within the deadline");
        assert!(matches!(
            report.result,
            Err(Error::Interrupted(InterruptReason::Cancelled))
        ));
        assert_eq!(report.usage.tokens.input, 10);
        assert_eq!(report.usage.tokens.output, 5);
        assert!(report.usage.cost.is_some_and(|cost| cost.usd_micros > 0));
        // Check before shutdown: prompt cancellation must own this cleanup.
        for pid in pids {
            assert_eq!(
                test_kill_process(pid),
                Err(Errno::SRCH),
                "process {pid:?} was reaped"
            );
        }
        group.0 = None;
        agent.flush_events().await.unwrap();
        let published = events.take();
        let terminal: Vec<_> = published
            .iter()
            .filter_map(|event| match &event.event {
                CodingEvent::ToolProcessCompleted { termination, .. } => {
                    assert_eq!(*termination, CommandTermination::Cancelled);
                    Some("process")
                }
                CodingEvent::ToolCallCompleted {
                    tool_call_id,
                    is_error,
                    ..
                } => {
                    assert_eq!(tool_call_id, "worker");
                    assert!(*is_error);
                    Some("tool")
                }
                CodingEvent::ProcessingEnd => Some("prompt"),
                CodingEvent::SessionEnded => {
                    panic!("caller cancellation must leave the session reusable")
                }
                _ => None,
            })
            .collect();
        assert_eq!(terminal, ["process", "tool", "prompt"]);

        let next = timeout(PATIENCE, agent.prompt("Continue after cancellation"))
            .await
            .unwrap();
        assert_eq!(
            next.result.unwrap().text.as_deref(),
            Some("Ready for more work.")
        );
        assert_eq!(
            next.usage.tokens.input, 10,
            "usage belongs to this invocation"
        );
        assert_eq!(next.usage.tokens.output, 5);
        agent.shutdown(ShutdownReason::Completed).await.unwrap();
        let requests = twin.requests();
        assert_eq!(
            requests.len(),
            2,
            "the cancelled prompt makes no further model call"
        );
        // fabro-c98c: the cancelled call's result is error-marked, and the
        // openai-wire request render wraps error-marked results — the next
        // prompt's model call sees the cancellation as the failure it was
        // instead of a successful-looking "Cancelled".
        assert_eq!(tool_output(&requests[1], "worker"), "[TOOL ERROR] Cancelled");
        let end = events.take();
        assert_eq!(
            end.iter()
                .filter(|event| matches!(event.event, CodingEvent::SessionEnded))
                .count(),
            1
        );
        assert!(
            published
                .iter()
                .chain(&end)
                .map(|event| event.seq)
                .collect::<Vec<_>>()
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        );
        twin.stop().await;
    }
}
