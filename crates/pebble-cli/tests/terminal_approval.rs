//! The exported terminal service used by embedders, through a real PTY.
//!
//! The ordinary Pebble exec binary does not install this service. A child test
//! drives the same middleware and real local tools with an in-memory model.

#![cfg(unix)]

use std::io::IsTerminal as _;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use std::{env, fs, io};

use lithos_llm::catalog::Catalog;
use pebble_cli_core::approval::TerminalApproval;
use pebble_coding_agent::environment::LocalEnvironment;
use pebble_coding_agent::events::CodingEvent;
use pebble_coding_agent::test_support::{
    ScriptedCall, ScriptedProvider, TEST_CATALOG, scripted_client_builder, text_response,
    tool_call_response,
};
use pebble_coding_agent::tools::{
    PermissionLevel, PermissionLevelPolicy, PermissionMiddleware, RegisteredTool,
};
use pebble_coding_agent::{CodingAgent, ShutdownReason};
use rustix::process::setsid;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::fs as async_fs;
use tokio::io::AsyncWriteExt as _;
use tokio::time::timeout;

#[path = "support/terminal.rs"]
#[expect(
    dead_code,
    reason = "the shared driver also provides TUI-only operations"
)]
mod terminal;

use terminal::{Terminal, child_command};

const DONE: &str = "TERMINAL-APPROVAL-DONE";
const PATIENCE: Duration = Duration::from_secs(20);

/// Entry point for both terminal and pipe children. No external provider,
/// ambient credentials, or production command is involved.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[expect(
    clippy::print_stdout,
    reason = "fixture completion is the PTY boundary"
)]
async fn launch_terminal_child() {
    let Ok(args) = env::var("PEBBLE_PTY_ARGS") else {
        return;
    };
    let args: Vec<String> = serde_json::from_str(&args).expect("fixture arguments");
    setsid().expect("detach child from runner terminal");
    let root = Path::new(&args[1]);
    let level: PermissionLevel =
        serde_json::from_value(Value::String(args[2].clone())).expect("known fixture permission");
    let scenario = &args[3];
    let aliases = scenario == "aliases";
    let (write, edit, shell) = if aliases {
        ("Write", "Edit", "Bash")
    } else {
        ("write_file", "edit_file", "shell")
    };
    let extension = match scenario.as_str() {
        "mcp-always" => Some("mcp__files__write"),
        "extension-alias" => Some("Write"),
        _ => None,
    };
    let mut calls = if scenario == "shell-always" {
        vec![shell_call(shell, "shell-first", "shell-first.txt")]
    } else {
        extension.map_or_else(Vec::new, |name| vec![call(name, "extension", json!({}))])
    };
    let path_key = if aliases { "path" } else { "file_path" };
    calls.push(call(
        write,
        "write-first",
        json!({path_key: "first.txt", "content": "first"}),
    ));
    calls.push(call(
        write,
        "write-second",
        json!({path_key: "second.txt", "content": "second"}),
    ));
    if matches!(scenario.as_str(), "write-edit" | "aliases") {
        calls.push(call(
            edit,
            "edit",
            json!({path_key: "second.txt", "old_string": "second", "new_string": "edited"}),
        ));
    }
    calls.push(shell_call(shell, "shell-last", "shell-last.txt"));
    if scenario == "write-edit" {
        calls.push(shell_call(shell, "shell-again", "shell-again.txt"));
    }
    let expected_calls = calls.len() + 1;
    calls.push(ScriptedCall::response(text_response(
        "permission sequence complete",
    )));
    let (mut builder, provider) = scripted_client_builder(ScriptedProvider::new(calls));
    if aliases {
        // The Kimi profile exposes the native tools as `Write`, `Edit`, and `Bash`.
        let catalog = Catalog::builder()
            .overlay_toml(&TEST_CATALOG.replace("profile = \"anthropic\"", "profile = \"kimi\""))
            .expect("fixture catalog layer")
            .build()
            .expect("fixture catalog");
        builder = builder.catalog(catalog);
    }
    let client = builder.build().expect("scripted client").client;
    let environment = LocalEnvironment::new(root);
    environment
        .prepare()
        .await
        .expect("prepare fixture environment");
    let interactive = io::stdin().is_terminal() && scenario != "auto-approve";
    let middleware = PermissionMiddleware::new(Arc::new(PermissionLevelPolicy::new(level)))
        .with_approval(Arc::new(TerminalApproval::new(level, interactive)));
    let mut builder = CodingAgent::builder(client, Arc::new(environment))
        .model("test/model")
        .tool_middleware(Arc::new(middleware));
    if let Some(name) = extension {
        let marker = root.join("extension.txt");
        builder = builder.tools([RegisteredTool::function(
            name,
            "Harmless extension marker",
            json!({"type": "object"}),
            move |_, _| {
                let marker = marker.clone();
                async move {
                    async_fs::write(marker, "extension")
                        .await
                        .expect("write extension fixture marker");
                    Ok("extension ran".to_owned())
                }
            },
        )]);
    }
    let mut agent = builder.build().await.expect("fixture agent");
    let mut events = agent.subscribe();
    let report = timeout(PATIENCE, agent.prompt("run the permission sequence"))
        .await
        .expect("fixture prompt finishes");
    assert!(report.result.is_ok(), "{report:?}");
    agent
        .shutdown(ShutdownReason::Completed)
        .await
        .expect("fixture shutdown");
    assert_eq!(provider.call_count(), expected_calls);
    let mut outcomes = Vec::new();
    while let Ok(event) = events.try_recv() {
        if let CodingEvent::ToolCallCompleted {
            tool_call_id,
            is_error,
            ..
        } = event.event
        {
            outcomes.push((tool_call_id, is_error));
        }
    }
    fs::write(
        root.join("outcomes.json"),
        serde_json::to_vec(&outcomes).unwrap(),
    )
    .unwrap();
    let requests = provider.requests();
    fs::write(
        root.join("last-request.json"),
        serde_json::to_vec(requests.last().unwrap()).unwrap(),
    )
    .unwrap();
    println!("{DONE}");
}

fn call(name: &str, id: &str, arguments: Value) -> ScriptedCall {
    ScriptedCall::response(tool_call_response(name, id, arguments))
}

fn shell_call(name: &str, id: &str, marker: &str) -> ScriptedCall {
    call(
        name,
        id,
        json!({"command": format!("printf shell > {marker}")}),
    )
}

fn arguments(root: &Path, level: PermissionLevel, scenario: &str) -> Vec<String> {
    vec![
        "--cwd".into(),
        root.display().to_string(),
        level.to_string(),
        scenario.into(),
    ]
}

fn start(root: &Path, level: PermissionLevel, scenario: &str) -> Terminal {
    Terminal::start(
        &arguments(root, level, scenario),
        "unused-scripted-provider",
        "unused",
    )
}

fn outcomes(root: &Path) -> Vec<(String, bool)> {
    let bytes = fs::read(root.join("outcomes.json")).expect("child wrote fixture outcomes");
    serde_json::from_slice(&bytes).expect("fixture outcomes are valid JSON")
}

fn marker(root: &Path, name: &str, expected: Option<&str>) {
    match expected {
        Some(expected) => assert_eq!(
            fs::read_to_string(root.join(name)).expect("approved call wrote fixture marker"),
            expected
        ),
        None => assert!(!root.join(name).exists(), "{name} must not be created"),
    }
}

/// Answers the `nth` prompt for `tool`. Counting prompts keeps an answer from
/// landing on an earlier prompt for the same tool that is still on screen.
async fn answer(terminal: &mut Terminal, tool: &str, nth: usize, response: &[u8]) {
    let prompt = format!("Allow {tool}?");
    terminal
        .until(|screen| screen.text().matches(prompt.as_str()).count() == nth)
        .await;
    terminal.contains("Choice: ").await;
    terminal.send(response).await;
}

#[tokio::test]
async fn write_always_allows_another_write_but_shell_still_requires_approval() {
    let root = TempDir::new().unwrap();
    let mut terminal = start(root.path(), PermissionLevel::ReadOnly, "writes-shell");
    terminal
        .contains("[a]lways: read-write for the rest of this session")
        .await;
    terminal
        .contains("shell, web and MCP still require approval")
        .await;
    answer(&mut terminal, "write_file", 1, b"a\n").await;
    terminal
        .contains("Allow shell? Current permission: read-write.")
        .await;
    marker(root.path(), "first.txt", Some("first"));
    marker(root.path(), "second.txt", Some("second"));
    marker(root.path(), "shell-last.txt", None);
    terminal
        .contains("ALL tools, including shell, web and MCP")
        .await;
    answer(&mut terminal, "shell", 1, b"n\n").await;
    let screen = terminal.finish_with(DONE, true).await;
    assert_eq!(screen.text().matches("Allow write_file?").count(), 1);
    assert_eq!(screen.text().matches("Allow shell?").count(), 1);
    assert_eq!(outcomes(root.path()), [
        ("write-first".into(), false),
        ("write-second".into(), false),
        ("shell-last".into(), true)
    ]);
    marker(root.path(), "shell-last.txt", None);
    assert!(
        fs::read_to_string(root.path().join("last-request.json"))
            .unwrap()
            .contains("shell tool denied by user")
    );
}

#[tokio::test]
async fn write_always_covers_edits_and_shell_yes_does_not_grant_full() {
    let root = TempDir::new().unwrap();
    let mut terminal = start(root.path(), PermissionLevel::ReadOnly, "write-edit");
    answer(&mut terminal, "write_file", 1, b"always\n").await;
    answer(&mut terminal, "shell", 1, b"yes\n").await;
    answer(&mut terminal, "shell", 2, b"no\n").await;
    terminal.finish_with(DONE, true).await;
    marker(root.path(), "second.txt", Some("edited"));
    marker(root.path(), "shell-last.txt", Some("shell"));
    marker(root.path(), "shell-again.txt", None);
    assert_eq!(
        outcomes(root.path()).last().unwrap(),
        &("shell-again".into(), true)
    );
}

#[tokio::test]
async fn yes_once_and_denial_do_not_upgrade_the_session() {
    let root = TempDir::new().unwrap();
    let mut terminal = start(root.path(), PermissionLevel::ReadOnly, "writes-shell");
    answer(&mut terminal, "write_file", 1, b"y\n").await;
    answer(&mut terminal, "write_file", 2, b"n\n").await;
    answer(&mut terminal, "shell", 1, b"n\n").await;
    terminal.finish_with(DONE, true).await;
    marker(root.path(), "first.txt", Some("first"));
    marker(root.path(), "second.txt", None);
    marker(root.path(), "shell-last.txt", None);
    assert_eq!(outcomes(root.path()), [
        ("write-first".into(), false),
        ("write-second".into(), true),
        ("shell-last".into(), true)
    ]);
}

#[tokio::test]
async fn empty_invalid_and_terminal_eof_answers_deny_without_upgrading() {
    let root = TempDir::new().unwrap();
    let mut terminal = start(root.path(), PermissionLevel::ReadOnly, "writes-shell");
    answer(&mut terminal, "write_file", 1, b"\n").await;
    answer(&mut terminal, "write_file", 2, b"invalid\n").await;
    answer(&mut terminal, "shell", 1, b"\x04").await;
    terminal.finish_with(DONE, true).await;
    marker(root.path(), "first.txt", None);
    marker(root.path(), "second.txt", None);
    marker(root.path(), "shell-last.txt", None);
    assert!(outcomes(root.path()).iter().all(|(_, denied)| *denied));
}

#[tokio::test]
async fn native_profile_aliases_use_the_write_scope() {
    let root = TempDir::new().unwrap();
    let mut terminal = start(root.path(), PermissionLevel::ReadOnly, "aliases");
    terminal.contains("[a]lways: read-write").await;
    answer(&mut terminal, "Write", 1, b"a\n").await;
    answer(&mut terminal, "Bash", 1, b"n\n").await;
    let screen = terminal.finish_with(DONE, true).await;
    assert_eq!(screen.text().matches("Allow Write?").count(), 1);
    assert!(!screen.text().contains("Allow Edit?"));
    marker(root.path(), "second.txt", Some("edited"));
    marker(root.path(), "shell-last.txt", None);
}

#[tokio::test]
async fn shell_mcp_and_native_named_extensions_offer_full_explicitly() {
    for (scenario, tool) in [
        ("shell-always", "shell"),
        ("mcp-always", "mcp__files__write"),
        ("extension-alias", "Write"),
    ] {
        let root = TempDir::new().unwrap();
        let mut terminal = start(root.path(), PermissionLevel::ReadOnly, scenario);
        terminal
            .contains("[a]lways: full for the rest of this session")
            .await;
        terminal
            .contains("ALL tools, including shell, web and MCP")
            .await;
        answer(&mut terminal, tool, 1, b"a\n").await;
        let screen = terminal.finish_with(DONE, true).await;
        assert_eq!(screen.text().matches("Allow ").count(), 1);
        assert!(outcomes(root.path()).iter().all(|(_, denied)| !denied));
        marker(root.path(), "first.txt", Some("first"));
        marker(root.path(), "second.txt", Some("second"));
        marker(root.path(), "shell-last.txt", Some("shell"));
        if scenario != "shell-always" {
            marker(root.path(), "extension.txt", Some("extension"));
        }
    }
}

#[tokio::test]
async fn a_fresh_session_does_not_remember_always() {
    let root = TempDir::new().unwrap();
    let mut terminal = start(root.path(), PermissionLevel::ReadOnly, "shell-always");
    answer(&mut terminal, "shell", 1, b"a\n").await;
    terminal.finish_with(DONE, true).await;
    let root = TempDir::new().unwrap();
    let mut terminal = start(root.path(), PermissionLevel::ReadOnly, "writes-shell");
    answer(&mut terminal, "write_file", 1, b"n\n").await;
    answer(&mut terminal, "write_file", 2, b"n\n").await;
    answer(&mut terminal, "shell", 1, b"n\n").await;
    terminal.finish_with(DONE, true).await;
    assert!(outcomes(root.path()).iter().all(|(_, denied)| *denied));
}

#[tokio::test]
async fn noninteractive_levels_ignore_piped_approval_answers() {
    for level in [
        PermissionLevel::ReadOnly,
        PermissionLevel::ReadWrite,
        PermissionLevel::Full,
    ] {
        let root = TempDir::new().unwrap();
        let mut child = child_command(&arguments(root.path(), level, "writes-shell"), root.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(b"always\nalways\nalways\n")
            .await
            .unwrap();
        let output = timeout(PATIENCE, child.wait_with_output())
            .await
            .unwrap()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!String::from_utf8_lossy(&output.stderr).contains("Allow "));
        marker(
            root.path(),
            "first.txt",
            (level != PermissionLevel::ReadOnly).then_some("first"),
        );
        marker(
            root.path(),
            "second.txt",
            (level != PermissionLevel::ReadOnly).then_some("second"),
        );
        marker(
            root.path(),
            "shell-last.txt",
            (level == PermissionLevel::Full).then_some("shell"),
        );
        assert_eq!(outcomes(root.path()), [
            ("write-first".into(), level == PermissionLevel::ReadOnly),
            ("write-second".into(), level == PermissionLevel::ReadOnly),
            ("shell-last".into(), level != PermissionLevel::Full)
        ]);
    }
}

#[tokio::test]
async fn explicit_full_and_disabled_prompting_do_not_ask_in_a_terminal() {
    for (level, scenario) in [
        (PermissionLevel::Full, "writes-shell"),
        (PermissionLevel::ReadWrite, "auto-approve"),
    ] {
        let root = TempDir::new().unwrap();
        let terminal = start(root.path(), level, scenario);
        let screen = terminal.finish_with(DONE, true).await;
        assert!(!screen.text().contains("Allow "));
        marker(root.path(), "first.txt", Some("first"));
        marker(root.path(), "second.txt", Some("second"));
        marker(
            root.path(),
            "shell-last.txt",
            (level == PermissionLevel::Full).then_some("shell"),
        );
    }
}
