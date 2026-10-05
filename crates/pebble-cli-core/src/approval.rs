//! Asking the person at the terminal before a tool the permission level does
//! not allow outright.

use std::io::{self, Write as _};
use std::sync::{Mutex, PoisonError};

use async_trait::async_trait;
use pebble_agent::{ToolCallRequest, ToolDescriptor, ToolSystemError};
use pebble_coding_agent::tools::{
    ApprovalDecision, PermissionLevel, PermissionLevelPolicy, ToolApprovalService,
};
use tokio::task::spawn_blocking;

/// Approval at the terminal: a tool the level allows runs; otherwise the
/// person is asked on standard error and answers on standard input. Without
/// a terminal, or when asking was turned off, such tools are refused.
///
/// Answering "always" grants the least permission level that allows the tool:
/// [`PermissionLevel::ReadWrite`] for native writes, [`PermissionLevel::Full`]
/// for shell, web, and unrecognized tools (including MCP). The grant covers all
/// tools allowed by that level for the lifetime of this instance, including
/// calls from any sessions that share it. A new instance starts at its
/// configured level; grants are not persisted.
pub struct TerminalApproval {
    level:       Mutex<PermissionLevel>,
    interactive: bool,
}

impl TerminalApproval {
    /// Approval that asks when `interactive` and refuses otherwise.
    #[must_use]
    pub fn new(level: PermissionLevel, interactive: bool) -> Self {
        Self {
            level: Mutex::new(level),
            interactive,
        }
    }

    fn level(&self) -> PermissionLevel {
        *self.level.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn raise_level(&self, escalation: Escalation) {
        let mut level = self.level.lock().unwrap_or_else(PoisonError::into_inner);
        // Another approval may have raised the level while this prompt waited.
        *level = (*level).max(escalation.level());
    }
}

/// The step up the ladder one tool call needs. It is made only from the call,
/// so answering "always" never grants more than that call required.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Escalation {
    ReadWrite,
    Full,
}

impl Escalation {
    /// The step `tool` needs from `current`, or `None` when `current` already
    /// allows it.
    fn needed(current: PermissionLevel, tool: &ToolDescriptor) -> Option<Self> {
        match PermissionLevelPolicy::required_level(tool) {
            required if required <= current => None,
            PermissionLevel::ReadWrite => Some(Self::ReadWrite),
            PermissionLevel::Full => Some(Self::Full),
            // Every level allows what read-only allows.
            PermissionLevel::ReadOnly => None,
        }
    }

    const fn level(self) -> PermissionLevel {
        match self {
            Self::ReadWrite => PermissionLevel::ReadWrite,
            Self::Full => PermissionLevel::Full,
        }
    }
}

#[async_trait]
impl ToolApprovalService for TerminalApproval {
    async fn approve(
        &self,
        request: &ToolCallRequest,
    ) -> Result<ApprovalDecision, ToolSystemError> {
        let tool_name = request.call().name.clone();
        let current = self.level();
        let Some(escalation) = Escalation::needed(current, request.descriptor()) else {
            return Ok(ApprovalDecision::Allow);
        };
        if !self.interactive {
            return Ok(ApprovalDecision::Deny {
                reason: format!("{tool_name} tool denied at current permission level"),
            });
        }
        let asked = tool_name.clone();
        let answer = spawn_blocking(move || ask(&asked, current, escalation))
            .await
            .map_err(|error| ToolSystemError::new(format!("approval prompt failed: {error}")))?;
        Ok(match answer {
            Ok(Answer::Allow) => ApprovalDecision::Allow,
            Ok(Answer::AllowAlways) => {
                self.raise_level(escalation);
                ApprovalDecision::Allow
            }
            Ok(Answer::Deny) => ApprovalDecision::Deny {
                reason: format!("{tool_name} tool denied by user"),
            },
            Err(reason) => ApprovalDecision::Deny { reason },
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Answer {
    Allow,
    AllowAlways,
    Deny,
}

/// Asks on standard error and reads one line from standard input. Runs on a
/// blocking task: the person may take their time.
#[expect(
    clippy::print_stderr,
    reason = "the approval prompt is the command's stderr boundary"
)]
fn ask(
    tool_name: &str,
    current: PermissionLevel,
    escalation: Escalation,
) -> Result<Answer, String> {
    eprint!("{}", prompt(tool_name, current, escalation));
    io::stderr().flush().ok();
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .map_err(|error| format!("failed to read the answer: {error}"))?;
    Ok(parse_answer(&input))
}

fn parse_answer(input: &str) -> Answer {
    match input.trim().to_lowercase().as_str() {
        "y" | "yes" => Answer::Allow,
        "a" | "always" => Answer::AllowAlways,
        _ => Answer::Deny,
    }
}

fn prompt(tool_name: &str, current: PermissionLevel, escalation: Escalation) -> String {
    let target = escalation.level();
    let scope = match escalation {
        Escalation::ReadWrite => "All read/write tools; shell, web and MCP still require approval.",
        Escalation::Full => "ALL tools, including shell, web and MCP.",
    };
    format!(
        "Allow {tool_name}? Current permission: {current}.\n\
         [y]es: this call only / [n]o: deny this call\n\
         [a]lways: {target} for the rest of this session\n\
         {scope}\nChoice: "
    )
}

#[cfg(test)]
mod tests {
    use lithos_llm::types::{ToolCall, ToolDefinition};
    use pebble_agent::{SessionScope, ToolCatalog, ToolDescriptor, ToolId, TurnContext};
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use super::*;

    fn request(identity: &str, name: &str) -> ToolCallRequest {
        ToolCatalog::new([ToolDescriptor::new(
            ToolId::try_new(identity).expect("valid fixture identity"),
            ToolDefinition::function(name, "Fixture tool", json!({"type": "object"})),
        )])
        .resolve(
            TurnContext::new(&SessionScope::default(), "test/model", 0, &[]),
            ToolCall::function("call_1", name, json!({})),
            CancellationToken::new(),
        )
        .unwrap_or_else(|_| panic!("valid fixture call"))
    }

    #[test]
    fn a_call_escalates_only_as_far_as_its_tool_requires() {
        use PermissionLevel::{Full, ReadOnly, ReadWrite};
        for (identity, current, expected) in [
            ("read_file", ReadOnly, None),
            ("write_file", ReadOnly, Some(Escalation::ReadWrite)),
            ("write_file", ReadWrite, None),
            ("shell", ReadOnly, Some(Escalation::Full)),
            ("shell", ReadWrite, Some(Escalation::Full)),
            ("shell", Full, None),
            ("mcp__files__write", ReadWrite, Some(Escalation::Full)),
        ] {
            let request = request(identity, identity);
            assert_eq!(
                Escalation::needed(current, request.descriptor()),
                expected,
                "{identity} at {current}"
            );
        }
    }

    #[test]
    fn the_prompt_states_the_scope_of_always() {
        let text = prompt(
            "write_file",
            PermissionLevel::ReadOnly,
            Escalation::ReadWrite,
        );
        assert!(text.contains("Current permission: read-only."));
        assert!(text.contains("this call only"));
        assert!(text.contains("[a]lways: read-write for the rest of this session"));
        assert!(text.contains("shell, web and MCP still require approval"));
        let text = prompt("shell", PermissionLevel::ReadWrite, Escalation::Full);
        assert!(text.contains("[a]lways: full for the rest of this session"));
        assert!(text.contains("ALL tools, including shell, web and MCP"));
    }

    #[test]
    fn a_delayed_grant_never_lowers_the_level() {
        for current in [
            PermissionLevel::ReadOnly,
            PermissionLevel::ReadWrite,
            PermissionLevel::Full,
        ] {
            for escalation in [Escalation::ReadWrite, Escalation::Full] {
                let approval = TerminalApproval::new(current, true);
                approval.raise_level(escalation);
                assert_eq!(approval.level(), current.max(escalation.level()));
            }
        }
    }

    #[test]
    fn only_explicit_yes_or_always_grants_permission() {
        for input in ["", "\n", "n", "no", "invalid", "yes please", "all"] {
            assert_eq!(parse_answer(input), Answer::Deny);
        }
        for input in ["y", "yes", " YES\n"] {
            assert_eq!(parse_answer(input), Answer::Allow);
        }
        for input in ["a", "always", " Always\n"] {
            assert_eq!(parse_answer(input), Answer::AllowAlways);
        }
    }

    #[tokio::test]
    async fn native_aliases_and_extensions_use_the_same_policy_for_auto_approval() {
        let approval = TerminalApproval::new(PermissionLevel::ReadWrite, false);
        assert_eq!(
            approval
                .approve(&request("write_file", "Write"))
                .await
                .unwrap(),
            ApprovalDecision::Allow
        );
        assert!(matches!(
            approval.approve(&request("Write", "Write")).await.unwrap(),
            ApprovalDecision::Deny { .. }
        ));
        assert_eq!(approval.level(), PermissionLevel::ReadWrite);
    }
}
