//! Tool identity and composable permission middleware.

use std::fmt;
use std::result::Result as StdResult;
use std::sync::Arc;

use async_trait::async_trait;
use pebble_agent::{
    ToolCallNext, ToolCallRequest, ToolCatalog, ToolDiscoveryNext, ToolErrorKind, ToolMiddleware,
    ToolOutcome, ToolSystemError, TurnContext,
};

use super::native::NativeTool;
use crate::SessionScope;
use crate::types::{PermissionLevel, ToolCategory};

/// What one permission policy requires for a tool.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ToolPermission {
    /// The tool can be discovered and called without approval.
    Allow,
    /// The tool can be discovered only when an approval service is installed.
    RequireApproval,
    /// The tool is hidden and every attempted call is refused.
    Deny {
        /// The explanation returned to the model.
        reason: String,
    },
}

/// Classifies tools for one permission middleware layer.
pub trait ToolPermissionPolicy: Send + Sync {
    /// Returns the permission for one stable tool identity.
    fn permission(
        &self,
        session: &SessionScope,
        tool: &pebble_agent::ToolDescriptor,
    ) -> ToolPermission;
}

/// What an approval service decided for one call and its actual arguments.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ApprovalDecision {
    /// Continue through the remaining middleware.
    Allow,
    /// Refuse this call.
    Deny {
        /// The explanation returned to the model.
        reason: String,
    },
}

/// Resolves human or application approval for a specific tool call.
#[async_trait]
pub trait ToolApprovalService: Send + Sync {
    /// Approves or refuses a call after its arguments have been validated.
    async fn approve(
        &self,
        request: &ToolCallRequest,
    ) -> StdResult<ApprovalDecision, ToolSystemError>;
}

/// A permission layer that governs both discovery and invocation.
pub struct PermissionMiddleware {
    policy:   Arc<dyn ToolPermissionPolicy>,
    approval: Option<Arc<dyn ToolApprovalService>>,
}

impl PermissionMiddleware {
    /// Creates a permission layer with no approval path.
    #[must_use]
    pub fn new(policy: Arc<dyn ToolPermissionPolicy>) -> Self {
        Self {
            policy,
            approval: None,
        }
    }

    /// Sets the service for argument-sensitive or human approval.
    #[must_use]
    pub fn with_approval(mut self, approval: Arc<dyn ToolApprovalService>) -> Self {
        self.approval = Some(approval);
        self
    }

    fn permission(
        &self,
        session: &SessionScope,
        tool: &pebble_agent::ToolDescriptor,
    ) -> ToolPermission {
        self.policy.permission(session, tool)
    }

    fn is_visible(&self, session: &SessionScope, tool: &pebble_agent::ToolDescriptor) -> bool {
        match self.permission(session, tool) {
            ToolPermission::Allow => true,
            ToolPermission::RequireApproval => self.approval.is_some(),
            ToolPermission::Deny { .. } => false,
        }
    }
}

#[async_trait]
impl ToolMiddleware for PermissionMiddleware {
    async fn discover(
        &self,
        context: TurnContext<'_>,
        next: ToolDiscoveryNext<'_>,
    ) -> StdResult<ToolCatalog, ToolSystemError> {
        let mut catalog = next.run(context).await?;
        catalog.retain(|tool| self.is_visible(context.session(), tool));
        Ok(catalog)
    }

    async fn call(
        &self,
        request: ToolCallRequest,
        next: ToolCallNext<'_>,
    ) -> StdResult<ToolOutcome, ToolSystemError> {
        match self.permission(request.session(), request.descriptor()) {
            ToolPermission::Allow => next.run(request).await,
            ToolPermission::Deny { reason } => {
                Ok(ToolOutcome::failure(ToolErrorKind::Denied, reason))
            }
            ToolPermission::RequireApproval => {
                let Some(approval) = &self.approval else {
                    return Ok(ToolOutcome::failure(
                        ToolErrorKind::Denied,
                        format!(
                            "{} requires approval, but no approval service is installed",
                            request.call().name
                        ),
                    ));
                };
                match approval.approve(&request).await? {
                    ApprovalDecision::Allow => next.run(request).await,
                    ApprovalDecision::Deny { reason } => {
                        Ok(ToolOutcome::failure(ToolErrorKind::Denied, reason))
                    }
                }
            }
        }
    }
}

impl fmt::Debug for PermissionMiddleware {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PermissionMiddleware")
            .field("has_approval", &self.approval.is_some())
            .finish_non_exhaustive()
    }
}

/// A policy backed by Pebble's three built-in permission levels.
#[derive(Clone, Copy, Debug)]
pub struct PermissionLevelPolicy {
    level: PermissionLevel,
}

impl PermissionLevelPolicy {
    /// Uses `level` for every tool.
    #[must_use]
    pub const fn new(level: PermissionLevel) -> Self {
        Self { level }
    }

    /// The least level that runs `tool` without asking.
    ///
    /// The tool is judged by its canonical identity, not the name a profile
    /// exposes it under, so an extension that reuses a native alias such as
    /// `Write` still needs [`PermissionLevel::Full`].
    #[must_use]
    pub fn required_level(tool: &pebble_agent::ToolDescriptor) -> PermissionLevel {
        let category = gate_category(tool);
        [PermissionLevel::ReadOnly, PermissionLevel::ReadWrite]
            .into_iter()
            .find(|level| level.auto_approves(category))
            .unwrap_or(PermissionLevel::Full)
    }
}

/// The permission-gate category of a tool, by its canonical identity.
fn gate_category(tool: &pebble_agent::ToolDescriptor) -> ToolCategory {
    NativeTool::from_canonical_name(tool.id().as_str())
        .and_then(NativeTool::category)
        .unwrap_or(ToolCategory::Shell)
}

impl ToolPermissionPolicy for PermissionLevelPolicy {
    fn permission(
        &self,
        _session: &SessionScope,
        tool: &pebble_agent::ToolDescriptor,
    ) -> ToolPermission {
        if self.level.auto_approves(gate_category(tool)) {
            ToolPermission::Allow
        } else {
            ToolPermission::RequireApproval
        }
    }
}

/// Resolves a tool name in any profile's vocabulary to the canonical name the
/// rest of the crate reasons about.
///
/// A profile may expose a built-in tool under the vocabulary its model was
/// trained against — the Kimi profile uses Kimi Code's `Read`/`Edit`/`Bash`
/// names — but permissions, categories, and output limits must not depend on
/// which profile is running. Names that are not built-in (MCP, skill,
/// application-registered) pass through unchanged.
#[must_use]
pub fn canonical_tool_name(name: &str) -> &str {
    match NativeTool::from_any_name(name) {
        Some(tool) => tool.canonical_name(),
        None => name,
    }
}

/// The coarse access category of an exposed tool.
///
/// `None` means the name is outside the permission taxonomy, so callers can
/// decide what that means: a permission gate defaults it to
/// [`ToolCategory::Shell`], a tool summary reports [`ToolCategory::Other`].
#[must_use]
pub(crate) fn known_tool_category(name: &str) -> Option<ToolCategory> {
    NativeTool::from_any_name(name).and_then(NativeTool::category)
}

/// The permission-gate category of a tool.
///
/// An unknown tool falls back to [`ToolCategory::Shell`], so it needs explicit
/// approval at any level below [`PermissionLevel::Full`].
#[must_use]
pub(crate) fn tool_category(name: &str) -> ToolCategory {
    known_tool_category(name).unwrap_or(ToolCategory::Shell)
}

impl PermissionLevel {
    /// Whether this level runs `category` without asking.
    ///
    /// | category | `ReadOnly` | `ReadWrite` | `Full` |
    /// | --- | --- | --- | --- |
    /// | `Read`, `Subagent` | yes | yes | yes |
    /// | `Write` | no | yes | yes |
    /// | `Shell`, `Other` | no | no | yes |
    ///
    /// [`Full`](PermissionLevel::Full) approves everything, a category pebble
    /// adds later included; the levels below it approve only what they name,
    /// so a category they have never heard of needs approval.
    ///
    /// The `Full`/`Other` answer is the one place this table differs from
    /// fabro's, which refused that pair. No tool name reaches it either way —
    /// `tool_category` answers [`Shell`](ToolCategory::Shell) for a name it
    /// does not know, and [`Other`](ToolCategory::Other) is what a tool summary
    /// reports rather than what a gate is asked about.
    #[must_use]
    pub const fn auto_approves(self, category: ToolCategory) -> bool {
        match self {
            Self::Full => true,
            Self::ReadWrite => matches!(
                category,
                ToolCategory::Read | ToolCategory::Subagent | ToolCategory::Write
            ),
            Self::ReadOnly => matches!(category, ToolCategory::Read | ToolCategory::Subagent),
        }
    }

    /// Whether this level runs the named tool without asking.
    ///
    /// The name is resolved through `tool_category`, so a tool pebble does
    /// not know needs approval below [`PermissionLevel::Full`].
    #[must_use]
    pub fn auto_approves_tool(self, tool_name: &str) -> bool {
        self.auto_approves(tool_category(tool_name))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use lithos_llm::types::{ToolArguments, ToolCall, ToolDefinition, ToolInput};
    use pebble_agent::{ToolDescriptor, ToolId, ToolService, ToolSystem};
    use serde_json::{Value, json};
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::test_support::FixedPermission;

    struct RecordingApproval {
        arguments: Mutex<Vec<Value>>,
    }

    #[async_trait]
    impl ToolApprovalService for RecordingApproval {
        async fn approve(
            &self,
            request: &ToolCallRequest,
        ) -> StdResult<ApprovalDecision, ToolSystemError> {
            self.arguments
                .lock()
                .expect("the argument lock is healthy")
                .push(
                    request
                        .call()
                        .input
                        .to_value()
                        .expect("valid fixture arguments"),
                );
            Ok(ApprovalDecision::Allow)
        }
    }

    struct CountingService {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ToolService for CountingService {
        async fn discover(
            &self,
            _context: TurnContext<'_>,
        ) -> StdResult<ToolCatalog, ToolSystemError> {
            Ok(ToolCatalog::new([descriptor()]))
        }

        async fn call(&self, _request: ToolCallRequest) -> StdResult<ToolOutcome, ToolSystemError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ToolOutcome::success("ran".into()))
        }
    }

    #[test]
    fn permission_uses_stable_identity_instead_of_visible_alias() {
        let policy = PermissionLevelPolicy::new(PermissionLevel::ReadOnly);
        for (identity, name, expected) in [
            ("read_file", "Read", ToolPermission::Allow),
            ("Read", "Read", ToolPermission::RequireApproval),
        ] {
            let tool = ToolDescriptor::new(
                ToolId::try_new(identity).expect("valid identity"),
                ToolDefinition::function(name, "Read", json!({})),
            );
            assert_eq!(policy.permission(&SessionScope::default(), &tool), expected);
        }
    }

    fn descriptor() -> ToolDescriptor {
        ToolDescriptor::new(
            ToolId::try_new("shell").expect("the identity is valid"),
            ToolDefinition::function("shell", "Run a command", json!({})),
        )
    }

    fn request(descriptor: ToolDescriptor) -> ToolCallRequest {
        ToolCatalog::new([descriptor])
            .resolve(
                TurnContext::new(&SessionScope::default(), "test/model", 0, &[]),
                ToolCall {
                    id:                "call_1".to_owned(),
                    name:              "shell".to_owned(),
                    input:             ToolInput::Function(ToolArguments::from_json(
                        json!({"command": "cargo test"}),
                    )),
                    provider_metadata: BTreeMap::new(),
                },
                CancellationToken::new(),
            )
            .unwrap_or_else(|_| panic!("the test request is valid"))
    }

    #[tokio::test]
    async fn approval_required_tools_need_an_approval_service() {
        let calls = Arc::new(AtomicUsize::new(0));
        let system = ToolSystem::new(Arc::new(CountingService {
            calls: Arc::clone(&calls),
        }))
        .middleware(Arc::new(PermissionMiddleware::new(Arc::new(
            FixedPermission(ToolPermission::RequireApproval),
        ))));
        let messages = [];
        let catalog = system
            .discover(TurnContext::new(
                &SessionScope::default(),
                "test/model",
                0,
                &messages,
            ))
            .await
            .expect("discovery succeeds");

        assert_eq!(catalog.visible_tools().count(), 0);
        let outcome = system
            .call(request(
                catalog
                    .find_by_name("shell")
                    .expect("the hidden tool remains resolvable")
                    .clone(),
            ))
            .await
            .expect("the refusal succeeds");
        assert!(matches!(outcome, ToolOutcome::Failure {
            kind: ToolErrorKind::Denied,
            ..
        }));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn approval_sees_the_actual_call_arguments() {
        let calls = Arc::new(AtomicUsize::new(0));
        let approval = Arc::new(RecordingApproval {
            arguments: Mutex::new(Vec::new()),
        });
        let system = ToolSystem::new(Arc::new(CountingService {
            calls: Arc::clone(&calls),
        }))
        .middleware(Arc::new(
            PermissionMiddleware::new(Arc::new(FixedPermission(ToolPermission::RequireApproval)))
                .with_approval(approval.clone()),
        ));
        let messages = [];
        let catalog = system
            .discover(TurnContext::new(
                &SessionScope::default(),
                "test/model",
                0,
                &messages,
            ))
            .await
            .expect("discovery succeeds");
        let descriptor = catalog
            .find_by_name("shell")
            .expect("the tool is present")
            .clone();

        let outcome = system
            .call(request(descriptor))
            .await
            .expect("the call succeeds");

        assert!(matches!(outcome, ToolOutcome::Success { .. }));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(*approval.arguments.lock().expect("the lock is healthy"), [
            json!({"command": "cargo test"})
        ]);
    }

    #[test]
    fn canonical_names_pass_through() {
        assert_eq!(canonical_tool_name("read_file"), "read_file");
        assert_eq!(canonical_tool_name("shell"), "shell");
    }

    #[test]
    fn harness_names_resolve_to_canonical_names() {
        assert_eq!(canonical_tool_name("Read"), "read_file");
        assert_eq!(canonical_tool_name("Bash"), "shell");
        assert_eq!(canonical_tool_name("shell_command"), "shell");
        assert_eq!(canonical_tool_name("FetchURL"), "web_fetch");
    }

    #[test]
    fn unknown_names_pass_through() {
        assert_eq!(
            canonical_tool_name("mcp__filesystem__read_file"),
            "mcp__filesystem__read_file"
        );
    }

    #[test]
    fn known_categories_come_from_the_built_in_table() {
        assert_eq!(known_tool_category("Read"), Some(ToolCategory::Read));
        assert_eq!(
            known_tool_category("apply_patch"),
            Some(ToolCategory::Write)
        );
        assert_eq!(known_tool_category("Bash"), Some(ToolCategory::Shell));
        assert_eq!(
            known_tool_category("spawn_agent"),
            Some(ToolCategory::Subagent)
        );
        // Inside the built-in set, but outside the permission taxonomy.
        assert_eq!(known_tool_category("web_search"), None);
        assert_eq!(known_tool_category("mcp__files__read"), None);
    }

    #[test]
    fn uncategorized_tools_gate_as_shell() {
        assert_eq!(tool_category("web_search"), ToolCategory::Shell);
        assert_eq!(tool_category("mcp__files__read"), ToolCategory::Shell);
        assert_eq!(tool_category("Read"), ToolCategory::Read);
    }

    #[test]
    fn auto_approval_matches_the_permission_ladder() {
        let cases = [
            (ToolCategory::Read, [true, true, true]),
            (ToolCategory::Subagent, [true, true, true]),
            (ToolCategory::Write, [false, true, true]),
            (ToolCategory::Shell, [false, false, true]),
            (ToolCategory::Other, [false, false, true]),
        ];
        let levels = [
            PermissionLevel::ReadOnly,
            PermissionLevel::ReadWrite,
            PermissionLevel::Full,
        ];
        for (category, expected) in cases {
            for (level, expected) in levels.into_iter().zip(expected) {
                assert_eq!(
                    level.auto_approves(category),
                    expected,
                    "{level} should{} auto-approve {category}",
                    if expected { "" } else { " not" }
                );
            }
        }
    }

    #[test]
    fn the_required_level_follows_the_tool_identity() {
        for (identity, name, expected) in [
            ("read_file", "Read", PermissionLevel::ReadOnly),
            ("spawn_agent", "spawn_agent", PermissionLevel::ReadOnly),
            ("write_file", "Write", PermissionLevel::ReadWrite),
            ("edit_file", "Edit", PermissionLevel::ReadWrite),
            ("apply_patch", "apply_patch", PermissionLevel::ReadWrite),
            ("shell", "Bash", PermissionLevel::Full),
            ("web_search", "WebSearch", PermissionLevel::Full),
            ("web_fetch", "FetchURL", PermissionLevel::Full),
            ("unknown", "unknown", PermissionLevel::Full),
            (
                "mcp__files__write",
                "mcp__files__write",
                PermissionLevel::Full,
            ),
            // An extension that reuses a native alias is not that native tool.
            ("Write", "Write", PermissionLevel::Full),
            ("Read", "Read", PermissionLevel::Full),
        ] {
            let tool = ToolDescriptor::new(
                ToolId::try_new(identity).expect("valid fixture identity"),
                ToolDefinition::function(name, "Fixture tool", json!({"type": "object"})),
            );
            let required = PermissionLevelPolicy::required_level(&tool);
            assert_eq!(required, expected, "{identity} exposed as {name}");
            for level in [
                PermissionLevel::ReadOnly,
                PermissionLevel::ReadWrite,
                PermissionLevel::Full,
            ] {
                let allowed = matches!(
                    PermissionLevelPolicy::new(level).permission(&SessionScope::default(), &tool),
                    ToolPermission::Allow
                );
                assert_eq!(allowed, level >= required, "{identity} at {level}");
            }
        }
    }

    #[test]
    fn tool_auto_approval_resolves_the_name_first() {
        // `Read` is Kimi Code's name for `read_file`.
        assert!(PermissionLevel::ReadOnly.auto_approves_tool("Read"));
        assert!(!PermissionLevel::ReadOnly.auto_approves_tool("Write"));
        assert!(PermissionLevel::ReadWrite.auto_approves_tool("Write"));
        assert!(!PermissionLevel::ReadWrite.auto_approves_tool("Bash"));
        assert!(PermissionLevel::Full.auto_approves_tool("Bash"));
        // Unknown tools need approval below `Full`.
        assert!(!PermissionLevel::ReadWrite.auto_approves_tool("mcp__files__write"));
        assert!(PermissionLevel::Full.auto_approves_tool("mcp__files__write"));
    }
}
