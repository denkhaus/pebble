//! Vocabulary describing the tools a session exposes and the permission
//! ladder an embedder applies to them.

use std::fmt;

pub use pebble_agent::ToolErrorKind;
use serde::{Deserialize, Serialize};

/// Where a registered tool came from.
///
/// The registry uses this to decide how a tool's name is qualified, and the
/// context-window accounting uses it to bucket token counts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolSource {
    /// A tool pebble itself implements.
    #[default]
    Native,
    /// A tool implemented directly by the embedding application.
    Application,
    /// A tool proxied from an MCP server by the embedding application.
    Mcp {
        /// The server the tool came from.
        server_name:   String,
        /// The raw upstream name, before the `mcp__<server>__` qualification
        /// the registry applies.
        original_name: String,
    },
    /// A tool contributed by a skill.
    Skill,
}

/// The behavioral class of a tool, used by the permission table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolCategory {
    /// Reads state without changing it.
    Read,
    /// Writes files or other durable state.
    Write,
    /// Runs a subordinate process.
    Shell,
    /// Spawns or drives a child session.
    Subagent,
    /// Anything the table does not classify.
    Other,
}

impl ToolCategory {
    /// The wire spelling of this category.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Shell => "shell",
            Self::Subagent => "subagent",
            Self::Other => "other",
        }
    }
}

impl fmt::Display for ToolCategory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One tool as advertised to an observer of the session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSummary {
    /// The name the model calls.
    pub name:        String,
    /// The description sent to the model.
    pub description: String,
    /// Where the tool came from.
    pub source:      ToolSource,
    /// The behavioral class used by the permission table.
    pub category:    ToolCategory,
    /// Whether this tool has been called during the session.
    ///
    /// Pebble reports `false` at registration; a consumer that reduces the
    /// event stream flips it.
    #[serde(default)]
    pub invoked:     bool,
}

/// The permission ladder an embedder applies to a session's tools.
///
/// Each level is a superset of the one before it, and the levels order that
/// way: `ReadOnly < ReadWrite < Full`. Pebble ships the ladder and an
/// auto-approval table as an optional helper; the loop consults only the
/// permission middleware an embedder installs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PermissionLevel {
    /// Reads only.
    ReadOnly,
    /// Reads and writes, but no subordinate processes.
    ReadWrite,
    /// Everything, including subordinate processes.
    Full,
}

impl PermissionLevel {
    /// The wire spelling of this level, in kebab-case.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::ReadWrite => "read-write",
            Self::Full => "full",
        }
    }
}

impl fmt::Display for PermissionLevel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// The `(server, tool)` an MCP tool's qualified name `mcp__{server}__{tool}`
/// was built from, or `None` when the name is not in that shape.
///
/// Defined here, beside [`ToolSource::Mcp`], rather than with the MCP client:
/// the projection is always compiled and reads server names back out of the
/// tool names the registry produced.
#[must_use]
pub(crate) fn parse_qualified_name(qualified: &str) -> Option<(String, String)> {
    let rest = qualified.strip_prefix("mcp__")?;
    let (server, tool) = rest.split_once("__")?;
    if server.is_empty() || tool.is_empty() {
        return None;
    }
    Some((server.to_owned(), tool.to_owned()))
}

/// A server or tool name as a qualified tool name spells it: every character
/// outside `[A-Za-z0-9_]` replaced by `_`.
pub(crate) fn sanitize_mcp_name(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn qualified_names_parse_back() {
        assert_eq!(
            parse_qualified_name("mcp__filesystem__read_file"),
            Some(("filesystem".to_owned(), "read_file".to_owned()))
        );
        assert_eq!(parse_qualified_name("not_mcp__server__tool"), None);
        assert_eq!(parse_qualified_name("mcp__serveronly"), None);
        assert_eq!(parse_qualified_name("mcp____tool"), None);
    }

    #[test]
    fn a_name_is_sanitized_to_the_qualified_alphabet() {
        assert_eq!(sanitize_mcp_name("my-server.v2"), "my_server_v2");
        assert_eq!(sanitize_mcp_name("plain_name"), "plain_name");
    }

    #[test]
    fn tool_source_defaults_to_native() {
        assert_eq!(ToolSource::default(), ToolSource::Native);
    }

    #[test]
    fn tool_source_is_kind_tagged() {
        assert_eq!(
            serde_json::to_value(ToolSource::Native).expect("serializes"),
            json!({"kind": "native"})
        );
        assert_eq!(
            serde_json::to_value(ToolSource::Application).expect("serializes"),
            json!({"kind": "application"})
        );
        assert_eq!(
            serde_json::to_value(ToolSource::Mcp {
                server_name:   "filesystem".into(),
                original_name: "read_file".into(),
            })
            .expect("serializes"),
            json!({
                "kind": "mcp",
                "server_name": "filesystem",
                "original_name": "read_file",
            })
        );
        assert_eq!(
            serde_json::to_value(ToolSource::Skill).expect("serializes"),
            json!({"kind": "skill"})
        );
    }

    #[test]
    fn tool_summary_round_trips() {
        let summary = ToolSummary {
            name:        "read_file".into(),
            description: "Read a file".into(),
            source:      ToolSource::Native,
            category:    ToolCategory::Read,
            invoked:     false,
        };
        let value = serde_json::to_value(&summary).expect("serializes");
        assert_eq!(
            value,
            json!({
                "name": "read_file",
                "description": "Read a file",
                "source": {"kind": "native"},
                "category": "read",
                "invoked": false,
            })
        );
        assert_eq!(
            serde_json::from_value::<ToolSummary>(value).expect("parses"),
            summary
        );
    }

    #[test]
    fn tool_summary_defaults_invoked_when_absent() {
        let summary: ToolSummary = serde_json::from_value(json!({
            "name": "shell",
            "description": "Run a command",
            "source": {"kind": "native"},
            "category": "shell",
        }))
        .expect("parses");
        assert!(!summary.invoked);
    }

    #[test]
    fn permission_level_is_kebab_case_on_the_wire() {
        assert_eq!(
            serde_json::to_value(PermissionLevel::ReadOnly).expect("serializes"),
            json!("read-only")
        );
        assert_eq!(
            serde_json::to_value(PermissionLevel::ReadWrite).expect("serializes"),
            json!("read-write")
        );
        assert_eq!(
            serde_json::to_value(PermissionLevel::Full).expect("serializes"),
            json!("full")
        );
        assert_eq!(PermissionLevel::ReadWrite.to_string(), "read-write");
    }

    #[test]
    fn tool_error_kind_is_snake_case_on_the_wire() {
        assert_eq!(
            serde_json::to_value(ToolErrorKind::InvalidArguments).expect("serializes"),
            json!("invalid_arguments")
        );
        assert_eq!(
            serde_json::to_value(ToolErrorKind::Timeout).expect("serializes"),
            json!("timeout")
        );
    }
}
