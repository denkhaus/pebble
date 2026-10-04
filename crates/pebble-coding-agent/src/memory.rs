//! Project instructions loaded into the system prompt.
//!
//! Memory is the standing instruction a repository keeps for whoever works in
//! it — `AGENTS.md` and its relatives. Pebble reads the files an application
//! names and nothing else: no conventional filename, no walk up to a
//! repository root, no home directory. An application that wants fabro's
//! conventions builds the path list itself and hands it over.
//!
//! What this module owns is the budget. Memory competes with the conversation
//! for the context window, so the loader spends at most
//! [`ProjectMemory::BUDGET_BYTES`] across every file, skips a file whose text
//! it has already loaded, and cuts the file that crosses the line rather than
//! dropping it.
//!
//! A coding agent loads its memory through [`ProjectMemory::load`] when it
//! initializes. The same loader is public, so an application that puts project
//! instructions into a plain model call of its own reads them by the rules the
//! agent uses.

use std::collections::HashSet;

use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use crate::char_boundary::floor_char_boundary;
use crate::environment::Environment;
use crate::error::{Error, InterruptReason, Result};
use crate::profiles::join_sections;
use crate::types::MemoryFileSummary;

/// What replaces the text the budget could not fit.
const TRUNCATION_MARKER: &str = "[Project instructions truncated at 32KB]";

/// One loaded memory file.
///
/// [`content`](Self::content) is what goes into the system prompt. The rest
/// describes the file for the
/// [`MemoryLoaded`](crate::events::CodingEvent::MemoryLoaded) event, which
/// deliberately carries the description and never the text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryDocument {
    /// The path the file was read from.
    pub(crate) path:         String,
    /// The text loaded into the system prompt, already cut to the budget.
    pub(crate) content:      String,
    /// The file's full size, in bytes.
    pub(crate) byte_count:   usize,
    /// How many bytes of it were loaded.
    pub(crate) loaded_bytes: usize,
    /// Whether the budget cut the file short.
    pub(crate) truncated:    bool,
}

impl MemoryDocument {
    /// The path the file was read from, as it was given.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The text that goes into the system prompt, already cut to the budget.
    ///
    /// A cut file ends with a marker saying so.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// The file's full size, in bytes.
    #[must_use]
    pub const fn byte_count(&self) -> usize {
        self.byte_count
    }

    /// How many bytes of the file were loaded: the length of
    /// [`content`](Self::content).
    #[must_use]
    pub const fn loaded_bytes(&self) -> usize {
        self.loaded_bytes
    }

    /// Whether the budget cut the file short.
    #[must_use]
    pub const fn truncated(&self) -> bool {
        self.truncated
    }

    /// The description of this file that the event stream carries.
    #[must_use]
    pub fn to_summary(&self) -> MemoryFileSummary {
        MemoryFileSummary {
            path:         self.path.clone(),
            byte_count:   self.byte_count,
            loaded_bytes: self.loaded_bytes,
            truncated:    self.truncated,
        }
    }
}

/// The project instructions loaded from an ordered list of files, within the
/// memory budget.
///
/// This is the loader a [`CodingAgent`](crate::CodingAgent) runs over
/// [`with_memory_files`](crate::CodingAgentOptions::with_memory_files) when it
/// initializes, and what it appends to the system prompt is
/// [`text`](Self::text). Load one directly to give the same instructions to a
/// model call the agent does not make.
///
/// ```no_run
/// use pebble_coding_agent::ProjectMemory;
/// use pebble_coding_agent::environment::LocalEnvironment;
/// use tokio_util::sync::CancellationToken;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let env = LocalEnvironment::new("/path/to/work");
/// let paths = vec!["AGENTS.md".to_owned(), "CLAUDE.md".to_owned()];
/// let memory = ProjectMemory::load(&env, &paths, &CancellationToken::new()).await?;
///
/// for document in memory.documents() {
///     println!(
///         "{}: {} of {} bytes",
///         document.path(),
///         document.loaded_bytes(),
///         document.byte_count()
///     );
/// }
/// let system_prompt = format!("Follow the project's instructions.\n\n{}", memory.text());
/// # let _ = system_prompt;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectMemory {
    documents: Vec<MemoryDocument>,
}

impl ProjectMemory {
    /// The total bytes of memory one load keeps.
    pub const BUDGET_BYTES: usize = 32_768;

    /// Loads `paths`, in order, within the budget.
    ///
    /// `paths` are resolved through `env`, so a session working in a container
    /// reads the container's files. A path that cannot be read, and a file
    /// that is empty, is skipped rather than failing the load: an application
    /// normally names more candidate paths than any one repository has.
    ///
    /// A file whose text has already been loaded is skipped, which is what
    /// makes a symlinked `CLAUDE.md` beside an `AGENTS.md` cost one budget's
    /// worth instead of two. Files are charged against the budget in the
    /// order given; the first one that does not fit is cut to what remains,
    /// with a marker in place of the rest, and the files after it are skipped.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Interrupted`] when `cancel` fires, which is checked
    /// around every read.
    pub async fn load(
        env: &dyn Environment,
        paths: &[String],
        cancel: &CancellationToken,
    ) -> Result<Self> {
        let mut documents: Vec<MemoryDocument> = Vec::new();
        let mut budget_remaining = Self::BUDGET_BYTES;
        let mut seen_content: HashSet<String> = HashSet::new();

        for path in paths {
            let Some(content) = read_memory_file(env, path, cancel).await? else {
                continue;
            };

            if !seen_content.insert(content.clone()) {
                debug!(path, "Memory file duplicates one already loaded, skipping");
                continue;
            }

            let byte_count = content.len();
            if byte_count <= budget_remaining {
                debug!(path, size_bytes = byte_count, "Memory file loaded");
                budget_remaining -= byte_count;
                documents.push(MemoryDocument {
                    path: path.clone(),
                    content,
                    byte_count,
                    loaded_bytes: byte_count,
                    truncated: false,
                });
            } else if budget_remaining > 0 {
                warn!(
                    path,
                    size_bytes = byte_count,
                    budget_remaining,
                    "Memory file truncated to fit the budget"
                );
                let content = truncate_to_budget(&content, budget_remaining);
                let loaded_bytes = content.len();
                budget_remaining = 0;
                documents.push(MemoryDocument {
                    path: path.clone(),
                    content,
                    byte_count,
                    loaded_bytes,
                    truncated: true,
                });
            } else {
                warn!(
                    path,
                    size_bytes = byte_count,
                    "Memory file skipped, budget exhausted"
                );
            }
        }

        Ok(Self { documents })
    }

    /// The files that were loaded, in the order they were charged against the
    /// budget. A skipped path has no entry.
    #[must_use]
    pub fn documents(&self) -> &[MemoryDocument] {
        &self.documents
    }

    /// Whether nothing was loaded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// How many bytes were loaded across every file: the length of
    /// [`text`](Self::text) less the blank lines between documents, and never
    /// more than [`BUDGET_BYTES`](Self::BUDGET_BYTES).
    #[must_use]
    pub fn loaded_bytes(&self) -> usize {
        self.documents
            .iter()
            .map(MemoryDocument::loaded_bytes)
            .sum()
    }

    /// The description of each loaded file, as the
    /// [`MemoryLoaded`](crate::events::CodingEvent::MemoryLoaded) event
    /// carries them.
    #[must_use]
    pub fn summaries(&self) -> Vec<MemoryFileSummary> {
        self.documents
            .iter()
            .map(MemoryDocument::to_summary)
            .collect()
    }

    /// The loaded text, one document after another with a blank line between
    /// them, and empty when nothing was loaded.
    ///
    /// This is what a coding agent appends to its system prompt, after a
    /// blank line of its own.
    #[must_use]
    pub fn text(&self) -> String {
        join_sections(self.documents.iter().map(MemoryDocument::content))
    }

    /// Takes the documents out, for the one caller that moves their text into
    /// a system prompt.
    pub(crate) fn into_documents(self) -> Vec<MemoryDocument> {
        self.documents
    }
}

/// Reads one memory file, answering `None` for anything the loader skips.
///
/// An absent candidate is probed, not read: the miss stays a DEBUG line
/// instead of a failed read the environment reports as an error. The
/// cancellation token is checked on both sides of the read, so a cancel
/// that arrives while a slow environment is reading is noticed before the next
/// file is opened.
async fn read_memory_file(
    env: &dyn Environment,
    path: &str,
    cancel: &CancellationToken,
) -> Result<Option<String>> {
    if cancel.is_cancelled() {
        return Err(Error::Interrupted(InterruptReason::Cancelled));
    }

    // Memory candidates are OPTIONAL files: a convention path such as
    // `.codex/instructions.md` is usually absent, and a blind read makes
    // the sandbox driver surface the miss as a failed operation
    // (`read{...} fs: error=File ... was not found`), burying real
    // failures in the run view. Probing existence first keeps the
    // expected miss at DEBUG and never issues the failing read; a probe
    // that itself errors falls through to the read, preserving the old
    // behavior where the read can still succeed.
    if matches!(env.file_exists(path).await, Ok(false)) {
        debug!(path, "Memory file absent, skipping");
        return Ok(None);
    }

    let read = env.read_file_text(path).await;

    if cancel.is_cancelled() {
        return Err(Error::Interrupted(InterruptReason::Cancelled));
    }

    let Ok(content) = read else {
        debug!(path, "Memory file could not be read, skipping");
        return Ok(None);
    };

    if content.is_empty() {
        warn!(path, "Memory file is empty, skipping");
        return Ok(None);
    }

    Ok(Some(content))
}

/// Cuts `content` to `budget` bytes and marks where it was cut.
///
/// The marker itself is charged against the budget, so the result never
/// exceeds it. A budget too small for the marker returns as much of the marker
/// as fits, which tells a reader that something was dropped even when nothing
/// of the file survived.
fn truncate_to_budget(content: &str, budget: usize) -> String {
    if budget <= TRUNCATION_MARKER.len() {
        // The marker is ASCII, so any byte index inside it is a boundary.
        return TRUNCATION_MARKER[..budget].to_owned();
    }

    let usable = budget - TRUNCATION_MARKER.len();
    let end = floor_char_boundary(content, usable);
    format!("{}{TRUNCATION_MARKER}", &content[..end])
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::test_support::MockEnvironment;

    fn environment(files: &[(&str, &str)]) -> MockEnvironment {
        MockEnvironment {
            files: files
                .iter()
                .map(|(path, content)| ((*path).to_owned(), (*content).to_owned()))
                .collect::<HashMap<_, _>>(),
            ..MockEnvironment::default()
        }
    }

    fn paths(paths: &[&str]) -> Vec<String> {
        paths.iter().map(|path| (*path).to_owned()).collect()
    }

    async fn load(env: &MockEnvironment, paths: &[String]) -> ProjectMemory {
        ProjectMemory::load(env, paths, &CancellationToken::new())
            .await
            .expect("the load succeeds")
    }

    #[tokio::test]
    async fn a_configured_file_is_loaded_with_its_measurements() {
        let env = environment(&[("/repo/AGENTS.md", "Agent instructions")]);

        let memory = load(&env, &paths(&["/repo/AGENTS.md"])).await;

        let documents = memory.documents();
        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].path(), "/repo/AGENTS.md");
        assert_eq!(documents[0].content(), "Agent instructions");
        assert_eq!(documents[0].byte_count(), "Agent instructions".len());
        assert_eq!(documents[0].loaded_bytes(), documents[0].byte_count());
        assert!(!documents[0].truncated());
        assert!(!memory.is_empty());
        assert_eq!(memory.loaded_bytes(), "Agent instructions".len());
    }

    #[tokio::test]
    async fn exactly_the_configured_paths_are_read_in_order() {
        let env = environment(&[
            ("/repo/AGENTS.md", "agents"),
            ("/repo/CLAUDE.md", "claude"),
            ("/repo/GEMINI.md", "gemini"),
        ]);

        let memory = load(&env, &paths(&["/repo/GEMINI.md", "/repo/AGENTS.md"])).await;

        assert_eq!(
            memory
                .documents()
                .iter()
                .map(MemoryDocument::content)
                .collect::<Vec<_>>(),
            ["gemini", "agents"]
        );
    }

    #[tokio::test]
    async fn no_configured_paths_load_nothing() {
        let env = environment(&[("/repo/AGENTS.md", "agents")]);

        let memory = load(&env, &[]).await;

        assert!(memory.is_empty());
        assert_eq!(memory.loaded_bytes(), 0);
        assert_eq!(memory.text(), "");
        assert!(memory.summaries().is_empty());
    }

    #[tokio::test]
    async fn a_missing_or_empty_file_is_skipped() {
        let env = environment(&[("/repo/EMPTY.md", ""), ("/repo/AGENTS.md", "agents")]);

        let memory = load(
            &env,
            &paths(&["/repo/MISSING.md", "/repo/EMPTY.md", "/repo/AGENTS.md"]),
        )
        .await;

        assert_eq!(memory.documents().len(), 1);
        assert_eq!(memory.documents()[0].path(), "/repo/AGENTS.md");
    }

    #[tokio::test]
    async fn the_file_that_crosses_the_budget_is_cut_rather_than_dropped() {
        let first = "x".repeat(30_000);
        let env = environment(&[
            ("/repo/AGENTS.md", first.as_str()),
            ("/repo/CLAUDE.md", &"y".repeat(5_000)),
        ]);

        let memory = load(&env, &paths(&["/repo/AGENTS.md", "/repo/CLAUDE.md"])).await;

        let documents = memory.documents();
        assert_eq!(documents.len(), 2);
        assert_eq!(documents[0].content(), first);
        assert!(!documents[0].truncated());
        assert!(documents[1].truncated());
        assert!(documents[1].content().ends_with(TRUNCATION_MARKER));
        assert!(documents[1].byte_count() > documents[1].content().len());
        assert!(
            documents[0].content().len() + documents[1].content().len()
                <= ProjectMemory::BUDGET_BYTES
        );
        assert_eq!(memory.loaded_bytes(), ProjectMemory::BUDGET_BYTES);
    }

    #[tokio::test]
    async fn a_file_past_an_exhausted_budget_is_skipped() {
        let env = environment(&[
            ("/repo/A.md", &"x".repeat(ProjectMemory::BUDGET_BYTES)),
            ("/repo/B.md", "still worth reading"),
        ]);

        let memory = load(&env, &paths(&["/repo/A.md", "/repo/B.md"])).await;

        assert_eq!(memory.documents().len(), 1);
        assert_eq!(memory.documents()[0].path(), "/repo/A.md");
    }

    #[tokio::test]
    async fn a_file_repeating_text_already_loaded_is_skipped() {
        let env = environment(&[
            ("/repo/AGENTS.md", "shared instructions"),
            ("/repo/CLAUDE.md", "shared instructions"),
        ]);

        let memory = load(&env, &paths(&["/repo/AGENTS.md", "/repo/CLAUDE.md"])).await;

        assert_eq!(memory.documents().len(), 1);
        assert_eq!(memory.documents()[0].content(), "shared instructions");
    }

    #[tokio::test]
    async fn a_single_oversized_file_reports_both_sizes() {
        let whole = "x".repeat(ProjectMemory::BUDGET_BYTES + 1_024);
        let env = environment(&[("/repo/AGENTS.md", whole.as_str())]);

        let memory = load(&env, &paths(&["/repo/AGENTS.md"])).await;

        let document = &memory.documents()[0];
        assert_eq!(memory.documents().len(), 1);
        assert!(document.truncated());
        assert_eq!(document.byte_count(), whole.len());
        assert_eq!(document.loaded_bytes(), document.content().len());
        assert!(document.content().len() < document.byte_count());
        assert!(document.content().len() <= ProjectMemory::BUDGET_BYTES);
    }

    #[tokio::test]
    async fn a_cancelled_load_is_interrupted() {
        let env = environment(&[("/repo/AGENTS.md", "agents")]);
        let cancel = CancellationToken::new();
        cancel.cancel();

        let error = ProjectMemory::load(&env, &paths(&["/repo/AGENTS.md"]), &cancel)
            .await
            .expect_err("a cancelled load fails");

        assert!(matches!(
            error,
            Error::Interrupted(InterruptReason::Cancelled)
        ));
    }

    #[tokio::test]
    async fn loaded_documents_describe_themselves_for_the_event_stream() {
        let env = environment(&[("/repo/AGENTS.md", "agents")]);

        let memory = load(&env, &paths(&["/repo/AGENTS.md"])).await;

        let expected = MemoryFileSummary {
            path:         "/repo/AGENTS.md".to_owned(),
            byte_count:   6,
            loaded_bytes: 6,
            truncated:    false,
        };
        assert_eq!(memory.documents()[0].to_summary(), expected);
        assert_eq!(memory.summaries(), [expected]);
    }

    #[tokio::test]
    async fn the_text_separates_documents_with_a_blank_line() {
        let env = environment(&[
            ("/repo/AGENTS.md", "agents"),
            ("/repo/CLAUDE.md", "claude\n"),
        ]);

        let memory = load(&env, &paths(&["/repo/AGENTS.md", "/repo/CLAUDE.md"])).await;

        assert_eq!(memory.text(), "agents\n\nclaude\n");
    }

    #[test]
    fn truncation_keeps_whole_characters_and_fits_the_budget() {
        let content = "€".repeat(100);

        let truncated = truncate_to_budget(&content, TRUNCATION_MARKER.len() + 10);

        assert!(truncated.ends_with(TRUNCATION_MARKER));
        assert!(truncated.len() <= TRUNCATION_MARKER.len() + 10);
        // Nine of the ten usable bytes hold three whole three-byte characters.
        assert_eq!(truncated.len(), TRUNCATION_MARKER.len() + 9);
    }

    #[test]
    fn a_budget_smaller_than_the_marker_keeps_what_fits_of_it() {
        assert_eq!(truncate_to_budget("anything", 7), "[Projec");
        assert_eq!(truncate_to_budget("anything", 0), "");
    }
}
