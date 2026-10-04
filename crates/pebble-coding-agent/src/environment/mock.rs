//! Scriptable [`Environment`] implementations for tests.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use super::capture::capture_collected_stream;
use super::support::{ExecFailure, classify_exec_error};
use super::{
    DirEntry, EnvResult, Environment, EnvironmentError, EnvironmentErrorKind, ExecOutcome,
    ExecOutputStream, ExecRequest, ExecResult, GrepOptions,
};
use crate::event::OutputCaptureStats;
use crate::types::CommandTermination;

/// An [`Environment`] whose answers are fixtures and whose calls are recorded.
///
/// Construct it with struct-update syntax, setting only the fixtures a test
/// cares about:
///
/// ```
/// use pebble_coding_agent::test_support::MockEnvironment;
///
/// let environment = MockEnvironment {
///     files: [("a.txt".to_owned(), "hello".to_owned())].into(),
///     ..MockEnvironment::default()
/// };
/// ```
///
/// Writes are recorded in [`written_files`](Self::written_files) but are *not*
/// visible to later reads: a test that needs a filesystem that remembers uses
/// [`MutableMockEnvironment`] instead.
pub struct MockEnvironment {
    /// Contents returned by reads and existence checks.
    pub files:                 HashMap<String, String>,
    /// Returned by every [`exec`](Environment::exec) call.
    pub exec_result:           ExecResult,
    /// Returned by every [`grep`](Environment::grep) call.
    pub grep_results:          Vec<String>,
    /// Returned by every [`glob`](Environment::glob) call.
    pub glob_results:          Vec<String>,
    /// Returned by every [`list_directory`](Environment::list_directory) call.
    pub dir_entries:           Vec<DirEntry>,
    /// Reported as the working directory.
    pub working_dir:           &'static str,
    /// Reported as the platform.
    pub platform_str:          &'static str,
    /// Reported as the operating system version.
    pub os_version_str:        String,
    /// Reported by [`ExecOutcome::streams_separated`]. Set it to `false` to
    /// model an environment that merges the two streams.
    pub streams_separated:     bool,
    /// When set, [`exec`](Environment::exec) fails with this message before a
    /// process would have run, modelling a transport failure rather than a
    /// command that exited badly.
    pub exec_error:            Option<String>,
    /// The `(path, content)` pair of every write.
    pub written_files:         Mutex<Vec<(String, String)>>,
    /// Every path [`read_file_bytes`](Environment::read_file_bytes) was
    /// called with, in call order — pins which reads a code path issues.
    pub read_attempts:         Mutex<Vec<String>>,
    /// How many times [`write_existing_file`](Environment::write_existing_file)
    /// was called.
    pub existing_file_writes:  AtomicUsize,
    /// The timeout of the last command, with no timeout recorded as
    /// [`u64::MAX`].
    pub captured_timeout:      Mutex<Option<u64>>,
    /// The last command.
    pub captured_command:      Mutex<Option<String>>,
    /// Every command, in call order.
    pub captured_commands:     Mutex<Vec<String>>,
    /// Every command's working directory, in call order.
    pub captured_working_dirs: Mutex<Vec<Option<String>>>,
    /// The environment variables of the last command.
    pub captured_env_vars:     Mutex<Option<HashMap<String, String>>>,
    /// Every directory listing's `(path, depth)`, in call order.
    pub captured_listings:     Mutex<Vec<(String, Option<usize>)>>,
    /// The retention cap of the last command.
    pub captured_output_cap:   Mutex<Option<usize>>,
}

impl MockEnvironment {
    /// A mock reporting a Linux host under `/home/test`.
    #[must_use]
    pub fn linux() -> Self {
        Self {
            working_dir: "/home/test",
            platform_str: "linux",
            os_version_str: "Linux 6.1.0".to_owned(),
            ..Self::default()
        }
    }

    /// How many times a caller wrote through
    /// [`write_existing_file`](Environment::write_existing_file).
    #[must_use]
    pub fn existing_file_write_count(&self) -> usize {
        self.existing_file_writes.load(Ordering::Relaxed)
    }
}

impl Default for MockEnvironment {
    fn default() -> Self {
        Self {
            files:                 HashMap::new(),
            exec_result:           ExecResult {
                stdout:      "mock output".to_owned(),
                stderr:      String::new(),
                exit_code:   Some(0),
                termination: CommandTermination::Exited,
                duration_ms: 10,
            },
            grep_results:          Vec::new(),
            glob_results:          Vec::new(),
            dir_entries:           Vec::new(),
            working_dir:           "/work",
            platform_str:          "darwin",
            os_version_str:        "Darwin 24.0.0".to_owned(),
            streams_separated:     true,
            exec_error:            None,
            written_files:         Mutex::new(Vec::new()),
            read_attempts:         Mutex::new(Vec::new()),
            existing_file_writes:  AtomicUsize::new(0),
            captured_timeout:      Mutex::new(None),
            captured_command:      Mutex::new(None),
            captured_commands:     Mutex::new(Vec::new()),
            captured_working_dirs: Mutex::new(Vec::new()),
            captured_env_vars:     Mutex::new(None),
            captured_listings:     Mutex::new(Vec::new()),
            captured_output_cap:   Mutex::new(None),
        }
    }
}

#[async_trait]
impl Environment for MockEnvironment {
    fn working_directory(&self) -> &str {
        self.working_dir
    }

    fn platform(&self) -> &str {
        self.platform_str
    }

    fn os_version(&self) -> String {
        self.os_version_str.clone()
    }

    async fn read_file_bytes(&self, path: &str) -> EnvResult<Vec<u8>> {
        self.read_attempts
            .lock()
            .expect("read attempts lock")
            .push(path.to_owned());
        self.files
            .get(path)
            .map(|content| content.as_bytes().to_vec())
            .ok_or_else(|| {
                EnvironmentError::new(
                    EnvironmentErrorKind::NotFound,
                    format!("File not found: {path}"),
                )
            })
    }

    async fn write_file(&self, path: &str, content: &str) -> EnvResult<()> {
        self.written_files
            .lock()
            .expect("written_files lock is not poisoned")
            .push((path.to_owned(), content.to_owned()));
        Ok(())
    }

    async fn write_existing_file(&self, path: &str, content: &str) -> EnvResult<()> {
        self.existing_file_writes.fetch_add(1, Ordering::Relaxed);
        self.write_file(path, content).await
    }

    async fn rename_file(&self, source: &str, destination: &str) -> EnvResult<()> {
        if source == destination {
            return Ok(());
        }
        let content = self.read_file_text(source).await?;
        self.write_file(destination, &content).await
    }

    async fn delete_file(&self, _path: &str) -> EnvResult<()> {
        Ok(())
    }

    async fn file_exists(&self, path: &str) -> EnvResult<bool> {
        Ok(self.files.contains_key(path))
    }

    async fn list_directory(&self, path: &str, depth: Option<usize>) -> EnvResult<Vec<DirEntry>> {
        self.captured_listings
            .lock()
            .expect("captured_listings lock is not poisoned")
            .push((path.to_owned(), depth));
        Ok(self.dir_entries.clone())
    }

    async fn grep(
        &self,
        _pattern: &str,
        _path: &str,
        _options: &GrepOptions,
    ) -> EnvResult<Vec<String>> {
        Ok(self.grep_results.clone())
    }

    async fn glob(&self, _pattern: &str, _path: Option<&str>) -> EnvResult<Vec<String>> {
        Ok(self.glob_results.clone())
    }

    async fn exec(&self, request: ExecRequest<'_>) -> EnvResult<ExecOutcome> {
        let ExecRequest {
            command,
            timeout_ms,
            working_dir,
            env_vars,
            cancel_token: _,
            output_bytes_cap,
            output_sink,
        } = request;

        *self
            .captured_timeout
            .lock()
            .expect("captured_timeout lock is not poisoned") = Some(timeout_ms.unwrap_or(u64::MAX));
        *self
            .captured_command
            .lock()
            .expect("captured_command lock is not poisoned") = Some(command.to_owned());
        self.captured_commands
            .lock()
            .expect("captured_commands lock is not poisoned")
            .push(command.to_owned());
        self.captured_working_dirs
            .lock()
            .expect("captured_working_dirs lock is not poisoned")
            .push(working_dir.map(ToOwned::to_owned));
        *self
            .captured_env_vars
            .lock()
            .expect("captured_env_vars lock is not poisoned") = env_vars.cloned();
        *self
            .captured_output_cap
            .lock()
            .expect("captured_output_cap lock is not poisoned") = output_bytes_cap;

        if let Some(error) = &self.exec_error {
            return Err(EnvironmentError::new(
                classify_exec_error(ExecFailure::Start),
                error.clone(),
            ));
        }

        let mut result = self.exec_result.clone();
        // The fixture is what the command "wrote", so the sink sees all of it
        // before the cap does, the way a drained pipe would deliver it.
        if let Some(sink) = output_sink {
            if !result.stdout.is_empty() {
                sink(ExecOutputStream::Stdout, result.stdout.as_bytes());
            }
            if !result.stderr.is_empty() {
                sink(ExecOutputStream::Stderr, result.stderr.as_bytes());
            }
        }
        let stdout_capture = capture_collected_stream(&mut result.stdout, output_bytes_cap);
        let stderr_capture = capture_collected_stream(&mut result.stderr, output_bytes_cap);

        Ok(ExecOutcome {
            result,
            streams_separated: self.streams_separated,
            stdout_capture,
            stderr_capture,
        })
    }
}

/// An [`Environment`] backed by a map that writes actually change.
///
/// Use it where a test drives a read-modify-write sequence — patching a file,
/// editing and reading back — and cares that the second read sees the first
/// write.
pub struct MutableMockEnvironment {
    /// The files this environment holds.
    pub files: Mutex<HashMap<String, String>>,
}

impl MutableMockEnvironment {
    /// An environment holding `files`.
    #[must_use]
    pub fn new(files: HashMap<String, String>) -> Self {
        Self {
            files: Mutex::new(files),
        }
    }
}

#[async_trait]
impl Environment for MutableMockEnvironment {
    fn working_directory(&self) -> &'static str {
        "/work"
    }

    fn platform(&self) -> &'static str {
        "linux"
    }

    fn os_version(&self) -> String {
        "Linux 6.1.0".to_owned()
    }

    async fn read_file_bytes(&self, path: &str) -> EnvResult<Vec<u8>> {
        self.files
            .lock()
            .expect("files lock is not poisoned")
            .get(path)
            .map(|content| content.as_bytes().to_vec())
            .ok_or_else(|| {
                EnvironmentError::new(
                    EnvironmentErrorKind::NotFound,
                    format!("File not found: {path}"),
                )
            })
    }

    async fn write_file(&self, path: &str, content: &str) -> EnvResult<()> {
        self.files
            .lock()
            .expect("files lock is not poisoned")
            .insert(path.to_owned(), content.to_owned());
        Ok(())
    }

    async fn rename_file(&self, source: &str, destination: &str) -> EnvResult<()> {
        let mut files = self.files.lock().expect("files lock is not poisoned");
        let content = files.get(source).cloned().ok_or_else(|| {
            EnvironmentError::new(
                EnvironmentErrorKind::NotFound,
                format!("File not found: {source}"),
            )
        })?;
        if source != destination {
            files.remove(source);
            files.insert(destination.to_owned(), content);
        }
        Ok(())
    }

    async fn delete_file(&self, path: &str) -> EnvResult<()> {
        self.files
            .lock()
            .expect("files lock is not poisoned")
            .remove(path);
        Ok(())
    }

    async fn file_exists(&self, path: &str) -> EnvResult<bool> {
        Ok(self
            .files
            .lock()
            .expect("files lock is not poisoned")
            .contains_key(path))
    }

    async fn list_directory(&self, _path: &str, _depth: Option<usize>) -> EnvResult<Vec<DirEntry>> {
        Ok(Vec::new())
    }

    async fn grep(
        &self,
        pattern: &str,
        _path: &str,
        _options: &GrepOptions,
    ) -> EnvResult<Vec<String>> {
        let files = self.files.lock().expect("files lock is not poisoned");
        let mut matches = Vec::new();
        for (path, content) in files.iter() {
            for (index, line) in content.lines().enumerate() {
                if line.contains(pattern) {
                    matches.push(format!("{path}:{}:{line}", index + 1));
                }
            }
        }
        Ok(matches)
    }

    async fn glob(&self, _pattern: &str, _path: Option<&str>) -> EnvResult<Vec<String>> {
        Ok(Vec::new())
    }

    async fn exec(&self, _request: ExecRequest<'_>) -> EnvResult<ExecOutcome> {
        Ok(ExecOutcome {
            result:            ExecResult {
                stdout:      String::new(),
                stderr:      String::new(),
                exit_code:   Some(0),
                termination: CommandTermination::Exited,
                duration_ms: 0,
            },
            streams_separated: true,
            stdout_capture:    OutputCaptureStats::complete(0),
            stderr_capture:    OutputCaptureStats::complete(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_with_files() -> MockEnvironment {
        MockEnvironment {
            files: [("a.txt".to_owned(), "hello".to_owned())].into(),
            ..MockEnvironment::default()
        }
    }

    #[tokio::test]
    async fn reads_come_from_the_file_fixtures() {
        let environment = mock_with_files();

        assert_eq!(
            environment
                .read_file_text("a.txt")
                .await
                .expect("the fixture exists"),
            "hello"
        );
        assert!(environment.file_exists("a.txt").await.expect("checked"));
        assert!(!environment.file_exists("b.txt").await.expect("checked"));
    }

    #[tokio::test]
    async fn a_missing_fixture_reads_as_not_found() {
        let error = MockEnvironment::default()
            .read_file_bytes("missing.txt")
            .await
            .expect_err("there is no fixture");

        assert_eq!(error.kind(), EnvironmentErrorKind::NotFound);
        assert_eq!(error.to_string(), "File not found: missing.txt");
    }

    #[tokio::test]
    async fn writes_are_recorded_but_not_visible_to_reads() {
        let environment = mock_with_files();

        environment
            .write_file("b.txt", "written")
            .await
            .expect("writes are recorded");

        assert_eq!(
            *environment
                .written_files
                .lock()
                .expect("written_files lock is not poisoned"),
            vec![("b.txt".to_owned(), "written".to_owned())]
        );
        assert!(environment.read_file_text("b.txt").await.is_err());
    }

    #[tokio::test]
    async fn writing_an_existing_file_is_counted() {
        let environment = MockEnvironment::default();

        environment
            .write_existing_file("a.txt", "one")
            .await
            .expect("writes are recorded");
        environment
            .write_existing_file("a.txt", "two")
            .await
            .expect("writes are recorded");
        environment
            .write_file("a.txt", "three")
            .await
            .expect("writes are recorded");

        assert_eq!(environment.existing_file_write_count(), 2);
        assert_eq!(
            environment
                .written_files
                .lock()
                .expect("written_files lock is not poisoned")
                .len(),
            3
        );
    }

    #[tokio::test]
    async fn commands_are_captured_and_answered_from_the_fixture() {
        let environment = MockEnvironment::default();
        let env_vars = HashMap::from([("KEY".to_owned(), "value".to_owned())]);

        let outcome = environment
            .exec(ExecRequest {
                timeout_ms: Some(1_234),
                working_dir: Some("/elsewhere"),
                env_vars: Some(&env_vars),
                output_bytes_cap: Some(64),
                ..ExecRequest::new("echo hello")
            })
            .await
            .expect("the mock answers");

        assert_eq!(outcome.result.stdout, "mock output");
        assert!(outcome.streams_separated);
        assert_eq!(
            *environment
                .captured_command
                .lock()
                .expect("captured_command lock is not poisoned"),
            Some("echo hello".to_owned())
        );
        assert_eq!(
            *environment
                .captured_commands
                .lock()
                .expect("captured_commands lock is not poisoned"),
            vec!["echo hello".to_owned()]
        );
        assert_eq!(
            *environment
                .captured_timeout
                .lock()
                .expect("captured_timeout lock is not poisoned"),
            Some(1_234)
        );
        assert_eq!(
            *environment
                .captured_working_dirs
                .lock()
                .expect("captured_working_dirs lock is not poisoned"),
            vec![Some("/elsewhere".to_owned())]
        );
        assert_eq!(
            *environment
                .captured_env_vars
                .lock()
                .expect("captured_env_vars lock is not poisoned"),
            Some(env_vars)
        );
        assert_eq!(
            *environment
                .captured_output_cap
                .lock()
                .expect("captured_output_cap lock is not poisoned"),
            Some(64)
        );
    }

    #[tokio::test]
    async fn a_command_without_a_timeout_records_the_widest_one() {
        let environment = MockEnvironment::default();

        environment
            .exec(ExecRequest::new("echo hello"))
            .await
            .expect("the mock answers");

        assert_eq!(
            *environment
                .captured_timeout
                .lock()
                .expect("captured_timeout lock is not poisoned"),
            Some(u64::MAX)
        );
    }

    #[tokio::test]
    async fn an_injected_exec_error_fails_before_a_process_runs() {
        let environment = MockEnvironment {
            exec_error: Some("sandbox is gone".to_owned()),
            ..MockEnvironment::default()
        };

        let error = environment
            .exec(ExecRequest::new("echo hello"))
            .await
            .expect_err("the injected failure wins");

        assert_eq!(error.to_string(), "sandbox is gone");
        assert_eq!(
            *environment
                .captured_command
                .lock()
                .expect("captured_command lock is not poisoned"),
            Some("echo hello".to_owned()),
            "the call is still recorded"
        );
    }

    #[tokio::test]
    async fn a_sink_is_fed_the_fixture_before_the_cap_applies() {
        use std::sync::Arc;

        let environment = MockEnvironment {
            exec_result: ExecResult {
                stdout: "abcdefghijklmnopqrst".to_owned(),
                stderr: "warned".to_owned(),
                ..MockEnvironment::default().exec_result
            },
            ..MockEnvironment::default()
        };
        let seen: Arc<Mutex<Vec<(ExecOutputStream, String)>>> = Arc::default();
        let recorded = Arc::clone(&seen);

        let outcome = environment
            .exec(ExecRequest {
                output_bytes_cap: Some(8),
                output_sink: Some(Arc::new(move |stream, chunk: &[u8]| {
                    recorded
                        .lock()
                        .expect("seen lock is not poisoned")
                        .push((stream, String::from_utf8_lossy(chunk).into_owned()));
                })),
                ..ExecRequest::new("noisy")
            })
            .await
            .expect("the mock answers");

        assert_eq!(outcome.result.stdout, "abcdqrst");
        assert_eq!(*seen.lock().expect("seen lock is not poisoned"), vec![
            (ExecOutputStream::Stdout, "abcdefghijklmnopqrst".to_owned()),
            (ExecOutputStream::Stderr, "warned".to_owned()),
        ]);
    }

    #[tokio::test]
    async fn scripted_output_is_bounded_by_the_retention_cap() {
        let environment = MockEnvironment {
            exec_result: ExecResult {
                stdout: "abcdefghijklmnopqrst".to_owned(),
                ..MockEnvironment::default().exec_result
            },
            ..MockEnvironment::default()
        };

        let outcome = environment
            .exec(ExecRequest {
                output_bytes_cap: Some(8),
                ..ExecRequest::new("noisy")
            })
            .await
            .expect("the mock answers");

        assert_eq!(outcome.result.stdout, "abcdqrst");
        assert_eq!(outcome.stdout_capture, OutputCaptureStats {
            observed_bytes: 20,
            retained_bytes: 8,
            omitted_bytes:  12,
        });
        assert_eq!(outcome.stderr_capture, OutputCaptureStats::complete(0));
    }

    #[tokio::test]
    async fn uncapped_scripted_output_is_reported_whole() {
        let outcome = MockEnvironment::default()
            .exec(ExecRequest::new("echo hello"))
            .await
            .expect("the mock answers");

        assert_eq!(
            outcome.output_capture(),
            OutputCaptureStats::complete("mock output".len())
        );
    }

    #[tokio::test]
    async fn search_answers_come_from_the_fixtures() {
        let environment = MockEnvironment {
            grep_results: vec!["a.txt:1:hello".to_owned()],
            glob_results: vec!["/work/a.txt".to_owned()],
            ..MockEnvironment::default()
        };

        assert_eq!(
            environment
                .grep("anything", ".", &GrepOptions::default())
                .await
                .expect("the mock answers"),
            vec!["a.txt:1:hello".to_owned()]
        );
        assert_eq!(
            environment
                .glob("anything", None)
                .await
                .expect("the mock answers"),
            vec!["/work/a.txt".to_owned()]
        );
        assert!(
            environment
                .list_directory(".", None)
                .await
                .expect("the mock answers")
                .is_empty()
        );
    }

    #[test]
    fn the_linux_mock_reports_a_linux_host() {
        let environment = MockEnvironment::linux();

        assert_eq!(environment.working_directory(), "/home/test");
        assert_eq!(environment.platform(), "linux");
        assert_eq!(environment.os_version(), "Linux 6.1.0");
    }

    #[test]
    fn the_default_mock_reports_a_darwin_host() {
        let environment = MockEnvironment::default();

        assert_eq!(environment.working_directory(), "/work");
        assert_eq!(environment.platform(), "darwin");
        assert_eq!(environment.os_version(), "Darwin 24.0.0");
    }

    #[tokio::test]
    async fn the_mutable_mock_shows_writes_to_later_reads() {
        let environment =
            MutableMockEnvironment::new(HashMap::from([("a.txt".to_owned(), "before".to_owned())]));

        environment
            .write_file("a.txt", "after")
            .await
            .expect("the write lands");

        assert_eq!(
            environment
                .read_file_text("a.txt")
                .await
                .expect("the file exists"),
            "after"
        );
        assert!(environment.file_exists("a.txt").await.expect("checked"));

        environment
            .delete_file("a.txt")
            .await
            .expect("the delete lands");

        assert!(!environment.file_exists("a.txt").await.expect("checked"));
    }

    #[tokio::test]
    async fn the_mutable_mock_greps_its_files_by_substring() {
        let environment = MutableMockEnvironment::new(HashMap::from([(
            "a.txt".to_owned(),
            "one\ntwo\nthree".to_owned(),
        )]));

        let matches = environment
            .grep("t", ".", &GrepOptions::default())
            .await
            .expect("the search runs");

        assert_eq!(matches, vec![
            "a.txt:2:two".to_owned(),
            "a.txt:3:three".to_owned(),
        ]);
    }

    #[tokio::test]
    async fn the_mutable_mock_runs_no_commands() {
        let outcome = MutableMockEnvironment::new(HashMap::new())
            .exec(ExecRequest::new("echo hello"))
            .await
            .expect("the mock answers");

        assert!(outcome.result.is_success());
        assert!(outcome.result.stdout.is_empty());
    }
}
