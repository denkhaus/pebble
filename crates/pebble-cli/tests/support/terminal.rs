//! A small terminal emulator and PTY driver for the real interactive binary.

use std::fs::File;
use std::os::fd::OwnedFd;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use std::{env, io, mem};

use rustix::fs::{Mode, OFlags, fcntl_getfl, fcntl_setfl, open};
use rustix::io as fd_io;
use rustix::process::{Pid, Signal, kill_process};
use rustix::pty::{OpenptFlags, grantpt, openpt, ptsname, unlockpt};
use rustix::termios::{Termios, Winsize, tcgetattr, tcsetwinsize};
use tokio::io::unix::AsyncFd;
use tokio::process::{Child, Command};
use tokio::time::timeout;
use unicode_width::UnicodeWidthChar as _;
use vte::{Params, Parser, Perform};

pub(crate) struct Terminal {
    master:     AsyncFd<OwnedFd>,
    slave:      File,
    original:   Termios,
    child:      Child,
    parser:     Parser,
    pub screen: Screen,
    pub output: Vec<u8>,
}

/// The test binary re-run as `launch_terminal_child`, with a clean
/// environment apart from `PATH`, `PEBBLE_HOME`, the child's arguments, and
/// coverage output. The caller chooses its standard streams.
pub(crate) fn child_command(args: &[String], home: &Path) -> Command {
    let mut command = Command::new(env::current_exe().expect("test executable"));
    command
        .args(["--exact", "launch_terminal_child", "--nocapture"])
        .env_clear()
        .env("PATH", env::var_os("PATH").unwrap_or_default())
        .env(
            "PEBBLE_PTY_ARGS",
            serde_json::to_string(args).expect("serialize args"),
        )
        .env("PEBBLE_HOME", home)
        .kill_on_drop(true);
    if let Some(profile) = env::var_os("LLVM_PROFILE_FILE") {
        command.env("LLVM_PROFILE_FILE", profile);
    }
    command
}

impl Terminal {
    pub(crate) fn start(args: &[String], base_url: &str, namespace: &str) -> Self {
        let cwd = args
            .iter()
            .position(|arg| arg == "--cwd")
            .expect("test working directory");
        Self::start_with_env(
            args,
            base_url,
            Some(namespace),
            Path::new(&args[cwd + 1]),
            &[],
        )
    }

    pub(crate) fn start_with_env(
        args: &[String],
        base_url: &str,
        namespace: Option<&str>,
        home: &Path,
        extra: &[(&str, &str)],
    ) -> Self {
        let master = openpt(OpenptFlags::RDWR | OpenptFlags::NOCTTY).expect("open PTY");
        fd_io::fcntl_setfd(&master, fd_io::FdFlags::CLOEXEC).expect("close master on exec");
        grantpt(&master).expect("grant PTY");
        unlockpt(&master).expect("unlock PTY");
        let name = ptsname(&master, Vec::new()).expect("PTY path");
        let slave = File::from(
            open(
                name.as_c_str(),
                OFlags::RDWR | OFlags::NOCTTY | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .expect("open slave"),
        );
        let original = tcgetattr(&slave).expect("original terminal modes");
        tcsetwinsize(&slave, Winsize {
            ws_row:    24,
            ws_col:    100,
            ws_xpixel: 0,
            ws_ypixel: 0,
        })
        .expect("set size");
        fcntl_setfl(
            &master,
            fcntl_getfl(&master).expect("get flags") | OFlags::NONBLOCK,
        )
        .expect("nonblocking master");
        let mut command = child_command(args, home);
        command
            .env("TERM", "xterm-256color")
            .env("NO_COLOR", "1")
            .env("PEBBLE_OPENAI_BASE_URL", base_url)
            .stdin(Stdio::from(slave.try_clone().expect("clone input")))
            .stdout(Stdio::from(slave.try_clone().expect("clone output")))
            .stderr(Stdio::from(slave.try_clone().expect("clone errors")));
        if let Some(namespace) = namespace {
            command.env("OPENAI_API_KEY", namespace);
        }
        command.envs(extra.iter().copied());
        if extra.contains(&("PEBBLE_PTY_COLOR", "1")) {
            command.env_remove("NO_COLOR");
        }
        let child = command.spawn().expect("launch terminal child");
        Self {
            master: AsyncFd::new(master).expect("register PTY"),
            slave,
            original,
            child,
            parser: Parser::new(),
            screen: Screen::new(100, 24),
            output: Vec::new(),
        }
    }

    pub(crate) async fn send(&self, bytes: &[u8]) {
        let mut remaining = bytes;
        while !remaining.is_empty() {
            let mut ready = self.master.writable().await.expect("PTY writable");
            if let Ok(result) =
                ready.try_io(|fd| fd_io::write(fd.get_ref(), remaining).map_err(io::Error::from))
            {
                let count = result.expect("write PTY");
                remaining = &remaining[count..];
            }
        }
    }

    async fn read(&mut self) {
        let mut bytes = vec![0; 65536];
        loop {
            let mut ready = self.master.readable().await.expect("PTY readable");
            if let Ok(result) =
                ready.try_io(|fd| fd_io::read(fd.get_ref(), &mut bytes).map_err(io::Error::from))
            {
                let count = result.expect("read PTY");
                assert!(count > 0, "terminal exited early: {}", self.screen.text());
                self.output.extend_from_slice(&bytes[..count]);
                self.parser.advance(&mut self.screen, &bytes[..count]);
                let replies = mem::take(&mut self.screen.replies);
                self.send(&replies).await;
                break;
            }
        }
    }

    pub(crate) async fn until(&mut self, condition: impl Fn(&Screen) -> bool) {
        let result = timeout(Duration::from_secs(20), async {
            // A synchronized update is one visible frame. Inspecting a
            // partially received frame can mistake cleared activity for idle.
            while self.screen.updating || !condition(&self.screen) {
                self.read().await;
            }
        })
        .await;
        assert!(
            result.is_ok(),
            "terminal condition timed out:\n{}\nRecent terminal bytes: {:?}",
            self.screen.text(),
            String::from_utf8_lossy(&self.output[self.output.len().saturating_sub(4000)..])
        );
    }

    pub(crate) async fn contains(&mut self, text: &str) {
        self.until(|screen| screen.text().contains(text)).await;
    }

    pub(crate) fn resize(&mut self, columns: u16, rows: u16) {
        self.screen.resize(usize::from(columns), usize::from(rows));
        tcsetwinsize(&self.slave, Winsize {
            ws_row:    rows,
            ws_col:    columns,
            ws_xpixel: 0,
            ws_ypixel: 0,
        })
        .expect("resize PTY");
        self.signal(Signal::WINCH);
    }

    pub(crate) fn signal(&self, signal: Signal) {
        let pid =
            Pid::from_raw(i32::try_from(self.child.id().expect("child pid")).expect("pid fits"))
                .expect("nonzero pid");
        kill_process(pid, signal).expect("signal child");
    }

    pub(crate) async fn finish(self) -> Screen {
        self.finish_with("Resume:", true).await
    }

    pub(crate) async fn finish_with(mut self, marker: &str, success: bool) -> Screen {
        // The final message and mode reset must both reach the emulator.
        self.until(|screen| !screen.bracketed_paste && screen.text().contains(marker))
            .await;
        let status = timeout(Duration::from_secs(10), self.child.wait())
            .await
            .expect("child exits")
            .expect("reap child");
        assert_eq!(
            status.success(),
            success,
            "unexpected exit: {}",
            self.screen.text()
        );
        let restored = tcgetattr(&self.slave).expect("restored modes");
        assert_eq!(restored.input_modes, self.original.input_modes);
        assert_eq!(restored.output_modes, self.original.output_modes);
        assert_eq!(restored.local_modes, self.original.local_modes);
        assert_eq!(restored.control_modes, self.original.control_modes);
        assert!(self.screen.cursor_visible);
        assert!(!self.screen.used_alternate);
        assert!(!self.screen.scrollback_cleared);
        self.screen
    }
}

pub(crate) struct Screen {
    rows:                Vec<Vec<char>>,
    history:             Vec<String>,
    columns:             usize,
    row:                 usize,
    column:              usize,
    replies:             Vec<u8>,
    pub frames:          usize,
    updating:            bool,
    pub cursor_visible:  bool,
    pub bracketed_paste: bool,
    used_alternate:      bool,
    scrollback_cleared:  bool,
}

impl Screen {
    fn new(columns: usize, rows: usize) -> Self {
        Self {
            rows: vec![vec![' '; columns]; rows],
            history: Vec::new(),
            columns,
            row: 0,
            column: 0,
            replies: Vec::new(),
            frames: 0,
            updating: false,
            cursor_visible: true,
            bracketed_paste: false,
            used_alternate: false,
            scrollback_cleared: false,
        }
    }

    pub(crate) fn text(&self) -> String {
        self.history
            .iter()
            .cloned()
            .chain(self.rows.iter().map(|row| row.iter().collect::<String>()))
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub(crate) fn live_text(&self) -> String {
        self.rows
            .iter()
            .map(|row| row.iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub(crate) fn cursor_line(&self) -> String {
        self.rows[self.row]
            .iter()
            .collect::<String>()
            .trim_end()
            .into()
    }

    fn resize(&mut self, columns: usize, rows: usize) {
        while self.row >= rows {
            self.history.push(self.rows.remove(0).iter().collect());
            self.row -= 1;
        }
        self.rows.resize(rows, vec![' '; columns]);
        for row in &mut self.rows {
            row.resize(columns, ' ');
        }
        self.columns = columns;
        self.column = self.column.min(columns - 1);
    }

    fn newline(&mut self) {
        self.row += 1;
        if self.row == self.rows.len() {
            self.history.push(self.rows.remove(0).iter().collect());
            self.rows.push(vec![' '; self.columns]);
            self.row -= 1;
        }
    }
}

impl Perform for Screen {
    fn print(&mut self, character: char) {
        let width = character.width().unwrap_or(0);
        if width == 0 {
            return;
        }
        if self.column + width > self.columns {
            self.column = 0;
            self.newline();
        }
        self.rows[self.row][self.column] = character;
        self.column += width;
    }

    fn execute(&mut self, byte: u8) {
        match byte {
            b'\r' => self.column = 0,
            b'\n' => self.newline(),
            8 => self.column = self.column.saturating_sub(1),
            _ => {}
        }
    }

    fn csi_dispatch(&mut self, params: &Params, intermediates: &[u8], _ignore: bool, action: char) {
        let params: Vec<_> = params.iter().map(|value| usize::from(value[0])).collect();
        let first = params.first().copied().unwrap_or(0);
        match (intermediates, action) {
            ([], 'H' | 'f') => {
                self.row = first.max(1).min(self.rows.len()) - 1;
                self.column = params.get(1).copied().unwrap_or(1).max(1).min(self.columns) - 1;
            }
            ([], 'J') => {
                if first == 3 {
                    self.scrollback_cleared = true;
                }
                if first == 0 {
                    self.rows[self.row][self.column.min(self.columns)..].fill(' ');
                    for row in &mut self.rows[self.row + 1..] {
                        row.fill(' ');
                    }
                }
                if first == 2 {
                    for row in &mut self.rows {
                        row.fill(' ');
                    }
                }
            }
            ([], 'n') if first == 6 => self
                .replies
                .extend(format!("\x1b[{};{}R", self.row + 1, self.column + 1).bytes()),
            ([], 'c') => self.replies.extend(b"\x1b[?1;2c"),
            ([b'?'], 'u') => self.replies.extend(b"\x1b[?0u"),
            ([b'?'], 'h' | 'l') => {
                if first == 2026 {
                    self.updating = action == 'h';
                    if action == 'l' {
                        self.frames += 1;
                    }
                }
                if first == 25 {
                    self.cursor_visible = action == 'h';
                }
                if first == 2004 {
                    self.bracketed_paste = action == 'h';
                }
                if [47, 1047, 1049].contains(&first) && action == 'h' {
                    self.used_alternate = true;
                }
            }
            _ => {}
        }
    }
}
