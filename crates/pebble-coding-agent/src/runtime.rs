//! Internal coding-agent state and adapters.
//!
//! [`CodingRuntime`] holds the coding-specific state around the
//! provider-neutral [`Agent`]. The public [`CodingAgent`](crate::CodingAgent)
//! facade owns its lifecycle and is the only application entry point.

mod builder;
mod control;
#[cfg(test)]
mod loop_tests;
mod retry;
#[cfg(test)]
pub(crate) mod testing;
#[cfg(test)]
mod tests;
mod turn;

use std::fmt;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::SystemTime;

use lithos_llm::Client;
use lithos_llm::types::{Error as LlmError, ErrorKind as LlmErrorKind, ReasoningEffort, Speed};
use pebble_agent::{Agent, AgentControlHandle, ToolMiddleware};
#[cfg(test)]
use pebble_agent::UserMessage;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

pub(crate) use self::builder::CodingRuntimeBuilder;
pub(crate) use self::control::{actor_from_attribution, input_message, steering_message};
pub use self::retry::RetryEventObserver;
use self::turn::{CodingAgentBridge, ConversationState};
pub(crate) use crate::coding_agent::{
    CodingAgentBuildError, CodingInput, FailoverOutlook, PromptTiming, ResumeMode, ShutdownReason,
};
use crate::compaction::{
    CompactionAccount, CompactionControl, CompactionOptions, CompactionOutcome, CompactionReason,
    CompactionRequest, compact_context, estimate_active_context_usage,
};
use crate::config::CodingAgentOptions;
use crate::context_window::{memory_prompt_tokens, skills_prompt_tokens};
use crate::environment::{Environment, ExecRequest};
use crate::error::{Error, ErrorData, ErrorKind, InterruptReason, Result, TaskKind};
use crate::event::Emitter;
use crate::file_tracker::FileTracker;
use crate::history::History;
use crate::human_input::HumanInputProvider;
use crate::memory::ProjectMemory;
use crate::policy::{CompactionPolicy, ContextPolicy};
use crate::profile::{AgentProfile, EnvContext, ModelFacts};
use crate::prompt_transform::{SystemPromptContext, SystemPromptTransform};
use crate::record::{SESSION_RECORD_FORMAT_VERSION, SessionRecord};
use crate::redact::Redactor;
use crate::skills::{Skill, SkillExpansion, discover_skills};
use crate::subagent::{SubagentEventCallback, SubagentSupervisor};
use crate::tool::{ToolDefinitionWithSource, ToolEnvProvider, ToolRegistry};
use crate::tools::skill::make_use_skill_tool_for_vocabulary;
#[cfg(test)]
use crate::types::PermissionLevel;
use crate::types::{
    AgentProfileKind, CodingAgentEvent, CodingAgentState, CodingEvent, ContextWindowSnapshot,
    MemoryFileSummary, Message, SkillSummary, ToolSummary, Usage, is_prompt_source, rfc3339_millis,
};
#[cfg(test)]
use crate::types::{InputContent, InputSource};
use crate::{SessionId, SessionScope, discovery};

/// How long a probe run inside the environment may take.
const PROBE_TIMEOUT_MS: u64 = 5_000;

/// A live session's state, warm, for a successor in the same process.
///
/// The record is the durable part. The rest is what
/// [`CodingRuntime::initialize`] derives from the environment and the options,
/// carried so the successor does not derive it again.
#[derive(Clone, Debug)]
pub(crate) struct WarmState {
    pub(crate) record: SessionRecord,
    pub(crate) system_prompt: String,
    pub(crate) skills: Vec<Skill>,
    pub(crate) skill_dirs: Vec<String>,
    pub(crate) memory_summaries: Vec<MemoryFileSummary>,
    pub(crate) memory_tokens: u64,
    pub(crate) skills_tokens: u64,
    pub(crate) file_tracker: FileTracker,
    pub(crate) activated_skill_context_observed: bool,
    pub(crate) context_window: Option<ContextWindowSnapshot>,
}

/// What one prompt accumulated across every input it processed.
///
/// `usage` includes the summary call of each entry in `compactions`, which
/// is its breakdown, not an addition to it.
#[derive(Clone, Debug, Default)]
struct PromptTotals {
    timing:          PromptTiming,
    usage:           Usage,
    /// The compactions this prompt completed, in order.
    compactions:     Vec<CompactionAccount>,
    /// The assistant turns and tool results this prompt committed to the
    /// history: what a failover leaves in the conversation for the next
    /// route to continue from.
    committed_turns: u64,
}

/// Where one prompt starts: with new input, or from where the history stopped.
enum PromptStart {
    /// Commit this input, then ask the model.
    Input(CodingInput),
    /// Ask the model on the history as it stands.
    Continue,
}

/// Where an outside task names why it is about to cancel a prompt.
///
/// [`CodingRuntime::interrupt_reason_handle`] hands one out before the prompt
/// starts, so a watchdog holding it can say what it is doing — a budget spent,
/// a deadline passed — and the prompt then ends with that reason rather than a
/// plain cancellation. Cloning is cheap and every clone names the same session.
///
/// First writer wins: the reason a prompt reports is the first one recorded,
/// and [`record`](Self::record) is the only way to write, so a later task
/// cannot overwrite what already explains the interrupt.
#[derive(Clone, Debug, Default)]
pub(crate) struct InterruptReasonHandle(Arc<Mutex<Option<InterruptReason>>>);

impl InterruptReasonHandle {
    /// Records `reason` unless one is already recorded.
    ///
    /// Answers whether this call is the one the prompt will report.
    pub(crate) fn record(&self, reason: InterruptReason) -> bool {
        let mut recorded = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if recorded.is_some() {
            return false;
        }
        *recorded = Some(reason);
        true
    }

    /// The reason recorded so far, if any.
    #[must_use]
    pub(crate) fn current(&self) -> Option<InterruptReason> {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Forgets the recorded reason, once the prompt it explained has
    /// reported it.
    fn clear(&self) {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = None;
    }
}

/// The resolved model and client for this session. Shared by the runtime,
/// the lifecycle bridge, and the model service so all calls use one route.
struct SessionModel {
    client:         Client,
    provider:       String,
    model:          String,
    model_selector: String,
    facts:          ModelFacts,
}

/// Resources prepared at initialization and shared unchanged with the bridge.
/// Token counts describe these exact prompt resources.
#[derive(Clone)]
struct PromptResources {
    registry:      ToolRegistry,
    skills:        Vec<Skill>,
    /// The directories the skills were discovered in, as
    /// [`SkillsDiscovered`](CodingEvent::SkillsDiscovered) reported them.
    skill_dirs:    Vec<String>,
    system_prompt: String,
    memory_tokens: u64,
    skills_tokens: u64,
}

/// One conversation with one model.
///
/// A session is used from one place at a time: [`CodingRuntime::prompt`]
/// borrows it for the length of a prompt. Everything that has to reach a live
/// prompt — steering, interrupts, follow-up input, the event stream — comes
/// from a handle taken before the prompt starts.
#[must_use = "call `shutdown` to stop the session and join what it owns"]
pub(crate) struct CodingRuntime {
    model_context:     Arc<SessionModel>,
    resources:         Arc<PromptResources>,
    session_scope:     SessionScope,
    created_at:        SystemTime,
    config:            CodingAgentOptions,
    /// The conversation and what the current prompt has accumulated, shared
    /// with the bridge that writes it.
    conversation:      Arc<Mutex<ConversationState>>,
    emitter:           Emitter,
    /// The event pump, until [`CodingRuntime::shutdown`] joins it.
    pump:              Option<JoinHandle<Result<()>>>,
    /// What the session is doing, shared with the bridge that runs the tool
    /// round.
    state:             StateMachine,
    ended:             bool,
    end_emitted:       bool,
    profile:           Arc<dyn AgentProfile>,
    knowledge_cutoff:  String,
    tool_middleware:   Vec<Arc<dyn ToolMiddleware>>,
    env:               Arc<dyn Environment>,
    human_input:       Option<Arc<dyn HumanInputProvider>>,
    tool_env_provider: Option<Arc<dyn ToolEnvProvider>>,
    /// What strips secrets out of the process output this session publishes.
    redactor:          Arc<dyn Redactor>,
    /// The control the generic agent is bound to when it is built, so a
    /// handle can be given out before the first prompt creates the agent.
    agent_control:     AgentControlHandle,
    /// Ends the whole prompt. Distinct from the round token, which ends one
    /// turn.
    cancel_token:      CancellationToken,
    /// Why the prompt in progress is being ended, when an outside task said.
    interrupt_reason:  InterruptReasonHandle,
    compaction:        CompactionControl,
    memory_summaries:  Vec<MemoryFileSummary>,
    subagents:         Option<SubagentSupervisor>,
    /// The application's adjustment to the system prompt, applied once when
    /// the session initializes.
    prompt_transform:  Option<Arc<dyn SystemPromptTransform>>,
    context_policy:    Option<Arc<dyn ContextPolicy>>,
    compaction_policy: Option<Arc<dyn CompactionPolicy>>,
    /// The provider-neutral conversation loop, created after initialization on
    /// the first prompt and retained for the rest of the session.
    coding_agent:      Option<Agent>,
    /// Coding state and durable projection shared with `coding_agent`.
    coding_bridge:     Option<Arc<CodingAgentBridge>>,
    /// The agent's fallback plan as it stands for this runtime, when the
    /// agent has one, so a model failure that will not move the prompt can
    /// be reported before the prompt ends.
    failover_outlook:  Option<FailoverOutlook>,
}

impl fmt::Debug for CodingRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodingRuntime")
            .field("id", &self.session_scope.session_id())
            .field("root_session_id", &self.session_scope.root_session_id())
            .field("provider", &self.model_context.provider)
            .field("model", &self.model_context.model)
            .field("profile", &self.profile.profile_kind())
            .field("state", &self.state.current())
            .field("ended", &self.ended)
            .field("turns", &self.conversation().history.len())
            .finish_non_exhaustive()
    }
}

impl CodingRuntime {
    /// Starts building a session that talks to the model through `client`.
    pub(crate) fn builder(client: Client) -> CodingRuntimeBuilder {
        CodingRuntimeBuilder::new(client)
    }

    /// Rebuilds a stored session, ready to carry on where it left off.
    ///
    /// The record supplies the identity, the conversation, and where the event
    /// stream had got to; `deps` supplies everything a record cannot hold — the
    /// client, the environment, the tools, the options. `mode` decides the
    /// model: the exact route the record names, restored without guessing, or
    /// a selector the caller chose for failover. Event numbering continues from
    /// the record, so the root session tree's events stay uniquely numbered
    /// across restarts.
    ///
    /// # Errors
    ///
    /// Returns [`CodingAgentBuildError`] for the same reasons
    /// [`CodingRuntimeBuilder::build`] does;
    /// [`UnsupportedRecord`](CodingAgentBuildError::UnsupportedRecord) for a
    /// format version this build does not read; and, when the recorded model is
    /// asked for,
    /// [`RecordedRouteMissing`](CodingAgentBuildError::RecordedRouteMissing)
    /// for a record that names no route,
    /// [`RecordedRouteUnavailable`](CodingAgentBuildError::RecordedRouteUnavailable)
    /// for one the client cannot reach, and
    /// [`RecordedRouteMismatch`](CodingAgentBuildError::RecordedRouteMismatch)
    /// if the resolver answered with a different model.
    pub(crate) fn from_record(
        record: SessionRecord,
        mode: &ResumeMode,
        deps: CodingRuntimeBuilder,
    ) -> StdResult<Self, CodingAgentBuildError> {
        if !record.is_supported() {
            return Err(CodingAgentBuildError::UnsupportedRecord {
                version:   record.format_version,
                supported: SESSION_RECORD_FORMAT_VERSION,
            });
        }

        // The selector the build resolves, and the exact route it must
        // answer with when the record's own model is asked for.
        let (selector, recorded) = match mode {
            ResumeMode::RecordedModel => {
                let route = record.recorded_route().ok_or_else(|| {
                    CodingAgentBuildError::RecordedRouteMissing {
                        session_id: record.scope.session_id().to_string(),
                    }
                })?;
                (route.clone(), Some(route))
            }
            ResumeMode::UseModel(selector) => (selector.clone(), None),
        };

        let mut deps = deps;
        deps.model = Some(selector);
        deps.events.resume_after_seq = record.last_event_seq;
        let built = deps.build_with_scope(record.scope, record.created_at);
        let session = match (built, &recorded) {
            // An exact route the client cannot reach is its own failure, and
            // never a reason to run the conversation somewhere else.
            (Err(CodingAgentBuildError::ModelSelection { source, .. }), Some(_)) => {
                return Err(CodingAgentBuildError::RecordedRouteUnavailable {
                    provider: record.provider.clone().unwrap_or_default(),
                    model: record.model.clone().unwrap_or_default(),
                    source,
                });
            }
            (built, _) => built?,
        };
        if let Some(recorded) = recorded {
            let resolved = session.route();
            if resolved != recorded {
                return Err(CodingAgentBuildError::RecordedRouteMismatch { recorded, resolved });
            }
        }

        session
            .conversation()
            .replace_history(History::from_stored_messages(&record.messages));
        Ok(session)
    }

    /// Rebuilds a live session's successor from its warm state, on the same
    /// route, without initializing again.
    ///
    /// Everything [`initialize`](Self::initialize) would compute is taken from
    /// the state instead: the system prompt, the discovered skills and the tool
    /// that loads one, the token counts the context snapshot reads, and the
    /// files the session touched. Call
    /// [`start_from_warm_state`](Self::start_from_warm_state) afterwards in
    /// place of `initialize`.
    ///
    /// # Errors
    ///
    /// As [`from_record`](Self::from_record) on the recorded model.
    pub(crate) fn from_warm_state(
        state: WarmState,
        deps: CodingRuntimeBuilder,
    ) -> StdResult<Self, CodingAgentBuildError> {
        let mut session = Self::from_record(state.record, &ResumeMode::RecordedModel, deps)?;
        let resources = Arc::make_mut(&mut session.resources);
        if !state.skills.is_empty() {
            let vocabulary = resources.registry.vocabulary();
            resources
                .registry
                .register(make_use_skill_tool_for_vocabulary(
                    Arc::from(state.skills.clone()),
                    vocabulary,
                ))?;
        }
        resources.skills = state.skills;
        resources.skill_dirs = state.skill_dirs;
        resources.system_prompt = state.system_prompt;
        resources.memory_tokens = state.memory_tokens;
        resources.skills_tokens = state.skills_tokens;
        session.memory_summaries = state.memory_summaries;
        {
            let mut conversation = session.conversation();
            conversation.file_tracker = state.file_tracker;
            conversation.activated_skill_context_observed = state.activated_skill_context_observed;
            conversation.context_window = state.context_window;
        }
        Ok(session)
    }

    /// Opens the event stream of a session built from warm state.
    ///
    /// Publishes [`SessionStarted`](CodingEvent::SessionStarted) and then
    /// [`SkillsDiscovered`](CodingEvent::SkillsDiscovered) with the skills the
    /// warm state carried, so a view that folds only the successor's events
    /// still lists them. Nothing was read for it: the directories are the
    /// ones the predecessor searched and nothing is reported skipped. No
    /// memory is loaded and no memory event is published, because the warm
    /// state carries what loading produced. The MCP servers the successor's
    /// builder names are started and announced by the build, as a fresh
    /// agent's are.
    pub(crate) async fn start_from_warm_state(&mut self) -> Result<()> {
        self.emit(CodingEvent::SessionStarted {
            provider: Some(self.model_context.provider.clone()),
            model:    Some(self.model_context.model.clone()),
        });
        self.emit(CodingEvent::SkillsDiscovered {
            profile:     self.profile.profile_kind().as_str().to_owned(),
            source_dirs: self.resources.skill_dirs.clone(),
            skills:      self.skill_summaries(),
            skipped:     Vec::new(),
        });
        self.flush_events().await.map(|_| ())
    }

    /// Everything a successor in this process needs to carry on without
    /// initializing again: the durable record plus the state initialization
    /// derived from it.
    pub(crate) fn warm_state(&self) -> WarmState {
        let record = self.to_record();
        let conversation = self.conversation();
        WarmState {
            record,
            system_prompt: self.resources.system_prompt.clone(),
            skills: self.resources.skills.clone(),
            skill_dirs: self.resources.skill_dirs.clone(),
            memory_summaries: self.memory_summaries.clone(),
            memory_tokens: self.resources.memory_tokens,
            skills_tokens: self.resources.skills_tokens,
            file_tracker: conversation.file_tracker.clone(),
            activated_skill_context_observed: conversation.activated_skill_context_observed,
            context_window: conversation.context_window.clone(),
        }
    }

    /// The session as it should be stored.
    ///
    /// Everything a resumed session needs and nothing an application could not
    /// supply again. A child records which session spawned it, so a stored tree
    /// can be read back in shape; the tree itself is the application's to
    /// rebuild, because a child's supervisor is not stored.
    ///
    /// The record stores the last sequence committed by the event pipeline.
    /// Public callers take records between prompts, after the prompt's event
    /// barrier has committed its complete history.
    pub(crate) fn to_record(&self) -> SessionRecord {
        let mut record = SessionRecord::new(self.session_scope.clone());
        record.provider = Some(self.model_context.provider.clone());
        record.model = Some(self.model_context.model.clone());
        record.created_at = self.created_at;
        record.last_event_seq = self.emitter.committed_seq();
        record.messages = self.conversation().history.to_stored_messages();
        record
    }

    /// Loads what the session was told, and captures where it is working.
    ///
    /// Call this once, before the first prompt. It publishes
    /// [`SessionStarted`](CodingEvent::SessionStarted), loads the memory files
    /// and skill directories the options name, probes the environment for the
    /// prompt's sake, and asks the profile for the system prompt the whole
    /// session will use. Naming no memory files and no skill directories is
    /// normal: pebble looks in no conventional location and guesses no
    /// filename.
    ///
    /// Discovering any skill also registers the tool that loads one, in the
    /// profile's own vocabulary. That is the only tool a session adds to
    /// itself: what it loads is not known until the directories have been
    /// read.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Interrupted`] when the session is cancelled while it is
    /// initializing, which is checked around every read and every probe.
    #[tracing::instrument(
        name = "coding_session_initialize",
        skip_all,
        fields(session_id = %self.session_scope.session_id(), provider = %self.model_context.provider, model = %self.model_context.model)
    )]
    pub(crate) async fn initialize(&mut self) -> Result<()> {
        let cancel = self.cancel_token.clone();

        self.emit(CodingEvent::SessionStarted {
            provider: Some(self.model_context.provider.clone()),
            model:    Some(self.model_context.model.clone()),
        });
        if cancel.is_cancelled() {
            return Err(Error::Interrupted(InterruptReason::Cancelled));
        }

        let profile_kind = self.profile.profile_kind();
        let profile = profile_kind.as_str().to_owned();

        let sources =
            discovery::resolve_sources(&self.config, self.env.as_ref(), profile_kind, &cancel)
                .await?;

        // Independent reads of the environment, overlapped; the events they
        // feed stay in their documented order below.
        let (memory, skills) = tokio::join!(
            ProjectMemory::load(self.env.as_ref(), &sources.memory_files, &cancel),
            discover_skills(self.env.as_ref(), &sources.skill_dirs, &cancel),
        );

        let memory = memory?;
        // The files are described, never quoted: the durable stream must not
        // carry the bytes of a project's own instructions. The same
        // descriptions are what a prompt transform is shown.
        let memory_summaries = memory.summaries();
        self.memory_summaries.clone_from(&memory_summaries);
        self.emit(CodingEvent::MemoryLoaded {
            profile:            profile.clone(),
            files:              memory_summaries.clone(),
            total_loaded_bytes: memory.loaded_bytes(),
            budget_bytes:       ProjectMemory::BUDGET_BYTES,
        });

        let discovered = skills?;
        let resources = Arc::make_mut(&mut self.resources);
        resources.skills = discovered.skills;
        resources.skill_dirs = sources.skill_dirs;
        let mut skipped = sources.skipped_skills;
        skipped.extend(discovered.skipped);
        self.emit(CodingEvent::SkillsDiscovered {
            profile,
            source_dirs: self.resources.skill_dirs.clone(),
            skills: self.skill_summaries(),
            skipped,
        });
        // The one tool that cannot be built by the builder: what it loads is
        // discovered here, and a session that discovered no skills advertises
        // no way to load one.
        if !self.resources.skills.is_empty() {
            let vocabulary = self.resources.registry.vocabulary();
            let tool = make_use_skill_tool_for_vocabulary(
                Arc::from(self.resources.skills.clone()),
                vocabulary,
            );
            Arc::make_mut(&mut self.resources).registry.register(tool)?;
        }

        // Measured once: memory and skills never change again, and the context
        // snapshot built every round reads these numbers instead of
        // re-tokenizing the same text.
        Arc::make_mut(&mut self.resources).memory_tokens = memory_prompt_tokens(memory.documents());
        Arc::make_mut(&mut self.resources).skills_tokens =
            skills_prompt_tokens(&self.resources.skills, self.resources.registry.vocabulary());

        let env_context = self.build_env_context(&cancel).await?;
        debug!(
            is_git_repo = env_context.is_git_repo,
            model = env_context.model.as_str(),
            "Environment context built"
        );

        // Built once and fixed for the session's life. Only the loaded text
        // reaches the profile; the file metadata is already on the stream.
        let memory: Vec<String> = memory
            .into_documents()
            .into_iter()
            .map(|document| document.content)
            .collect();
        let system_prompt = self.compose_system_prompt(&env_context, &memory, &memory_summaries);
        Arc::make_mut(&mut self.resources).system_prompt = system_prompt;

        self.flush_events().await.map(|_| ())
    }

    /// The words the model reads first: the profile's prompt for this
    /// session, adjusted once by the application's transform when it has one.
    ///
    /// Called once the registry is complete — the builder froze it and the
    /// skill tool is the last addition — so a profile that gates a prompt
    /// section on a tool reads the session's real answer. The transform sees
    /// the prompt as written and the session as the prompt describes it. Tool
    /// summaries are the registered starting set; per-turn middleware can
    /// narrow what the model sees later.
    fn compose_system_prompt(
        &self,
        env_context: &EnvContext,
        memory: &[String],
        memory_summaries: &[MemoryFileSummary],
    ) -> String {
        let default_prompt = self.profile.build_system_prompt(
            &self.resources.registry,
            env_context,
            memory,
            self.config.user_instructions.as_deref(),
            &self.resources.skills,
        );
        let Some(transform) = &self.prompt_transform else {
            return default_prompt;
        };
        let tools: Vec<_> = self
            .registered_tools()
            .iter()
            .map(ToolDefinitionWithSource::to_tool_summary)
            .collect();
        let skills = self.skill_summaries();
        let context = SystemPromptContext::new(
            &default_prompt,
            env_context,
            &tools,
            memory_summaries,
            &skills,
        );
        transform.transform(context).apply(default_prompt)
    }

    /// Gathers what the system prompt says about where the session is working.
    ///
    /// The three git answers come from running git in the session's own
    /// environment, so a session working in a container describes that
    /// container's checkout rather than the machine pebble runs on. A probe
    /// that fails is simply left out: an environment without git is a working
    /// environment.
    async fn build_env_context(&self, cancel: &CancellationToken) -> Result<EnvContext> {
        stop_if_cancelled(cancel)?;
        let (current_date, git_branch) = tokio::join!(
            self.probe_date(cancel),
            self.probe(cancel, "git rev-parse --abbrev-ref HEAD"),
        );
        let is_git_repo = git_branch.is_some();

        stop_if_cancelled(cancel)?;
        let (git_status_short, git_recent_commits) = if is_git_repo {
            tokio::join!(
                self.probe(cancel, "git status --short"),
                self.probe(cancel, "git log --oneline -10"),
            )
        } else {
            (None, None)
        };

        stop_if_cancelled(cancel)?;
        Ok(EnvContext {
            is_git_repo,
            git_branch,
            git_status_short,
            git_recent_commits,
            current_date,
            model: self.model_context.model.clone(),
            knowledge_cutoff: self.knowledge_cutoff.clone(),
            ..EnvContext::from_environment(self.env.as_ref())
        })
    }

    /// Runs one short command in the environment, answering with its trimmed
    /// output when it succeeded and produced any.
    async fn probe(&self, cancel: &CancellationToken, command: &str) -> Option<String> {
        let outcome = self
            .env
            .exec(ExecRequest {
                timeout_ms: Some(PROBE_TIMEOUT_MS),
                cancel_token: Some(cancel.child_token()),
                ..ExecRequest::new(command)
            })
            .await
            .ok()?;
        if !outcome.result.is_success() {
            return None;
        }
        let output = outcome.result.stdout.trim();
        (!output.is_empty()).then(|| output.to_owned())
    }

    /// Today's date where the session is working.
    ///
    /// Asked of the environment, because a session working somewhere else
    /// should date its prompt from there. An environment that cannot answer —
    /// no shell, no `date` — falls back to the UTC date on this machine, which
    /// is never wrong by more than a day.
    async fn probe_date(&self, cancel: &CancellationToken) -> String {
        let reported = self.probe(cancel, "date +%Y-%m-%d").await;
        match reported {
            Some(date) if is_iso_date(&date) => date,
            _ => rfc3339_millis::format(SystemTime::now())
                .get(..10)
                .unwrap_or_default()
                .to_owned(),
        }
    }

    /// This session's identifier.
    pub(crate) fn id(&self) -> &str {
        self.session_scope.session_id().as_str()
    }

    /// The session and root identities that place this session in its tree.
    ///
    /// Read-only for the same reason [`CodingRuntimeBuilder`]'s `child_of` is
    /// crate-internal: a session's place in its tree is settled when it is
    /// built, by whoever spawned it. Root-scoped tools, the shared event
    /// stream, and stored records all key on this, so a session that could
    /// be re-rooted afterwards could be detached from the tree that owns
    /// it.
    pub(crate) const fn session(&self) -> &SessionScope {
        &self.session_scope
    }

    /// Which harness this session runs.
    pub(crate) fn profile_kind(&self) -> AgentProfileKind {
        self.profile.profile_kind()
    }

    /// Descriptions of the memory files loaded into the system prompt.
    pub(crate) fn memory_summaries(&self) -> &[MemoryFileSummary] {
        &self.memory_summaries
    }

    /// Descriptions of the skills available to this session.
    pub(crate) fn skill_summaries(&self) -> Vec<SkillSummary> {
        self.resources
            .skills
            .iter()
            .map(Skill::to_summary)
            .collect()
    }

    /// Descriptions of registered tools in stable name order.
    pub(crate) fn tool_summaries(&self) -> Vec<ToolSummary> {
        let mut tools: Vec<_> = self
            .registered_tools()
            .iter()
            .map(ToolDefinitionWithSource::to_tool_summary)
            .collect();
        tools.sort_by(|left, right| left.name.cmp(&right.name));
        tools
    }

    /// The provider the session resolved to.
    pub(crate) fn provider(&self) -> &str {
        &self.model_context.provider
    }

    /// The catalog identifier of the model the session resolved to.
    pub(crate) fn model(&self) -> &str {
        &self.model_context.model
    }

    /// What the model is told about its context window and output budget.
    #[cfg(test)]
    pub(crate) fn model_facts(&self) -> ModelFacts {
        self.model_context.facts
    }

    /// Where this session's tools act, for the crate's own tests.
    #[cfg(test)]
    pub(crate) const fn environment(&self) -> &Arc<dyn Environment> {
        &self.env
    }

    /// How this session was configured, for the crate's own tests.
    #[cfg(test)]
    pub(crate) const fn config(&self) -> &CodingAgentOptions {
        &self.config
    }

    /// The permission level the application recorded for this session.
    #[cfg(test)]
    pub(crate) fn permission_level(&self) -> Option<PermissionLevel> {
        self.config.permission_level
    }

    /// What the session is doing right now.
    pub(crate) fn state(&self) -> CodingAgentState {
        self.state.current()
    }

    /// The state machine itself, for a test that reads it while a prompt has
    /// the session borrowed.
    pub(crate) fn state_machine(&self) -> StateMachine {
        self.state.clone()
    }

    /// Shared cancellation for the session's current compaction.
    pub(crate) fn compaction_control(&self) -> CompactionControl {
        self.compaction.clone()
    }

    /// A snapshot of the conversation so far.
    pub(crate) fn history(&self) -> History {
        self.conversation().history.clone()
    }

    /// The latest context-window measurement, when a model has answered.
    pub(crate) fn context_window(&self) -> Option<ContextWindowSnapshot> {
        self.conversation().context_window.clone()
    }

    /// The most recent assistant message, without cloning the full history.
    pub(crate) fn final_assistant_message(&self) -> Option<Message> {
        self.conversation()
            .history
            .turns()
            .iter()
            .rev()
            .find(|message| matches!(message, Message::Assistant { .. }))
            .cloned()
    }

    /// A snapshot of the files this session has read and changed.
    #[cfg(test)]
    pub(crate) fn file_tracker(&self) -> FileTracker {
        self.conversation().file_tracker.clone()
    }

    /// Where the last prompt spent its time.
    pub(crate) fn last_prompt_timing(&self) -> PromptTiming {
        self.conversation().totals.timing
    }

    /// What the last prompt used, summed over every response, and what it
    /// cost where the catalog or the provider priced every one of them.
    pub(crate) fn last_prompt_usage(&self) -> Usage {
        self.conversation().totals.usage
    }

    /// The files the last prompt wrote or edited, this session's and its
    /// children's, in touch order, and the most recent of them.
    pub(crate) fn last_prompt_files(&self) -> (Vec<String>, Option<String>) {
        self.conversation().files.snapshot()
    }

    /// The compactions the last prompt completed, this session's own, in
    /// order. A manual compaction between prompts belongs to no prompt.
    pub(crate) fn last_prompt_compactions(&self) -> Vec<CompactionAccount> {
        self.conversation().totals.compactions.clone()
    }

    /// How many assistant turns and tool results the last prompt committed
    /// to the history on this runtime.
    pub(crate) fn last_prompt_committed_turns(&self) -> u64 {
        self.conversation().totals.committed_turns
    }

    /// Tells this runtime where the agent's fallback plan stands, or that
    /// there is none. With a plan, a prompt whose model fails without moving
    /// publishes [`RouteFailoverStopped`](CodingEvent::RouteFailoverStopped)
    /// before it ends.
    pub(crate) fn set_failover_outlook(&mut self, outlook: Option<FailoverOutlook>) {
        self.failover_outlook = outlook;
    }

    /// Publishes that `error` ends the prompt on this route although the
    /// agent's plan named fallback routes, when that is so. Nothing is said
    /// without a plan, for a failure that is not the model's, for one the
    /// agent will move on, or when the caller cancelled.
    fn report_failover_stopped(&self, error: &Error, cancel: Option<&CancellationToken>) {
        if cancel.is_some_and(CancellationToken::is_cancelled) {
            return;
        }
        let Some(outlook) = self.failover_outlook else {
            return;
        };
        let Some(reason) = outlook.stop_for(error) else {
            return;
        };
        self.emit(CodingEvent::RouteFailoverStopped {
            route: self.route(),
            attempt: outlook.attempt,
            reason,
            error: ErrorData::from(error),
        });
    }

    /// The broadcast channel this session's live subscribers are on, for a
    /// replacement that should keep them.
    pub(crate) fn published_sender(&self) -> Option<broadcast::Sender<CodingAgentEvent>> {
        self.emitter.published_sender()
    }

    /// The `provider/model` this session runs on.
    pub(crate) fn route(&self) -> String {
        format!(
            "{}/{}",
            self.model_context.provider, self.model_context.model
        )
    }

    /// The shared conversation, locked.
    fn conversation(&self) -> MutexGuard<'_, ConversationState> {
        self.conversation
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// The tools registered for this session before middleware filters them.
    pub(crate) fn registered_tools(&self) -> Vec<ToolDefinitionWithSource> {
        self.resources.registry.definitions_with_source()
    }

    /// Watches the session's events from here on.
    ///
    /// The stream is lossy for a subscriber that falls behind; an application
    /// that must see everything configures
    /// [`CodingRuntimeBuilder::event_sink`] instead.
    ///
    /// The stream ends with the session, not with the value: once
    /// [`CodingRuntime::shutdown`] has returned, the receiver reads out
    /// whatever it still holds and then observes `RecvError::Closed`, with
    /// the session still alive. So a reader that loops until the stream
    /// closes can be joined before the session is dropped, which is the
    /// order an application wants — the renderer's summary is the last
    /// thing printed.
    pub(crate) fn subscribe(&self) -> broadcast::Receiver<CodingAgentEvent> {
        self.emitter.subscribe()
    }

    /// A handle that steers and interrupts this session from elsewhere.
    pub(crate) fn control_handle(&self) -> AgentControlHandle {
        self.agent_control.clone()
    }

    /// Queues guidance for the next round.
    #[cfg(test)]
    pub(crate) fn steer(&self, text: impl Into<String>) {
        let _ = self.agent_control.enqueue_steering(text.into());
    }

    /// Queues typed input to process once the current input is finished, as
    /// the steering bus queues a person's follow-up: sourced `FollowUp`, so a
    /// `/name` reference in it expands.
    #[cfg(test)]
    pub(crate) fn follow_up(&self, message: impl Into<String>) {
        let input = input_message(InputContent::text(message.into()), InputSource::FollowUp);
        let _ = self.agent_control.follow_up(input);
    }

    /// Queues synthesized input carrying no source, as a subagent's result
    /// envelope or a continuation arrives: nothing in it expands.
    #[cfg(test)]
    pub(crate) fn follow_up_unattributed(&self, message: impl Into<String>) {
        let text: String = message.into();
        let _ = self
            .agent_control
            .follow_up(UserMessage::new(InputContent::text(text).into_parts()));
    }

    /// Ends the prompt.
    ///
    /// The loop unwinds through its own checkpoints — every tool call still
    /// gets its result recorded — and then closes the session. This is the
    /// terminal gesture; [`AgentControlHandle::interrupt`] is the one that
    /// only abandons a round.
    #[cfg(test)]
    pub(crate) fn interrupt(&self) {
        self.set_interrupt_reason(InterruptReason::Cancelled);
        self.cancel_token.cancel();
    }

    /// The token that ends this session's prompt.
    pub(crate) fn cancel_token(&self) -> CancellationToken {
        self.cancel_token.clone()
    }

    /// Where this session's supervisor publishes child-lifecycle facts.
    ///
    /// A child's own events already use the tree's shared pipeline. Lifecycle
    /// facts are the parent's news, so this callback publishes them under the
    /// parent's session identity on that same pipeline.
    pub(crate) fn sub_agent_event_callback(&self) -> SubagentEventCallback {
        let emitter = self.emitter.clone();
        let parent_session_id = self.session_scope.session_id().to_string();
        Arc::new(move |event| {
            emitter.emit(parent_session_id.clone(), event);
        })
    }

    /// This session's children, for the crate's own tests.
    #[cfg(test)]
    pub(crate) const fn subagent_supervisor(&self) -> Option<&SubagentSupervisor> {
        self.subagents.as_ref()
    }

    /// Whether this session can ask a person a question, for the crate's own
    /// tests: a child never can.
    #[cfg(test)]
    pub(crate) const fn has_human_input(&self) -> bool {
        self.human_input.is_some()
    }

    /// The reason slot an outside task can fill before cancelling.
    ///
    /// First writer wins, so a watchdog that names its own reason before
    /// cancelling gets that reason reported instead of a plain cancellation.
    #[must_use]
    pub(crate) fn interrupt_reason_handle(&self) -> InterruptReasonHandle {
        self.interrupt_reason.clone()
    }

    /// Changes how hard the model is asked to think, from the next round on.
    pub(crate) fn set_reasoning_effort(&mut self, effort: Option<ReasoningEffort>) {
        self.config.reasoning_effort = effort;
        if let Some(agent) = &mut self.coding_agent {
            agent.set_reasoning_effort(effort);
        }
    }

    /// Changes which latency or cost tier the session asks for, from the next
    /// round on.
    pub(crate) fn set_speed(&mut self, speed: Option<Speed>) {
        self.config.speed = speed;
        if let Some(agent) = &mut self.coding_agent {
            agent.set_speed(speed);
        }
    }

    /// Processes one input to completion, answering with the assistant's final
    /// text when it ended with any.
    ///
    /// Questions go to the provider the session was built with.
    ///
    /// # Errors
    ///
    /// Returns [`Error::SessionClosed`] for a session that has ended,
    /// [`Error::Interrupted`] when the prompt was cancelled or ran out of
    /// wall-clock time, [`Error::Llm`] when the model call failed for good, and
    /// [`Error::EventSink`] when the configured sink refused an event.
    pub(crate) async fn prompt(&mut self, input: impl Into<CodingInput>) -> Result<Option<String>> {
        self.prompt_with_cancellation(input, None).await
    }

    /// Compacts older history while the session is idle.
    pub(crate) async fn compact(
        &mut self,
        options: CompactionOptions,
        cancel: Option<&CancellationToken>,
    ) -> Result<CompactionOutcome> {
        match self.state.current() {
            CodingAgentState::Closed => return Err(Error::SessionClosed),
            CodingAgentState::Idle => {}
            state => {
                return Err(Error::InvalidState(format!(
                    "cannot compact while the session is {state:?}"
                )));
            }
        }

        let (mut history, file_tracker) = {
            let state = self.conversation();
            (state.history.clone(), state.file_tracker.clone())
        };
        let preserve_turns = options
            .preserve_turns_value()
            .unwrap_or(self.config.compaction_preserve_turns)
            .max(1);
        if history.compact_preserve_start(preserve_turns) == 0 {
            return Ok(CompactionOutcome::Unchanged);
        }

        let mut operation_guard = OperationGuard::new(self);
        let estimate = estimate_active_context_usage(&self.resources.system_prompt, &history);
        let compaction_cancel = self.cancel_token.child_token();
        if cancel.is_some_and(CancellationToken::is_cancelled) {
            compaction_cancel.cancel();
        }
        let caller_link = cancel.map(|caller| link_cancellation(caller, &compaction_cancel));
        let event_failure = self.emitter.failure_token();
        let event_link = link_cancellation(&event_failure, &compaction_cancel);
        let operation = self.compaction.begin(&compaction_cancel);
        self.state.transition(CodingAgentState::Compacting);
        let request = CompactionRequest {
            policy: self.compaction_policy.as_deref(),
            model: &self.model_context.model_selector,
            facts: self.model_context.facts,
            preserve_turns,
            estimate,
            reason: CompactionReason::Manual,
            instructions: options.instructions_ref(),
            cancel: operation.token(),
        };
        let result = compact_context(
            &mut history,
            &self.model_context.client,
            &file_tracker,
            request,
            &self.emitter,
            self.session_scope.session_id().as_str(),
        )
        .await;
        drop(operation);
        event_link.stop().await;
        if let Some(link) = caller_link {
            link.stop().await;
        }

        if matches!(result, Ok(CompactionOutcome::Compacted(_))) {
            self.conversation().replace_history(history);
        }
        if self.state.current() == CodingAgentState::Closed {
            self.shutdown(ShutdownReason::Cancelled).await?;
        } else {
            self.state.transition(CodingAgentState::Idle);
            if let Err(error) = self.flush_events().await {
                let _ = self.shutdown(ShutdownReason::Error).await;
                return Err(error);
            }
        }
        operation_guard.finished = true;
        result
    }

    /// Processes one input until it completes or `cancel` fires.
    ///
    /// Cancelling `cancel` ends this prompt alone: the loop unwinds through its
    /// checkpoints so every tool call still gets its result, the prompt reports
    /// [`Error::Interrupted`], and the session returns to
    /// [`Idle`](CodingAgentState::Idle) ready for its next prompt. A call that
    /// is running is cancelled through its token and keeps the result it
    /// returns; a call the model asked for that has not started yet — the
    /// cancellation landed while the assistant turn was being committed or
    /// compacted — is answered `Cancelled` without running, so history stays
    /// paired. Only close or shutdown closes the session.
    ///
    /// # Errors
    ///
    /// As [`prompt`](Self::prompt).
    #[tracing::instrument(
        name = "coding_session_prompt",
        skip_all,
        fields(
            session_id = %self.session_scope.session_id(),
            provider = %self.model_context.provider,
            model = %self.model_context.model
        )
    )]
    pub(crate) async fn prompt_with_cancellation(
        &mut self,
        input: impl Into<CodingInput>,
        cancel: Option<&CancellationToken>,
    ) -> Result<Option<String>> {
        self.run_prompt(PromptStart::Input(input.into()), cancel)
            .await
    }

    /// Continues the prompt the history left unfinished, without new input.
    ///
    /// The history is unfinished when it ends with input the model has not
    /// answered: a user turn, or the results of the tool calls the model asked
    /// for. The model is asked again on the history as it stands and the
    /// prompt then runs to completion as
    /// [`prompt_with_cancellation`](Self::prompt_with_cancellation) would;
    /// the tool calls that already have their results are not run again.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Agent`] carrying
    /// [`AgentError::NothingToContinue`](pebble_agent::AgentError::NothingToContinue)
    /// when the history is empty or ends with the model's own turn, and
    /// otherwise fails as `prompt_with_cancellation` does.
    pub(crate) async fn continue_prompt(
        &mut self,
        cancel: Option<&CancellationToken>,
    ) -> Result<Option<String>> {
        self.run_prompt(PromptStart::Continue, cancel).await
    }

    /// Runs one prompt, from new input or from where the history stopped, with
    /// the bookkeeping every prompt shares: the wall-clock timer, the
    /// cancellation links, the state transitions, and the durability barrier.
    async fn run_prompt(
        &mut self,
        start: PromptStart,
        cancel: Option<&CancellationToken>,
    ) -> Result<Option<String>> {
        self.conversation().begin_prompt();
        if self.state.current() == CodingAgentState::Closed {
            return Err(Error::SessionClosed);
        }

        let mut operation_guard = OperationGuard::new(self);
        // A child of the terminal token, so a shutdown ends the prompt too.
        // Caller cancellation and a failed durable stream are joined in by
        // tasks because a cancellation token has only one parent.
        let prompt_cancel = self.cancel_token.child_token();
        let caller_link = cancel.map(|caller| link_cancellation(caller, &prompt_cancel));
        let event_failure = self.emitter.failure_token();
        let event_link = link_cancellation(&event_failure, &prompt_cancel);

        let timer = self.start_wall_clock_timer(&prompt_cancel);
        let result = match start {
            PromptStart::Input(input) => {
                // The input's source decides: a `/name` reference a person
                // typed expands, input the caller synthesized passes through
                // unchanged, as `SkillExpansion`'s contract says. Synthesized
                // input carries paths and result envelopes with slash-prefixed
                // words of their own; expanding it would fail on an unknown
                // skill or splice a template over it.
                let expansion = if is_prompt_source(&input.source()) {
                    SkillExpansion::Apply
                } else {
                    SkillExpansion::Skip
                };
                self.process_input(input, expansion, &prompt_cancel).await
            }
            PromptStart::Continue => self.continue_input(&prompt_cancel).await,
        };
        self.conversation().finish_prompt_timing();
        // Said here, while the stream is open and before a session that
        // closes itself on the failure publishes its end.
        if let Err(error) = &result {
            self.report_failover_stopped(error, cancel);
        }

        if let Some(link) = caller_link {
            link.stop().await;
        }
        event_link.stop().await;
        let mut task_failure = stop_wall_clock_timer(timer).await;
        // The reason has been reported by now. Clearing it here rather than at
        // the start of the next prompt keeps a reason a watchdog records just
        // before it cancels, and still stops one prompt's reason reaching the
        // next.
        if self.state.current() != CodingAgentState::Closed {
            self.interrupt_reason.clear();
        }

        if self.state.current() == CodingAgentState::Closed {
            let reason = if self.cancel_token.is_cancelled() {
                ShutdownReason::Cancelled
            } else {
                ShutdownReason::Error
            };
            if let Err(error) = self.shutdown(reason).await {
                remember_failure(&mut task_failure, error);
            }
        } else {
            self.state.transition(CodingAgentState::Idle);
            // `ProcessingEnd` is the prompt's durability barrier. The prompt
            // cannot finish before it and every earlier event reach the sink.
            if let Err(error) = self.flush_events().await {
                remember_failure(&mut task_failure, error);
            }
        }

        // A failure found by the final barrier moved the session to `Closed`
        // after the branch above began. Finish the shutdown now so children and
        // the pump do not outlive the prompt that reports it.
        if self.state.current() == CodingAgentState::Closed
            && !self.ended
            && let Err(error) = self.shutdown(ShutdownReason::Error).await
        {
            remember_failure(&mut task_failure, error);
        }

        operation_guard.finished = true;
        // The durable stream is the record of every other outcome. If it is
        // incomplete, that is the failure the caller must act on even when the
        // model or a cleanup task also failed.
        if task_failure
            .as_ref()
            .is_some_and(|error| error.kind() == ErrorKind::EventStream)
        {
            return Err(task_failure.expect("the failure was present"));
        }

        match (result, task_failure) {
            // The prompt's own failure is the story; a task that also failed on
            // the way out is reported rather than returned.
            (Err(error), Some(task)) => {
                warn!(error = ?task, "A session task failed while the prompt was already failing");
                Err(error)
            }
            (Err(error), None) => Err(error),
            (Ok(_), Some(task)) => Err(task),
            (Ok(output), None) => Ok(output),
        }
    }

    /// Starts the task that ends a prompt which has taken too long.
    ///
    /// The timer cancels the prompt rather than dropping it, so the loop
    /// unwinds through its own checkpoints and every tool call still has its
    /// result recorded: a running call is cancelled and keeps its own result,
    /// and a call not yet started is answered `Cancelled` without running. The
    /// session stays open: running out of time is the prompt's failure, and
    /// the next prompt gets a fresh budget.
    fn start_wall_clock_timer(&self, prompt_cancel: &CancellationToken) -> Option<WallClockTimer> {
        let duration = self.config.wall_clock_timeout?;
        let stop = CancellationToken::new();
        let cancel = prompt_cancel.clone();
        let reason = self.interrupt_reason_handle();
        let watched = stop.clone();
        let task = tokio::spawn(async move {
            tokio::select! {
                () = watched.cancelled() => {}
                () = sleep(duration) => {
                    reason.record(InterruptReason::WallClockTimeout);
                    cancel.cancel();
                }
            }
        });
        Some(WallClockTimer { stop, task })
    }

    /// Closes the session, joining everything it owns.
    ///
    /// Only the first call does anything, which is what lets the loop close a
    /// cancelled session at whichever checkpoint reaches it first. Children are
    /// closed before this session publishes its own end, so a reader sees a
    /// tree unwind from the leaves.
    ///
    /// Every stream [`CodingRuntime::subscribe`] handed out ends by the time
    /// this returns: joining the pump is what closes them, so a reader
    /// looping until `RecvError::Closed` finishes without the session
    /// having to be dropped first.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EventSink`] when the configured sink had refused an
    /// event, and [`Error::Task`] when a task the session owned failed
    /// outright. The session is closed either way.
    #[tracing::instrument(
        name = "coding_session_shutdown",
        skip_all,
        fields(session_id = %self.session_scope.session_id(), reason = ?reason)
    )]
    pub(crate) async fn shutdown(&mut self, reason: ShutdownReason) -> Result<bool> {
        if self.ended {
            return Ok(false);
        }
        if reason == ShutdownReason::Cancelled {
            self.set_interrupt_reason(InterruptReason::Cancelled);
        }
        self.state.transition(CodingAgentState::Closed);
        self.cancel_token.cancel();
        self.agent_control.close();
        if let Some(supervisor) = &self.subagents {
            supervisor.shutdown_all().await;
        }
        if let Some(agent) = &mut self.coding_agent {
            let _ = agent.shutdown();
        }
        if !self.end_emitted {
            self.emit(CodingEvent::SessionEnded);
            self.end_emitted = true;
        }
        let flushed = self.flush_events().await;
        let joined = self.join_pump().await;
        self.ended = true;
        flushed?;
        joined?;
        Ok(true)
    }

    /// Waits until every event currently queued has reached the durable sink.
    pub(crate) async fn flush_events(&mut self) -> Result<u64> {
        match self.emitter.flush().await {
            Ok(seq) => Ok(seq),
            Err(failure) => {
                self.state.transition(CodingAgentState::Closed);
                Err(failure.into_runtime_error())
            }
        }
    }

    /// The highest event sequence committed by the event pipeline.
    ///
    /// With a durable sink, commitment means the sink accepted the event.
    pub(crate) fn committed_event_seq(&self) -> u64 {
        self.emitter.committed_seq()
    }

    /// Publishes everything queued, then joins the pump.
    async fn join_pump(&mut self) -> Result<()> {
        let Some(pump) = self.pump.as_mut() else {
            return Ok(());
        };
        let _ = self.emitter.close().await;
        let outcome = pump.await;
        self.pump = None;
        match outcome {
            Ok(result) => result,
            Err(source) => Err(Error::Task {
                task: TaskKind::EventPump,
                source,
            }),
        }
    }

    /// Reports a pump that has already stopped, which only a refusing sink
    /// does while a prompt is in progress.
    ///
    /// Checked at round boundaries so a prompt ends promptly once its events
    /// stop being recorded, rather than working on against a stream nobody
    /// has.
    ///
    /// A sink failure closes the session as well as ending the prompt. The
    /// pipeline stops for good when the pump does — nothing is recorded, and no
    /// subscriber is served — so a session that kept answering would be working
    /// where nobody could see it. The application gets the failure from the
    /// prompt that found it, and [`Error::SessionClosed`] from every call
    /// after.
    async fn check_pump(&mut self) -> Result<()> {
        if let Some(failure) = self.emitter.failure() {
            self.state.transition(CodingAgentState::Closed);
            return Err(failure.into_runtime_error());
        }
        let Some(pump) = self.pump.as_ref() else {
            return Ok(());
        };
        if !pump.is_finished() {
            return Ok(());
        }
        let outcome = self.join_pump().await;
        if outcome.is_err() {
            self.state.transition(CodingAgentState::Closed);
        }
        outcome
    }

    /// Answers for a prompt the agent loop aborted.
    ///
    /// A terminal cancellation closes the session; anything else — the
    /// caller's prompt token, or the wall clock — ends only this prompt, and
    /// the session is left open for the next one.
    pub(super) async fn prompt_aborted(&mut self) -> Error {
        if self.cancel_token.is_cancelled() {
            self.close_cancelled().await
        } else {
            self.interrupted_error()
        }
    }

    /// Closes a cancelled session and answers with the error the prompt ends
    /// on.
    async fn close_cancelled(&mut self) -> Error {
        let interrupted = self.interrupted_error();
        match self.shutdown(ShutdownReason::Cancelled).await {
            Ok(_) => interrupted,
            Err(error) => error,
        }
    }

    fn set_interrupt_reason(&self, reason: InterruptReason) {
        self.interrupt_reason.record(reason);
    }

    /// The error an interrupted prompt ends with, naming whatever reason was
    /// recorded first.
    fn interrupted_error(&self) -> Error {
        Error::Interrupted(
            self.interrupt_reason
                .current()
                .unwrap_or(InterruptReason::Cancelled),
        )
    }

    /// Publishes one event on this session's stream.
    pub(crate) fn emit(&self, event: CodingEvent) {
        self.emitter
            .emit(self.session_scope.session_id().to_string(), event);
    }

    /// Publishes a model failure, closing the session when the credential is
    /// the problem.
    ///
    /// An authentication failure will not fix itself between rounds, so the
    /// session stops rather than spending the rest of the prompt failing the
    /// same way.
    fn emit_llm_error(&mut self, error: LlmError) -> Error {
        let credential_failure = is_auth_error(&error);
        // Projected from the error the caller receives, so what a reader sees
        // and what the prompt returns say the same thing.
        let error = Error::Llm(error);
        self.emit(CodingEvent::Error {
            error: ErrorData::from(&error),
        });
        if credential_failure {
            self.state.transition(CodingAgentState::Closed);
        }
        error
    }
}

/// The session's state, and the one place it moves.
///
/// Shared between the runtime, which moves it at the edges of a prompt, and
/// the bridge, which moves it into `Executing` around each tool round while
/// the runtime is borrowed by the prompt in progress. Cheap to clone; every
/// clone reads and moves the same state.
#[derive(Clone, Debug)]
pub(crate) struct StateMachine {
    state:      Arc<Mutex<CodingAgentState>>,
    emitter:    Emitter,
    session_id: SessionId,
}

impl StateMachine {
    /// A session that starts idle.
    fn new(emitter: Emitter, session_id: SessionId) -> Self {
        Self {
            state: Arc::new(Mutex::new(CodingAgentState::Idle)),
            emitter,
            session_id,
        }
    }

    /// What the session is doing right now.
    pub(crate) fn current(&self) -> CodingAgentState {
        *self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Moves the session's state machine, publishing the end of a processing
    /// cycle where one ends.
    ///
    /// Valid moves: Idle or Executing to Thinking, Thinking to Executing or
    /// Idle, Idle to Compacting, Compacting to Idle, and anything to Closed.
    /// Ending the session belongs to
    /// [`CodingRuntime::shutdown`], never here.
    pub(super) fn transition(&self, to: CodingAgentState) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let from = *state;
        if from == to {
            return;
        }

        debug_assert!(
            matches!(
                (from, to),
                (
                    CodingAgentState::Idle | CodingAgentState::Executing,
                    CodingAgentState::Thinking
                ) | (
                    CodingAgentState::Thinking,
                    CodingAgentState::Executing | CodingAgentState::Idle
                ) | (CodingAgentState::Idle, CodingAgentState::Compacting)
                    | (CodingAgentState::Compacting, CodingAgentState::Idle)
                    | (_, CodingAgentState::Closed)
            ),
            "invalid session state transition: {from:?} -> {to:?}"
        );

        *state = to;
        drop(state);
        if matches!(
            from,
            CodingAgentState::Thinking | CodingAgentState::Executing | CodingAgentState::Compacting
        ) && to == CodingAgentState::Idle
        {
            self.emitter
                .emit(self.session_id.to_string(), CodingEvent::ProcessingEnd);
        }
    }
}

/// Closes a session when an operation is abandoned before its cleanup boundary.
struct OperationGuard {
    state:     StateMachine,
    control:   AgentControlHandle,
    cancel:    CancellationToken,
    subagents: Option<SubagentSupervisor>,
    finished:  bool,
}

impl OperationGuard {
    fn new(runtime: &CodingRuntime) -> Self {
        Self {
            state:     runtime.state.clone(),
            control:   runtime.agent_control.clone(),
            cancel:    runtime.cancel_token.clone(),
            subagents: runtime.subagents.clone(),
            finished:  false,
        }
    }
}

impl Drop for OperationGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.cancel.cancel();
            if let Some(supervisor) = &self.subagents {
                supervisor.cancel_all();
            }
            self.control.close();
            self.state.transition(CodingAgentState::Closed);
        }
    }
}

/// Owns a disposable cancellation forwarding task, including on future drop.
struct CancellationLink(JoinHandle<()>);

impl CancellationLink {
    async fn stop(mut self) {
        self.0.abort();
        let _ = (&mut self.0).await;
    }
}

impl Drop for CancellationLink {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Cancels `prompt_cancel` when `caller` fires.
///
/// A token has one parent, and the prompt token's is the terminal token, so
/// the caller's is joined in by a task instead. The task ends on its own once
/// the prompt token fires for any reason, and the prompt aborts it when it
/// finishes.
fn link_cancellation(
    caller: &CancellationToken,
    prompt_cancel: &CancellationToken,
) -> CancellationLink {
    let caller = caller.clone();
    let prompt_cancel = prompt_cancel.clone();
    CancellationLink(tokio::spawn(async move {
        tokio::select! {
            () = caller.cancelled() => prompt_cancel.cancel(),
            () = prompt_cancel.cancelled() => {}
        }
    }))
}

/// Keeps the event-stream failure when cleanup finds more than one failure.
fn remember_failure(stored: &mut Option<Error>, failure: Error) {
    match stored.as_ref().map(Error::kind) {
        None => *stored = Some(failure),
        Some(ErrorKind::EventStream) if failure.kind() == ErrorKind::EventStream => {}
        Some(_) if failure.kind() == ErrorKind::EventStream => *stored = Some(failure),
        Some(_) => {
            warn!(error = ?failure, "A session task also failed while the prompt was ending");
        }
    }
}

/// The task watching one prompt's wall-clock budget.
struct WallClockTimer {
    stop: CancellationToken,
    task: JoinHandle<()>,
}

impl Drop for WallClockTimer {
    fn drop(&mut self) {
        self.stop.cancel();
        self.task.abort();
    }
}

/// Stops the timer and joins it, reporting a task that failed outright.
async fn stop_wall_clock_timer(timer: Option<WallClockTimer>) -> Option<Error> {
    let mut timer = timer?;
    timer.stop.cancel();
    (&mut timer.task).await.err().map(|source| Error::Task {
        task: TaskKind::WallClockTimer,
        source,
    })
}

/// Stops a step that a cancelled session should not take.
fn stop_if_cancelled(cancel: &CancellationToken) -> Result<()> {
    if cancel.is_cancelled() {
        return Err(Error::Interrupted(InterruptReason::Cancelled));
    }
    Ok(())
}

/// Whether a model failure means the credential, rather than the call.
fn is_auth_error(error: &LlmError) -> bool {
    matches!(
        error.kind(),
        LlmErrorKind::Authentication | LlmErrorKind::AccessDenied
    )
}

/// Whether a probe's answer looks like the `YYYY-MM-DD` it was asked for.
fn is_iso_date(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() == 10
        && bytes.iter().enumerate().all(|(index, byte)| match index {
            4 | 7 => *byte == b'-',
            _ => byte.is_ascii_digit(),
        })
}
