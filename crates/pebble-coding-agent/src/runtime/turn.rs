//! The coding layer that projects Pebble's durable behavior onto `Agent`.

use std::result::Result as StdResult;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Instant, SystemTime};

use async_trait::async_trait;
use lithos_llm::middleware::CallContext;
use lithos_llm::types::{
    ContentPart, Error as LlmError, FinishReason, Message as LlmMessage, Request, Response,
    ResponseStream, ToolCall, ToolResult,
};
use pebble_agent as agent;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::control::{actor_from_attribution, input_message, input_source_from_attribution};
use super::retry::RetryEventBridge;
use super::{CodingRuntime, PromptResources, PromptTotals, SessionModel, StateMachine};
use crate::coding_agent::CodingInput;
use crate::compaction::{
    CompactionControl, CompactionOutcome, CompactionReason, CompactionRequest, CompactionResult,
    check_context_usage, compact_context,
};
use crate::config::CodingAgentOptions;
use crate::context_window::{
    ContextWindowInput, build_local_snapshot, context_window_from_response_usage,
};
use crate::error::{Error, ErrorData, InterruptReason, Result};
use crate::event::Emitter;
use crate::file_tracker::{FileTracker, PromptFiles};
use crate::history::History;
use crate::loop_detection::detect_loop;
use crate::policy::{CompactionPolicy, ContextPolicy, ContextPreparation};
use crate::profile::ModelFacts;
use crate::skills::{ExpandedInput, SkillExpansion, expand_skill};
use crate::subagent::SubagentSupervisor;
use crate::task_reminder::maybe_task_reminder;
use crate::tool::{CodingToolService, NativeTool, canonical_tool_name};
use crate::types::{
    CodingAgentState, CodingEvent, ContextWindowSnapshot, ContextWindowStaleness, InputContent,
    InputSource, LlmOutputKind, LlmRetryPhase, Message, SkillActivationSource, Usage,
};
use crate::{SessionId, SessionScope};

/// How many failed response streams Pebble replays after the first attempt.
const STREAM_CONSUME_RETRIES: u32 = 3;

/// What the model is told when it keeps making the same calls.
const LOOP_WARNING: &str = "WARNING: Loop detected. You appear to be repeating the same tool \
                            calls. Please try a different approach or ask for clarification.";

/// The turn that asks the model to finish an answer its output limit cut short.
///
/// Model-visible API: the model reads it as a user turn, so the wording is
/// frozen.
const OUTPUT_LIMIT_CONTINUATION: &str = "Your previous answer stopped at the output limit. \
                                         Continue from exactly where it stopped, without \
                                         repeating what you already wrote.";

/// How many times one prompt asks the model to continue a cut-short answer.
///
/// The limit is the model's own maximum unless the application lowered it, so
/// a second request would spend the whole budget to reach the same place.
const MAX_OUTPUT_LIMIT_CONTINUATIONS: u32 = 1;

/// The `kind` of the warning that reports an answer cut at the output limit.
const OUTPUT_LIMIT_WARNING: &str = "output_limit";

#[derive(Clone)]
pub(super) struct CodingAgentBridge {
    model_context:     Arc<SessionModel>,
    resources:         Arc<PromptResources>,
    state:             Arc<Mutex<ConversationState>>,
    config:            Arc<CodingAgentOptions>,
    tools:             Arc<CodingToolService>,
    emitter:           Emitter,
    session_id:        SessionId,
    /// The session's state, moved to `Executing` for the length of a tool
    /// round and back to `Thinking` after it.
    state_machine:     StateMachine,
    /// The token that ends the prompt in progress. A child of the runtime's
    /// terminal token, so it also fires when the session is shut down, and set
    /// afresh by [`begin_prompt`](Self::begin_prompt) for every prompt.
    prompt_cancel:     Arc<Mutex<CancellationToken>>,
    compaction:        CompactionControl,
    session_scope:     SessionScope,
    context_policy:    Option<Arc<dyn ContextPolicy>>,
    compaction_policy: Option<Arc<dyn CompactionPolicy>>,
    subagents:         Option<SubagentSupervisor>,
}

/// The conversation and what one prompt accumulates around it.
///
/// The authoritative coding history is shared by the runtime and the bridge.
/// The generic loop holds a derived model-facing view, updated from the first
/// changed turn and rebuilt in full only when history is replaced.
///
/// One copy of coding history, shared by the runtime and the bridge: the
/// runtime reads it for records, exports, and the public history, and the
/// bridge writes it as the loop commits turns. The first four members outlive a
/// prompt; the rest are reset by [`ConversationState::begin_prompt`].
pub(super) struct ConversationState {
    pub(super) history: History,
    /// Earliest canonical turn whose model-facing projection needs replacement.
    dirty_from: Option<usize>,
    /// Whether the last applied projection includes a transient task reminder.
    projected_reminder: bool,
    pub(super) file_tracker: FileTracker,
    /// What this prompt wrote or edited, the children's work included.
    pub(super) files: PromptFiles,
    pub(super) totals: PromptTotals,
    pub(super) activated_skill_context_observed: bool,
    pub(super) context_window: Option<ContextWindowSnapshot>,
    compaction_failed: bool,
    pending_task_reminder: Option<Message>,
    local_context_window: Option<ContextWindowSnapshot>,
    inference_start: Option<Instant>,
    tool_start: Option<Instant>,
    boundary_error: Option<Error>,
    /// How many times this prompt has asked the model to continue an answer
    /// that stopped at the output limit.
    output_limit_continuations: u32,
}

impl ConversationState {
    /// A conversation that starts from `history`, with nothing accumulated.
    /// `files` is the collector this session's children report into.
    pub(super) fn new(history: History, files: PromptFiles) -> Self {
        Self {
            history,
            dirty_from: None,
            projected_reminder: false,
            file_tracker: FileTracker::default(),
            files,
            totals: PromptTotals::default(),
            activated_skill_context_observed: false,
            context_window: None,
            compaction_failed: false,
            pending_task_reminder: None,
            local_context_window: None,
            inference_start: None,
            tool_start: None,
            boundary_error: None,
            output_limit_continuations: 0,
        }
    }

    /// Resets prompt-local bookkeeping while retaining history and its
    /// projection.
    pub(super) fn begin_prompt(&mut self) {
        self.totals = PromptTotals::default();
        self.files.reset();
        self.compaction_failed = false;
        self.pending_task_reminder = None;
        self.local_context_window = None;
        self.inference_start = None;
        self.tool_start = None;
        self.boundary_error = None;
        self.output_limit_continuations = 0;
    }

    fn begin_inference(&mut self, local: ContextWindowSnapshot) {
        self.local_context_window = Some(local);
        self.inference_start = Some(Instant::now());
    }

    fn finish_inference(&mut self) {
        if let Some(started) = self.inference_start.take() {
            self.totals.timing.inference = self
                .totals
                .timing
                .inference
                .saturating_add(started.elapsed());
        }
    }

    fn begin_tools(&mut self) {
        self.tool_start.get_or_insert_with(Instant::now);
    }

    pub(super) fn finish_prompt_timing(&mut self) {
        self.finish_inference();
        self.finish_tool_timing();
    }

    fn finish_tool_timing(&mut self) {
        if let Some(started) = self.tool_start.take() {
            self.totals.timing.tool = self.totals.timing.tool.saturating_add(started.elapsed());
        }
    }

    fn finish_tools(&mut self, calls: &[ToolCall], results: &[ToolResult]) {
        self.finish_tool_timing();
        if activated_a_skill(calls, results) {
            self.activated_skill_context_observed = true;
        }
        let written = self.file_tracker.record_from_tool_calls(calls, results);
        self.files.record(written);
        self.totals.committed_turns += 1;
        self.push(Message::ToolResults {
            results:   results.to_vec(),
            timestamp: SystemTime::now(),
        });
    }

    fn interrupt_round(&mut self) {
        self.finish_inference();
        self.pending_task_reminder = None;
    }

    fn continue_after_output_limit(&mut self) -> bool {
        if self.output_limit_continuations >= MAX_OUTPUT_LIMIT_CONTINUATIONS {
            return false;
        }
        self.output_limit_continuations += 1;
        true
    }

    fn record_response_usage(&mut self, usage: Usage) -> Option<ContextWindowSnapshot> {
        let context_window = self
            .local_context_window
            .take()
            .map(|local| context_window_from_response_usage(&local, usage.tokens));
        if let Some(context_window) = &context_window {
            let mut stored = context_window.clone();
            stored.staleness = ContextWindowStaleness::Stored;
            self.context_window = Some(stored);
        }
        self.accumulate_usage(usage);
        context_window
    }

    fn accumulate_usage(&mut self, usage: Usage) {
        self.totals.usage = self.totals.usage.saturating_add(usage);
    }

    /// Bills a completed compaction's summary call to this prompt and lists
    /// the compaction on its report.
    fn record_compaction(&mut self, result: &CompactionResult) {
        self.accumulate_usage(result.usage());
        self.totals.compactions.push(result.account());
    }

    fn push_assistant(&mut self, message: Message) {
        if let Some(reminder) = self.pending_task_reminder.take() {
            self.push(reminder);
        }
        self.totals.committed_turns += 1;
        self.push(message);
    }

    fn stage_task_reminder(&mut self, tools: &agent::ToolCatalog) {
        let names = tools
            .visible_tools()
            .map(|tool| tool.definition().name.as_str())
            .collect::<Vec<_>>();
        self.pending_task_reminder =
            maybe_task_reminder(&self.history, &names).map(|content| Message::System {
                content,
                timestamp: SystemTime::now(),
            });
    }

    fn conversation_update(&self, include_reminder: bool) -> agent::ConversationUpdate {
        let has_reminder = include_reminder && self.pending_task_reminder.is_some();
        let mut start = self.dirty_from;
        if self.projected_reminder || has_reminder {
            let end = self.history.len();
            start = Some(start.map_or(end, |dirty| dirty.min(end)));
        }
        let Some(start) = start else {
            return agent::ConversationUpdate::unchanged();
        };
        let mut messages = self.history.turns()[start..]
            .iter()
            .map(Message::to_llm_message)
            .collect::<Vec<_>>();
        if has_reminder && let Some(reminder) = &self.pending_task_reminder {
            messages.push(reminder.to_llm_message());
        }
        // Clear dirtiness only when the generic loop acknowledges applying
        // this update. Cancellation may discard a prepared lifecycle result.
        agent::ConversationUpdate::replace_tail(start, messages)
    }

    fn push(&mut self, message: Message) {
        let start = self.history.len();
        self.dirty_from = Some(self.dirty_from.map_or(start, |dirty| dirty.min(start)));
        self.history.push(message);
    }

    pub(super) fn replace_history(&mut self, history: History) {
        self.history = history;
        self.dirty_from = Some(0);
    }

    fn acknowledge_projection(&mut self, messages: &[LlmMessage]) {
        self.projected(messages.len() > self.history.len());
    }

    fn projected(&mut self, has_reminder: bool) {
        self.dirty_from = None;
        self.projected_reminder = has_reminder;
    }
}

impl CodingAgentBridge {
    fn from_runtime(runtime: &CodingRuntime) -> Self {
        let registry = Arc::new(runtime.resources.registry.clone());
        let config = Arc::new(runtime.config.clone());
        let mut tools = CodingToolService::new(
            Arc::clone(&registry),
            Arc::clone(&runtime.env),
            Arc::clone(&config),
            runtime.emitter.clone(),
            runtime.session_scope.clone(),
            Arc::clone(&runtime.redactor),
        );
        if let Some(provider) = runtime.tool_env_provider.as_ref() {
            tools = tools.with_tool_env_provider(Arc::clone(provider));
        }
        if let Some(provider) = runtime.human_input.as_ref() {
            tools = tools.with_human_input(Arc::clone(provider));
        }
        Self {
            model_context: runtime.model_context.clone(),
            resources: runtime.resources.clone(),
            state: Arc::clone(&runtime.conversation),
            config,
            tools: Arc::new(tools),
            emitter: runtime.emitter.clone(),
            session_id: runtime.session_scope.session_id().clone(),
            state_machine: runtime.state.clone(),
            prompt_cancel: Arc::new(Mutex::new(runtime.cancel_token.clone())),
            compaction: runtime.compaction.clone(),
            session_scope: runtime.session_scope.clone(),
            context_policy: runtime.context_policy.clone(),
            compaction_policy: runtime.compaction_policy.clone(),
            subagents: runtime.subagents.clone(),
        }
    }

    fn begin_prompt(&self, prompt_cancel: CancellationToken) {
        *self
            .prompt_cancel
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = prompt_cancel;
    }

    /// Asks the model to finish an answer its output limit cut short.
    ///
    /// The answer as far as it got is already committed, so the request is one
    /// more user turn on top of it. Every cut is reported as a warning; only
    /// the first [`MAX_OUTPUT_LIMIT_CONTINUATIONS`] in a prompt earn a
    /// continuation, and after that the answer stands as it is.
    fn continue_after_output_limit(&self) -> Option<agent::UserMessage> {
        let continued = self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .continue_after_output_limit();
        self.emit(CodingEvent::Warning {
            kind:    OUTPUT_LIMIT_WARNING.to_owned(),
            message: if continued {
                "The answer stopped at the model's output limit; asking it to continue".to_owned()
            } else {
                "The answer stopped at the model's output limit again; leaving it as it stands"
                    .to_owned()
            },
            details: serde_json::json!({
                "max_output_tokens": max_output_tokens(&self.config, &self.model_context.facts),
                "continued": continued,
            }),
        });
        continued.then(|| agent::UserMessage::text(OUTPUT_LIMIT_CONTINUATION))
    }

    /// The token that ends the prompt in progress.
    fn prompt_cancel(&self) -> CancellationToken {
        self.prompt_cancel
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn take_boundary_error(&self) -> Option<Error> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .boundary_error
            .take()
    }

    /// Keeps a typed coding-layer failure while the generic loop receives its
    /// boundary representation.
    fn record_boundary_error(&self, error: Error) -> agent::LifecycleError {
        let message = ErrorData::from(&error).message;
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .boundary_error = Some(error);
        agent::LifecycleError::new(message)
    }

    /// Waits for the tree stream at a stable turn boundary.
    async fn flush_events(&self) -> StdResult<(), agent::LifecycleError> {
        self.emitter
            .flush()
            .await
            .map(|_| ())
            .map_err(|failure| self.record_boundary_error(failure.into_runtime_error()))
    }

    fn emit(&self, event: CodingEvent) {
        self.emitter.emit(self.session_id.to_string(), event);
    }

    pub(super) fn finish_inference(&self) {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .finish_inference();
    }

    async fn compact_if_needed(&self) -> bool {
        let estimate = {
            let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            check_context_usage(
                &self.resources.system_prompt,
                &state.history,
                self.model_context.facts.context_window_tokens,
                self.config.compaction_threshold_percent,
                &self.emitter,
                self.session_id.as_str(),
            )
        };
        let Some(estimate) = estimate else {
            return false;
        };
        if !self.config.enable_context_compaction {
            return false;
        }

        let (mut history, file_tracker) = {
            let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
            (state.history.clone(), state.file_tracker.clone())
        };
        let prompt_cancel = self.prompt_cancel();
        let operation = self.compaction.begin(&prompt_cancel);
        let request = CompactionRequest {
            policy: self.compaction_policy.as_deref(),
            model: &self.model_context.model_selector,
            facts: self.model_context.facts,
            preserve_turns: self.config.compaction_preserve_turns,
            estimate,
            reason: CompactionReason::Threshold,
            instructions: None,
            cancel: operation.token(),
        };
        let result = compact_context(
            &mut history,
            &self.model_context.client,
            &file_tracker,
            request,
            &self.emitter,
            self.session_id.as_str(),
        )
        .await;
        match result {
            Ok(CompactionOutcome::Compacted(result)) => {
                let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                // Summarization is part of this prompt's bill, but its usage
                // does not describe the coding request's context window.
                state.record_compaction(&result);
                state.replace_history(history);
            }
            Ok(CompactionOutcome::Unchanged) => {}
            Err(error) => {
                if !matches!(error, Error::Interrupted(InterruptReason::Cancelled)) {
                    self.emit(CodingEvent::Error {
                        error: ErrorData::from(&error),
                    });
                }
                return true;
            }
        }
        false
    }

    /// Compacts once per prompt, remembering a failure so it is not retried.
    async fn compact_once_if_needed(&self) {
        let compaction_failed = self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .compaction_failed;
        if !compaction_failed && self.compact_if_needed().await {
            self.state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .compaction_failed = true;
        }
    }

    fn stage_task_reminder(&self, tools: &agent::ToolCatalog) {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .stage_task_reminder(tools);
    }

    fn conversation_update(&self, include_reminder: bool) -> agent::ConversationUpdate {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .conversation_update(include_reminder)
    }

    fn commit_user_message(&self, message: &LlmMessage, attribution: Option<&Value>) {
        let content = InputContent::from(message.content());
        let text = content.text_content().to_owned();
        let source = input_source_from_attribution(attribution).unwrap_or(InputSource::Agent);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.compaction_failed = false;
        state.push(Message::User {
            content:   content.clone(),
            timestamp: SystemTime::now(),
        });
        drop(state);
        self.emit(CodingEvent::UserInput {
            text,
            content: content.event_content(),
            source,
        });
    }

    fn commit_steering(&self, message: &LlmMessage, attribution: Option<&Value>) {
        let content = InputContent::from(message.content());
        let text = content.text_content().to_owned();
        let actor = actor_from_attribution(attribution);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.push(Message::Steering {
            content:   content.clone(),
            timestamp: SystemTime::now(),
        });
        drop(state);
        self.emit(CodingEvent::SteeringInjected {
            text,
            content: content.event_content(),
            actor,
        });
    }

    fn commit_assistant(&self, response: &Response) {
        self.finish_inference();
        let text = response.text();
        let tool_calls = tool_calls_of(response);
        let reasoning = response.reasoning();
        let provider_parts = provider_parts_of(response);
        let usage = response.usage_with_cost();
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let context_window = state.record_response_usage(usage);
        state.push_assistant(Message::Assistant {
            content: text.clone(),
            tool_calls: tool_calls.clone(),
            provider_parts,
            usage,
            response_id: response.id.clone().unwrap_or_default(),
            timestamp: SystemTime::now(),
        });
        drop(state);

        // The model layer's own warnings — a tool call the output limit cut
        // short, an option the protocol could not carry — ride the response
        // and are reported the way pebble reports its own.
        for warning in &response.warnings {
            self.emit(CodingEvent::Warning {
                kind:    warning.code.clone(),
                message: warning.message.clone(),
                details: Value::Null,
            });
        }

        let answering_model = response.model.model().as_str();
        self.emit(CodingEvent::AssistantMessage {
            text,
            model: if answering_model.is_empty() {
                self.model_context.model.clone()
            } else {
                answering_model.to_owned()
            },
            usage,
            tool_call_count: tool_calls.len(),
            context_window,
            reasoning,
        });
    }

    fn commit_tool_results(&self, calls: &[ToolCall], results: &[ToolResult], cancelled: bool) {
        self.state_machine.transition(CodingAgentState::Thinking);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.finish_tools(calls, results);

        if cancelled || self.prompt_cancel().is_cancelled() {
            return;
        }
        let loop_detected = self.config.enable_loop_detection
            && detect_loop(&state.history, self.config.loop_detection_window);
        if loop_detected {
            state.push(Message::Steering {
                content:   LOOP_WARNING.into(),
                timestamp: SystemTime::now(),
            });
        }
        drop(state);
        if loop_detected {
            self.emit(CodingEvent::LoopDetected);
        }
    }

    fn measure_request(&self, request: &Request) -> ContextWindowSnapshot {
        let tools = self.resources.registry.sources_for(request.tools());
        let activated = self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .activated_skill_context_observed;
        build_local_snapshot(ContextWindowInput {
            request,
            tools: &tools,
            system_prompt: &self.resources.system_prompt,
            memory_tokens: self.resources.memory_tokens,
            skills_tokens: self.resources.skills_tokens,
            activated_skill_context_observed: activated,
            provider: &self.model_context.provider,
            model: &self.model_context.model,
            context_window_tokens: self.model_context.facts.context_window_tokens,
        })
    }
}

impl agent::EventProjection for CodingAgentBridge {
    fn project(&self, event: &agent::AgentEvent) {
        match event {
            agent::AgentEvent::ModelRequestStarted { request, .. } => {
                let local = self.measure_request(request);
                let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                state.begin_inference(local);
                drop(state);
                self.emit(CodingEvent::LlmRequestStarted {
                    requested_model: self.model_context.model.clone(),
                });
            }
            agent::AgentEvent::FirstOutput { kind } => {
                let kind = match kind {
                    agent::FirstOutputKind::Text => LlmOutputKind::Text,
                    agent::FirstOutputKind::Reasoning => LlmOutputKind::Reasoning,
                    agent::FirstOutputKind::ToolCall => LlmOutputKind::ToolCall,
                    _ => return,
                };
                self.emit(CodingEvent::LlmFirstOutput { kind });
            }
            agent::AgentEvent::TextDelta { delta } => {
                self.emit(CodingEvent::TextDelta {
                    delta: delta.clone(),
                });
            }
            agent::AgentEvent::ReasoningDelta { delta } => {
                self.emit(CodingEvent::ReasoningDelta {
                    delta: delta.clone(),
                });
            }
            agent::AgentEvent::OutputReplaced => {
                self.emit(CodingEvent::AssistantOutputReplace {
                    text:      String::new(),
                    reasoning: None,
                });
            }
            agent::AgentEvent::TurnReplay {
                failed_attempt,
                delay_seconds,
                error,
            } => {
                self.emit(CodingEvent::LlmRetry {
                    provider:   self.model_context.provider.clone(),
                    model:      self.model_context.model.clone(),
                    attempt:    usize::try_from(failed_attempt.saturating_sub(1))
                        .unwrap_or(usize::MAX),
                    delay_secs: *delay_seconds,
                    error:      ErrorData::from(error),
                    phase:      LlmRetryPhase::Consume,
                });
            }
            agent::AgentEvent::ToolStarted { call } => {
                self.state_machine.transition(CodingAgentState::Executing);
                self.state
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .begin_tools();
                self.tools.emit_started(call);
            }
            agent::AgentEvent::ToolCompleted {
                result,
                error_kind,
                output_stats,
                metadata,
            } => {
                self.tools
                    .emit_result(result, *output_stats, *error_kind, metadata);
            }
            agent::AgentEvent::TurnInterrupted { generation } => {
                self.state
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .interrupt_round();
                self.emit(CodingEvent::RoundInterrupted {
                    generation: *generation,
                });
            }
            agent::AgentEvent::ToolRoundsExhausted { limit } => {
                self.emit(CodingEvent::ToolRoundsExhausted { limit: *limit });
            }
            _ => {}
        }
    }
}

#[async_trait]
impl agent::AgentLifecycle for CodingAgentBridge {
    async fn prepare_context(
        &self,
        turn: agent::TurnContext<'_>,
        tools: &agent::ToolCatalog,
        cancel: &CancellationToken,
    ) -> StdResult<Option<Vec<LlmMessage>>, agent::LifecycleError> {
        match &self.context_policy {
            Some(policy) => {
                policy
                    .prepare(
                        ContextPreparation {
                            session: &self.session_scope,
                            turn,
                            tools,
                        },
                        cancel,
                    )
                    .await
            }
            None => Ok(None),
        }
    }

    async fn before_model(
        &self,
        context: agent::TurnContext<'_>,
        cancel: &CancellationToken,
    ) -> StdResult<agent::ConversationUpdate, agent::LifecycleError> {
        if cancel.is_cancelled() || self.prompt_cancel().is_cancelled() {
            return Ok(agent::ConversationUpdate::unchanged());
        }
        // `turn` counts the model turns this prompt has completed. Nothing is
        // open here — the last turn's calls were answered before the loop came
        // back around — so ending the prompt leaves the conversation paired,
        // the way the wall clock does.
        if let Some(max_turns) = self.config.max_turns
            && context.turn() >= max_turns
        {
            return Err(self.record_boundary_error(Error::Interrupted(InterruptReason::TurnLimit)));
        }

        // Commit the input and any steering before work on the next request.
        self.flush_events().await?;
        self.compact_once_if_needed().await;
        let update = self.conversation_update(false);
        // Compaction can publish its own result or failure.
        self.flush_events().await?;
        Ok(update)
    }

    async fn after_tool_discovery(
        &self,
        _context: agent::TurnContext<'_>,
        tools: &agent::ToolCatalog,
        cancel: &CancellationToken,
    ) -> StdResult<agent::ConversationUpdate, agent::LifecycleError> {
        if cancel.is_cancelled() || self.prompt_cancel().is_cancelled() {
            return Ok(agent::ConversationUpdate::unchanged());
        }
        self.stage_task_reminder(tools);
        Ok(self.conversation_update(true))
    }

    async fn after_model(
        &self,
        _context: agent::TurnContext<'_>,
        _response: &Response,
        _cancel: &CancellationToken,
    ) -> StdResult<agent::ConversationUpdate, agent::LifecycleError> {
        // Do not run tools until the response that requested them is durable.
        self.flush_events().await?;
        self.compact_once_if_needed().await;
        let update = self.conversation_update(false);
        self.flush_events().await?;
        Ok(update)
    }

    async fn prepare_follow_up(
        &self,
        _context: agent::TurnContext<'_>,
        message: agent::UserMessage,
        _cancel: &CancellationToken,
    ) -> StdResult<agent::UserMessage, agent::LifecycleError> {
        let attribution = message.attribution().cloned();
        let content = InputContent::from(message.content());
        let text = content.text_content();
        // A queued turn a person typed carries the Prompt or FollowUp source
        // and its `/name` reference expands. Every synthesized turn — a
        // subagent's result envelope, a continuation, an integration's
        // assembled text — passes through unchanged, as `SkillExpansion`'s
        // contract says: expanding it would fail on an unknown skill or
        // splice a template over it.
        let typed = matches!(
            input_source_from_attribution(attribution.as_ref()),
            Some(InputSource::Prompt | InputSource::FollowUp)
        );
        let expanded = if self.resources.skills.is_empty() || !typed {
            ExpandedInput {
                text:       text.to_owned(),
                skill_name: None,
            }
        } else {
            expand_skill(&self.resources.skills, text)
                .map_err(|source| self.record_boundary_error(Error::SkillExpansion(source)))?
        };
        if let Some(name) = &expanded.skill_name {
            self.state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .activated_skill_context_observed = true;
            self.emit(CodingEvent::SkillActivated {
                skill_name: name.clone(),
                source:     SkillActivationSource::Slash,
            });
        }
        let content = content_after_skill_expansion(content, &expanded);
        let message = agent::UserMessage::new(content.into_parts());
        Ok(match attribution {
            Some(attribution) => message.with_attribution(attribution),
            None => message,
        })
    }

    async fn after_answer(
        &self,
        _context: agent::TurnContext<'_>,
        response: &Response,
        cancel: &CancellationToken,
    ) -> StdResult<agent::AfterAnswerAction, agent::LifecycleError> {
        if response.finish_reason == FinishReason::Length
            && let Some(continuation) = self.continue_after_output_limit()
        {
            return Ok(agent::AfterAnswerAction::ContinueWith(continuation));
        }
        let Some(supervisor) = self.subagents.as_ref() else {
            return Ok(agent::AfterAnswerAction::Complete);
        };
        match supervisor.next_parent_notification_turn(cancel).await {
            Ok(Some(turn)) => Ok(agent::AfterAnswerAction::ContinueWith(turn.into())),
            Ok(None) => Ok(agent::AfterAnswerAction::Complete),
            Err(error) => Err(self.record_boundary_error(error)),
        }
    }
}

impl agent::ConversationProjection for CodingAgentBridge {
    fn conversation_replaced(&self, messages: &[LlmMessage]) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.acknowledge_projection(messages);
    }

    fn user_message_committed(&self, message: &LlmMessage, attribution: Option<&Value>) {
        self.commit_user_message(message, attribution);
    }

    fn steering_message_committed(&self, message: &LlmMessage, attribution: Option<&Value>) {
        self.commit_steering(message, attribution);
    }

    fn assistant_message_committed(&self, response: &Response) {
        self.commit_assistant(response);
    }

    fn tool_results_committed(&self, calls: &[ToolCall], results: &[ToolResult], cancelled: bool) {
        self.commit_tool_results(calls, results, cancelled);
    }
}

#[derive(Clone)]
struct CodingModelService {
    model_context: Arc<SessionModel>,
    emitter:       Emitter,
    session_id:    SessionId,
}

#[async_trait]
impl agent::ModelService for CodingModelService {
    async fn stream(
        &self,
        request: Request,
        mut context: CallContext,
    ) -> StdResult<ResponseStream, LlmError> {
        context.extensions_mut().insert(RetryEventBridge::new(
            self.emitter.clone(),
            self.session_id.to_string(),
            self.model_context.provider.clone(),
            self.model_context.model.clone(),
        ));
        self.model_context
            .client
            .stream_with_context(request, context)
            .await
    }
}

impl CodingRuntime {
    /// Processes one input through the shared generic agent loop.
    ///
    /// `prompt_cancel` ends this prompt alone. It is a child of the runtime's
    /// terminal token, so a shutdown ends the prompt too; the two are told
    /// apart afterwards, because only the terminal one closes the session.
    pub(super) async fn process_input(
        &mut self,
        input: CodingInput,
        skill_expansion: SkillExpansion,
        prompt_cancel: &CancellationToken,
    ) -> Result<Option<String>> {
        if self.state.current() == CodingAgentState::Closed {
            return Err(Error::SessionClosed);
        }
        self.check_pump().await?;
        self.state.transition(CodingAgentState::Thinking);

        let source = input.source();
        let content = input.content().clone();
        let expanded = self.expand_input(content.text_content(), skill_expansion)?;
        if let Some(name) = &expanded.skill_name {
            self.conversation().activated_skill_context_observed = true;
            self.emit(CodingEvent::SkillActivated {
                skill_name: name.clone(),
                source:     SkillActivationSource::Slash,
            });
        }

        let message = input_message(content_after_skill_expansion(content, &expanded), source);
        self.drive_agent(Some(message), prompt_cancel).await
    }

    /// Continues the unfinished prompt through the shared generic agent loop,
    /// committing nothing before the model is asked.
    pub(super) async fn continue_input(
        &mut self,
        prompt_cancel: &CancellationToken,
    ) -> Result<Option<String>> {
        if self.state.current() == CodingAgentState::Closed {
            return Err(Error::SessionClosed);
        }
        self.check_pump().await?;
        self.state.transition(CodingAgentState::Thinking);
        self.drive_agent(None, prompt_cancel).await
    }

    /// Runs the generic loop, from `message` or from the conversation as it
    /// stands, and maps how it ended onto this session.
    async fn drive_agent(
        &mut self,
        message: Option<agent::UserMessage>,
        prompt_cancel: &CancellationToken,
    ) -> Result<Option<String>> {
        self.ensure_coding_agent()?;
        let bridge = self
            .coding_bridge
            .clone()
            .ok_or_else(|| Error::InvalidState("the coding bridge was not built".to_owned()))?;
        bridge.begin_prompt(prompt_cancel.clone());
        let mut agent = self
            .coding_agent
            .take()
            .ok_or_else(|| Error::InvalidState("the coding agent was not built".to_owned()))?;

        let result = match message {
            Some(message) => agent.prompt_with_cancellation(message, prompt_cancel).await,
            None => agent.continue_prompt_with_cancellation(prompt_cancel).await,
        };
        self.coding_agent = Some(agent);
        bridge.finish_inference();
        // A failed stream cancels the generic loop. Recover its exact error
        // before an abort can be mistaken for caller cancellation.
        self.check_pump().await?;
        if let Some(error) = bridge.take_boundary_error() {
            // The supervisor's wait answers a cancellation with a plain
            // `Cancelled`, whoever cancelled and whatever reason they recorded
            // first. The prompt ends the way an abort at any other checkpoint
            // does: with the recorded reason, and closed when the cancellation
            // was the session's own.
            if matches!(error, Error::Interrupted(InterruptReason::Cancelled)) {
                return Err(self.prompt_aborted().await);
            }
            return Err(error);
        }

        match result {
            Ok(outcome) => {
                let text = outcome.text();
                Ok((!text.trim().is_empty()).then_some(text))
            }
            Err(agent::AgentError::Aborted) => Err(self.prompt_aborted().await),
            Err(agent::AgentError::Model { source }) => Err(self.emit_llm_error(source)),
            Err(agent::AgentError::ToolRoundsExhausted { limit }) => {
                self.check_pump().await?;
                Err(Error::ToolRoundsExhausted { limit })
            }
            Err(error) => {
                self.check_pump().await?;
                Err(Error::Agent(error))
            }
        }
    }

    fn ensure_coding_agent(&mut self) -> Result<()> {
        if self.coding_agent.is_some() {
            return Ok(());
        }

        let bridge = Arc::new(CodingAgentBridge::from_runtime(self));
        let model_service = CodingModelService {
            model_context: self.model_context.clone(),
            emitter:       self.emitter.clone(),
            session_id:    self.session_scope.session_id().clone(),
        };
        let messages = bridge
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .history
            .to_llm_messages();
        let config = agent::AgentConfig {
            max_output_tokens: max_output_tokens(&self.config, &self.model_context.facts),
            reasoning_effort: self.config.reasoning_effort,
            speed: self.config.speed,
            turn_replay: self.config.turn_replay,
            max_turn_replays: STREAM_CONSUME_RETRIES,
            max_tool_rounds: self.config.max_tool_rounds,
            ..agent::AgentConfig::default()
        };
        let mut builder =
            agent::Agent::builder(model_service, self.model_context.model_selector.clone())
                .session(self.session_scope.clone())
                .system_prompt(self.resources.system_prompt.clone())
                .messages(messages)
                .tool_service(bridge.tools.clone())
                .tool_middleware(bridge.tools.clone())
                .control_handle(self.agent_control.clone())
                .lifecycle(bridge.clone())
                .conversation_projection(bridge.clone())
                .event_projection(bridge.clone())
                .config(config);
        for middleware in &self.tool_middleware {
            builder = builder.tool_middleware(Arc::clone(middleware));
        }
        let agent = builder.build().map_err(Error::AgentBuild)?;

        bridge
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .projected(false);
        self.coding_bridge = Some(bridge);
        self.coding_agent = Some(agent);
        Ok(())
    }

    /// Expands a `/name` reference in the input, where one is allowed.
    fn expand_input(&self, input: &str, expansion: SkillExpansion) -> Result<ExpandedInput> {
        if self.resources.skills.is_empty() || expansion == SkillExpansion::Skip {
            return Ok(ExpandedInput {
                text:       input.to_owned(),
                skill_name: None,
            });
        }
        expand_skill(&self.resources.skills, input).map_err(Error::SkillExpansion)
    }
}

/// Keeps attachments unchanged when a slash command replaces readable text.
fn content_after_skill_expansion(content: InputContent, expanded: &ExpandedInput) -> InputContent {
    if expanded.skill_name.is_none() {
        return content;
    }

    let mut parts = vec![ContentPart::Text {
        text: expanded.text.clone(),
    }];
    parts.extend(
        content
            .into_parts()
            .into_iter()
            .filter(|part| !matches!(part, ContentPart::Text { .. })),
    );
    InputContent::new(parts)
}

/// The most tokens the model may produce in one turn: the configured cap, or
/// the catalog's maximum for the model.
fn max_output_tokens(config: &CodingAgentOptions, facts: &ModelFacts) -> Option<u32> {
    let configured = config
        .max_tokens
        .map(|tokens| u32::try_from(tokens).expect("token limits were validated at build"));
    let from_catalog = || {
        facts
            .max_output_tokens
            .map(|tokens| u32::try_from(tokens).unwrap_or(u32::MAX))
    };
    configured
        .or_else(from_catalog)
        .filter(|tokens| *tokens > 0)
}

fn tool_calls_of(response: &Response) -> Vec<ToolCall> {
    response
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect()
}

fn provider_parts_of(response: &Response) -> Vec<ContentPart> {
    response
        .content
        .iter()
        .filter(|part| matches!(part, ContentPart::Opaque { .. } | ContentPart::Reasoning(_)))
        .cloned()
        .collect()
}

fn activated_a_skill(calls: &[ToolCall], results: &[ToolResult]) -> bool {
    calls.iter().zip(results).any(|(call, result)| {
        !result.is_error && canonical_tool_name(&call.name) == NativeTool::UseSkill.canonical_name()
    })
}

#[cfg(test)]
mod tests {
    use std::slice;

    use lithos_llm::catalog::{ModelId, ProviderId};
    use lithos_llm::types::ReasoningContent;

    use super::*;
    use crate::coding_agent::ShutdownReason;
    use crate::runtime::testing;
    use crate::test_support::scripted_client;

    #[tokio::test]
    async fn projection_updates_only_the_changed_tail_and_waits_for_acknowledgement() {
        use agent::ConversationProjection as _;
        use lithos_llm::types::Role;

        let (client, _) = scripted_client(vec![]);
        let mut runtime = testing::builder(client).build().expect("builds");
        let bridge = CodingAgentBridge::from_runtime(&runtime);
        let old = LlmMessage::text(Role::User, "old");
        bridge.commit_user_message(&old, None);
        bridge.conversation_replaced(slice::from_ref(&old));
        assert_eq!(
            bridge.conversation_update(false),
            agent::ConversationUpdate::unchanged()
        );

        let new = LlmMessage::text(Role::User, "new");
        bridge.commit_user_message(&new, None);
        let expected = agent::ConversationUpdate::replace_tail(1, vec![new.clone()]);
        assert_eq!(bridge.conversation_update(false), expected);
        // Preparing an update does not acknowledge it: an interrupted stage
        // must offer the same update at its next boundary.
        assert_eq!(bridge.conversation_update(false), expected);
        bridge.conversation_replaced(&[old, new]);
        assert_eq!(
            bridge.conversation_update(false),
            agent::ConversationUpdate::unchanged()
        );

        bridge
            .state
            .lock()
            .expect("healthy lock")
            .pending_task_reminder = Some(Message::System {
            content:   "remember".to_owned(),
            timestamp: SystemTime::now(),
        });
        let reminder = LlmMessage::text(Role::System, "remember");
        assert_eq!(
            bridge.conversation_update(true),
            agent::ConversationUpdate::replace_tail(2, vec![reminder.clone()])
        );
        let mut projected = bridge
            .state
            .lock()
            .expect("healthy lock")
            .history
            .to_llm_messages();
        projected.push(reminder);
        bridge.conversation_replaced(&projected);
        assert_eq!(
            bridge.conversation_update(false),
            agent::ConversationUpdate::replace_tail(2, vec![])
        );
        runtime
            .shutdown(ShutdownReason::Completed)
            .await
            .expect("shutdown");
    }

    fn response_with(content: Vec<ContentPart>) -> Response {
        Response::new(ProviderId::new("test"), ModelId::new("model"), content)
    }

    #[test]
    fn tool_calls_come_out_in_the_order_they_were_asked() {
        let response = response_with(vec![
            ContentPart::Text {
                text: "working".to_owned(),
            },
            ContentPart::ToolCall(ToolCall::function("call_1", "read", serde_json::json!({}))),
            ContentPart::ToolCall(ToolCall::function("call_2", "shell", serde_json::json!({}))),
        ]);

        let calls = tool_calls_of(&response);

        assert_eq!(
            calls
                .iter()
                .map(|call| call.id.as_str())
                .collect::<Vec<_>>(),
            ["call_1", "call_2"]
        );
    }

    #[test]
    fn only_provider_native_parts_are_kept_for_replay() {
        let response = response_with(vec![
            ContentPart::Reasoning(ReasoningContent {
                text:             "thinking".to_owned(),
                signature:        None,
                signature_origin: None,
                redacted:         false,
            }),
            ContentPart::Text {
                text: "answer".to_owned(),
            },
            ContentPart::Opaque {
                kind: "openai.reasoning".to_owned(),
                data: serde_json::json!({"id": "rs_1"}),
            },
            ContentPart::ToolCall(ToolCall::function("call_1", "read", serde_json::json!({}))),
        ]);

        let parts = provider_parts_of(&response);

        assert_eq!(parts.len(), 2);
        assert!(matches!(parts[0], ContentPart::Reasoning(_)));
        assert!(matches!(parts[1], ContentPart::Opaque { .. }));
    }

    #[test]
    fn a_successful_skill_call_counts_as_an_activation() {
        let call = ToolCall::function(
            "call_1",
            NativeTool::UseSkill.canonical_name(),
            serde_json::json!({}),
        );
        let ok = ToolResult {
            tool_call_id: "call_1".to_owned(),
            name:         None,
            content:      vec![ContentPart::Text {
                text: "loaded".to_owned(),
            }],
            is_error:     false,
        };
        let failed = ToolResult {
            is_error: true,
            ..ok.clone()
        };

        assert!(activated_a_skill(slice::from_ref(&call), &[ok]));
        assert!(!activated_a_skill(&[call], &[failed]));
    }
}
