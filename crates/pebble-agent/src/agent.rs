//! The agent facade and turn loop.

use std::collections::HashSet;
use std::result::Result as StdResult;
use std::sync::Arc;
use std::time::Duration;

use futures_util::future::join_all;
use lithos_llm::middleware::{CallContext, RetryPolicy};
use lithos_llm::types::{
    ContentPart, Error as LlmError, Message, ReasoningEffort, Request, Response, Role, Speed,
    ToolCall, ToolChoice, ToolResult,
};
use serde_json::Value;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::SessionScope;
use crate::control::{AgentControlHandle, CompletionReadiness, Control};
use crate::conversation::ConversationProjection;
use crate::error::{AgentBuildError, AgentError, Result};
use crate::event::{AgentEvent, EventHub, EventProjection, FirstOutputKind};
use crate::integration::{StreamObserver, StreamOutcome, stream_response};
use crate::model::ModelService;
#[cfg(test)]
use crate::tool::ToolCallRequest;
use crate::tool::{
    CANCELLED, StaticToolService, Tool, ToolCatalog, ToolErrorKind, ToolMiddleware, ToolOutcome,
    ToolScheduling, ToolService, ToolSystem,
};
use crate::turn::{
    AfterAnswerAction, AgentLifecycle, ConversationUpdate, LifecycleError, TurnContext,
};
use crate::validation::validate_context;

/// The default number of lifecycle events held for each subscriber.
const DEFAULT_EVENT_CAPACITY: usize = 256;

/// How many attempts the default [`AgentConfig::turn_replay`] allows: one
/// opening attempt plus the three replays of the default
/// [`AgentConfig::max_turn_replays`].
const DEFAULT_REPLAY_ATTEMPTS: u32 = 4;

/// The wait before the first replay under the default
/// [`AgentConfig::turn_replay`]. Each later wait doubles the one before.
const DEFAULT_REPLAY_INITIAL_DELAY: Duration = Duration::from_secs(1);

/// The longest computed wait between replays under the default
/// [`AgentConfig::turn_replay`].
const DEFAULT_REPLAY_MAX_DELAY: Duration = Duration::from_secs(60);

/// A user message accepted by [`Agent::prompt`], steering, or follow-up.
///
/// A message may carry an attribution: an opaque value the layer that queued
/// it reads back when the message is committed, which this crate never
/// interprets and never sends to the model. A coding layer uses it to say who
/// wrote a steering message.
#[derive(Clone, Debug, PartialEq)]
pub struct UserMessage {
    content:     Vec<ContentPart>,
    attribution: Option<Value>,
}

impl UserMessage {
    /// Creates a message from provider-neutral content parts.
    #[must_use]
    pub fn new(content: impl IntoIterator<Item = ContentPart>) -> Self {
        Self {
            content:     content.into_iter().collect(),
            attribution: None,
        }
    }

    /// Creates a text message.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::new([ContentPart::Text { text: text.into() }])
    }

    /// Attaches an opaque attribution, reported back on the message's
    /// [`SteeringMessage`](AgentEvent::SteeringMessage) event.
    #[must_use]
    pub fn with_attribution(mut self, attribution: Value) -> Self {
        self.attribution = Some(attribution);
        self
    }

    /// The message content.
    #[must_use]
    pub fn content(&self) -> &[ContentPart] {
        &self.content
    }

    /// The attribution attached to the message, if any.
    #[must_use]
    pub const fn attribution(&self) -> Option<&Value> {
        self.attribution.as_ref()
    }

    /// The readable text in the message.
    #[must_use]
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn into_message(self) -> Message {
        Message::new(Role::User, self.content)
    }
}

impl From<String> for UserMessage {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<&str> for UserMessage {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

/// Closes an abandoned prompt instead of exposing incomplete history as
/// reusable.
struct PromptRun {
    control:  Arc<Control>,
    events:   EventHub,
    finished: bool,
}

impl Drop for PromptRun {
    fn drop(&mut self) {
        if !self.finished {
            self.control.close();
            self.control.finish_prompt();
            self.events.emit(AgentEvent::PromptAborted);
            self.events.emit(AgentEvent::AgentClosed);
        }
    }
}

/// Turn-loop policy for an [`Agent`].
#[derive(Clone, Debug)]
pub struct AgentConfig {
    /// The most tokens one model turn may produce.
    pub max_output_tokens: Option<u32>,
    /// The requested reasoning effort.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// The requested latency or cost tier.
    pub speed: Option<Speed>,
    /// Replay policy for a response stream that fails after it opens.
    ///
    /// The default waits one second before the first replay, doubles the wait
    /// for each replay after that, and caps a wait at sixty seconds. Each wait
    /// is jittered: it lands between half of the computed wait and the whole
    /// of it, so the three replays together take between about 3.5 and 7
    /// seconds. That outlasts a provider blip of a few seconds, which the 100
    /// millisecond schedule of [`RetryPolicy::exponential`] does not. A
    /// `Retry-After` from the provider replaces the computed wait; one longer
    /// than sixty seconds ends the replays instead.
    pub turn_replay: RetryPolicy,
    /// Hard limit on replays after the first response stream opens.
    pub max_turn_replays: u32,
    /// The most tool rounds one prompt may run. `None` sets no limit.
    ///
    /// A tool round is one model turn that asks for tools, followed by the
    /// execution of those calls; it is not a count of individual calls. The
    /// budget covers one call to [`Agent::prompt`] or
    /// [`Agent::continue_prompt`], queued follow-up input included, and starts
    /// over with the next one. Once `limit` rounds have run, the model is
    /// still asked for its answer: a turn that answers with text completes
    /// the prompt as usual, and a turn that asks for tools again ends it with
    /// [`AgentError::ToolRoundsExhausted`] instead. The calls of that turn
    /// are answered as `Cancelled` without running, so history stays paired,
    /// and [`AgentEvent::ToolRoundsExhausted`] is emitted. With `Some(0)`,
    /// the first turn that asks for tools ends the prompt this way.
    pub max_tool_rounds: Option<usize>,
    /// The consecutive all-error tool rounds a prompt tolerates before it
    /// ends with [`AgentError::ConsecutiveToolErrors`] (fabro-9c79). Every
    /// result of each counted round was an error; a single successful
    /// result resets the count. `Some(0)` never trips (a first round
    /// cannot be consecutive); the default is 5.
    pub max_consecutive_error_rounds: usize,
    /// Events held for each live subscriber.
    pub event_capacity: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_output_tokens: None,
            reasoning_effort: None,
            speed: None,
            turn_replay: RetryPolicy::exponential()
                .initial_delay(DEFAULT_REPLAY_INITIAL_DELAY)
                .max_delay(DEFAULT_REPLAY_MAX_DELAY)
                .jitter(true)
                .max_attempts(DEFAULT_REPLAY_ATTEMPTS),
            max_turn_replays: 3,
            max_tool_rounds: None,
            max_consecutive_error_rounds: 5,
            event_capacity: DEFAULT_EVENT_CAPACITY,
        }
    }
}

/// Collects the required model service and optional agent behavior.
#[must_use = "a builder does nothing until `build` is called"]
pub struct AgentBuilder {
    session:          SessionScope,
    model_service:    Arc<dyn ModelService>,
    model:            String,
    system_prompt:    String,
    messages:         Vec<Message>,
    tools:            Vec<Tool>,
    tool_service:     Option<Arc<dyn ToolService>>,
    tool_middleware:  Vec<Arc<dyn ToolMiddleware>>,
    lifecycle:        Option<Arc<dyn AgentLifecycle>>,
    conversation:     Option<Arc<dyn ConversationProjection>>,
    event_projection: Option<Arc<dyn EventProjection>>,
    config:           AgentConfig,
    control:          Option<AgentControlHandle>,
}

impl AgentBuilder {
    fn new(service: impl ModelService + 'static, model: impl Into<String>) -> Self {
        Self {
            session:          SessionScope::default(),
            model_service:    Arc::new(service),
            model:            model.into(),
            system_prompt:    String::new(),
            messages:         Vec::new(),
            tools:            Vec::new(),
            tool_service:     None,
            tool_middleware:  Vec::new(),
            lifecycle:        None,
            conversation:     None,
            event_projection: None,
            config:           AgentConfig::default(),
            control:          None,
        }
    }

    /// Sets the identity shared by this agent's lifecycle and tool calls.
    pub fn session(mut self, session: SessionScope) -> Self {
        self.session = session;
        self
    }

    /// Binds the agent to a control handle created before it, with
    /// [`AgentControlHandle::detached`].
    ///
    /// Steering already queued on the handle is applied by the agent's first
    /// prompt. Without this, the agent creates its own control.
    pub fn control_handle(mut self, handle: AgentControlHandle) -> Self {
        self.control = Some(handle);
        self
    }

    /// Sets the system prompt sent before conversation history.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Starts with existing provider-neutral conversation history.
    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// Adds model-visible tools.
    pub fn tools(mut self, tools: impl IntoIterator<Item = Tool>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Sets the terminal service for dynamic tool discovery and invocation.
    ///
    /// A custom service cannot be combined with tools added through
    /// [`tools`](Self::tools).
    pub fn tool_service(mut self, service: Arc<dyn ToolService>) -> Self {
        self.tool_service = Some(service);
        self
    }

    /// Adds one tool middleware inside every middleware already installed.
    ///
    /// The first installed middleware is outermost.
    pub fn tool_middleware(mut self, middleware: Arc<dyn ToolMiddleware>) -> Self {
        self.tool_middleware.push(middleware);
        self
    }

    /// Sets the lifecycle stages around model turns and natural answers.
    pub fn lifecycle(mut self, lifecycle: Arc<dyn AgentLifecycle>) -> Self {
        self.lifecycle = Some(lifecycle);
        self
    }

    /// Projects explicit canonical-conversation commits into another model.
    pub fn conversation_projection(mut self, projection: Arc<dyn ConversationProjection>) -> Self {
        self.conversation = Some(projection);
        self
    }

    /// Projects lifecycle events into an embedding layer's event model.
    pub fn event_projection(mut self, projection: Arc<dyn EventProjection>) -> Self {
        self.event_projection = Some(projection);
        self
    }

    /// Replaces the turn-loop policy.
    pub fn config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    /// Builds the agent.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid configuration or duplicate static tools.
    pub fn build(self) -> StdResult<Agent, AgentBuildError> {
        if self.model.trim().is_empty() {
            return Err(AgentBuildError::EmptyModel);
        }
        if self.config.max_output_tokens == Some(0) {
            return Err(AgentBuildError::ZeroOutputTokens);
        }
        if self.config.event_capacity == 0 {
            return Err(AgentBuildError::ZeroEventCapacity);
        }
        if self.tool_service.is_some() && !self.tools.is_empty() {
            return Err(AgentBuildError::ConflictingToolSources);
        }
        let mut names = HashSet::new();
        let mut ids = HashSet::new();
        for tool in &self.tools {
            let name = tool.definition().name.clone();
            if !names.insert(name.clone()) {
                return Err(AgentBuildError::DuplicateTool { name });
            }
            let id = tool.descriptor().id().clone();
            if !ids.insert(id.clone()) {
                return Err(AgentBuildError::DuplicateToolId { id });
            }
        }

        let events = EventHub::new(self.config.event_capacity, self.event_projection);
        let terminal = self.tool_service.unwrap_or_else(|| {
            Arc::new(StaticToolService::new(self.tools)) as Arc<dyn ToolService>
        });
        let mut tool_system = ToolSystem::new(terminal);
        for middleware in self.tool_middleware {
            tool_system = tool_system.middleware(middleware);
        }
        Ok(Agent {
            session: self.session,
            model_service: self.model_service,
            model: self.model,
            system_prompt: self.system_prompt,
            messages: self.messages,
            tool_system,
            lifecycle: self.lifecycle,
            conversation: self.conversation,
            config: self.config,
            events,
            control: self
                .control
                .map_or_else(Control::new, |handle| handle.control()),
        })
    }
}

/// What an agent is doing now.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AgentState {
    /// Waiting for input.
    Idle,
    /// Processing a prompt.
    Running,
    /// Shut down.
    Closed,
}

/// An owned, immutable view of an agent's current state.
#[derive(Clone, Debug, PartialEq)]
pub struct AgentSnapshot {
    session:  SessionScope,
    state:    AgentState,
    model:    String,
    messages: Vec<Message>,
}

impl AgentSnapshot {
    /// The identity and ancestry captured by this snapshot.
    #[must_use]
    pub const fn session(&self) -> &SessionScope {
        &self.session
    }

    /// The lifecycle state captured by this snapshot.
    #[must_use]
    pub const fn state(&self) -> AgentState {
        self.state
    }

    /// The configured model selector.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The conversation history captured by this snapshot.
    #[must_use]
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }
}

/// The final result of a prompt and any queued follow-up input.
#[derive(Clone, Debug, PartialEq)]
pub struct PromptOutcome {
    response:        Response,
    turn_count:      usize,
    tool_call_count: usize,
}

impl PromptOutcome {
    /// The final complete model response.
    #[must_use]
    pub const fn response(&self) -> &Response {
        &self.response
    }

    /// Consumes the outcome and returns the final response.
    #[must_use]
    pub fn into_response(self) -> Response {
        self.response
    }

    /// The readable text in the final response.
    #[must_use]
    pub fn text(&self) -> String {
        self.response.text()
    }

    /// The number of model turns completed for this prompt.
    #[must_use]
    pub const fn turn_count(&self) -> usize {
        self.turn_count
    }

    /// The total number of tool calls executed for this prompt.
    #[must_use]
    pub const fn tool_call_count(&self) -> usize {
        self.tool_call_count
    }
}

/// One provider-neutral active conversation.
pub struct Agent {
    session:       SessionScope,
    model_service: Arc<dyn ModelService>,
    model:         String,
    system_prompt: String,
    messages:      Vec<Message>,
    tool_system:   ToolSystem,
    lifecycle:     Option<Arc<dyn AgentLifecycle>>,
    conversation:  Option<Arc<dyn ConversationProjection>>,
    config:        AgentConfig,
    events:        EventHub,
    control:       Arc<Control>,
}

impl Agent {
    /// The identity and ancestry of this agent.
    #[must_use]
    pub const fn session(&self) -> &SessionScope {
        &self.session
    }

    /// Starts a builder with an injected model service and model selector.
    pub fn builder(service: impl ModelService + 'static, model: impl Into<String>) -> AgentBuilder {
        AgentBuilder::new(service, model)
    }

    /// Subscribes to lifecycle events.
    ///
    /// The stream is bounded and lossy for a receiver that falls behind.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.events.subscribe()
    }

    /// Returns a handle that can steer, follow up, or abort an active prompt.
    #[must_use]
    pub fn control_handle(&self) -> AgentControlHandle {
        AgentControlHandle::new(Arc::clone(&self.control))
    }

    /// Queues steering for the next turn and interrupts the current turn.
    pub fn steer(&self, message: impl Into<UserMessage>) -> bool {
        self.control_handle().steer(message)
    }

    /// Queues input to process after the current answer.
    pub fn follow_up(&self, message: impl Into<UserMessage>) -> bool {
        self.control_handle().follow_up(message)
    }

    /// Aborts the active prompt.
    pub fn abort(&self) -> bool {
        self.control_handle().abort()
    }

    /// Interrupts the current turn and waits for steering before another.
    pub fn interrupt(&self) -> bool {
        self.control_handle().interrupt()
    }

    /// Claims the next turn boundary for steering without interrupting now.
    pub fn park_for_steer(&self) -> bool {
        self.control_handle().park_for_steer()
    }

    /// Waits until no prompt is running.
    pub async fn wait_for_idle(&self) {
        self.control_handle().wait_for_idle().await;
    }

    /// The current lifecycle state.
    #[must_use]
    pub fn state(&self) -> AgentState {
        if self.control.is_closed() {
            AgentState::Closed
        } else if self.control.is_running() {
            AgentState::Running
        } else {
            AgentState::Idle
        }
    }

    /// The current conversation history.
    #[must_use]
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Captures an owned view of the current state and history.
    #[must_use]
    pub fn snapshot(&self) -> AgentSnapshot {
        AgentSnapshot {
            session:  self.session.clone(),
            state:    self.state(),
            model:    self.model.clone(),
            messages: self.messages.clone(),
        }
    }

    /// Changes the reasoning effort used by later model turns.
    pub fn set_reasoning_effort(&mut self, effort: Option<ReasoningEffort>) {
        self.config.reasoning_effort = effort;
    }

    /// Changes the latency or cost tier used by later model turns.
    pub fn set_speed(&mut self, speed: Option<Speed>) {
        self.config.speed = speed;
    }

    /// Clears conversation history and queued input while the agent is idle.
    ///
    /// Returns `false` while a prompt is active or after shutdown.
    pub fn reset(&mut self) -> bool {
        if self.state() != AgentState::Idle {
            return false;
        }
        self.messages.clear();
        self.control.clear_queues();
        true
    }

    /// Processes one input and every queued follow-up to completion.
    ///
    /// Dropping this future after work starts permanently closes the agent.
    /// Use [`prompt_with_cancellation`](Self::prompt_with_cancellation) to
    /// cancel cooperatively and keep the agent reusable.
    ///
    /// # Errors
    ///
    /// Returns an error when the input is empty, the agent is closed, the
    /// prompt is aborted, context preparation fails, or the model call
    /// fails.
    pub async fn prompt(&mut self, message: impl Into<UserMessage>) -> Result<PromptOutcome> {
        self.prompt_inner(Some(message.into()), None).await
    }

    /// Processes one input until it completes or `cancel` is cancelled.
    ///
    /// The external signal is combined with cancellation from this agent's
    /// control handle. A cancellation that is already set still commits the
    /// input before the prompt aborts.
    ///
    /// The conversation stays paired however the prompt ends, short of the
    /// caller dropping the future. A cancellation observed after the model
    /// asked for tools answers every pending call with a `Cancelled` error
    /// result, without running it, and records those results before the
    /// prompt returns [`AgentError::Aborted`]; an `after_model` stage that
    /// fails is answered the same way before its error is returned. A call
    /// that was already running is cancelled through its own token and keeps
    /// the result it returns. Dropping this future permanently closes the
    /// agent because incomplete history cannot safely be reused.
    pub async fn prompt_with_cancellation(
        &mut self,
        message: impl Into<UserMessage>,
        cancel: &CancellationToken,
    ) -> Result<PromptOutcome> {
        self.prompt_inner(Some(message.into()), Some(cancel)).await
    }

    /// Continues the prompt the conversation left unfinished, without new
    /// input.
    ///
    /// A prompt is unfinished when the conversation ends with something the
    /// model has not answered: a user message, or the results of the tool
    /// calls the model asked for. That is what a conversation looks like after
    /// a model call failed for good, or after it was restored from a record
    /// taken at that point. The model is asked again on the conversation as it
    /// stands, nothing is committed before that call, and the prompt then runs
    /// to completion exactly as [`prompt`](Self::prompt) would, follow-ups
    /// included. Tool calls that already have their results are not run again.
    ///
    /// Dropping the future after work starts permanently closes the agent, as
    /// with `prompt`.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::NothingToContinue`] when the conversation is
    /// empty or ends with the model's own message, and otherwise fails as
    /// `prompt` does.
    pub async fn continue_prompt(&mut self) -> Result<PromptOutcome> {
        self.prompt_inner(None, None).await
    }

    /// Continues the unfinished prompt until it completes or `cancel` is
    /// cancelled.
    ///
    /// # Errors
    ///
    /// As [`continue_prompt`](Self::continue_prompt), with the cancellation
    /// contract of
    /// [`prompt_with_cancellation`](Self::prompt_with_cancellation).
    pub async fn continue_prompt_with_cancellation(
        &mut self,
        cancel: &CancellationToken,
    ) -> Result<PromptOutcome> {
        self.prompt_inner(None, Some(cancel)).await
    }

    /// Whether the conversation ends with input the model has not answered.
    fn has_unfinished_prompt(&self) -> bool {
        self.messages
            .last()
            .is_some_and(|message| matches!(message.role(), Role::User | Role::Tool))
    }

    #[tracing::instrument(
        name = "agent_prompt",
        skip_all,
        fields(
            model = %self.model,
            input_part_count = message.as_ref().map_or(0, |message| message.content.len()),
            continued = message.is_none(),
        )
    )]
    async fn prompt_inner(
        &mut self,
        message: Option<UserMessage>,
        parent_cancel: Option<&CancellationToken>,
    ) -> Result<PromptOutcome> {
        match &message {
            Some(message) if message.content.is_empty() => return Err(AgentError::EmptyInput),
            None if !self.has_unfinished_prompt() => return Err(AgentError::NothingToContinue),
            Some(_) | None => {}
        }
        let Some(prompt_cancel) = self.control.begin_prompt(parent_cancel) else {
            return Err(AgentError::Closed);
        };

        let mut run = PromptRun {
            control:  Arc::clone(&self.control),
            events:   self.events.clone(),
            finished: false,
        };
        self.emit(AgentEvent::PromptStarted);
        let result = self.process_prompt(message, &prompt_cancel).await;
        match &result {
            Ok(outcome) => self.emit(AgentEvent::PromptCompleted {
                response: outcome.response.clone(),
            }),
            Err(AgentError::Aborted) => self.emit(AgentEvent::PromptAborted),
            Err(_) => {}
        }
        self.control.finish_prompt();
        run.finished = true;
        result
    }

    /// Closes an idle agent.
    ///
    /// Returns whether this call performed the transition.
    pub fn shutdown(&mut self) -> bool {
        if !self.control.close() {
            return false;
        }
        self.emit(AgentEvent::AgentClosed);
        true
    }

    /// Runs the loop from `first_message`, or from the conversation as it
    /// stands when there is none to commit first.
    async fn process_prompt(
        &mut self,
        first_message: Option<UserMessage>,
        prompt_cancel: &CancellationToken,
    ) -> Result<PromptOutcome> {
        let mut next_message = first_message;
        let mut turn_count = 0;
        let mut tool_call_count = 0;
        let mut tool_round_count = 0;
        // fabro-9c79: consecutive rounds in which EVERY tool result was an
        // error. A single successful result resets it.
        let mut consecutive_error_rounds = 0usize;

        'prompt: loop {
            if let Some(message) = next_message.take() {
                self.commit_user_message(message);
            }

            let final_response = loop {
                if prompt_cancel.is_cancelled() {
                    return Err(AgentError::Aborted);
                }

                if !self.control.wait_until_resumed(prompt_cancel).await {
                    return Err(AgentError::Aborted);
                }
                self.emit_pending_interrupts();

                let (round_cancel, steering) = self.control.begin_round();
                for steering in steering {
                    let attribution = steering.attribution().cloned();
                    let message = steering.into_message();
                    self.commit_steering_message(message, attribution);
                }
                if let Some(lifecycle) = self.lifecycle.clone() {
                    let context =
                        TurnContext::new(&self.session, &self.model, turn_count, &self.messages);
                    let prepared = lifecycle.before_model(context, &round_cancel).await;
                    // Nothing is open here: the last turn's tool calls were
                    // answered before the loop came back around, so aborting
                    // leaves the conversation paired.
                    if prompt_cancel.is_cancelled() {
                        return Err(AgentError::Aborted);
                    }
                    if round_cancel.is_cancelled() {
                        self.emit_pending_interrupts();
                        continue;
                    }
                    let update = prepared.map_err(|source| AgentError::Lifecycle { source })?;
                    self.apply_conversation_update(update)
                        .map_err(|source| AgentError::Lifecycle { source })?;
                }

                let tools = self.discover_tools(turn_count).await?;
                if let Some(lifecycle) = self.lifecycle.clone() {
                    let context =
                        TurnContext::new(&self.session, &self.model, turn_count, &self.messages);
                    let prepared = lifecycle
                        .after_tool_discovery(context, &tools, &round_cancel)
                        .await;
                    if prompt_cancel.is_cancelled() {
                        return Err(AgentError::Aborted);
                    }
                    if round_cancel.is_cancelled() {
                        self.emit_pending_interrupts();
                        continue;
                    }
                    let update = prepared.map_err(|source| AgentError::Lifecycle { source })?;
                    self.apply_conversation_update(update)
                        .map_err(|source| AgentError::Lifecycle { source })?;
                }
                let prepared = if let Some(lifecycle) = self.lifecycle.as_ref() {
                    let context =
                        TurnContext::new(&self.session, &self.model, turn_count, &self.messages);
                    tokio::select! {
                        biased;
                        () = prompt_cancel.cancelled() => return Err(AgentError::Aborted),
                        () = round_cancel.cancelled() => continue,
                        result = lifecycle.prepare_context(context, &tools, &round_cancel) => {
                            result.map_err(|source| AgentError::Lifecycle { source })?
                        }
                    }
                } else {
                    None
                };
                if let Some(messages) = &prepared {
                    validate_context(messages)
                        .map_err(|source| AgentError::Lifecycle { source })?;
                }
                let request = self.build_request_with_messages(
                    &tools,
                    prepared.as_deref().unwrap_or(&self.messages),
                )?;
                self.emit(AgentEvent::TurnStarted { turn: turn_count });
                self.emit(AgentEvent::ModelRequestStarted {
                    model:   self.model.clone(),
                    request: request.clone(),
                });

                let response = match self
                    .stream_response(request, prompt_cancel, &round_cancel)
                    .await?
                {
                    StreamResult::Completed(response) => *response,
                    StreamResult::Interrupted => {
                        self.emit_pending_interrupts();
                        continue;
                    }
                };
                let turn = turn_count;
                turn_count += 1;

                self.commit_assistant_message(&response);

                // The assistant turn is committed, and the lifecycle can take a
                // while (the coding layer compacts here). A prompt cancelled
                // by now is not honored until the turn's tool calls have their
                // results: `execute_tools` sees the fired token and answers
                // every call as `Cancelled` without running it, so the
                // conversation the abort leaves behind stays paired. A stage
                // that fails is held to the same rule: the calls are answered
                // as `Cancelled` before its error ends the prompt.
                if let Some(lifecycle) = self.lifecycle.clone() {
                    let context =
                        TurnContext::new(&self.session, &self.model, turn, &self.messages);
                    match lifecycle
                        .after_model(context, &response, &round_cancel)
                        .await
                        .and_then(|update| self.apply_conversation_update(update))
                    {
                        Ok(()) => {}
                        Err(source) => {
                            let calls = tool_calls(&response);
                            if !calls.is_empty() {
                                let results =
                                    self.answer_calls_as_cancelled(turn, &calls, &tools).await;
                                self.commit_tool_results(&calls, &results, true);
                            }
                            return Err(AgentError::Lifecycle { source });
                        }
                    }
                }

                let calls = tool_calls(&response);
                if calls.is_empty() {
                    // An answer opens nothing, so the abort can land here.
                    if prompt_cancel.is_cancelled() {
                        return Err(AgentError::Aborted);
                    }
                    if round_cancel.is_cancelled() || self.control.is_paused() {
                        if round_cancel.is_cancelled() {
                            self.emit_pending_interrupts();
                        }
                        continue;
                    }

                    if matches!(
                        self.control.wait_for_completion(prompt_cancel).await,
                        CompletionReadiness::SteeringQueued
                    ) {
                        self.emit_pending_interrupts();
                        continue;
                    }
                    if prompt_cancel.is_cancelled() {
                        return Err(AgentError::Aborted);
                    }
                    if round_cancel.is_cancelled() {
                        self.emit_pending_interrupts();
                        continue;
                    }
                    if let Some(follow_up) = self.control.pop_follow_up() {
                        next_message = Some(
                            self.prepare_follow_up(follow_up, turn, prompt_cancel)
                                .await?,
                        );
                        continue 'prompt;
                    }
                    if let Some(lifecycle) = self.lifecycle.clone() {
                        let context =
                            TurnContext::new(&self.session, &self.model, turn, &self.messages);
                        let boundary_action = lifecycle
                            .after_answer(context, &response, prompt_cancel)
                            .await
                            .map_err(|source| AgentError::Lifecycle { source })?;
                        // Still nothing open: the answer had no tool calls.
                        if prompt_cancel.is_cancelled() {
                            return Err(AgentError::Aborted);
                        }
                        if round_cancel.is_cancelled() {
                            self.emit_pending_interrupts();
                            continue;
                        }
                        match boundary_action {
                            AfterAnswerAction::Complete => {}
                            AfterAnswerAction::Continue => continue,
                            AfterAnswerAction::ContinueWith(message) => {
                                self.commit_user_message(message);
                                continue;
                            }
                        }
                    }
                    break response;
                }

                // The budget counts rounds that ran. A turn that asks for
                // tools once it is spent is committed, like any other, but
                // its calls are answered as `Cancelled` without running, so
                // the history the prompt leaves behind stays paired.
                if let Some(limit) = self.config.max_tool_rounds
                    && tool_round_count >= limit
                {
                    let results = self.answer_calls_as_cancelled(turn, &calls, &tools).await;
                    self.commit_tool_results(&calls, &results, true);
                    self.emit(AgentEvent::ToolRoundsExhausted { limit });
                    return Err(AgentError::ToolRoundsExhausted { limit });
                }
                tool_round_count += 1;
                tool_call_count += calls.len();

                let execution = self
                    .execute_tools(turn, &calls, &tools, prompt_cancel, &round_cancel)
                    .await;
                let cancelled = prompt_cancel.is_cancelled() || round_cancel.is_cancelled();
                self.commit_tool_results(&calls, &execution.results, cancelled);

                // fabro-9c79: the repeat-until-interrupt backstop. A round
                // where every result is an error counts; one success resets.
                if !execution.results.is_empty()
                    && execution.results.iter().all(|result| result.is_error)
                {
                    consecutive_error_rounds += 1;
                    if consecutive_error_rounds >= self.config.max_consecutive_error_rounds {
                        let first_error = execution
                            .results
                            .iter()
                            .find(|result| result.is_error)
                            .map(|result| {
                                result
                                    .content
                                    .iter()
                                    .filter_map(|part| match part {
                                        ContentPart::Text { text } => Some(text.as_str()),
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            })
                            .unwrap_or_default();
                        self.emit(AgentEvent::ConsecutiveToolErrors {
                            rounds:      consecutive_error_rounds,
                            first_error: first_error.clone(),
                        });
                        return Err(AgentError::ConsecutiveToolErrors {
                            rounds: consecutive_error_rounds,
                            first_error,
                        });
                    }
                } else {
                    consecutive_error_rounds = 0;
                }

                if let Some(source) = execution.system_error {
                    return Err(AgentError::ToolSystem { source });
                }

                if prompt_cancel.is_cancelled() {
                    return Err(AgentError::Aborted);
                }
                if round_cancel.is_cancelled() {
                    self.emit_pending_interrupts();
                }
            };

            if let Some(follow_up) = self.control.pop_follow_up() {
                next_message = Some(
                    self.prepare_follow_up(follow_up, turn_count.saturating_sub(1), prompt_cancel)
                        .await?,
                );
                continue 'prompt;
            }

            return Ok(PromptOutcome {
                response: final_response,
                turn_count,
                tool_call_count,
            });
        }
    }

    async fn discover_tools(&self, turn: usize) -> Result<ToolCatalog> {
        let tools = self
            .tool_system
            .discover(TurnContext::new(
                &self.session,
                &self.model,
                turn,
                &self.messages,
            ))
            .await
            .map_err(|source| AgentError::ToolSystem { source })?;
        let mut names = HashSet::new();
        let mut ids = HashSet::new();
        for tool in tools.tools() {
            let name = tool.definition().name.clone();
            if !names.insert(name.clone()) {
                return Err(AgentError::DuplicateTool { name });
            }
            let id = tool.id().clone();
            if !ids.insert(id.clone()) {
                return Err(AgentError::DuplicateToolId { id });
            }
        }
        Ok(tools)
    }

    async fn prepare_follow_up(
        &self,
        message: UserMessage,
        turn: usize,
        cancel: &CancellationToken,
    ) -> Result<UserMessage> {
        let Some(lifecycle) = self.lifecycle.clone() else {
            return Ok(message);
        };
        let context = TurnContext::new(&self.session, &self.model, turn, &self.messages);
        lifecycle
            .prepare_follow_up(context, message, cancel)
            .await
            .map_err(|source| AgentError::Lifecycle { source })
    }

    fn build_request_with_messages(
        &self,
        tools: &ToolCatalog,
        messages: &[Message],
    ) -> Result<Request> {
        let mut builder = Request::builder().model(self.model.clone());
        if !self.system_prompt.trim().is_empty() {
            builder = builder.system(self.system_prompt.clone());
        }
        for message in messages {
            builder = builder.message(message.clone());
        }
        for tool in tools.visible_tools() {
            builder = builder.tool(tool.definition().clone());
        }
        if tools.visible_tools().next().is_some() {
            builder = builder.tool_choice(ToolChoice::Auto);
        }
        if let Some(tokens) = self.config.max_output_tokens {
            builder = builder.max_output_tokens(tokens);
        }
        if let Some(effort) = self.config.reasoning_effort {
            builder = builder.reasoning_effort(effort);
        }
        if let Some(speed) = self.config.speed {
            builder = builder.speed(speed);
        }
        builder
            .build()
            .map_err(|source| AgentError::Request { source })
    }

    async fn stream_response(
        &self,
        request: Request,
        prompt_cancel: &CancellationToken,
        round_cancel: &CancellationToken,
    ) -> Result<StreamResult> {
        let observer = AgentStreamObserver {
            events: &self.events,
        };
        match stream_response(
            self.model_service.as_ref(),
            request,
            prompt_cancel,
            round_cancel,
            self.config.turn_replay,
            self.config.max_turn_replays,
            CallContext::new,
            &observer,
        )
        .await
        {
            StreamOutcome::Completed(response) => Ok(StreamResult::Completed(response)),
            StreamOutcome::Aborted => Err(AgentError::Aborted),
            StreamOutcome::Interrupted => Ok(StreamResult::Interrupted),
            StreamOutcome::Failed(source) => Err(AgentError::Model { source: *source }),
        }
    }

    /// Answers every call as `Cancelled` without running one.
    async fn answer_calls_as_cancelled(
        &self,
        turn: usize,
        calls: &[ToolCall],
        tools: &ToolCatalog,
    ) -> Vec<ToolResult> {
        let fired = CancellationToken::new();
        fired.cancel();
        self.execute_tools(turn, calls, tools, &fired, &fired)
            .await
            .results
    }

    fn commit_user_message(&mut self, message: UserMessage) {
        let attribution = message.attribution().cloned();
        let message = message.into_message();
        self.messages.push(message.clone());
        if let Some(projection) = &self.conversation {
            projection.user_message_committed(&message, attribution.as_ref());
        }
        self.emit(AgentEvent::UserMessage { message });
    }

    fn emit_pending_interrupts(&self) {
        if let Some(generations) = self.control.settle_interrupts() {
            for generation in generations {
                self.emit(AgentEvent::TurnInterrupted { generation });
            }
        }
    }

    fn apply_conversation_update(
        &mut self,
        update: ConversationUpdate,
    ) -> StdResult<(), LifecycleError> {
        if update.apply(&mut self.messages)?
            && let Some(projection) = &self.conversation
        {
            projection.conversation_replaced(&self.messages);
        }
        Ok(())
    }

    fn commit_steering_message(&mut self, message: Message, attribution: Option<Value>) {
        self.messages.push(message.clone());
        if let Some(projection) = &self.conversation {
            projection.steering_message_committed(&message, attribution.as_ref());
        }
        self.emit(AgentEvent::SteeringMessage {
            message,
            attribution,
        });
    }

    fn commit_assistant_message(&mut self, response: &Response) {
        self.messages.push(response_message(response));
        if let Some(projection) = &self.conversation {
            projection.assistant_message_committed(response);
        }
        self.emit(AgentEvent::AssistantMessage {
            response: response.clone(),
        });
    }

    fn commit_tool_results(&mut self, calls: &[ToolCall], results: &[ToolResult], cancelled: bool) {
        self.messages.push(tool_results_message(results));
        if let Some(projection) = &self.conversation {
            projection.tool_results_committed(calls, results, cancelled);
        }
    }

    async fn execute_tools(
        &self,
        turn: usize,
        calls: &[ToolCall],
        tools: &ToolCatalog,
        prompt_cancel: &CancellationToken,
        round_cancel: &CancellationToken,
    ) -> ToolRoundExecution {
        // A round that opens after either token fired runs already cancelled:
        // each call is answered as `Cancelled`, and every call stays paired.
        let cancel = CancellationToken::new();
        let _cancel_on_drop = cancel.clone().drop_guard();
        if prompt_cancel.is_cancelled() || round_cancel.is_cancelled() {
            cancel.cancel();
        }
        let running = self.schedule_tools(turn, calls, tools, &cancel);
        tokio::pin!(running);

        let mut prompt_cancelled = false;
        let mut round_cancelled = false;
        loop {
            tokio::select! {
                biased;
                () = prompt_cancel.cancelled(), if !prompt_cancelled => {
                    prompt_cancelled = true;
                    cancel.cancel();
                }
                () = round_cancel.cancelled(), if !round_cancelled => {
                    round_cancelled = true;
                    cancel.cancel();
                }
                results = &mut running => return ToolRoundExecution::from_calls(results),
            }
        }
    }

    async fn schedule_tools(
        &self,
        turn: usize,
        calls: &[ToolCall],
        tools: &ToolCatalog,
        cancel: &CancellationToken,
    ) -> Vec<ExecutedCall> {
        let exclusive = calls.iter().position(|call| {
            tools
                .find_by_name(&call.name)
                .is_some_and(|tool| tool.scheduling() == ToolScheduling::ExclusiveRound)
        });
        if let Some(exclusive) = exclusive {
            let mut results = Vec::with_capacity(calls.len());
            for (index, call) in calls.iter().enumerate() {
                if cancel.is_cancelled() {
                    results.push(ExecutedCall::completed(cancelled_tool_result(call)));
                } else if index == exclusive {
                    results.push(self.execute_tool(turn, call, tools, cancel).await);
                } else {
                    results.push(self.refuse_exclusive_peer(call));
                }
            }
            return results;
        }

        let sequential = calls.iter().any(|call| {
            tools
                .find_by_name(&call.name)
                .is_some_and(|tool| tool.scheduling() == ToolScheduling::Sequential)
        });
        if sequential {
            let mut results = Vec::with_capacity(calls.len());
            for call in calls {
                results.push(self.execute_tool(turn, call, tools, cancel).await);
            }
            results
        } else {
            join_all(
                calls
                    .iter()
                    .map(|call| self.execute_tool(turn, call, tools, cancel)),
            )
            .await
        }
    }

    #[tracing::instrument(
        name = "tool_call",
        skip_all,
        fields(turn, tool = %call.name, tool_call_id = %call.id)
    )]
    async fn execute_tool(
        &self,
        turn: usize,
        call: &ToolCall,
        tools: &ToolCatalog,
        cancel: &CancellationToken,
    ) -> ExecutedCall {
        // A call that finds its round already cancelled is answered without
        // starting, and publishes nothing: there was no execution to report.
        if cancel.is_cancelled() {
            return ExecutedCall::completed(cancelled_tool_result(call));
        }
        self.emit(AgentEvent::ToolStarted { call: call.clone() });
        let called = self
            .tool_system
            .execute_with(
                tools,
                TurnContext::new(&self.session, &self.model, turn, &self.messages),
                call.clone(),
                cancel.child_token(),
                Some(self.events.clone()),
            )
            .await;
        match called {
            Ok(outcome) => self.complete_tool(call, outcome, None),
            Err(error) => {
                let message = error.message().to_owned();
                self.complete_tool(
                    call,
                    ToolOutcome::failure(ToolErrorKind::Execution, message),
                    Some(error),
                )
            }
        }
    }

    fn refuse_exclusive_peer(&self, call: &ToolCall) -> ExecutedCall {
        self.emit(AgentEvent::ToolStarted { call: call.clone() });
        self.complete_tool(
            call,
            ToolOutcome::failure(
                ToolErrorKind::Denied,
                "this tool did not run because an exclusive tool must run alone in its round",
            ),
            None,
        )
    }

    fn complete_tool(
        &self,
        call: &ToolCall,
        outcome: ToolOutcome,
        system_error: Option<crate::ToolSystemError>,
    ) -> ExecutedCall {
        let error_kind = outcome.error_kind();
        let output_stats = outcome.output_stats();
        let metadata = outcome.metadata().clone();
        let result = outcome.into_result(call);
        self.emit(AgentEvent::ToolCompleted {
            result: result.clone(),
            error_kind,
            output_stats,
            metadata,
        });
        ExecutedCall {
            result,
            system_error,
        }
    }

    fn emit(&self, event: AgentEvent) {
        self.events.emit(event);
    }
}

struct ExecutedCall {
    result:       ToolResult,
    system_error: Option<crate::ToolSystemError>,
}

impl ExecutedCall {
    const fn completed(result: ToolResult) -> Self {
        Self {
            result,
            system_error: None,
        }
    }
}

struct ToolRoundExecution {
    results:      Vec<ToolResult>,
    system_error: Option<crate::ToolSystemError>,
}

impl ToolRoundExecution {
    fn from_calls(calls: Vec<ExecutedCall>) -> Self {
        let mut results = Vec::with_capacity(calls.len());
        let mut system_error = None;
        for call in calls {
            results.push(call.result);
            if system_error.is_none() {
                system_error = call.system_error;
            }
        }
        Self {
            results,
            system_error,
        }
    }
}

enum StreamResult {
    Completed(Box<Response>),
    Interrupted,
}

struct AgentStreamObserver<'a> {
    events: &'a EventHub,
}

impl StreamObserver for AgentStreamObserver<'_> {
    fn first_output(&self, kind: FirstOutputKind) {
        self.emit(AgentEvent::FirstOutput { kind });
    }

    fn text_delta(&self, delta: &str) {
        self.emit(AgentEvent::TextDelta {
            delta: delta.to_owned(),
        });
    }

    fn reasoning_delta(&self, delta: &str) {
        self.emit(AgentEvent::ReasoningDelta {
            delta: delta.to_owned(),
        });
    }

    fn output_replaced(&self) {
        self.emit(AgentEvent::OutputReplaced);
    }

    fn replay(&self, failed_attempt: u32, delay: Duration, error: &LlmError) {
        self.emit(AgentEvent::TurnReplay {
            failed_attempt,
            delay_seconds: delay.as_secs_f64(),
            error: error.data(),
        });
    }
}

impl AgentStreamObserver<'_> {
    fn emit(&self, event: AgentEvent) {
        self.events.emit(event);
    }
}

fn response_message(response: &Response) -> Message {
    Message::new(Role::Assistant, nonempty_content(response.content.clone()))
}

fn tool_results_message(results: &[ToolResult]) -> Message {
    let message = Message::new(
        Role::Tool,
        results.iter().cloned().map(ContentPart::ToolResult),
    );
    match results.first() {
        Some(result) => message.with_tool_call_id(result.tool_call_id.clone()),
        None => message,
    }
}

fn nonempty_content(mut content: Vec<ContentPart>) -> Vec<ContentPart> {
    if content.is_empty() {
        content.push(ContentPart::Text {
            text: String::new(),
        });
    }
    content
}

/// The result of a call that was cancelled before it started.
fn cancelled_tool_result(call: &ToolCall) -> ToolResult {
    ToolOutcome::failure(ToolErrorKind::Cancelled, CANCELLED).into_result(call)
}

fn tool_calls(response: &Response) -> Vec<ToolCall> {
    response
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use futures_util::stream;
    use lithos_llm::catalog::{ModelId, ProviderId};
    use lithos_llm::middleware::CallContext;
    use lithos_llm::types::{
        ErrorKind, ResponseStream, RetryClassification, StreamEvent, ToolDefinition,
    };
    use serde_json::json;
    use tokio::sync::Notify;
    use tokio::time::timeout;

    use super::*;
    use crate::tool::ToolError;
    use crate::{ToolDescriptor, ToolId};

    /// How long a test waits for a prompt another task has to unblock.
    const PATIENCE: Duration = Duration::from_secs(5);

    /// A dropped connection, which the client may repeat.
    fn dropped_stream() -> LlmError {
        LlmError::new(ErrorKind::Network, "connection reset").with_retry(RetryClassification::Safe)
    }

    struct ScriptedModel {
        responses: Mutex<VecDeque<Response>>,
        requests:  Option<Arc<Mutex<Vec<Request>>>>,
    }

    impl ScriptedModel {
        fn new(responses: impl IntoIterator<Item = Response>) -> Self {
            Self {
                responses: Mutex::new(responses.into_iter().collect()),
                requests:  None,
            }
        }

        fn recording(
            responses: impl IntoIterator<Item = Response>,
        ) -> (Self, Arc<Mutex<Vec<Request>>>) {
            let requests = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    responses: Mutex::new(responses.into_iter().collect()),
                    requests:  Some(Arc::clone(&requests)),
                },
                requests,
            )
        }
    }

    #[async_trait]
    impl ModelService for ScriptedModel {
        async fn stream(
            &self,
            request: Request,
            _context: CallContext,
        ) -> StdResult<ResponseStream, LlmError> {
            if let Some(requests) = &self.requests {
                requests
                    .lock()
                    .expect("the request lock is healthy")
                    .push(request);
            }
            let response = self
                .responses
                .lock()
                .expect("the script lock is healthy")
                .pop_front()
                .expect("the script has another response");
            Ok(ResponseStream::new(stream::iter([Ok(
                StreamEvent::Ended {
                    response: Box::new(response),
                },
            )])))
        }
    }

    fn response(content: impl IntoIterator<Item = ContentPart>) -> Response {
        Response::new(
            ProviderId::new("test"),
            ModelId::new("model"),
            content.into_iter().collect(),
        )
    }

    fn text_response(text: &str) -> Response {
        response([ContentPart::Text {
            text: text.to_owned(),
        }])
    }

    /// A tool that says when it starts and then waits for the test to release
    /// it or for its round to be cancelled.
    fn parking_tool(reached: Arc<Notify>, release: Arc<Notify>) -> Tool {
        Tool::function(
            "park",
            "Waits for the test",
            json!({"type": "object"}),
            move |context, _arguments| {
                let reached = Arc::clone(&reached);
                let release = Arc::clone(&release);
                async move {
                    reached.notify_one();
                    tokio::select! {
                        () = release.notified() => Ok("released".into()),
                        () = context.cancellation().cancelled() => {
                            Err(ToolError::new("cancelled"))
                        }
                    }
                }
            },
        )
        .expect("the parking tool is valid")
    }

    /// An agent whose first turn calls the parking tool and whose second
    /// answers `steered`, with the notifiers the test drives the tool through.
    fn parking_agent() -> (Agent, Arc<Notify>, Arc<Notify>) {
        let calls_tool = response([ContentPart::ToolCall(ToolCall::function(
            "call_1",
            "park",
            json!({}),
        ))]);
        let reached = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let agent = Agent::builder(
            ScriptedModel::new([calls_tool, text_response("steered")]),
            "test/model",
        )
        .tools([parking_tool(Arc::clone(&reached), Arc::clone(&release))])
        .build()
        .expect("the agent builds");
        (agent, reached, release)
    }

    /// Everything the receiver already holds.
    fn drained(receiver: &mut broadcast::Receiver<AgentEvent>) -> Vec<AgentEvent> {
        let mut events = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            events.push(event);
        }
        events
    }

    /// Where the first event matching `matcher` was published.
    fn position(events: &[AgentEvent], matcher: impl Fn(&AgentEvent) -> bool) -> Option<usize> {
        events.iter().position(matcher)
    }

    struct DynamicTools {
        hidden_executions: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ToolService for DynamicTools {
        async fn discover(
            &self,
            context: TurnContext<'_>,
        ) -> StdResult<ToolCatalog, crate::ToolSystemError> {
            let names = if context.turn() == 0 {
                ["allowed", "hidden"].as_slice()
            } else {
                ["later"].as_slice()
            };
            Ok(ToolCatalog::new(names.iter().map(|name| {
                ToolDescriptor::new(
                    ToolId::try_new(*name).expect("the test identity is valid"),
                    ToolDefinition::function(*name, *name, json!({})),
                )
            })))
        }

        async fn call(
            &self,
            request: ToolCallRequest,
        ) -> StdResult<ToolOutcome, crate::ToolSystemError> {
            if request.descriptor().id().as_str() == "hidden" {
                self.hidden_executions.fetch_add(1, Ordering::SeqCst);
            }
            Ok(ToolOutcome::success(request.call().name.clone().into()))
        }
    }

    struct HideTool;

    #[async_trait]
    impl ToolMiddleware for HideTool {
        async fn discover(
            &self,
            context: TurnContext<'_>,
            next: crate::ToolDiscoveryNext<'_>,
        ) -> StdResult<ToolCatalog, crate::ToolSystemError> {
            let mut catalog = next.run(context).await?;
            catalog.retain(|tool| tool.id().as_str() != "hidden");
            Ok(catalog)
        }

        async fn call(
            &self,
            request: ToolCallRequest,
            next: crate::ToolCallNext<'_>,
        ) -> StdResult<ToolOutcome, crate::ToolSystemError> {
            if request.descriptor().id().as_str() == "hidden" {
                Ok(ToolOutcome::failure(
                    ToolErrorKind::Denied,
                    "hidden for this turn",
                ))
            } else {
                next.run(request).await
            }
        }
    }

    struct RecordingToolMiddleware {
        trace: Arc<Mutex<Vec<&'static str>>>,
    }

    #[async_trait]
    impl ToolMiddleware for RecordingToolMiddleware {
        async fn call(
            &self,
            request: ToolCallRequest,
            next: crate::ToolCallNext<'_>,
        ) -> StdResult<ToolOutcome, crate::ToolSystemError> {
            self.trace
                .lock()
                .expect("the trace lock is healthy")
                .push("middleware:before");
            let outcome = next.run(request).await;
            self.trace
                .lock()
                .expect("the trace lock is healthy")
                .push("middleware:after");
            outcome
        }
    }

    struct BackgroundBoundary {
        trace: Arc<Mutex<Vec<String>>>,
    }

    struct PreparingLifecycle {
        prepared: AtomicBool,
    }

    #[async_trait]
    impl AgentLifecycle for PreparingLifecycle {
        async fn before_model(
            &self,
            _context: TurnContext<'_>,
            _cancel: &CancellationToken,
        ) -> StdResult<ConversationUpdate, crate::LifecycleError> {
            if self.prepared.swap(true, Ordering::SeqCst) {
                return Ok(ConversationUpdate::unchanged());
            }
            Ok(ConversationUpdate::replace(vec![Message::new(
                Role::User,
                [ContentPart::Text {
                    text: "prepared".to_owned(),
                }],
            )]))
        }
    }

    struct RecordingConversation {
        commits: Arc<Mutex<Vec<String>>>,
    }

    impl ConversationProjection for RecordingConversation {
        fn user_message_committed(&self, message: &Message, _attribution: Option<&Value>) {
            self.commits
                .lock()
                .expect("the commit lock is healthy")
                .push(format!("user:{}", message_text_content(message)));
        }

        fn assistant_message_committed(&self, response: &Response) {
            self.commits
                .lock()
                .expect("the commit lock is healthy")
                .push(format!("assistant:{}", response.text()));
        }

        fn tool_results_committed(
            &self,
            calls: &[ToolCall],
            results: &[ToolResult],
            cancelled: bool,
        ) {
            self.commits
                .lock()
                .expect("the commit lock is healthy")
                .push(format!(
                    "tools:{}:{}:{cancelled}",
                    calls.len(),
                    results.len()
                ));
        }

        fn conversation_replaced(&self, messages: &[Message]) {
            self.commits
                .lock()
                .expect("the commit lock is healthy")
                .push(format!("replace:{}", messages.len()));
        }
    }

    fn message_text_content(message: &Message) -> String {
        message
            .content()
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[async_trait]
    impl AgentLifecycle for BackgroundBoundary {
        async fn before_model(
            &self,
            context: TurnContext<'_>,
            _cancel: &CancellationToken,
        ) -> StdResult<crate::ConversationUpdate, crate::LifecycleError> {
            self.trace
                .lock()
                .expect("the trace lock is healthy")
                .push(format!("before:{}", context.turn()));
            Ok(crate::ConversationUpdate::unchanged())
        }

        async fn after_model(
            &self,
            context: TurnContext<'_>,
            _response: &Response,
            _cancel: &CancellationToken,
        ) -> StdResult<crate::ConversationUpdate, crate::LifecycleError> {
            self.trace
                .lock()
                .expect("the trace lock is healthy")
                .push(format!("after:{}", context.turn()));
            Ok(crate::ConversationUpdate::unchanged())
        }

        async fn after_answer(
            &self,
            context: TurnContext<'_>,
            _response: &Response,
            _cancel: &CancellationToken,
        ) -> StdResult<AfterAnswerAction, crate::LifecycleError> {
            self.trace
                .lock()
                .expect("the trace lock is healthy")
                .push(format!("answer:{}", context.turn()));
            Ok(if context.turn() == 0 {
                AfterAnswerAction::ContinueWith(UserMessage::text("background result"))
            } else {
                AfterAnswerAction::Complete
            })
        }
    }

    #[tokio::test]
    async fn prompt_returns_the_final_response() {
        let mut agent = Agent::builder(ScriptedModel::new([text_response("done")]), "test/model")
            .build()
            .expect("the agent builds");

        let outcome = agent.prompt("work").await.expect("the prompt succeeds");

        assert_eq!(outcome.text(), "done");
        assert_eq!(outcome.turn_count(), 1);
        assert_eq!(outcome.tool_call_count(), 0);
        assert_eq!(agent.messages().len(), 2);
        assert_eq!(agent.state(), AgentState::Idle);
    }

    /// A conversation as a failed model call leaves it: the model asked for a
    /// tool, the tool answered, and the next call never came back.
    fn conversation_paused_on_a_tool_result() -> Vec<Message> {
        vec![
            Message::text(Role::User, "work"),
            Message::new(Role::Assistant, [ContentPart::ToolCall(
                ToolCall::function("call_1", "inspect", json!({})),
            )]),
            Message::new(Role::Tool, [ContentPart::ToolResult(ToolResult {
                tool_call_id: "call_1".to_owned(),
                name:         Some("inspect".to_owned()),
                content:      vec![ContentPart::Text {
                    text: "inspected".to_owned(),
                }],
                is_error:     false,
            })])
            .with_tool_call_id("call_1"),
        ]
    }

    #[tokio::test]
    async fn an_unfinished_prompt_continues_from_its_committed_tool_results() {
        let ran = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&ran);
        let tool = Tool::function(
            "inspect",
            "Inspect",
            json!({"type": "object"}),
            move |_context, _arguments| {
                counted.fetch_add(1, Ordering::SeqCst);
                async { Ok("inspected".into()) }
            },
        )
        .expect("the inspect tool is valid");
        let (model, requests) = ScriptedModel::recording([text_response("finished")]);
        let mut agent = Agent::builder(model, "test/model")
            .tools([tool])
            .messages(conversation_paused_on_a_tool_result())
            .build()
            .expect("the agent builds");
        let mut events = agent.subscribe();

        let outcome = agent
            .continue_prompt()
            .await
            .expect("the continuation succeeds");

        assert_eq!(outcome.text(), "finished");
        assert_eq!(outcome.turn_count(), 1);
        assert_eq!(outcome.tool_call_count(), 0);
        assert_eq!(
            ran.load(Ordering::SeqCst),
            0,
            "the answered call is not run again"
        );
        let requests = requests.lock().expect("the request lock is healthy");
        assert_eq!(requests.len(), 1);
        let roles: Vec<Role> = requests[0].messages().iter().map(Message::role).collect();
        assert_eq!(roles, [Role::User, Role::Assistant, Role::Tool]);
        assert_eq!(agent.messages().len(), 4, "only the answer was committed");
        assert_eq!(agent.messages()[3].role(), Role::Assistant);
        assert_eq!(agent.state(), AgentState::Idle);
        let events = drained(&mut events);
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AgentEvent::UserMessage { .. })),
            "no user message was committed: {events:?}"
        );
        assert!(matches!(events.first(), Some(AgentEvent::PromptStarted)));
    }

    #[tokio::test]
    async fn a_finished_conversation_has_nothing_to_continue() {
        let mut agent = Agent::builder(ScriptedModel::new([text_response("done")]), "test/model")
            .build()
            .expect("the agent builds");
        let empty = agent.continue_prompt().await;
        assert!(
            matches!(empty, Err(AgentError::NothingToContinue)),
            "{empty:?}"
        );

        agent.prompt("work").await.expect("the prompt succeeds");
        let mut events = agent.subscribe();

        let answered = agent.continue_prompt().await;

        assert!(
            matches!(answered, Err(AgentError::NothingToContinue)),
            "{answered:?}"
        );
        assert_eq!(agent.messages().len(), 2, "nothing was committed");
        assert_eq!(agent.state(), AgentState::Idle);
        assert!(drained(&mut events).is_empty(), "no prompt was started");
    }

    #[tokio::test]
    async fn lifecycle_updates_and_conversation_commits_are_explicit() {
        let calls_tool = response([ContentPart::ToolCall(ToolCall::function(
            "call_1",
            "inspect",
            json!({}),
        ))]);
        let (model, requests) = ScriptedModel::recording([calls_tool, text_response("finished")]);
        let tool = Tool::function(
            "inspect",
            "Inspect",
            json!({}),
            |_context, _arguments| async { Ok("done".into()) },
        )
        .expect("the inspect tool is valid");
        let commits = Arc::new(Mutex::new(Vec::new()));
        let mut agent = Agent::builder(model, "test/model")
            .tools([tool])
            .lifecycle(Arc::new(PreparingLifecycle {
                prepared: AtomicBool::new(false),
            }))
            .conversation_projection(Arc::new(RecordingConversation {
                commits: Arc::clone(&commits),
            }))
            .build()
            .expect("the agent builds");

        agent.prompt("original").await.expect("the prompt succeeds");

        let requests = requests.lock().expect("the request lock is healthy");
        assert_eq!(message_text_content(&requests[0].messages()[0]), "prepared");
        assert_eq!(agent.messages().len(), 4);
        assert_eq!(*commits.lock().expect("the commit lock is healthy"), [
            "user:original",
            "replace:1",
            "assistant:",
            "tools:1:1:false",
            "assistant:finished",
        ]);
    }

    #[tokio::test]
    async fn tools_run_until_the_model_answers() {
        let calls_tool = response([ContentPart::ToolCall(ToolCall::function(
            "call_1",
            "inspect",
            json!({"name": "parser"}),
        ))]);
        let tool =
            Tool::function(
                "inspect",
                "Inspect a value",
                json!({"type": "object"}),
                |_context, arguments| async move {
                    Ok(format!("inspected {}", arguments["name"]).into())
                },
            )
            .expect("the inspect tool is valid");
        let mut agent = Agent::builder(
            ScriptedModel::new([calls_tool, text_response("finished")]),
            "test/model",
        )
        .tools([tool])
        .build()
        .expect("the agent builds");

        let outcome = agent.prompt("work").await.expect("the prompt succeeds");

        assert_eq!(outcome.text(), "finished");
        assert_eq!(outcome.turn_count(), 2);
        assert_eq!(outcome.tool_call_count(), 1);
        assert_eq!(agent.messages().len(), 4);
        let tool_message = &agent.messages()[2];
        assert_eq!(tool_message.role(), Role::Tool);
        assert!(matches!(
            &tool_message.content()[0],
            ContentPart::ToolResult(result) if !result.is_error
        ));
    }

    /// fabro-9c79: a model that retries the same failing tool call forever
    /// is stopped by the consecutive all-error round budget. The prompt
    /// fails with the diagnostics after exactly the budget's rounds; one
    /// successful result resets the count.
    #[tokio::test]
    async fn consecutive_all_error_rounds_end_the_prompt_with_diagnostics() {
        let mut script: Vec<Response> = (0..10)
            .map(|index| counting_call(&format!("call_{index}")))
            .collect();
        script.push(text_response("done"));
        let executions = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&executions);
        let failing = Tool::function(
            "count",
            "Counts its executions",
            json!({"type": "object"}),
            move |_context, _arguments| {
                let executions = Arc::clone(&counter);
                async move {
                    executions.fetch_add(1, Ordering::SeqCst);
                    Err(ToolError::new("the tool returned an unsupported outcome"))
                }
            },
        )
        .expect("the failing counting tool is valid");
        let mut agent = Agent::builder(ScriptedModel::new(script), "test/model")
            .tools([failing])
            .config(AgentConfig {
                max_consecutive_error_rounds: 3,
                ..AgentConfig::default()
            })
            .build()
            .expect("the agent builds");
        let mut events = agent.subscribe();

        let error = agent.prompt("work").await.expect_err("the budget trips");
        let AgentError::ConsecutiveToolErrors {
            rounds,
            first_error,
        } = &error
        else {
            panic!("expected ConsecutiveToolErrors, got {error:?}");
        };
        assert_eq!(*rounds, 3, "exactly the budget's rounds ran");
        assert!(
            first_error.contains("unsupported outcome"),
            "the diagnostics name the first error, got {first_error}"
        );
        assert_eq!(executions.load(Ordering::SeqCst), 3);
        let saw_event = &mut false;
        while let Ok(event) = events.try_recv() {
            if matches!(&event, AgentEvent::ConsecutiveToolErrors { rounds: 3, .. }) {
                *saw_event = true;
            }
        }
        assert!(*saw_event, "the diagnostic event was published");
    }

    /// A response that asks for the counting tool once.
    fn counting_call(id: &str) -> Response {
        response([ContentPart::ToolCall(ToolCall::function(
            id,
            "count",
            json!({}),
        ))])
    }

    /// A tool that records how many times it ran.
    fn counting_tool(executions: Arc<AtomicUsize>) -> Tool {
        Tool::function(
            "count",
            "Counts its executions",
            json!({"type": "object"}),
            move |_context, _arguments| {
                let executions = Arc::clone(&executions);
                async move {
                    executions.fetch_add(1, Ordering::SeqCst);
                    Ok("counted".into())
                }
            },
        )
        .expect("the counting tool is valid")
    }

    /// An agent under `budget` whose model asks for the counting tool
    /// `tool_turns` times in a row and then answers `done`, plus the counter
    /// its tool bumps and the agent's event stream.
    fn budgeted_agent(
        budget: Option<usize>,
        tool_turns: usize,
        answers_afterwards: bool,
    ) -> (Agent, Arc<AtomicUsize>, broadcast::Receiver<AgentEvent>) {
        let mut script: Vec<Response> = (0..tool_turns)
            .map(|index| counting_call(&format!("call_{index}")))
            .collect();
        if answers_afterwards {
            script.push(text_response("done"));
        }
        let executions = Arc::new(AtomicUsize::new(0));
        let agent = Agent::builder(ScriptedModel::new(script), "test/model")
            .tools([counting_tool(Arc::clone(&executions))])
            .config(AgentConfig {
                max_tool_rounds: budget,
                ..AgentConfig::default()
            })
            .build()
            .expect("the agent builds");
        let events = agent.subscribe();
        (agent, executions, events)
    }

    /// Checks the shape of a prompt that ran out of tool rounds under `limit`:
    /// exactly `limit` rounds ran, the refused turn is committed with its
    /// calls answered as `Cancelled`, the event is published once, and the
    /// agent is idle again.
    async fn assert_exhausted_after(limit: usize) {
        let (mut agent, executions, mut events) = budgeted_agent(Some(limit), limit + 1, false);

        let error = agent
            .prompt("work")
            .await
            .expect_err("the budget ends the prompt");

        assert!(
            matches!(error, AgentError::ToolRoundsExhausted { limit: seen } if seen == limit),
            "unexpected error under a limit of {limit}: {error:?}"
        );
        assert_eq!(executions.load(Ordering::SeqCst), limit);
        // The input, then one assistant turn and one tool-result turn for
        // every round that ran and for the refused one.
        let messages = agent.messages();
        assert_eq!(messages.len(), 1 + 2 * (limit + 1));
        let refused = messages
            .last()
            .expect("the refused turn's results are committed");
        assert_eq!(refused.role(), Role::Tool);
        assert!(matches!(
            &refused.content()[0],
            ContentPart::ToolResult(result)
                if result.is_error && result.tool_call_id == format!("call_{limit}")
        ));
        let events = drained(&mut events);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentEvent::ToolStarted { .. }))
                .count(),
            limit,
            "the refused calls never start"
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(
                    event,
                    AgentEvent::ToolRoundsExhausted { limit: seen } if *seen == limit
                ))
                .count(),
            1
        );
        assert!(
            !events.iter().any(|event| matches!(
                event,
                AgentEvent::PromptCompleted { .. } | AgentEvent::PromptAborted
            )),
            "an exhausted prompt neither completes nor aborts: {events:?}"
        );
        assert_eq!(agent.state(), AgentState::Idle);
    }

    #[tokio::test]
    async fn a_zero_round_budget_refuses_the_first_tool_turn() {
        assert_exhausted_after(0).await;
    }

    #[tokio::test]
    async fn one_tool_round_runs_before_the_budget_ends_the_prompt() {
        assert_exhausted_after(1).await;
    }

    #[tokio::test]
    async fn several_tool_rounds_run_before_the_budget_ends_the_prompt() {
        assert_exhausted_after(3).await;
    }

    #[tokio::test]
    async fn an_answer_within_the_budget_completes_the_prompt() {
        let (mut agent, executions, mut events) = budgeted_agent(Some(2), 2, true);

        let outcome = agent.prompt("work").await.expect("the prompt succeeds");

        assert_eq!(outcome.text(), "done");
        assert_eq!(outcome.turn_count(), 3);
        assert_eq!(outcome.tool_call_count(), 2);
        assert_eq!(executions.load(Ordering::SeqCst), 2);
        assert!(
            !drained(&mut events)
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolRoundsExhausted { .. }))
        );
    }

    #[tokio::test]
    async fn the_budget_starts_over_with_each_prompt() {
        let executions = Arc::new(AtomicUsize::new(0));
        let mut agent = Agent::builder(
            ScriptedModel::new([
                counting_call("call_a"),
                text_response("first"),
                counting_call("call_b"),
                text_response("second"),
            ]),
            "test/model",
        )
        .tools([counting_tool(Arc::clone(&executions))])
        .config(AgentConfig {
            max_tool_rounds: Some(1),
            ..AgentConfig::default()
        })
        .build()
        .expect("the agent builds");

        let first = agent.prompt("one").await.expect("the first prompt fits");
        let second = agent.prompt("two").await.expect("the second prompt fits");

        assert_eq!(first.text(), "first");
        assert_eq!(second.text(), "second");
        assert_eq!(executions.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn without_a_budget_tool_rounds_are_unlimited() {
        let (mut agent, executions, mut events) = budgeted_agent(None, 12, true);

        let outcome = agent.prompt("work").await.expect("the prompt succeeds");

        assert_eq!(outcome.text(), "done");
        assert_eq!(outcome.tool_call_count(), 12);
        assert_eq!(executions.load(Ordering::SeqCst), 12);
        assert!(
            !drained(&mut events)
                .iter()
                .any(|event| matches!(event, AgentEvent::ToolRoundsExhausted { .. }))
        );
    }

    #[tokio::test]
    async fn tool_discovery_and_policy_run_for_each_turn() {
        let calls_hidden = response([ContentPart::ToolCall(ToolCall::function(
            "call_1",
            "hidden",
            json!({}),
        ))]);
        let (model, requests) = ScriptedModel::recording([calls_hidden, text_response("finished")]);
        let hidden_executions = Arc::new(AtomicUsize::new(0));
        let mut agent = Agent::builder(model, "test/model")
            .tool_service(Arc::new(DynamicTools {
                hidden_executions: Arc::clone(&hidden_executions),
            }))
            .tool_middleware(Arc::new(HideTool))
            .build()
            .expect("the agent builds");

        agent.prompt("work").await.expect("the prompt succeeds");

        assert_eq!(hidden_executions.load(Ordering::SeqCst), 0);
        let requests = requests.lock().expect("the request lock is healthy");
        let names = |request: &Request| {
            request
                .tools()
                .iter()
                .map(|tool| tool.name.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&requests[0]), ["allowed".to_owned()]);
        assert_eq!(names(&requests[1]), ["later".to_owned()]);
        let ContentPart::ToolResult(result) = &agent.messages()[2].content()[0] else {
            panic!("the denied tool call has a result");
        };
        assert!(result.is_error);
        assert!(matches!(
            &result.content[0],
            ContentPart::Text { text } if text == "hidden for this turn"
        ));
    }

    #[tokio::test]
    async fn tool_middleware_is_inside_kernel_events() {
        let calls_tool = response([ContentPart::ToolCall(ToolCall::function(
            "call_1",
            "inspect",
            json!({}),
        ))]);
        let trace = Arc::new(Mutex::new(Vec::new()));
        let projected = Arc::clone(&trace);
        let projection = Arc::new(move |event: &AgentEvent| {
            let item = match event {
                AgentEvent::ToolStarted { .. } => Some("event:start"),
                AgentEvent::ToolCompleted { .. } => Some("event:complete"),
                _ => None,
            };
            if let Some(item) = item {
                projected
                    .lock()
                    .expect("the trace lock is healthy")
                    .push(item);
            }
        });
        let executed = Arc::clone(&trace);
        let tool = Tool::function(
            "inspect",
            "inspect",
            json!({}),
            move |_context, _arguments| {
                executed
                    .lock()
                    .expect("the trace lock is healthy")
                    .push("tool");
                async { Ok("done".into()) }
            },
        )
        .expect("the test tool is valid");
        let mut agent = Agent::builder(
            ScriptedModel::new([calls_tool, text_response("finished")]),
            "test/model",
        )
        .tools([tool])
        .tool_middleware(Arc::new(RecordingToolMiddleware {
            trace: Arc::clone(&trace),
        }))
        .event_projection(projection)
        .build()
        .expect("the agent builds");

        agent.prompt("work").await.expect("the prompt succeeds");

        assert_eq!(*trace.lock().expect("the trace lock is healthy"), [
            "event:start",
            "middleware:before",
            "tool",
            "middleware:after",
            "event:complete",
        ]);
    }

    #[tokio::test]
    async fn an_exclusive_tool_runs_alone() {
        let calls_tool = response([
            ContentPart::ToolCall(ToolCall::function("call_1", "question", json!({}))),
            ContentPart::ToolCall(ToolCall::function("call_2", "peer", json!({}))),
        ]);
        let executions = Arc::new(Mutex::new(Vec::new()));
        let question_executions = Arc::clone(&executions);
        let question = Tool::function(
            "question",
            "question",
            json!({}),
            move |_context, _arguments| {
                question_executions
                    .lock()
                    .expect("the execution lock is healthy")
                    .push("question");
                async { Ok("answered".into()) }
            },
        )
        .expect("the question tool is valid")
        .with_scheduling(ToolScheduling::ExclusiveRound);
        let peer_executions = Arc::clone(&executions);
        let peer = Tool::function("peer", "peer", json!({}), move |_context, _arguments| {
            peer_executions
                .lock()
                .expect("the execution lock is healthy")
                .push("peer");
            async { Ok("unexpected".into()) }
        })
        .expect("the peer tool is valid");
        let mut agent = Agent::builder(
            ScriptedModel::new([calls_tool, text_response("finished")]),
            "test/model",
        )
        .tools([question, peer])
        .build()
        .expect("the agent builds");

        agent.prompt("work").await.expect("the prompt succeeds");

        assert_eq!(
            *executions.lock().expect("the execution lock is healthy"),
            ["question"]
        );
        let ContentPart::ToolResult(result) = &agent.messages()[2].content()[1] else {
            panic!("the peer call has a result");
        };
        assert!(result.is_error);
        assert!(matches!(
            &result.content[0],
            ContentPart::Text { text } if text.contains("exclusive tool")
        ));
    }

    #[tokio::test]
    async fn lifecycle_can_continue_with_a_background_result() {
        let trace = Arc::new(Mutex::new(Vec::new()));
        let mut agent = Agent::builder(
            ScriptedModel::new([text_response("first"), text_response("second")]),
            "test/model",
        )
        .lifecycle(Arc::new(BackgroundBoundary {
            trace: Arc::clone(&trace),
        }))
        .build()
        .expect("the agent builds");

        let outcome = agent.prompt("work").await.expect("the prompt succeeds");

        assert_eq!(outcome.text(), "second");
        assert_eq!(agent.messages().len(), 4);
        assert_eq!(*trace.lock().expect("the trace lock is healthy"), [
            "before:0", "after:0", "answer:0", "before:1", "after:1", "answer:1"
        ]);
    }

    /// Aborts the prompt once, from the stage that runs after the assistant
    /// turn is committed and before its tool calls are answered.
    struct AbortingBoundary {
        control: AgentControlHandle,
        fired:   AtomicBool,
    }

    #[async_trait]
    impl AgentLifecycle for AbortingBoundary {
        async fn after_model(
            &self,
            _context: TurnContext<'_>,
            _response: &Response,
            _cancel: &CancellationToken,
        ) -> StdResult<crate::ConversationUpdate, crate::LifecycleError> {
            if !self.fired.swap(true, Ordering::SeqCst) {
                assert!(self.control.abort(), "a prompt is running to abort");
            }
            Ok(crate::ConversationUpdate::unchanged())
        }
    }

    #[tokio::test]
    async fn an_abort_during_after_model_answers_the_tool_call_as_cancelled() {
        let calls_tool = response([ContentPart::ToolCall(ToolCall::function(
            "call_1",
            "count",
            json!({}),
        ))]);
        let (model, requests) = ScriptedModel::recording([calls_tool, text_response("done")]);
        let executions = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&executions);
        let counting_tool = Tool::function(
            "count",
            "Counts its runs",
            json!({"type": "object"}),
            move |_context, _arguments| {
                let counted = Arc::clone(&counted);
                async move {
                    counted.fetch_add(1, Ordering::SeqCst);
                    Ok("ran".into())
                }
            },
        )
        .expect("the count tool is valid");
        let control = AgentControlHandle::detached();
        let mut agent = Agent::builder(model, "test/model")
            .control_handle(control.clone())
            .tools([counting_tool])
            .lifecycle(Arc::new(AbortingBoundary {
                control,
                fired: AtomicBool::new(false),
            }))
            .build()
            .expect("the agent builds");
        let mut events = agent.subscribe();

        let error = agent
            .prompt("work")
            .await
            .expect_err("the prompt was aborted");

        assert!(matches!(error, AgentError::Aborted), "{error:?}");
        assert_eq!(
            executions.load(Ordering::SeqCst),
            0,
            "a tool the caller stopped before never runs"
        );
        // The abort waits for the turn's calls to be answered, so what it
        // leaves behind is a conversation the next request can carry.
        let messages = agent.messages();
        assert_eq!(messages.len(), 3, "input, tool call, cancelled result");
        assert_eq!(messages[1].role(), Role::Assistant);
        assert_eq!(messages[2].role(), Role::Tool);
        let ContentPart::ToolResult(result) = &messages[2].content()[0] else {
            panic!("the cancelled call has a result");
        };
        assert_eq!(result.tool_call_id, "call_1");
        assert!(result.is_error);
        assert!(matches!(
            &result.content[0],
            ContentPart::Text { text } if text == "Cancelled"
        ));
        let events = drained(&mut events);
        assert!(
            position(&events, |event| matches!(
                event,
                AgentEvent::ToolStarted { .. }
            ))
            .is_none(),
            "a call that never started publishes nothing"
        );
        assert!(
            position(&events, |event| matches!(
                event,
                AgentEvent::TurnInterrupted { .. }
            ))
            .is_none(),
            "an abort is not a round interrupt"
        );
        assert!(position(&events, |event| matches!(event, AgentEvent::PromptAborted)).is_some());
        assert_eq!(agent.state(), AgentState::Idle);

        let outcome = agent
            .prompt("again")
            .await
            .expect("the next prompt succeeds");

        assert_eq!(outcome.text(), "done");
        let requests = requests.lock().expect("the request lock is healthy");
        let roles = requests[1]
            .messages()
            .iter()
            .map(Message::role)
            .collect::<Vec<_>>();
        assert_eq!(
            roles,
            [Role::User, Role::Assistant, Role::Tool, Role::User],
            "the next request carries the paired conversation"
        );
    }

    /// Fails the boundary that runs after the assistant turn is committed.
    struct FailingBoundary;

    #[async_trait]
    impl AgentLifecycle for FailingBoundary {
        async fn after_model(
            &self,
            _context: TurnContext<'_>,
            _response: &Response,
            _cancel: &CancellationToken,
        ) -> StdResult<crate::ConversationUpdate, crate::LifecycleError> {
            Err(crate::LifecycleError::new("the lifecycle failed"))
        }
    }

    #[tokio::test]
    async fn a_failing_after_model_stage_answers_the_tool_call_before_failing_the_prompt() {
        let calls_tool = response([ContentPart::ToolCall(ToolCall::function(
            "call_1",
            "count",
            json!({}),
        ))]);
        let executions = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&executions);
        let counting_tool = Tool::function(
            "count",
            "Counts its runs",
            json!({"type": "object"}),
            move |_context, _arguments| {
                let counted = Arc::clone(&counted);
                async move {
                    counted.fetch_add(1, Ordering::SeqCst);
                    Ok("ran".into())
                }
            },
        )
        .expect("the count tool is valid");
        let mut agent = Agent::builder(ScriptedModel::new([calls_tool]), "test/model")
            .tools([counting_tool])
            .lifecycle(Arc::new(FailingBoundary))
            .build()
            .expect("the agent builds");
        let mut events = agent.subscribe();

        let error = agent
            .prompt("work")
            .await
            .expect_err("the boundary failed the prompt");

        assert!(matches!(error, AgentError::Lifecycle { .. }), "{error:?}");
        assert_eq!(
            executions.load(Ordering::SeqCst),
            0,
            "a failed boundary runs no tool"
        );
        // The failure is reported only after the turn's calls are answered, so
        // the conversation it leaves is one the next request can carry.
        let messages = agent.messages();
        assert_eq!(messages.len(), 3, "input, tool call, cancelled result");
        assert_eq!(messages[1].role(), Role::Assistant);
        assert_eq!(messages[2].role(), Role::Tool);
        let ContentPart::ToolResult(result) = &messages[2].content()[0] else {
            panic!("the cancelled call has a result");
        };
        assert_eq!(result.tool_call_id, "call_1");
        assert!(result.is_error);
        assert!(matches!(
            &result.content[0],
            ContentPart::Text { text } if text == "Cancelled"
        ));
        let events = drained(&mut events);
        assert!(
            position(&events, |event| matches!(
                event,
                AgentEvent::ToolStarted { .. } | AgentEvent::TurnInterrupted { .. }
            ))
            .is_none(),
            "nothing started and nothing was interrupted"
        );
        assert_eq!(agent.state(), AgentState::Idle);
    }

    #[tokio::test]
    async fn invalid_tool_arguments_are_answered_without_running_the_tool() {
        let calls_tool = response([ContentPart::ToolCall(ToolCall::function(
            "call_1",
            "inspect",
            json!({}),
        ))]);
        let executions = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&executions);
        let tool = Tool::function(
            "inspect",
            "Inspect a value",
            json!({
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"]
            }),
            move |_context, _arguments| {
                observed.fetch_add(1, Ordering::SeqCst);
                async { Ok("unexpected".into()) }
            },
        )
        .expect("the inspect tool is valid");
        let mut agent = Agent::builder(
            ScriptedModel::new([calls_tool, text_response("corrected")]),
            "test/model",
        )
        .tools([tool])
        .build()
        .expect("the agent builds");

        agent.prompt("work").await.expect("the prompt succeeds");

        assert_eq!(executions.load(Ordering::SeqCst), 0);
        let ContentPart::ToolResult(result) = &agent.messages()[2].content()[0] else {
            panic!("the tool call has a result");
        };
        let ContentPart::Text { text } = &result.content[0] else {
            panic!("the validation result is text");
        };
        assert!(result.is_error);
        assert!(text.contains("missing required property"));
    }

    #[test]
    fn the_default_turn_replay_waits_seconds_not_milliseconds() {
        // One second, doubling, jittered. Jitter lands each wait between half
        // of the computed delay and the whole of it, so the three replays
        // together outlast a provider blip of a few seconds.
        let config = AgentConfig::default();
        let error = dropped_stream();
        let expected = [
            (1, Duration::from_millis(500), Duration::from_secs(1)),
            (2, Duration::from_secs(1), Duration::from_secs(2)),
            (3, Duration::from_secs(2), Duration::from_secs(4)),
        ];

        for (attempt, floor, ceiling) in expected {
            let delay = config
                .turn_replay
                .next_delay(attempt, &error)
                .unwrap_or_else(|| panic!("attempt {attempt} is replayed"));
            assert!(
                (floor..=ceiling).contains(&delay),
                "attempt {attempt} waited {delay:?}, outside {floor:?}..={ceiling:?}"
            );
        }
        assert!(
            config.turn_replay.next_delay(4, &error).is_none(),
            "the fourth failure spends the budget"
        );
    }

    #[test]
    fn the_default_turn_replay_caps_its_wait_at_a_minute() {
        // The default budget is spent long before the doubling reaches the
        // cap, so a wider budget on the same schedule shows it.
        let policy = AgentConfig::default().turn_replay.max_attempts(u32::MAX);

        let delay = policy
            .next_delay(20, &dropped_stream())
            .expect("a replay is still allowed");
        assert!(
            (Duration::from_secs(30)..=Duration::from_secs(60)).contains(&delay),
            "attempt 20 waited {delay:?}, outside the jittered sixty second cap"
        );
    }

    #[test]
    fn duplicate_tool_names_are_rejected() {
        let make_tool = || {
            Tool::function("same", "same", json!({}), |_context, _arguments| async {
                Ok("ok".into())
            })
            .expect("the test tool is valid")
        };
        let result = Agent::builder(ScriptedModel::new([]), "test/model")
            .tools([make_tool(), make_tool()])
            .build();
        let Err(error) = result else {
            panic!("duplicate names are ambiguous");
        };

        assert!(matches!(
            error,
            AgentBuildError::DuplicateTool { ref name } if name == "same"
        ));
    }

    #[tokio::test]
    async fn an_interrupt_with_steering_queued_delivers_the_steer_without_parking() {
        let (mut agent, reached, _release) = parking_agent();
        let control = agent.control_handle();
        let mut events = agent.subscribe();

        let controller = tokio::spawn(async move {
            reached.notified().await;
            assert!(control.enqueue_steering("change course"));
            assert!(control.interrupt(), "a prompt was running to interrupt");
            assert!(
                !control.is_paused(),
                "queued steering keeps the prompt from parking"
            );
            assert_eq!(control.pending_steering(), 1);
        });

        let outcome = timeout(PATIENCE, agent.prompt("start"))
            .await
            .expect("the queued steer resumes the prompt without a second gesture")
            .expect("the prompt succeeds");
        controller.await.expect("the controller finishes");

        assert_eq!(outcome.text(), "steered");
        assert_eq!(outcome.turn_count(), 2);
        let events = drained(&mut events);
        let interrupted = position(&events, |event| {
            matches!(event, AgentEvent::TurnInterrupted { .. })
        })
        .expect("the round was interrupted");
        let steered = position(&events, |event| {
            matches!(event, AgentEvent::SteeringMessage { .. })
        })
        .expect("the steer was delivered");
        assert!(
            interrupted < steered,
            "the steer opens the turn after the interrupted one"
        );
        assert_eq!(agent.messages().len(), 5);
        assert_eq!(agent.messages()[3].role(), Role::User);
        assert!(!agent.control_handle().is_paused());
    }

    #[tokio::test]
    async fn parking_for_steer_with_steering_queued_does_not_hang_the_prompt() {
        let (mut agent, reached, release) = parking_agent();
        let control = agent.control_handle();
        let mut events = agent.subscribe();

        let controller = tokio::spawn(async move {
            reached.notified().await;
            assert!(control.enqueue_steering("change course"));
            assert!(control.park_for_steer(), "a prompt was running to park");
            assert!(
                !control.is_paused(),
                "queued steering is what a park waits for"
            );
            assert_eq!(control.pending_steering(), 1);
            release.notify_one();
        });

        let outcome = timeout(PATIENCE, agent.prompt("start"))
            .await
            .expect("the queued steer resumes the prompt without a second gesture")
            .expect("the prompt succeeds");
        controller.await.expect("the controller finishes");

        assert_eq!(outcome.text(), "steered");
        assert_eq!(outcome.turn_count(), 2);
        let events = drained(&mut events);
        assert!(
            position(&events, |event| matches!(
                event,
                AgentEvent::TurnInterrupted { .. }
            ))
            .is_none(),
            "a park interrupts nothing"
        );
        assert!(
            position(&events, |event| matches!(
                event,
                AgentEvent::SteeringMessage { .. }
            ))
            .is_some(),
            "the steer was delivered"
        );
        assert_eq!(agent.messages().len(), 5);
        assert!(matches!(
            &agent.messages()[2].content()[0],
            ContentPart::ToolResult(result) if !result.is_error
        ));
        assert!(!agent.control_handle().is_paused());
    }

    #[tokio::test]
    async fn a_bare_interrupt_parks_until_steering_arrives() {
        let (mut agent, reached, _release) = parking_agent();
        let control = agent.control_handle();
        let mut events = agent.subscribe();

        let controller = tokio::spawn(async move {
            reached.notified().await;
            assert!(control.interrupt(), "a prompt was running to interrupt");
            assert!(control.is_paused(), "a bare interrupt parks");
            assert!(control.enqueue_steering("change course"));
            assert!(!control.is_paused(), "the steer releases the park");
        });

        let outcome = timeout(PATIENCE, agent.prompt("start"))
            .await
            .expect("the later steer resumes the parked prompt")
            .expect("the prompt succeeds");
        controller.await.expect("the controller finishes");

        assert_eq!(outcome.text(), "steered");
        let events = drained(&mut events);
        let interrupted = position(&events, |event| {
            matches!(event, AgentEvent::TurnInterrupted { .. })
        })
        .expect("the round was interrupted");
        let steered = position(&events, |event| {
            matches!(event, AgentEvent::SteeringMessage { .. })
        })
        .expect("the steer was delivered");
        assert!(interrupted < steered);
        assert!(!agent.control_handle().is_paused());
    }

    #[tokio::test]
    async fn steer_interrupts_and_delivers_in_one_gesture() {
        let (mut agent, reached, _release) = parking_agent();
        let control = agent.control_handle();
        let mut events = agent.subscribe();

        let controller = tokio::spawn(async move {
            reached.notified().await;
            assert!(control.steer("change course"));
            assert!(!control.is_paused(), "an atomic steer never parks");
            assert_eq!(control.pending_steering(), 1);
        });

        let outcome = timeout(PATIENCE, agent.prompt("start"))
            .await
            .expect("the steer resumes the prompt")
            .expect("the prompt succeeds");
        controller.await.expect("the controller finishes");

        assert_eq!(outcome.text(), "steered");
        assert_eq!(outcome.turn_count(), 2);
        let events = drained(&mut events);
        let interrupted = position(&events, |event| {
            matches!(event, AgentEvent::TurnInterrupted { .. })
        })
        .expect("the round was interrupted");
        let steered = position(&events, |event| {
            matches!(event, AgentEvent::SteeringMessage { .. })
        })
        .expect("the steer was delivered");
        assert!(interrupted < steered);
        assert_eq!(agent.messages().len(), 5);
        assert!(!agent.control_handle().is_paused());
    }

    #[tokio::test]
    async fn follow_up_is_processed_before_the_agent_becomes_idle() {
        let mut agent = Agent::builder(
            ScriptedModel::new([text_response("first"), text_response("second")]),
            "test/model",
        )
        .build()
        .expect("the agent builds");
        assert!(agent.follow_up("next"));

        let outcome = agent.prompt("start").await.expect("the prompt succeeds");

        assert_eq!(outcome.text(), "second");
        assert_eq!(outcome.turn_count(), 2);
        assert_eq!(agent.messages().len(), 4);
    }

    #[tokio::test]
    async fn reset_clears_history_and_queued_input() {
        let mut agent = Agent::builder(ScriptedModel::new([text_response("done")]), "test/model")
            .build()
            .expect("the agent builds");
        assert!(agent.follow_up("later"));
        assert!(agent.steer("change course"));

        assert!(agent.reset());
        let outcome = agent.prompt("start").await.expect("the prompt succeeds");

        assert_eq!(outcome.turn_count(), 1);
        assert_eq!(agent.messages().len(), 2);
    }

    #[tokio::test]
    async fn shutdown_refuses_new_prompts() {
        let mut agent = Agent::builder(ScriptedModel::new([]), "test/model")
            .build()
            .expect("the agent builds");

        assert!(agent.shutdown());
        assert!(!agent.shutdown());
        assert!(matches!(
            agent.prompt("work").await,
            Err(AgentError::Closed)
        ));
    }
}
