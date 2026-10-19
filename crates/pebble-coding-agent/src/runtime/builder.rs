//! Collects everything a session needs and builds a [`CodingRuntime`].
//!
//! The builder owns the tool registry: it asks the profile for its tools,
//! merges whatever the application registered, and freezes the result into
//! the session. Nothing mutates a registry afterwards. It also resolves the
//! model selector once, so every round of every prompt reaches the model the
//! catalog answered with.

use std::collections::HashMap;
use std::mem;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use lithos_llm::Client;
use lithos_llm::catalog::{Metadata, ModelHandle};
use lithos_llm::resolver::ResolvedRoute;
use lithos_llm::types::{ReasoningEffort, Request, Speed};
use pebble_agent::{AgentControlHandle, ToolMiddleware};
use serde::Deserialize;
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::turn::ConversationState;
use super::{CodingRuntime, InterruptReasonHandle, PromptResources, SessionModel, StateMachine};
use crate::coding_agent::CodingAgentBuildError;
use crate::compaction::CompactionControl;
use crate::config::CodingAgentOptions;
use crate::environment::Environment;
use crate::error::Result;
use crate::event::{Emitter, EventCapacity, EventOptions, EventPump, EventSink, EventSinkTimeout};
use crate::file_tracker::PromptFiles;
use crate::history::History;
use crate::human_input::HumanInputProvider;
use crate::policy::{CompactionPolicy, ContextPolicy};
use crate::profile::{AgentProfile, ModelFacts, SubagentSupport, builtin_profile};
use crate::profiles::{FileEditToolKind, ProfileDeps};
use crate::prompt_transform::SystemPromptTransform;
use crate::redact::{NoRedaction, Redactor};
use crate::search::seam::SearchProvider;
use crate::subagent::{
    ChildDeps, ChildIdentity, ChildObserver, OpenSessions, SubagentOptions, SubagentSupervisor,
};
use crate::tool::{
    NativeTool, PermissionLevelPolicy, PermissionMiddleware, RegisteredTool, StaticEnvProvider,
    ToolEnvProvider, ToolRegistry,
};
use crate::tools::{WebFetchSummarizer, make_question_tool, make_web_search_tool};
use crate::types::{AgentProfileKind, CodingAgentEvent, PermissionLevel};
use crate::{SessionId, SessionScope};

/// The catalog metadata namespace every agent runtime reads.
const METADATA_NAMESPACE: &str = "agent";

/// The `agent` namespace of a catalog entry.
///
/// The namespace is shared by every agent runtime that consumes the catalog.
/// Unknown keys are ignored, because the namespace grows and an older pebble
/// must keep reading a catalog a newer one wrote.
#[derive(Debug, Default, Deserialize)]
struct AgentMetadata {
    /// Which harness the model expects.
    #[serde(default)]
    profile:              Option<String>,
    /// Whether the model reasons without being asked to, where the capabilities
    /// alone do not say.
    #[serde(default)]
    reasoning_by_default: Option<bool>,
}

/// Collects everything a session needs and builds it.
///
/// The builder owns the tool registry: it asks the profile for its tools,
/// merges whatever the application registered, and freezes the result into the
/// session. Nothing mutates a registry afterwards.
#[must_use = "a builder does nothing until `build` is called"]
#[derive(Clone)]
pub(crate) struct CodingRuntimeBuilder {
    client:               Client,
    /// Set by the runtime when it resumes a record on a selector of its own.
    pub(super) model:     Option<String>,
    environment:          Option<Arc<dyn Environment>>,
    tools:                Vec<RegisteredTool>,
    tool_replacements:    Vec<(String, RegisteredTool)>,
    tool_middleware:      Vec<Arc<dyn ToolMiddleware>>,
    human_input:          Option<Arc<dyn HumanInputProvider>>,
    tool_env_provider:    Option<Arc<dyn ToolEnvProvider>>,
    redactor:             Arc<dyn Redactor>,
    web_fetch_summarizer: Option<String>,
    search_provider:      Option<Arc<dyn SearchProvider>>,
    options:              CodingAgentOptions,
    permission_level:     Option<PermissionLevel>,
    /// The runtime continues a record's event numbering through these.
    pub(super) events:    EventOptions,
    profile:              Option<Arc<dyn AgentProfile>>,
    prompt_transform:     Option<Arc<dyn SystemPromptTransform>>,
    context_policy:       Option<Arc<dyn ContextPolicy>>,
    compaction_policy:    Option<Arc<dyn CompactionPolicy>>,
    subagents:            SubagentOptions,
    child_observer:       Option<ChildObserver>,
    child:                Option<ChildIdentity>,
}

impl CodingRuntimeBuilder {
    /// Starts a session that talks to the model through `client`.
    pub(super) fn new(client: Client) -> Self {
        Self {
            client,
            model: None,
            environment: None,
            tools: Vec::new(),
            tool_replacements: Vec::new(),
            tool_middleware: Vec::new(),
            human_input: None,
            tool_env_provider: None,
            redactor: Arc::new(NoRedaction),
            web_fetch_summarizer: None,
            search_provider: None,
            options: CodingAgentOptions::default(),
            permission_level: None,
            events: EventOptions::default(),
            profile: None,
            prompt_transform: None,
            context_policy: None,
            compaction_policy: None,
            subagents: SubagentOptions::disabled(),
            child_observer: None,
            child: None,
        }
    }

    /// Lets the application adjust the system prompt the profile writes.
    ///
    /// The transform sees the default prompt and the context it was written
    /// from, and answers with the default, an addition, or a replacement. It
    /// applies to this root session only: a child runs its parent's profile and
    /// prompt.
    pub(crate) fn system_prompt_transform(
        mut self,
        transform: Arc<dyn SystemPromptTransform>,
    ) -> Self {
        self.prompt_transform = Some(transform);
        self
    }

    pub(crate) fn context_policy(mut self, policy: Arc<dyn ContextPolicy>) -> Self {
        self.context_policy = Some(policy);
        self
    }

    pub(crate) fn compaction_policy(mut self, policy: Arc<dyn CompactionPolicy>) -> Self {
        self.compaction_policy = Some(policy);
        self
    }

    /// Names the model, as the client's catalog spells it.
    ///
    /// Anything the catalog resolver accepts works — a model id, an alias, a
    /// `provider/model` pair, or `default`. The session pins whatever it
    /// resolves to, so every round of the prompt reaches the same model.
    pub(crate) fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets where the session's tools act.
    pub(crate) fn environment(mut self, environment: Arc<dyn Environment>) -> Self {
        self.environment = Some(environment);
        self
    }

    /// The environment the session will act through, once named.
    #[cfg(feature = "mcp")]
    pub(crate) const fn environment_ref(&self) -> Option<&Arc<dyn Environment>> {
        self.environment.as_ref()
    }

    /// Whether a model has been named on this builder.
    pub(crate) const fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Adds tools on top of the ones the profile contributes.
    ///
    /// The registry renames pebble's own tools into the profile's vocabulary as
    /// they arrive, so a built-in registered here still reaches the model under
    /// the name that model expects.
    pub(crate) fn tools(mut self, tools: impl IntoIterator<Item = RegisteredTool>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Records an explicit replacement of a registered identity.
    pub(crate) fn replace_tool(mut self, id: impl Into<String>, tool: RegisteredTool) -> Self {
        let id = id.into();
        if let Some((_, selected)) = self
            .tool_replacements
            .iter_mut()
            .find(|(key, _)| *key == id)
        {
            *selected = tool;
        } else {
            self.tool_replacements.push((id, tool));
        }
        self
    }

    /// Selects the built-in policy and records its level at build time.
    pub(crate) fn permission_level(mut self, level: PermissionLevel) -> Self {
        self.permission_level = Some(level);
        self
    }

    /// Adds one tool middleware to this session and its descendants.
    pub(crate) fn tool_middleware(mut self, middleware: Arc<dyn ToolMiddleware>) -> Self {
        self.tool_middleware.push(middleware);
        self
    }

    /// Sets where the session asks a person a question.
    ///
    /// Without one, no question tool is registered, so the model cannot park a
    /// prompt waiting for an answer nobody will give.
    pub(crate) fn human_input(mut self, provider: Arc<dyn HumanInputProvider>) -> Self {
        self.human_input = Some(provider);
        self
    }

    /// Sets where a tool call's extra environment variables come from.
    ///
    /// Resolved once per tool round, so a credential that expires mid-prompt is
    /// fetched again rather than reused.
    pub(crate) fn tool_env_provider(mut self, provider: Arc<dyn ToolEnvProvider>) -> Self {
        self.tool_env_provider = Some(provider);
        self
    }

    /// Sets fixed extra environment variables for every tool call.
    pub(crate) fn tool_env(self, env: HashMap<String, String>) -> Self {
        self.tool_env_provider(Arc::new(StaticEnvProvider(env)))
    }

    /// Sets what strips secrets out of the process output the session
    /// publishes.
    ///
    /// Pebble ships no secret detector, so without one the tail of a command's
    /// output reaches the event stream exactly as the command wrote it. What
    /// the model reads is never redacted: it is the same text the person at a
    /// terminal would have seen.
    pub(crate) fn redactor(mut self, redactor: Arc<dyn Redactor>) -> Self {
        self.redactor = redactor;
        self
    }

    /// Lets `web_fetch` answer a prompt about a page by asking `model`.
    ///
    /// The selector is resolved through this session's client, so the
    /// summarizing model can be smaller and cheaper than the one running the
    /// session. Without one, a `web_fetch` call carrying a prompt returns the
    /// page and says the summary was unavailable.
    ///
    /// The built-in profiles' fetch tool captures the summarizer when the
    /// profile is constructed; an application registering
    /// [`make_web_fetch_tool`](crate::tools::make_web_fetch_tool) itself passes
    /// a [`WebFetchSummarizer`](crate::tools::WebFetchSummarizer) directly
    /// instead.
    pub(crate) fn web_fetch_summarizer(mut self, model: impl Into<String>) -> Self {
        self.web_fetch_summarizer = Some(model.into());
        self
    }

    /// Sets where the session's web searches go.
    ///
    /// Pebble talks to no search engine of its own: the application implements
    /// [`SearchProvider`] over whatever it has. Registering one is the whole
    /// switch — the session advertises `web_search`, built on pebble's schema
    /// and answering in pebble's format whichever engine is underneath — and a
    /// session without one advertises no search tool, so a model is never told
    /// it can search and then refused. A child inherits its parent's provider.
    ///
    /// A profile whose model family expects a different search tool registers
    /// its own, which replaces this one.
    pub(crate) fn search_provider(mut self, provider: Arc<dyn SearchProvider>) -> Self {
        self.search_provider = Some(provider);
        self
    }

    /// Sets how the session behaves.
    pub(crate) fn options(mut self, options: CodingAgentOptions) -> Self {
        self.options = options;
        self
    }

    /// Keeps a predecessor's live subscribers: the new pipeline publishes on
    /// `published` instead of opening a channel of its own.
    pub(crate) fn continue_publishing_on(
        mut self,
        published: broadcast::Sender<CodingAgentEvent>,
    ) -> Self {
        self.events.published = Some(published);
        self
    }

    /// Replaces the request controls with a fallback route's, keeping every
    /// other option as it was.
    pub(crate) fn with_route_controls(
        mut self,
        reasoning_effort: Option<ReasoningEffort>,
        speed: Option<Speed>,
        max_tokens: Option<i64>,
    ) -> Self {
        self.options = self
            .options
            .with_reasoning_effort(reasoning_effort)
            .with_speed(speed)
            .with_max_tokens(max_tokens);
        self
    }

    /// Records every event durably before any subscriber sees it.
    ///
    /// A sink that refuses an event stops the prompt and closes the session: a
    /// session that cannot record what it did is worse than one that stops. The
    /// prompt that noticed reports [`Error::EventSink`]; every call after it
    /// reports [`Error::SessionClosed`].
    pub(crate) fn event_sink(mut self, sink: Arc<dyn EventSink>) -> Self {
        self.events.sink = Some(sink);
        self
    }

    /// Sets the pending event queue and live subscription capacity.
    pub(crate) fn event_capacity(mut self, capacity: impl Into<EventCapacity>) -> Self {
        self.events.capacity = capacity.into();
        self
    }

    /// Sets the longest one durable sink write may take.
    pub(crate) fn event_sink_timeout(mut self, timeout: impl Into<EventSinkTimeout>) -> Self {
        self.events.sink_timeout = timeout.into();
        self
    }

    /// Lets this session spawn children, built by `factory`.
    ///
    /// Registering a factory is the whole switch: the profile's subagent tools
    /// are registered only when there is one, and a session without one answers
    /// every spawn with a tool error. The builder also wires the supervisor to
    /// this session's event pipeline, so a child's events cannot be lost by an
    /// application that forgot to connect them.
    ///
    /// Pebble builds the [`ChildAgentSpec`](crate::subagent::ChildAgentSpec)
    /// each call receives from this session, so a child inherits the
    /// environment, the inheritable tools, the
    /// tool middleware its parent had, and never a
    /// [`HumanInputProvider`]: a child cannot ask a person a question.
    pub(crate) fn subagents(mut self, options: SubagentOptions) -> Self {
        self.subagents = options;
        self
    }

    /// Sees each child this session's tree builds, for the crate's own tests.
    #[cfg(test)]
    pub(crate) fn observe_children(mut self, observer: ChildObserver) -> Self {
        self.subagents = self.subagents.turned_on();
        self.child_observer = Some(observer);
        self
    }

    /// Builds this session as a child of another.
    ///
    /// Crate-internal: the identity, the depth, and the tree's open-session
    /// budget come from the spawning session, never from an application.
    pub(crate) fn child_of(mut self, child: ChildIdentity) -> Self {
        self.child = Some(child);
        self
    }

    /// Overrides the profile the catalog would select.
    ///
    /// Crate-internal on purpose: an application picks a harness by picking a
    /// model, never by naming one, so this exists for pebble's own tests and
    /// for a child session, which is built with the harness its parent already
    /// resolved.
    pub(crate) fn with_profile(mut self, profile: Arc<dyn AgentProfile>) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Builds the session.
    ///
    /// # Errors
    ///
    /// Returns [`CodingAgentBuildError`] when a dependency is missing, the
    /// model selector resolves to nothing, or the resolved model's catalog
    /// entry names no harness pebble can run.
    pub(crate) fn build(self) -> StdResult<CodingRuntime, CodingAgentBuildError> {
        let id = SessionId::fresh();
        let scope = match self.child.as_ref() {
            Some(child) => child.parent.child(id),
            None => SessionScope::root(id),
        };
        self.build_with_scope(scope, SystemTime::now())
    }

    /// Builds the session in `session_scope`, created at `created_at`: the
    /// model is resolved, the profile chosen, the event pipeline opened, the
    /// supervisor spawned when children are allowed, and the tool registry
    /// assembled, in that order.
    pub(super) fn build_with_scope(
        mut self,
        session_scope: SessionScope,
        created_at: SystemTime,
    ) -> StdResult<CodingRuntime, CodingAgentBuildError> {
        let id = session_scope.session_id().clone();
        // Restored children retain the same root-only integration rules as
        // children built by the supervisor.
        if !session_scope.is_root() {
            self.human_input = None;
        }
        if let Some(level) = self.permission_level {
            self.options.permission_level = Some(level);
            self.tool_middleware
                .push(Arc::new(PermissionMiddleware::new(Arc::new(
                    PermissionLevelPolicy::new(level),
                ))));
        }
        self.options
            .validate()
            .map_err(|source| CodingAgentBuildError::InvalidOptions { source })?;
        let selector = self
            .model
            .take()
            .ok_or(CodingAgentBuildError::MissingModel)?;
        let environment = self
            .environment
            .take()
            .ok_or(CodingAgentBuildError::MissingEnvironment)?;

        let model = ResolvedModel::resolve(&self.client, &selector)?;
        // Built here rather than inside the profile: the tool is the
        // application's answer — someone to ask — crossed with the harness's,
        // and a harness with no question tool of its own, Gemini, answers
        // `None` however the session was configured. The prompt reads it back
        // out of the registry rather than being told, because a child session
        // runs this same profile with nobody to ask.
        let question_tool = self
            .human_input
            .as_ref()
            .and_then(|_| make_question_tool(model.kind));
        let profile = self.profile(&model);

        // A child was placed in its tree by whoever spawned it; a root names
        // itself and starts the tree's budget.
        let depth = session_scope.depth();
        let (open_sessions, observer, inherited_emitter) = match self.child.take() {
            Some(child) => (
                child.open_sessions,
                child.observer,
                Some(child.event_emitter),
            ),
            None => (
                OpenSessions::root(self.subagents.limits()),
                self.child_observer.take(),
                None,
            ),
        };
        let (emitter, pump) = open_event_pipeline(
            mem::take(&mut self.events),
            inherited_emitter,
            &session_scope,
        );

        // One collector for the whole tree below this session: children report
        // what they touched into it, and the prompt's report reads it.
        let prompt_files = PromptFiles::default();
        let supervisor = self.subagents.is_enabled().then(|| {
            SubagentSupervisor::new(Arc::new(ChildDeps {
                parent_files: prompt_files.clone(),
                client: self.client.clone(),
                model_selector: model.handle.to_string(),
                profile: Arc::clone(&profile),
                environment: Arc::clone(&environment),
                tools: self.tools.clone(),
                tool_replacements: self.tool_replacements.clone(),
                tool_middleware: self.tool_middleware.clone(),
                context_policy: self.context_policy.clone(),
                compaction_policy: self.compaction_policy.clone(),
                options: child_options(&self.options, &self.subagents),
                tool_env_provider: self.tool_env_provider.clone(),
                redactor: Arc::clone(&self.redactor),
                search_provider: self.search_provider.clone(),
                event_emitter: emitter.clone(),
                observer,
                open_sessions,
            }))
        });
        let registry = self.assemble_registry(
            profile.as_ref(),
            question_tool,
            &SubagentSupport::new(depth, supervisor.clone()),
        )?;

        let state = StateMachine::new(emitter.clone(), id);
        let session = CodingRuntime {
            model_context: Arc::new(model.session_model(self.client)),
            resources: Arc::new(PromptResources {
                registry,
                skills: Vec::new(),
                skill_dirs: Vec::new(),
                system_prompt: String::new(),
                memory_tokens: 0,
                skills_tokens: 0,
            }),
            session_scope,
            created_at,
            config: self.options,
            conversation: Arc::new(Mutex::new(ConversationState::new(
                History::default(),
                prompt_files,
            ))),
            emitter,
            pump,
            state,
            ended: false,
            end_emitted: false,
            profile,
            knowledge_cutoff: model
                .route
                .model()
                .knowledge_cutoff()
                .unwrap_or_default()
                .to_owned(),
            tool_middleware: self.tool_middleware,
            env: environment,
            human_input: self.human_input,
            tool_env_provider: self.tool_env_provider,
            redactor: self.redactor,
            agent_control: AgentControlHandle::detached(),
            cancel_token: CancellationToken::new(),
            interrupt_reason: InterruptReasonHandle::default(),
            compaction: CompactionControl::default(),
            memory_summaries: Vec::new(),
            subagents: supervisor,
            prompt_transform: self.prompt_transform,
            context_policy: self.context_policy,
            compaction_policy: self.compaction_policy,
            coding_agent: None,
            coding_bridge: None,
            failover_outlook: None,
        };

        // Wired here rather than by the application: a supervisor with no
        // callback loses every child event, silently.
        if let Some(supervisor) = session.subagents.as_ref() {
            supervisor.set_event_callback(session.sub_agent_event_callback());
        }

        Ok(session)
    }

    /// The harness the session runs: the injected one, else the built-in
    /// profile the catalog named, constructed with what it captures.
    fn profile(&mut self, model: &ResolvedModel) -> Arc<dyn AgentProfile> {
        // Built before the profile, because the profile's `web_fetch` tool
        // captures it at construction the way a search tool captures its
        // engine.
        let web_fetch_summarizer = self
            .web_fetch_summarizer
            .take()
            .map(|model| Arc::new(WebFetchSummarizer::new(self.client.clone(), model)));
        let deps = ProfileDeps {
            provider_display_name: model.route.provider().display_name().to_owned(),
            file_edit_tool: FileEditToolKind::for_codecs(model.route.model().codecs()),
            search_provider: self.search_provider.clone(),
            web_fetch_summarizer,
        };
        self.profile
            .take()
            .unwrap_or_else(|| builtin_profile(model.kind, &deps))
    }

    /// The session's tools, frozen, in the order the model sees them: the
    /// profile's, the question tool, the application's, the profile's
    /// subagent family, and then the explicit replacements applied over the
    /// lot.
    fn assemble_registry(
        &self,
        profile: &dyn AgentProfile,
        question_tool: Option<RegisteredTool>,
        subagents: &SubagentSupport,
    ) -> StdResult<ToolRegistry, CodingAgentBuildError> {
        let mut registry = ToolRegistry::with_vocabulary(profile.tool_vocabulary());
        let profile_tools = profile.base_tools();
        // Built-in profiles contribute the search shape their models expect.
        // An injected profile that contributes no search tool gets the
        // canonical one, preserving the builder's public search-provider
        // contract without replacing a profile's own definition.
        if let Some(provider) = &self.search_provider
            && !profile_tools
                .iter()
                .any(|tool| tool.definition.name == NativeTool::WebSearch.canonical_name())
        {
            registry.register(make_web_search_tool(Arc::clone(provider)))?;
        }
        for tool in profile_tools {
            registry.register(tool)?;
        }
        // Root-only, and only where the application named somewhere to ask: a
        // child reports back to its parent rather than interrupting a person,
        // and a spec carries no `HumanInputProvider` for exactly that reason.
        if let Some(tool) = question_tool {
            registry.register(tool)?;
        }
        for tool in &self.tools {
            registry.register(tool.clone())?;
        }
        // Asked for whether or not there is a supervisor: a profile answers
        // with no tools when subagents are off, and this is the only place a
        // profile's subagent family reaches the registry.
        for tool in profile.subagent_tools(subagents) {
            registry.register(tool)?;
        }
        for (identity, tool) in &self.tool_replacements {
            registry.replace(identity, tool.clone())?;
        }
        Ok(registry)
    }
}

/// The model a selector resolved to, and what the catalog says about it.
struct ResolvedModel {
    route:  ResolvedRoute,
    handle: ModelHandle,
    facts:  ModelFacts,
    kind:   AgentProfileKind,
}

impl ResolvedModel {
    /// Resolves `selector` the way the session's own calls will, and reads
    /// the harness and the reasoning default off the catalog entry.
    fn resolve(client: &Client, selector: &str) -> StdResult<Self, CodingAgentBuildError> {
        let route = resolve_route(client, selector)?;
        let handle = route.handle();
        let metadata = effective_metadata(&route, &handle)?;
        // The catalog's capabilities say what the model can do; the `agent`
        // namespace is where a row says what it does by default, for the
        // always-reasoning models whose capabilities cannot tell.
        let mut facts = ModelFacts::from_catalog_model(route.model());
        if let Some(reasons_by_default) = metadata.reasoning_by_default {
            facts = facts.with_reasons_by_default(reasons_by_default);
        }
        // The catalog decides which harness the model expects whether or not
        // an implementation was injected, so a model that names none is
        // refused the same way either way.
        let kind = profile_kind(&metadata, &handle)?;
        Ok(Self {
            route,
            handle,
            facts,
            kind,
        })
    }

    /// The route pinned for the session's calls, through `client`.
    fn session_model(&self, client: Client) -> SessionModel {
        SessionModel {
            client,
            provider: self.handle.provider().as_str().to_owned(),
            model: self.handle.model().as_str().to_owned(),
            model_selector: self.handle.to_string(),
            facts: self.facts,
            profile_kind: self.kind,
        }
    }
}

/// The session's event pipeline: a child publishes on the emitter its tree
/// handed it and runs no pump of its own; a root opens the pipeline and
/// spawns the pump that drains it.
fn open_event_pipeline(
    events: EventOptions,
    inherited: Option<Emitter>,
    session_scope: &SessionScope,
) -> (Emitter, Option<JoinHandle<Result<()>>>) {
    if let Some(emitter) = inherited {
        return (emitter, None);
    }
    let (emitter, pump) = EventPump::new(events);
    let mut emitter = emitter.in_stream(session_scope.root_session_id().to_string());
    if let Some(parent) = session_scope.parent_session_id() {
        emitter = emitter.for_child(parent.as_str());
    }
    (emitter, Some(tokio::spawn(pump.run())))
}

/// What a child session inherits from its parent's options.
///
/// Everything that bounds or governs the child comes across unchanged — the
/// tool middleware, the permission level, the output budgets, the
/// wall-clock budget — so a factory cannot be handed anything wider than the
/// parent had. What does not come across by default is what the root loads
/// once: the memory files and the skill directories. A child is given a task,
/// not a project briefing, and paying for the briefing again in every child is
/// how a tree of agents spends a context window on nothing. An application
/// whose children must read the project's documents and see its skills the way
/// the root did says so on its [`SubagentOptions`], and the child then
/// initializes from the same paths its parent was given.
fn child_options(parent: &CodingAgentOptions, subagents: &SubagentOptions) -> CodingAgentOptions {
    CodingAgentOptions {
        memory_files: if subagents.inherits_memory() {
            parent.memory_files.clone()
        } else {
            Vec::new()
        },
        memory_discovery: if subagents.inherits_memory() {
            parent.memory_discovery.clone()
        } else {
            None
        },
        skill_dirs: if subagents.inherits_skills() {
            parent.skill_dirs.clone()
        } else {
            Vec::new()
        },
        skill_discovery: if subagents.inherits_skills() {
            parent.skill_discovery.clone()
        } else {
            None
        },
        ..parent.clone()
    }
}

/// Resolves the selector the way the session's own calls will.
fn resolve_route(
    client: &Client,
    selector: &str,
) -> StdResult<ResolvedRoute, CodingAgentBuildError> {
    let probe = Request::builder()
        .model(selector)
        .user("probe")
        .build()
        .map_err(|source| CodingAgentBuildError::Selector {
            selector: selector.to_owned(),
            source,
        })?;
    client
        .resolve_route(&probe)
        .map_err(|source| CodingAgentBuildError::ModelSelection {
            selector: selector.to_owned(),
            source,
        })
}

/// The `agent` namespace for a route, with the model's answers taking
/// precedence over the provider's.
///
/// Precedence is per member, not per namespace: a model that carries an
/// `agent` block saying only whether it reasons by default still takes its
/// profile from the provider.
fn effective_metadata(
    route: &ResolvedRoute,
    handle: &ModelHandle,
) -> StdResult<AgentMetadata, CodingAgentBuildError> {
    let model = read_metadata(route.model().metadata(), handle)?;
    let provider = read_metadata(route.provider().metadata(), handle)?;
    Ok(AgentMetadata {
        profile:              model.profile.or(provider.profile),
        reasoning_by_default: model.reasoning_by_default.or(provider.reasoning_by_default),
    })
}

fn read_metadata(
    metadata: &Metadata,
    handle: &ModelHandle,
) -> StdResult<AgentMetadata, CodingAgentBuildError> {
    metadata
        .namespace::<AgentMetadata>(METADATA_NAMESPACE)
        .map(Option::unwrap_or_default)
        .map_err(|source| CodingAgentBuildError::InvalidProfileMetadata {
            model: handle.to_string(),
            source,
        })
}

/// The harness the catalog says this model expects.
fn profile_kind(
    metadata: &AgentMetadata,
    handle: &ModelHandle,
) -> StdResult<AgentProfileKind, CodingAgentBuildError> {
    let named = metadata.profile.as_deref().ok_or_else(|| {
        CodingAgentBuildError::MissingProfileMetadata {
            model: handle.to_string(),
        }
    })?;
    AgentProfileKind::ALL
        .iter()
        .copied()
        .find(|kind| kind.as_str() == named)
        .ok_or_else(|| CodingAgentBuildError::UnknownProfile {
            model:   handle.to_string(),
            profile: named.to_owned(),
        })
}
