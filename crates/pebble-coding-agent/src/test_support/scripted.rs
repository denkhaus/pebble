//! A model that answers from a script.
//!
//! [`ScriptedProvider`] is a lithos [`ProviderAdapter`] registered on a real
//! [`Client`], so a session under test runs the same code path it runs against
//! a provider: the client resolves the route, builds the call, and hands the
//! session a stream. What differs is only what comes back, which is whatever
//! the script says — text, tool calls, reasoning, a stream that breaks
//! mid-turn, or a call that never answers at all.
//!
//! ```
//! use pebble_coding_agent::test_support::{ScriptedCall, scripted_client, text_response};
//!
//! let (client, provider) = scripted_client(vec![ScriptedCall::response(text_response("done"))]);
//! # let _ = (client, provider.call_count());
//! ```
//!
//! Every [`Request`] the script was asked with is captured, so a test can
//! assert on what the session sent as well as on what it did with the answer.

use std::future::pending;
use std::result::Result as StdResult;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use futures_util::{StreamExt as _, stream};
use lithos_llm::Client;
use lithos_llm::adapter::{ProviderAdapter, ResolvedCall};
use lithos_llm::catalog::{AdapterId, Catalog, ModelId, ProviderId};
use lithos_llm::client::{ClientBuild, ClientBuilder};
use lithos_llm::types::{
    ContentBlockId, ContentBlockKind, ContentPart, Cost, CostSource, Error as LlmError,
    ErrorKind as LlmErrorKind, FinishReason, Message, ReasoningContent, Request, Response,
    ResponseStream, RetryClassification, StreamEvent, TokenCounts, ToolCall, ToolCallKind,
};
use serde_json::{Value, json};
use tokio::sync::Notify;
use tokio::time::sleep;

use crate::types::message_text as message_text_of;

/// The catalog the scripted client resolves against.
///
/// Two providers, because a session's harness is chosen from catalog metadata
/// and the interesting cases are what a row does and does not say: `test`
/// names a profile at both levels, and `bare` names none anywhere.
pub const TEST_CATALOG: &str = r#"
schema_version = 1

[providers.test]
display_name = "Test"
adapter = "test-adapter"
codecs = ["test-codec"]
base_url = "http://127.0.0.1"
default_model = "model"

[providers.test.auth]
type = "none"

[providers.test.metadata.agent]
profile = "openai"

[providers.test.models.model]
display_name = "Test model"
api_model = "model"
capabilities = { text = true, tools = true }
limits = { context_tokens = 200000, max_output_tokens = 32000 }
knowledge_cutoff = "May 2026"

[providers.test.models.model.metadata.agent]
profile = "anthropic"

[providers.test.models.vision]
display_name = "Test vision model"
api_model = "vision"
capabilities = { text = true, images = true, tools = true }
limits = { context_tokens = 200000, max_output_tokens = 32000 }

[providers.test.models.vision.metadata.agent]
profile = "anthropic"

[providers.test.models.thinking]
display_name = "Thinking model"
api_model = "thinking"
capabilities = { text = true, tools = true, reasoning = true }
protocol_options = { reasoning_effort_levels = true }
limits = { context_tokens = 200000, max_output_tokens = 32000 }

[providers.test.models.thinking.metadata.agent]
profile = "anthropic"

[providers.test.models.always-thinking]
display_name = "Always thinking"
api_model = "always-thinking"
capabilities = { text = true, tools = true, reasoning = true }
limits = { context_tokens = 200000, max_output_tokens = 32000 }

# A model whose capabilities cannot say that it always reasons: it takes a
# thinking budget rather than a named effort level, so the row says so itself.
[providers.test.models.always-thinking.metadata.agent]
profile = "anthropic"
reasoning_by_default = true

[providers.test.models.small]
display_name = "Small window"
api_model = "small"
capabilities = { text = true, tools = true }
limits = { context_tokens = 100, max_output_tokens = 100 }

[providers.test.models.small.metadata.agent]
profile = "anthropic"

[providers.test.models.inherited]
display_name = "Inherited"
api_model = "inherited"

# A tools-capable row that names no profile of its own, so it inherits the
# provider's `openai`: the openai-wire arm of harness-conditional behavior
# (the tool-error wrap, fabro-c98c) with tools in play.
[providers.test.models.openai-tools]
display_name = "OpenAI-wire tools model"
api_model = "openai-tools"
capabilities = { text = true, tools = true }
limits = { context_tokens = 200000, max_output_tokens = 32000 }

[providers.test.models.strange]
display_name = "Strange"
api_model = "strange"

[providers.test.models.strange.metadata.agent]
profile = "nonesuch"

[providers.bare]
display_name = "Bare"
adapter = "test-adapter"
codecs = ["test-codec"]
base_url = "http://127.0.0.1"
default_model = "plain"

[providers.bare.auth]
type = "none"

[providers.bare.models.plain]
display_name = "Plain"
api_model = "plain"
"#;

/// A model failure a script can produce, over and over.
///
/// A lithos [`Error`](LlmError) carries a boxed source and cannot be cloned, so
/// a script holds the facts of a failure and builds a fresh error each time the
/// call it belongs to is answered.
#[derive(Debug, Clone)]
pub struct ScriptedFailure {
    kind:          LlmErrorKind,
    message:       String,
    retry:         RetryClassification,
    status:        Option<u16>,
    provider_code: Option<String>,
}

impl ScriptedFailure {
    /// A failure the client and the session may both repeat.
    #[must_use]
    pub fn retryable(kind: LlmErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            retry: RetryClassification::Safe,
            status: None,
            provider_code: None,
        }
    }

    /// A failure that repeating would only produce again.
    #[must_use]
    pub fn terminal(kind: LlmErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            retry: RetryClassification::Never,
            status: None,
            provider_code: None,
        }
    }

    /// Records the HTTP status the provider answered with.
    #[must_use]
    pub fn with_status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }

    /// Records the provider's own error code.
    #[must_use]
    pub fn with_provider_code(mut self, code: impl Into<String>) -> Self {
        self.provider_code = Some(code.into());
        self
    }

    /// Builds the error this failure stands for.
    pub fn to_error(&self) -> LlmError {
        let mut error =
            LlmError::new(self.kind.clone(), self.message.clone()).with_retry(self.retry);
        if let Some(status) = self.status {
            error = error.with_status(status);
        }
        if let Some(code) = &self.provider_code {
            error = error.with_provider_code(code.clone());
        }
        error
    }
}

/// One item of a scripted stream: an event, or the failure that ends it.
pub type ScriptedItem = StdResult<StreamEvent, ScriptedFailure>;

/// What the scripted provider does with one call.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ScriptedCall {
    /// Streams a whole response: its text, its tool calls, then the response
    /// itself.
    Response(Box<Response>),
    /// Streams exactly these items, then ends.
    Events(Vec<ScriptedItem>),
    /// Streams these items and then never ends, so only a cancellation stops
    /// the reader.
    EventsThenPending(Vec<ScriptedItem>),
    /// Fails to open.
    Failure(ScriptedFailure),
    /// Never answers, so the call hangs until something cancels it.
    PendingOpen,
}

impl ScriptedCall {
    /// Streams one response.
    #[must_use]
    pub fn response(response: Response) -> Self {
        Self::Response(Box::new(response))
    }

    /// Streams one response's text, then breaks mid-stream.
    #[must_use]
    pub fn fails_after(text: &str, failure: ScriptedFailure) -> Self {
        let mut events = text_delta_events(text);
        events.push(Err(failure));
        Self::Events(events)
    }
}

/// A provider that answers from a script.
///
/// The last scripted call repeats once the script runs out, which is what lets
/// a test say "then answer this way from now on" without counting rounds.
#[derive(Debug)]
pub struct ScriptedProvider {
    id:                  AdapterId,
    calls:               Vec<ScriptedCall>,
    completions:         Vec<ScriptedCompletion>,
    delay:               Duration,
    requests:            Mutex<Vec<Request>>,
    completion_requests: Mutex<Vec<Request>>,
    call_index:          AtomicUsize,
    completion_index:    AtomicUsize,
    started:             Notify,
}

/// What the scripted provider answers one non-streaming call with.
///
/// Only compaction calls this path, so a script that never compacts can leave
/// it empty.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ScriptedCompletion {
    /// The response the call returns.
    Response(Box<Response>),
    /// The failure the call returns.
    Failure(ScriptedFailure),
    /// Never answers, so the call hangs until the caller gives up on it.
    Pending,
    /// Answers with the response once the gate is opened, so a test can act
    /// while the call is in flight.
    Gated {
        /// The response the call returns once the gate is open.
        response: Box<Response>,
        /// Opened with [`Notify::notify_one`]; stays open once opened.
        gate:     Arc<Notify>,
    },
}

impl ScriptedCompletion {
    /// Answers with one response.
    #[must_use]
    pub fn response(response: Response) -> Self {
        Self::Response(Box::new(response))
    }

    /// Answers with `response` once the returned gate is opened with
    /// [`Notify::notify_one`].
    ///
    /// An opened gate stays open, so a script that repeats this completion
    /// answers every later call at once.
    #[must_use]
    pub fn gated(response: Response) -> (Self, Arc<Notify>) {
        let gate = Arc::new(Notify::new());
        (
            Self::Gated {
                response: Box::new(response),
                gate:     Arc::clone(&gate),
            },
            gate,
        )
    }
}

impl ScriptedProvider {
    /// A provider that answers streaming calls from `calls`.
    #[must_use]
    pub fn new(calls: Vec<ScriptedCall>) -> Self {
        Self {
            id: AdapterId::new("test-adapter"),
            calls,
            completions: Vec::new(),
            delay: Duration::ZERO,
            requests: Mutex::new(Vec::new()),
            completion_requests: Mutex::new(Vec::new()),
            call_index: AtomicUsize::new(0),
            completion_index: AtomicUsize::new(0),
            started: Notify::new(),
        }
    }

    /// Scripts what the non-streaming calls — compaction's — answer with.
    #[must_use]
    pub fn completing(mut self, completions: Vec<ScriptedCompletion>) -> Self {
        self.completions = completions;
        self
    }

    /// Makes every call take `delay` to answer, so a test can tell inference
    /// time from tool time.
    #[must_use]
    pub fn delayed(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    /// How many streaming calls the script has answered.
    #[must_use]
    pub fn call_count(&self) -> usize {
        self.call_index.load(Ordering::SeqCst)
    }

    /// How many non-streaming calls the script has answered.
    #[must_use]
    pub fn completion_count(&self) -> usize {
        self.completion_index.load(Ordering::SeqCst)
    }

    /// Every streaming request the session sent, in order.
    #[must_use]
    pub fn requests(&self) -> Vec<Request> {
        self.requests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Every non-streaming request the session sent, in order.
    #[must_use]
    pub fn completion_requests(&self) -> Vec<Request> {
        self.completion_requests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Waits until a streaming call has begun.
    ///
    /// One permit per call is kept, so a test that asks after the fact is not
    /// left waiting for a call that already started.
    pub async fn wait_for_call(&self) {
        self.started.notified().await;
    }

    /// The scripted answer for the call at `index`, repeating the last one.
    fn call_at(&self, index: usize) -> Option<ScriptedCall> {
        if self.calls.is_empty() {
            return None;
        }
        Some(
            self.calls
                .get(index)
                .unwrap_or_else(|| &self.calls[self.calls.len() - 1])
                .clone(),
        )
    }
}

#[async_trait]
impl ProviderAdapter for ScriptedProvider {
    fn id(&self) -> &AdapterId {
        &self.id
    }

    async fn complete(&self, call: &ResolvedCall) -> StdResult<Response, LlmError> {
        self.completion_requests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(call.request().clone());
        let index = self.completion_index.fetch_add(1, Ordering::SeqCst);
        let scripted = self
            .completions
            .get(index)
            .or_else(|| self.completions.last())
            .cloned();
        match scripted {
            Some(ScriptedCompletion::Response(response)) => Ok(*response),
            Some(ScriptedCompletion::Failure(failure)) => Err(failure.to_error()),
            Some(ScriptedCompletion::Pending) => pending().await,
            Some(ScriptedCompletion::Gated { response, gate }) => {
                gate.notified().await;
                // Hand the permit back, so the gate stays open for the next
                // call to this completion.
                gate.notify_one();
                Ok(*response)
            }
            None => Err(LlmError::new(
                LlmErrorKind::Middleware,
                "this scripted provider was given no completion script",
            )),
        }
    }

    async fn stream(&self, call: &ResolvedCall) -> StdResult<ResponseStream, LlmError> {
        if !self.delay.is_zero() {
            sleep(self.delay).await;
        }
        self.requests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(call.request().clone());
        let index = self.call_index.fetch_add(1, Ordering::SeqCst);
        self.started.notify_one();

        match self.call_at(index) {
            Some(ScriptedCall::Response(response)) => Ok(ResponseStream::new(stream::iter(
                realize(events_for(&response)),
            ))),
            Some(ScriptedCall::Events(events)) => {
                Ok(ResponseStream::new(stream::iter(realize(events))))
            }
            Some(ScriptedCall::EventsThenPending(events)) => Ok(ResponseStream::new(
                stream::iter(realize(events)).chain(stream::pending()),
            )),
            Some(ScriptedCall::Failure(failure)) => Err(failure.to_error()),
            Some(ScriptedCall::PendingOpen) => pending().await,
            None => Ok(ResponseStream::new(stream::iter(Vec::new()))),
        }
    }
}

/// Turns scripted items into the stream items lithos delivers.
fn realize(items: Vec<ScriptedItem>) -> Vec<StdResult<StreamEvent, LlmError>> {
    items
        .into_iter()
        .map(|item| item.map_err(|failure| failure.to_error()))
        .collect()
}

/// The events that carry `text` to a reader, short of ending the block.
#[must_use]
pub fn text_delta_events(text: &str) -> Vec<ScriptedItem> {
    let block = ContentBlockId::new("block-text");
    vec![
        Ok(StreamEvent::ContentBlockStart {
            id:   block.clone(),
            kind: ContentBlockKind::Text,
        }),
        Ok(StreamEvent::TextDelta {
            id:   block,
            text: text.to_owned(),
        }),
    ]
}

/// The events that carry `text` as reasoning.
#[must_use]
pub fn reasoning_delta_events(text: &str) -> Vec<ScriptedItem> {
    let block = ContentBlockId::new("block-reasoning");
    vec![
        Ok(StreamEvent::ContentBlockStart {
            id:   block.clone(),
            kind: ContentBlockKind::Reasoning,
        }),
        Ok(StreamEvent::ReasoningDelta {
            id:   block,
            text: text.to_owned(),
        }),
    ]
}

/// The events that announce one tool call.
#[must_use]
pub fn tool_call_events(call: &ToolCall) -> Vec<ScriptedItem> {
    let block = ContentBlockId::new(format!("block-{}", call.id));
    vec![
        Ok(StreamEvent::ContentBlockStart {
            id:   block.clone(),
            kind: ContentBlockKind::ToolCall {
                id:   call.id.clone(),
                name: Some(call.name.clone()),
                kind: ToolCallKind::Function,
            },
        }),
        Ok(StreamEvent::ToolCallDelta {
            id:        block.clone(),
            arguments: call.input.raw().to_owned(),
        }),
        Ok(StreamEvent::ContentBlockEnd {
            id:   block,
            part: ContentPart::ToolCall(call.clone()),
        }),
    ]
}

/// Every event that streaming `response` produces, ending with the response
/// itself.
#[must_use]
pub fn events_for(response: &Response) -> Vec<ScriptedItem> {
    let mut events: Vec<ScriptedItem> = Vec::new();
    let text = response.text();
    if !text.is_empty() {
        events.extend(text_delta_events(&text));
        events.push(Ok(StreamEvent::ContentBlockEnd {
            id:   ContentBlockId::new("block-text"),
            part: ContentPart::Text { text },
        }));
    }
    for part in &response.content {
        if let ContentPart::ToolCall(call) = part {
            events.extend(tool_call_events(call));
        }
    }
    events.push(Ok(StreamEvent::Ended {
        response: Box::new(response.clone()),
    }));
    events
}

/// The token counts every scripted response reports.
fn scripted_usage() -> TokenCounts {
    TokenCounts {
        input: 10,
        output: 5,
        ..TokenCounts::default()
    }
}

/// A response that is only text.
#[must_use]
pub fn text_response(text: &str) -> Response {
    let mut response = Response::new(ProviderId::new("test"), ModelId::new("model"), vec![
        ContentPart::Text {
            text: text.to_owned(),
        },
    ]);
    response.id = Some(format!("resp_{text}"));
    response.finish_reason = FinishReason::Stop;
    response.usage = scripted_usage();
    response
}

/// A response that asks for one tool, with a sentence of text in front of it.
#[must_use]
pub fn tool_call_response(tool_name: &str, tool_call_id: &str, arguments: Value) -> Response {
    multi_tool_call_response(vec![(tool_name, tool_call_id, arguments)])
}

/// A response that asks for one custom tool, whose input is free-form text
/// rather than JSON.
///
/// `apply_patch` is the one built-in tool shaped this way: the model writes a
/// patch, not an object.
#[must_use]
pub fn custom_tool_call_response(tool_name: &str, tool_call_id: &str, input: &str) -> Response {
    let content = vec![
        ContentPart::Text {
            text: "Let me use a tool.".to_owned(),
        },
        ContentPart::ToolCall(ToolCall::custom(tool_call_id, tool_name, input)),
    ];
    let mut response = Response::new(ProviderId::new("test"), ModelId::new("model"), content);
    response.id = Some(format!("resp_{tool_call_id}"));
    response.finish_reason = FinishReason::ToolCall;
    response.usage = scripted_usage();
    response
}

/// A response that asks for several tools at once.
#[must_use]
pub fn multi_tool_call_response(calls: Vec<(&str, &str, Value)>) -> Response {
    let mut content = vec![ContentPart::Text {
        text: "Let me use a tool.".to_owned(),
    }];
    let first = calls
        .first()
        .map_or_else(|| "multi".to_owned(), |(_, id, _)| (*id).to_owned());
    for (tool_name, tool_call_id, arguments) in calls {
        content.push(ContentPart::ToolCall(ToolCall::function(
            tool_call_id,
            tool_name,
            arguments,
        )));
    }
    let mut response = Response::new(ProviderId::new("test"), ModelId::new("model"), content);
    response.id = Some(format!("resp_{first}"));
    response.finish_reason = FinishReason::ToolCall;
    response.usage = scripted_usage();
    response
}

/// A response carrying both readable reasoning channels.
#[must_use]
pub fn reasoning_response(text: &str, summary: &str, trace: &str) -> Response {
    let mut response = text_response(text);
    let mut content = vec![ContentPart::opaque(
        "openai_compatible.reasoning_details",
        json!([
            {"type": "reasoning.summary", "summary": summary},
            {"type": "reasoning.text", "text": trace},
        ]),
    )];
    content.extend(response.content);
    response.content = content;
    response
}

/// A response carrying an OpenAI Responses `reasoning` item the way the
/// lithos codec decodes one: a readable `Reasoning` part holding `trace` when
/// there is one, then the whole item as the opaque `openai.reasoning` part.
///
/// The codec derives `trace` from the item, so a test names it explicitly to
/// pin what the normalizer does with a given pairing rather than to mirror
/// the codec's join rule.
#[must_use]
pub fn responses_reasoning_response(text: &str, trace: Option<&str>, item: Value) -> Response {
    let mut response = text_response(text);
    let mut content = Vec::new();
    if let Some(trace) = trace {
        content.push(ContentPart::Reasoning(ReasoningContent {
            text:             trace.to_owned(),
            signature:        None,
            signature_origin: None,
            redacted:         false,
        }));
    }
    content.push(ContentPart::opaque("openai.reasoning", item));
    content.extend(response.content);
    response.content = content;
    response
}

/// The same response, with the token counts a test wants.
#[must_use]
pub fn with_usage(mut response: Response, usage: TokenCounts) -> Response {
    response.usage = usage;
    response
}

/// The same response, reporting `input` prompt tokens and nothing else.
#[must_use]
pub fn with_input_tokens(response: Response, input: u64) -> Response {
    with_usage(response, TokenCounts {
        input,
        ..TokenCounts::default()
    })
}

/// The same response, priced by the catalog at `usd_micros`.
#[must_use]
pub fn with_cost(mut response: Response, usd_micros: u64) -> Response {
    response.cost = Some(Cost {
        usd_micros,
        source: CostSource::Catalog,
    });
    response
}

/// The same response, ending for the given reason.
#[must_use]
pub fn with_finish_reason(mut response: Response, finish_reason: FinishReason) -> Response {
    response.finish_reason = finish_reason;
    response
}

/// The catalog the scripted client resolves against.
///
/// # Panics
///
/// Panics if [`TEST_CATALOG`] stops parsing, which is a bug in this crate.
#[must_use]
pub fn test_catalog() -> Catalog {
    Catalog::builder()
        .overlay_toml(TEST_CATALOG)
        .expect("the test catalog layer parses")
        .build()
        .expect("the test catalog validates")
}

/// A client whose two test providers both answer from one script.
///
/// # Panics
///
/// Panics if the client cannot be built, which is a bug in this crate.
#[must_use]
pub fn scripted_client(calls: Vec<ScriptedCall>) -> (Client, Arc<ScriptedProvider>) {
    client_from(ScriptedProvider::new(calls))
}

/// A client answering from `provider`, which the caller keeps a handle on.
///
/// # Panics
///
/// Panics if the client cannot be built, which is a bug in this crate.
#[must_use]
pub fn client_from(provider: ScriptedProvider) -> (Client, Arc<ScriptedProvider>) {
    let (builder, provider) = scripted_client_builder(provider);
    let ClientBuild { client, .. } = builder.build().expect("the scripted client builds");
    (client, provider)
}

/// A half-built client answering from `provider`, for a test that adds
/// middleware of its own.
pub fn scripted_client_builder(
    provider: ScriptedProvider,
) -> (ClientBuilder, Arc<ScriptedProvider>) {
    let provider = Arc::new(provider);
    let shared: Arc<dyn ProviderAdapter> = Arc::clone(&provider) as Arc<dyn ProviderAdapter>;
    let builder = Client::builder()
        .catalog(test_catalog())
        .adapter_arc("test", Arc::clone(&shared))
        .adapter_arc("bare", shared);
    (builder, provider)
}

/// The concatenated text of one request message.
#[must_use]
pub fn message_text(message: &Message) -> String {
    message_text_of(message)
}
