mod client;
mod conversions;

use crate::client::{ChatCompletionChunk, CompletionsApi, CompletionsRequest, StreamOptions};
use crate::conversions::{
    convert_finish_reason, convert_tool_call, convert_usage, messages_to_request, process_response,
    tool_results_to_messages,
};
use golem_llm::config::with_config_key;
use golem_llm::durability::{DurableLLM, ExtendedGuest};
use golem_llm::event_source::{Event, EventSource, MessageEvent};
use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, Config, ContentPart, Error, ErrorCode, FinishReason, Guest,
    GuestChatStream, Message, Pollable, ResponseMetadata, StreamDelta, StreamEvent, ToolCall,
    ToolResult,
};
use golem_llm::LOGGING_STATE;
use log::trace;
use std::cell::RefCell;
use std::task::Poll;

struct GrokChatStream {
    stream: RefCell<Option<EventSource>>,
    failure: Option<Error>,
    finished: RefCell<bool>,
    finish_reason: RefCell<Option<FinishReason>>,
}

impl GrokChatStream {
    pub fn new(stream: EventSource) -> Self {
        GrokChatStream {
            stream: RefCell::new(Some(stream)),
            failure: None,
            finished: RefCell::new(false),
            finish_reason: RefCell::new(None),
        }
    }

    pub fn failed(error: Error) -> Self {
        GrokChatStream {
            stream: RefCell::new(None),
            failure: Some(error),
            finished: RefCell::new(false),
            finish_reason: RefCell::new(None),
        }
    }

    fn decode_message(&self, raw: &str) -> Result<Option<StreamEvent>, String> {
        trace!("Received raw stream event: {raw}");
        let json: serde_json::Value = serde_json::from_str(raw)
            .map_err(|err| format!("Failed to deserialize stream event: {err}"))?;

        let typ = json
            .as_object()
            .and_then(|obj| obj.get("object"))
            .and_then(|v| v.as_str());
        match typ {
            Some("chat.completion.chunk") => {
                let message: ChatCompletionChunk = serde_json::from_value(json)
                    .map_err(|err| format!("Failed to parse stream event: {err}"))?;
                if let Some(choice) = message.choices.into_iter().next() {
                    if let Some(finish_reason) = choice.finish_reason {
                        *self.finish_reason.borrow_mut() =
                            Some(convert_finish_reason(&finish_reason));
                    }
                    Ok(Some(StreamEvent::Delta(StreamDelta {
                        content: choice
                            .delta
                            .content
                            .map(|text| vec![ContentPart::Text(text)]),
                        tool_calls: choice
                            .delta
                            .tool_calls
                            .map(|calls| calls.iter().map(convert_tool_call).collect()),
                    })))
                } else if let Some(usage) = message.usage {
                    let finish_reason = self.finish_reason.borrow();
                    Ok(Some(StreamEvent::Finish(ResponseMetadata {
                        finish_reason: *finish_reason,
                        usage: Some(convert_usage(&usage)),
                        provider_id: None,
                        timestamp: Some(message.created.to_string()),
                        provider_metadata_json: None,
                    })))
                } else {
                    Ok(None)
                }
            }
            Some(_) => Ok(None),
            None => Err("Unexpected stream event format, does not have 'object' field".to_string()),
        }
    }
}

impl GuestChatStream for GrokChatStream {
    fn get_next(&self) -> Option<Vec<StreamEvent>> {
        if *self.finished.borrow() {
            return Some(vec![]);
        }

        let mut stream = self.stream.borrow_mut();
        if let Some(stream) = stream.as_mut() {
            match stream.poll_next() {
                Poll::Ready(None) => {
                    *self.finished.borrow_mut() = true;
                    Some(vec![])
                }
                Poll::Ready(Some(Err(golem_llm::event_source::error::Error::StreamEnded))) => {
                    *self.finished.borrow_mut() = true;
                    Some(vec![])
                }
                Poll::Ready(Some(Err(error))) => Some(vec![StreamEvent::Error(Error {
                    code: ErrorCode::InternalError,
                    message: error.to_string(),
                    provider_error_json: None,
                })]),
                Poll::Ready(Some(Ok(event))) => {
                    let mut events = vec![];

                    match event {
                        Event::Open => {}
                        Event::Message(MessageEvent { data, .. }) => {
                            if data != "[DONE]" {
                                match self.decode_message(&data) {
                                    Ok(Some(stream_event)) => {
                                        if matches!(stream_event, StreamEvent::Finish(_)) {
                                            *self.finished.borrow_mut() = true;
                                        }
                                        events.push(stream_event);
                                    }
                                    Ok(None) => {
                                        // Ignored event
                                    }
                                    Err(error) => {
                                        events.push(StreamEvent::Error(Error {
                                            code: ErrorCode::InternalError,
                                            message: error,
                                            provider_error_json: None,
                                        }));
                                    }
                                }
                            }
                        }
                    }

                    if events.is_empty() {
                        None
                    } else {
                        Some(events)
                    }
                }
                Poll::Pending => None,
            }
        } else if let Some(error) = self.failure.clone() {
            *self.finished.borrow_mut() = true;
            Some(vec![StreamEvent::Error(error)])
        } else {
            None
        }
    }

    fn blocking_get_next(&self) -> Vec<StreamEvent> {
        let pollable = self.subscribe();
        let mut result = Vec::new();
        loop {
            pollable.block();
            match self.get_next() {
                Some(events) => {
                    result.extend(events);
                    break result;
                }
                None => continue,
            }
        }
    }

    fn subscribe(&self) -> Pollable {
        if let Some(stream) = self.stream.borrow().as_ref() {
            stream.subscribe()
        } else {
            golem_rust::bindings::wasi::clocks::monotonic_clock::subscribe_duration(0)
        }
    }
}

struct GrokComponent;

impl GrokComponent {
    const ENV_VAR_NAME: &'static str = "XAI_API_KEY";

    fn request(client: CompletionsApi, request: CompletionsRequest) -> ChatEvent {
        match client.send_messages(request) {
            Ok(response) => process_response(response),
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn streaming_request(
        client: CompletionsApi,
        mut request: CompletionsRequest,
    ) -> GrokChatStream {
        request.stream = Some(true);
        request.stream_options = Some(StreamOptions {
            include_usage: true,
        });
        match client.stream_send_messages(request) {
            Ok(stream) => GrokChatStream::new(stream),
            Err(err) => GrokChatStream::failed(err),
        }
    }
}

impl Guest for GrokComponent {
    type ChatStream = GrokChatStream;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENV_VAR_NAME, ChatEvent::Error, |xai_api_key| {
            let client = CompletionsApi::new(xai_api_key);

            match messages_to_request(messages, config) {
                Ok(request) => Self::request(client, request),
                Err(err) => ChatEvent::Error(err),
            }
        })
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENV_VAR_NAME, ChatEvent::Error, |xai_api_key| {
            let client = CompletionsApi::new(xai_api_key);

            match messages_to_request(messages, config) {
                Ok(mut request) => {
                    request
                        .messages
                        .extend(tool_results_to_messages(tool_results));
                    Self::request(client, request)
                }
                Err(err) => ChatEvent::Error(err),
            }
        })
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        ChatStream::new(Self::unwrapped_stream(messages, config))
    }
}

impl ExtendedGuest for GrokComponent {
    fn unwrapped_stream(messages: Vec<Message>, config: Config) -> GrokChatStream {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENV_VAR_NAME, GrokChatStream::failed, |xai_api_key| {
            let client = CompletionsApi::new(xai_api_key);

            match messages_to_request(messages, config) {
                Ok(request) => Self::streaming_request(client, request),
                Err(err) => GrokChatStream::failed(err),
            }
        })
    }
}

type DurableGrokComponent = DurableLLM<GrokComponent>;

golem_llm::export_llm!(DurableGrokComponent with_types_in golem_llm);
