mod client;
mod conversions;

use crate::client::{
    Content, ContentBlockDelta, ErrorResponse, MessagesApi, MessagesRequest, StopReason, Usage,
};
use crate::conversions::{
    convert_usage, messages_to_request, process_response, stop_reason_to_finish_reason,
    tool_results_to_messages,
};
use golem_llm::config::with_config_key;
use golem_llm::durability::{DurableLLM, ExtendedGuest};
use golem_llm::event_source::{Event, EventSource, MessageEvent};
use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, Config, ContentPart, Error, ErrorCode, Guest, GuestChatStream, Message,
    Pollable, ResponseMetadata, Role, StreamDelta, StreamEvent, ToolCall, ToolResult,
};
use golem_llm::LOGGING_STATE;
use log::trace;
use std::cell::RefCell;
use std::collections::HashMap;
use std::task::Poll;

#[derive(Default)]
struct JsonFragment {
    id: String,
    name: String,
    json: String,
}

struct AnthropicChatStream {
    stream: RefCell<Option<EventSource>>,
    failure: Option<Error>,
    finished: RefCell<bool>,
    json_fragments: RefCell<HashMap<u64, JsonFragment>>,
    response_metadata: RefCell<ResponseMetadata>,
}

impl AnthropicChatStream {
    pub fn new(stream: EventSource) -> Self {
        AnthropicChatStream {
            stream: RefCell::new(Some(stream)),
            failure: None,
            finished: RefCell::new(false),
            json_fragments: RefCell::new(HashMap::new()),
            response_metadata: RefCell::new(ResponseMetadata {
                finish_reason: None,
                usage: None,
                provider_id: None,
                timestamp: None,
                provider_metadata_json: None,
            }),
        }
    }

    pub fn failed(error: Error) -> Self {
        AnthropicChatStream {
            stream: RefCell::new(None),
            failure: Some(error),
            finished: RefCell::new(false),
            json_fragments: RefCell::new(HashMap::new()),
            response_metadata: RefCell::new(ResponseMetadata {
                finish_reason: None,
                usage: None,
                provider_id: None,
                timestamp: None,
                provider_metadata_json: None,
            }),
        }
    }

    fn decode_message(&self, raw: &str) -> Result<Option<StreamEvent>, String> {
        trace!("Received raw stream event: {raw}");
        let json: serde_json::Value = serde_json::from_str(raw)
            .map_err(|err| format!("Failed to deserialize stream event: {err}"))?;

        let typ = json
            .as_object()
            .and_then(|obj| obj.get("type"))
            .and_then(|v| v.as_str());
        match typ {
            Some("error") => {
                let error = serde_json::from_value::<ErrorResponse>(json)
                    .map_err(|err| format!("Failed to deserialize stream event: {err}"))?;
                Ok(Some(StreamEvent::Error(Error {
                    code: ErrorCode::InternalError,
                    message: error.error.message,
                    provider_error_json: None,
                })))
            }
            Some("content_block_start") => {
                let index = json
                    .as_object()
                    .and_then(|obj| obj.get("index"))
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| {
                        "Unexpected stream event format, does not have 'index' field".to_string()
                    })?;

                let raw_content_block = json
                    .as_object()
                    .and_then(|obj| obj.get("content_block"))
                    .ok_or_else(|| {
                    "Unexpected stream event format, does not have 'content_block' field"
                        .to_string()
                })?;

                let content_block = serde_json::from_value::<Content>(raw_content_block.clone())
                    .map_err(|err| format!("Failed to deserialize stream event: {err}"))?;

                if let Content::ToolUse { id, name, .. } = content_block {
                    self.json_fragments.borrow_mut().insert(
                        index,
                        JsonFragment {
                            id,
                            name,
                            json: String::new(),
                        },
                    );
                }

                Ok(None)
            }
            Some("content_block_delta") => {
                let raw_delta = json
                    .as_object()
                    .and_then(|obj| obj.get("delta"))
                    .ok_or_else(|| {
                        "Unexpected stream event format, does not have 'delta' field".to_string()
                    })?;
                let delta = serde_json::from_value::<ContentBlockDelta>(raw_delta.clone())
                    .map_err(|err| format!("Failed to deserialize stream event: {err}"))?;

                match delta {
                    ContentBlockDelta::TextDelta { text } => {
                        Ok(Some(StreamEvent::Delta(StreamDelta {
                            content: Some(vec![ContentPart::Text(text)]),
                            tool_calls: None,
                        })))
                    }
                    ContentBlockDelta::InputJsonDelta { partial_json } => {
                        let index = json
                            .as_object()
                            .and_then(|obj| obj.get("index"))
                            .and_then(|v| v.as_u64())
                            .ok_or_else(|| {
                                "Unexpected stream event format, does not have 'index' field"
                                    .to_string()
                            })?;

                        let mut json_fragments = self.json_fragments.borrow_mut();
                        let fragment = json_fragments.entry(index).or_default();
                        fragment.json.push_str(&partial_json);

                        Ok(None)
                    }
                }
            }
            Some("content_block_stop") => {
                let index = json
                    .as_object()
                    .and_then(|obj| obj.get("index"))
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| {
                        "Unexpected stream event format, does not have 'index' field".to_string()
                    })?;

                if let Some(tool_use) = self.json_fragments.borrow_mut().remove(&index) {
                    Ok(Some(StreamEvent::Delta(StreamDelta {
                        content: None,
                        tool_calls: Some(vec![ToolCall {
                            id: tool_use.id,
                            name: tool_use.name,
                            arguments_json: tool_use.json,
                        }]),
                    })))
                } else {
                    Ok(None)
                }
            }
            Some("message_delta") => {
                let stop_reason = json
                    .as_object()
                    .and_then(|obj| obj.get("delta"))
                    .and_then(|v| v.as_object())
                    .and_then(|obj| obj.get("stop_reason"))
                    .and_then(|v| serde_json::from_value::<StopReason>(v.clone()).ok());
                let usage = json
                    .as_object()
                    .and_then(|obj| obj.get("usage"))
                    .and_then(|v| serde_json::from_value::<Usage>(v.clone()).ok());

                if let Some(stop_reason) = stop_reason {
                    self.response_metadata.borrow_mut().finish_reason =
                        Some(stop_reason_to_finish_reason(stop_reason));
                }
                if let Some(usage) = usage {
                    self.response_metadata.borrow_mut().usage = Some(convert_usage(usage));
                }
                Ok(None)
            }
            Some("message_stop") => {
                let response_metadata = self.response_metadata.borrow().clone();
                Ok(Some(StreamEvent::Finish(response_metadata)))
            }
            Some(_) => Ok(None),
            None => Err("Unexpected stream event format, does not have 'type' field".to_string()),
        }
    }
}

// TODO: probably all ChatStream implementations can be the same just with different decode_message implementation
impl GuestChatStream for AnthropicChatStream {
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

struct AnthropicComponent;

impl AnthropicComponent {
    const ENV_VAR_NAME: &'static str = "ANTHROPIC_API_KEY";

    fn request(client: MessagesApi, request: MessagesRequest) -> ChatEvent {
        match client.send_messages(request) {
            Ok(response) => process_response(response),
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn streaming_request(client: MessagesApi, mut request: MessagesRequest) -> AnthropicChatStream {
        request.stream = true;
        match client.stream_send_messages(request) {
            Ok(stream) => AnthropicChatStream::new(stream),
            Err(err) => AnthropicChatStream::failed(err),
        }
    }
}

impl Guest for AnthropicComponent {
    type ChatStream = AnthropicChatStream;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENV_VAR_NAME, ChatEvent::Error, |anthropic_api_key| {
            let client = MessagesApi::new(anthropic_api_key);

            match messages_to_request(messages, config) {
                Ok(request) => Self::request(client, request),
                Err(err) => return ChatEvent::Error(err),
            }
        })
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENV_VAR_NAME, ChatEvent::Error, |anthropic_api_key| {
            let client = MessagesApi::new(anthropic_api_key);

            match messages_to_request(messages, config) {
                Ok(mut request) => {
                    request
                        .messages
                        .extend(tool_results_to_messages(tool_results));
                    Self::request(client, request)
                }
                Err(err) => return ChatEvent::Error(err),
            }
        })
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        ChatStream::new(Self::unwrapped_stream(messages, config))
    }
}

impl ExtendedGuest for AnthropicComponent {
    fn unwrapped_stream(messages: Vec<Message>, config: Config) -> AnthropicChatStream {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(
            Self::ENV_VAR_NAME,
            AnthropicChatStream::failed,
            |anthropic_api_key| {
                let client = MessagesApi::new(anthropic_api_key);

                match messages_to_request(messages, config) {
                    Ok(request) => Self::streaming_request(client, request),
                    Err(err) => AnthropicChatStream::failed(err),
                }
            },
        )
    }

    fn retry_prompt(original_messages: &[Message], partial_result: &[StreamDelta]) -> Vec<Message> {
        let mut extended_messages = Vec::new();
        extended_messages.push(Message {
            role: Role::System,
            name: None,
            content: vec![
                ContentPart::Text(
                    "You were asked the same question previously, but the response was interrupted before completion. \
                     Please continue your response from where you left off. \
                     Do not include the part of the response that was already seen.".to_string()),
            ],
        });
        extended_messages.push(Message {
            role: Role::User,
            name: None,
            content: vec![ContentPart::Text(
                "Here is the original question:".to_string(),
            )],
        });
        extended_messages.extend_from_slice(original_messages);

        let mut partial_result_as_content = Vec::new();
        for delta in partial_result {
            if let Some(contents) = &delta.content {
                partial_result_as_content.extend_from_slice(contents);
            }
            if let Some(tool_calls) = &delta.tool_calls {
                for tool_call in tool_calls {
                    partial_result_as_content.push(ContentPart::Text(format!(
                        "<tool-call id=\"{}\" name=\"{}\" arguments=\"{}\"/>",
                        tool_call.id, tool_call.name, tool_call.arguments_json,
                    )));
                }
            }
        }

        extended_messages.push(Message {
            role: Role::User,
            name: None,
            content: vec![ContentPart::Text(
                "Here is the partial response that was successfully received:".to_string(),
            )]
            .into_iter()
            .chain(partial_result_as_content)
            .collect(),
        });
        extended_messages
    }
}

type DurableAnthropicComponent = DurableLLM<AnthropicComponent>;

golem_llm::export_llm!(DurableAnthropicComponent with_types_in golem_llm);
