use crate::client::{
    CreateModelResponseRequest, CreateModelResponseResponse, Input, InputItem, OutputItem,
    ResponseOutputItemDone, ResponseOutputTextDelta, ResponsesApi,
};
use crate::conversions::{
    create_response_metadata, messages_to_input_items, parse_error_code, process_model_response,
    tool_defs_to_tools, tool_results_to_input_items,
};
use golem_llm::config::with_config_key;
use golem_llm::durability::{DurableLLM, ExtendedGuest};
use golem_llm::event_source::{Event, EventSource, MessageEvent};
use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, Config, ContentPart, Error, ErrorCode, Guest, GuestChatStream, Message,
    Pollable, StreamDelta, StreamEvent, ToolCall, ToolResult,
};
use golem_llm::LOGGING_STATE;
use log::trace;
use std::cell::RefCell;
use std::task::Poll;

mod client;
mod conversions;

struct OpenAIChatStream {
    stream: RefCell<Option<EventSource>>,
    failure: Option<Error>,
    finished: RefCell<bool>,
}

impl OpenAIChatStream {
    pub fn new(stream: EventSource) -> Self {
        OpenAIChatStream {
            stream: RefCell::new(Some(stream)),
            failure: None,
            finished: RefCell::new(false),
        }
    }

    pub fn failed(error: Error) -> Self {
        OpenAIChatStream {
            stream: RefCell::new(None),
            failure: Some(error),
            finished: RefCell::new(false),
        }
    }

    fn decode_message(raw: &str) -> Result<Option<StreamEvent>, String> {
        trace!("Received raw stream event: {raw}");
        let json: serde_json::Value = serde_json::from_str(raw)
            .map_err(|err| format!("Failed to deserialize stream event: {err}"))?;

        let typ = json
            .as_object()
            .and_then(|obj| obj.get("type"))
            .and_then(|v| v.as_str());
        match typ {
            Some("response.failed") => {
                let response = json
                    .as_object()
                    .and_then(|obj| obj.get("response"))
                    .ok_or_else(|| {
                        "Unexpected stream event format, does not have 'response' field".to_string()
                    })?;
                let decoded =
                    serde_json::from_value::<CreateModelResponseResponse>(response.clone())
                        .map_err(|err| {
                            format!("Failed to deserialize stream event's response field: {err}")
                        })?;

                if let Some(error) = decoded.error {
                    Ok(Some(StreamEvent::Error(Error {
                        code: parse_error_code(error.code),
                        message: error.message,
                        provider_error_json: None,
                    })))
                } else {
                    Ok(Some(StreamEvent::Error(Error {
                        code: ErrorCode::InternalError,
                        message: "Unknown error".to_string(),
                        provider_error_json: None,
                    })))
                }
            }
            Some("response.completed") => {
                let response = json
                    .as_object()
                    .and_then(|obj| obj.get("response"))
                    .ok_or_else(|| {
                        "Unexpected stream event format, does not have 'response' field".to_string()
                    })?;
                let decoded =
                    serde_json::from_value::<CreateModelResponseResponse>(response.clone())
                        .map_err(|err| {
                            format!("Failed to deserialize stream event's response field: {err}")
                        })?;
                Ok(Some(StreamEvent::Finish(create_response_metadata(
                    &decoded,
                ))))
            }
            Some("response.output_text.delta") => {
                let decoded = serde_json::from_value::<ResponseOutputTextDelta>(json)
                    .map_err(|err| format!("Failed to deserialize stream event: {err}"))?;
                Ok(Some(StreamEvent::Delta(StreamDelta {
                    content: Some(vec![ContentPart::Text(decoded.delta)]),
                    tool_calls: None,
                })))
            }
            Some("response.output_item.done") => {
                let decoded = serde_json::from_value::<ResponseOutputItemDone>(json)
                    .map_err(|err| format!("Failed to deserialize stream event: {err}"))?;
                if let OutputItem::ToolCall {
                    arguments,
                    call_id,
                    name,
                    ..
                } = decoded.item
                {
                    Ok(Some(StreamEvent::Delta(StreamDelta {
                        content: None,
                        tool_calls: Some(vec![ToolCall {
                            id: call_id,
                            name,
                            arguments_json: arguments,
                        }]),
                    })))
                } else {
                    Ok(None)
                }
            }
            Some(_) => Ok(None),
            None => Err("Unexpected stream event format, does not have 'type' field".to_string()),
        }
    }
}

impl GuestChatStream for OpenAIChatStream {
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
                                match Self::decode_message(&data) {
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

struct OpenAIComponent;

impl OpenAIComponent {
    const ENV_VAR_NAME: &'static str = "OPENAI_API_KEY";

    fn request(client: ResponsesApi, items: Vec<InputItem>, config: Config) -> ChatEvent {
        match tool_defs_to_tools(config.tools) {
            Ok(tools) => {
                let request = CreateModelResponseRequest {
                    input: Input::List(items),
                    model: config.model,
                    temperature: config.temperature,
                    max_output_tokens: config.max_tokens,
                    tools,
                    tool_choice: config.tool_choice,
                    stream: false,
                };
                match client.create_model_response(request) {
                    Ok(response) => process_model_response(response),
                    Err(error) => ChatEvent::Error(error),
                }
            }
            Err(error) => ChatEvent::Error(error),
        }
    }

    fn streaming_request(
        client: ResponsesApi,
        items: Vec<InputItem>,
        config: Config,
    ) -> OpenAIChatStream {
        match tool_defs_to_tools(config.tools) {
            Ok(tools) => {
                let request = CreateModelResponseRequest {
                    input: Input::List(items),
                    model: config.model,
                    temperature: config.temperature,
                    max_output_tokens: config.max_tokens,
                    tools,
                    tool_choice: config.tool_choice,
                    stream: true,
                };
                match client.stream_model_response(request) {
                    Ok(stream) => OpenAIChatStream::new(stream),
                    Err(error) => OpenAIChatStream::failed(error),
                }
            }
            Err(error) => OpenAIChatStream::failed(error),
        }
    }
}

impl Guest for OpenAIComponent {
    type ChatStream = OpenAIChatStream;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENV_VAR_NAME, ChatEvent::Error, |openai_api_key| {
            let client = ResponsesApi::new(openai_api_key);

            let items = messages_to_input_items(messages);
            Self::request(client, items, config)
        })
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENV_VAR_NAME, ChatEvent::Error, |openai_api_key| {
            let client = ResponsesApi::new(openai_api_key);

            let mut items = messages_to_input_items(messages);
            items.extend(tool_results_to_input_items(tool_results));
            Self::request(client, items, config)
        })
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        ChatStream::new(Self::unwrapped_stream(messages, config))
    }
}

impl ExtendedGuest for OpenAIComponent {
    fn unwrapped_stream(messages: Vec<Message>, config: Config) -> Self::ChatStream {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(
            Self::ENV_VAR_NAME,
            OpenAIChatStream::failed,
            |openai_api_key| {
                let client = ResponsesApi::new(openai_api_key);

                let items = messages_to_input_items(messages);
                Self::streaming_request(client, items, config)
            },
        )
    }
}

type DurableOpenAIComponent = DurableLLM<OpenAIComponent>;

golem_llm::export_llm!(DurableOpenAIComponent with_types_in golem_llm);
