mod client;
mod conversions;

use crate::client::{
    BedrockRuntimeApi, BedrockRuntimeConfig, ErrorResponse, InvokeModelRequest,
    InvokeModelResponse, InvokeResult, OutputItem, ResponseOutputItemDone, ResponseOutputTextDelta,
};
use crate::conversions::{
    create_request, create_response_metadata, parse_error_code, process_model_response,
    tool_defs_to_tools, tool_results_to_messages,
};
use golem_llm::chat_stream::{LlmChatStream, LlmChatStreamState};
use golem_llm::config::with_config_key;
use golem_llm::durability::{DurableLLM, ExtendedGuest};
use golem_llm::event_source::EventSource;
use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, Config, ContentPart, Error, ErrorCode, Guest, Message, StreamDelta,
    StreamEvent, ToolCall, ToolResult,
};
use golem_llm::LOGGING_STATE;
use log::trace;
use std::cell::{Ref, RefCell, RefMut};
use std::collections::HashMap;

struct BedrockChatStream {
    stream: RefCell<Option<EventSource>>,
    failure: Option<Error>,
    finished: RefCell<bool>,
}

impl BedrockChatStream {
    pub fn new(stream: EventSource) -> LlmChatStream<Self> {
        LlmChatStream::new(BedrockChatStream {
            stream: RefCell::new(Some(stream)),
            failure: None,
            finished: RefCell::new(false),
        })
    }

    pub fn failed(err: Error) -> LlmChatStream<Self> {
        LlmChatStream::new(BedrockChatStream {
            stream: RefCell::new(None),
            failure: Some(err),
            finished: RefCell::new(false),
        })
    }
}

impl LlmChatStreamState for BedrockChatStream {
    fn failure(&self) -> &Option<Error> {
        &self.failure
    }

    fn is_finished(&self) -> bool {
        *self.finished.borrow()
    }

    fn set_finished(&self) {
        *self.finished.borrow_mut() = true;
    }

    fn stream(&self) -> Ref<Option<EventSource>> {
        self.stream.borrow()
    }

    fn stream_mut(&self) -> RefMut<Option<EventSource>> {
        self.stream.borrow_mut()
    }

    fn decode_message(&self, raw: &str) -> Result<Option<StreamEvent>, String> {
        trace!("Received raw stream event: {raw}");
        let json: serde_json::Value = serde_json::from_str(raw)
            .map_err(|err| format!("Failed to deserialize stream event: {err}"))?;

        // Bedrock has a different event structure compared to OpenAI
        // Let's handle the specific Bedrock event types

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
                let err_resp: ErrorResponse = serde_json::from_value(response.clone())
                    .map_err(|e| format!("Failed to parse ErrorResponse: {}", e))?;

                let details = err_resp.error;
                Ok(Some(StreamEvent::Error(Error {
                    code: parse_error_code(details.typ),
                    message: details.message,
                    provider_error_json: None,
                })))
            }
            Some("response.completed") => {
                let response = json
                    .as_object()
                    .and_then(|obj| obj.get("response"))
                    .ok_or_else(|| {
                        "Unexpected stream event format, does not have 'response' field".to_string()
                    })?;
                let decoded = serde_json::from_value::<InvokeModelResponse>(response.clone())
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
                if let OutputItem::FunctionCall {
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
                            arguments_json: arguments.to_string(),
                        }]),
                    })))
                } else {
                    Ok(None)
                }
            }
            Some("chunk.start") | Some("chunk.end") => Ok(None),

            Some(_) => Ok(None),
            None => Err("Unexpected stream event format, does not have 'type' field".to_string()),
        }
    }
}

struct BedrockComponent;

impl BedrockComponent {
    const ENV_ACCESS_KEY: &'static str = "AWS_ACCESS_KEY_ID";
    const ENV_SECRET_KEY: &'static str = "AWS_SECRET_ACCESS_KEY";
    const ENV_REGION: &'static str = "AWS_REGION";

    fn make_client() -> Result<BedrockRuntimeApi, Error> {
        let access_key_id = std::env::var(Self::ENV_ACCESS_KEY).map_err(|_| Error {
            code: ErrorCode::InternalError,
            message: format!("{} missing", Self::ENV_ACCESS_KEY),
            provider_error_json: None,
        })?;
        let secret_access_key = std::env::var(Self::ENV_SECRET_KEY).map_err(|_| Error {
            code: ErrorCode::InternalError,
            message: format!("{} missing", Self::ENV_SECRET_KEY),
            provider_error_json: None,
        })?;
        let region = std::env::var(Self::ENV_REGION).map_err(|_| Error {
            code: ErrorCode::InternalError,
            message: format!("{} missing", Self::ENV_REGION),
            provider_error_json: None,
        })?;
        let endpoint = format!("bedrock-runtime.{}.amazonaws.com", region);

        Ok(BedrockRuntimeApi::new(BedrockRuntimeConfig {
            access_key_id,
            secret_access_key,
            session_token: None,
            region,
            endpoint,
        }))
    }

    fn request(client: BedrockRuntimeApi, msgs: Vec<Message>, config: Config) -> ChatEvent {
        match tool_defs_to_tools(&config.tools) {
            Ok(tools) => {
                let request = create_request(msgs, config.clone());
                match client.invoke_model(&config.model, &request) {
                    Ok(response) => process_model_response(response),
                    Err(error) => ChatEvent::Error(error),
                }
            }
            Err(error) => ChatEvent::Error(error),
        }
    }

    fn streaming_request(
        client: BedrockRuntimeApi,
        msgs: Vec<Message>,
        config: Config,
    ) -> LlmChatStream<BedrockChatStream> {
        match tool_defs_to_tools(&config.tools) {
            Ok(tools) => {
                let mut request = create_request(msgs, config.clone());
                match client.stream_invoke_model(&config.model, &request) {
                    Ok(stream) => BedrockChatStream::new(stream),
                    Err(error) => BedrockChatStream::failed(error),
                }
            }
            Err(error) => BedrockChatStream::failed(error),
        }
    }
}

impl Guest for BedrockComponent {
    type ChatStream = LlmChatStream<BedrockChatStream>;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|st| st.init());
        match Self::make_client() {
            Ok(client) => Self::request(client, messages, config),
            Err(e) => ChatEvent::Error(e),
        }
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|st| st.init());
        match Self::make_client() {
            Ok(client) => {
                let mut msgs = messages;
                msgs.extend(tool_results_to_messages(&tool_results));
                Self::request(client, msgs, config)
            }
            Err(e) => ChatEvent::Error(e),
        }
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        ChatStream::new(Self::unwrapped_stream(messages, config))
    }
}

impl ExtendedGuest for BedrockComponent {
    fn unwrapped_stream(messages: Vec<Message>, config: Config) -> Self::ChatStream {
        LOGGING_STATE.with_borrow_mut(|st| st.init());
        match Self::make_client() {
            Ok(client) => BedrockComponent::streaming_request(client, messages, config),
            Err(e) => BedrockChatStream::failed(e),
        }
    }
}

type DurableBedrockComponent = DurableLLM<BedrockComponent>;

golem_llm::export_llm!(DurableBedrockComponent with_types_in golem_llm);
