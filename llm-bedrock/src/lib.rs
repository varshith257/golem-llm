use crate::client::{
    BedrockModelResponseResponse, InputItem, OutputItem, ResponseOutputItemDone,
    ResponseOutputTextDelta, ResponsesApi,
};
use crate::conversions::{
    create_request, create_response_metadata, messages_to_input_items, parse_error_code,
    process_model_response, tool_defs_to_tools, tool_results_to_input_items,
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

mod client;
mod conversions;

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

    pub fn failed(error: Error) -> LlmChatStream<Self> {
        LlmChatStream::new(BedrockChatStream {
            stream: RefCell::new(None),
            failure: Some(error),
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
        
        let event_type = json
            .as_object()
            .and_then(|obj| obj.get("type"))
            .and_then(|v| v.as_str());
            
        match event_type {
            Some("response.failed") => {
                let response = json
                    .as_object()
                    .and_then(|obj| obj.get("response"))
                    .ok_or_else(|| {
                        "Unexpected stream event format, does not have 'response' field".to_string()
                    })?;
                let decoded =
                    serde_json::from_value::<BedrockModelResponseResponse>(response.clone())
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
                    serde_json::from_value::<BedrockModelResponseResponse>(response.clone())
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
            // Add any Bedrock-specific event types here
            Some("chunk.start") => {
                // Handle chunk start event, typically just logging/metadata
                Ok(None)
            }
            Some("chunk.end") => {
                // Handle chunk end event, typically just logging/metadata
                Ok(None)
            }
            Some(_) => Ok(None),
            None => Err("Unexpected stream event format, does not have 'type' field".to_string()),
        }
    }
}

struct BedrockComponent;

impl BedrockComponent {
    const AWS_ACCESS_KEY_ID: &'static str = "AWS_ACCESS_KEY_ID";
    const AWS_SECRET_ACCESS_KEY: &'static str = "AWS_SECRET_ACCESS_KEY";
    const AWS_REGION: &'static str = "AWS_REGION";
    
    fn get_required_aws_configs() -> Result<HashMap<String, String>, Error> {
        let mut configs = HashMap::new();
        
        let access_key = std::env::var(Self::AWS_ACCESS_KEY_ID).map_err(|_| Error {
            code: ErrorCode::ConfigurationError,
            message: format!("Environment variable {} is required", Self::AWS_ACCESS_KEY_ID),
            provider_error_json: None,
        })?;
        
        let secret_key = std::env::var(Self::AWS_SECRET_ACCESS_KEY).map_err(|_| Error {
            code: ErrorCode::ConfigurationError,
            message: format!("Environment variable {} is required", Self::AWS_SECRET_ACCESS_KEY),
            provider_error_json: None,
        })?;
        
        let region = std::env::var(Self::AWS_REGION).map_err(|_| Error {
            code: ErrorCode::ConfigurationError,
            message: format!("Environment variable {} is required", Self::AWS_REGION),
            provider_error_json: None,
        })?;
        
        configs.insert(Self::AWS_ACCESS_KEY_ID.to_string(), access_key);
        configs.insert(Self::AWS_SECRET_ACCESS_KEY.to_string(), secret_key);
        configs.insert(Self::AWS_REGION.to_string(), region);
        
        Ok(configs)
    }

    fn request(client: ResponsesApi, items: Vec<InputItem>, config: Config) -> ChatEvent {
        match tool_defs_to_tools(&config.tools) {
            Ok(tools) => {
                let request = create_request(items, config, tools);
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
    ) -> LlmChatStream<BedrockChatStream> {
        match tool_defs_to_tools(&config.tools) {
            Ok(tools) => {
                let mut request = create_request(items, config, tools);
                request.stream = true;
                match client.stream_model_response(request) {
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
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        // Unlike OpenAI which uses a single API key, AWS requires multiple credentials
        match BedrockComponent::get_required_aws_configs() {
            Ok(aws_configs) => {
                let client = ResponsesApi::new(aws_configs);
                let items = messages_to_input_items(messages);
                Self::request(client, items, config)
            }
            Err(error) => ChatEvent::Error(error),
        }
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        match BedrockComponent::get_required_aws_configs() {
            Ok(aws_configs) => {
                let client = ResponsesApi::new(aws_configs);
                let mut items = messages_to_input_items(messages);
                items.extend(tool_results_to_input_items(tool_results));
                Self::request(client, items, config)
            }
            Err(error) => ChatEvent::Error(error),
        }
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        ChatStream::new(Self::unwrapped_stream(messages, config))
    }
}

impl ExtendedGuest for BedrockComponent {
    fn unwrapped_stream(messages: Vec<Message>, config: Config) -> Self::ChatStream {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        match BedrockComponent::get_required_aws_configs() {
            Ok(aws_configs) => {
                let client = ResponsesApi::new(aws_configs);
                let items = messages_to_input_items(messages);
                Self::streaming_request(client, items, config)
            }
            Err(error) => BedrockChatStream::failed(error),
        }
    }
}

type DurableBedrockComponent = DurableLLM<BedrockComponent>;

golem_llm::export_llm!(DurableBedrockComponent with_types_in golem_llm);