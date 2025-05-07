use crate::client::{OllamaApi, OllamaChatRequest, OllamaChatResponse};
use crate::conversions::{messages_to_ollama_request, process_ollama_response};
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

mod client;
mod conversions;

pub struct OllamaChatStream {
    stream: RefCell<Option<EventSource>>,
    failure: Option<Error>,
    finished: RefCell<bool>,
    response_metadata: RefCell<ResponseMetadata>,
}

impl OllamaChatStream {
    pub fn new(stream: EventSource) -> LlmChatStream<Self> {
        LlmChatStream::new(OllamaChatStream {
            stream: RefCell::new(Some(stream)),
            failure: None,
            finished: RefCell::new(false),
            response_metadata: RefCell::new(ResponseMetadata {
                finish_reason: None,
                usage: None,
                provider_id: None,
                timestamp: None,
                provider_metadata_json: None,
            }),
        })
    }

    pub fn failed(error: Error) -> LlmChatStream<Self> {
        LlmChatStream::new(OllamaChatStream {
            stream: RefCell::new(None),
            failure: Some(error),
            finished: RefCell::new(false),
            response_metadata: RefCell::new(ResponseMetadata {
                finish_reason: None,
                usage: None,
                provider_id: None,
                timestamp: None,
                provider_metadata_json: None,
            }),
        })
    }
}

impl LlmChatStreamState for OllamaChatStream {
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
        trace!("Received raw Ollama response: {raw}");
        let response: OllamaChatResponse = serde_json::from_str(raw)
            .map_err(|err| format!("Failed to deserialize Ollama response: {err}"))?;

        let event = process_ollama_response(response);
        Ok(Some(StreamEvent::ChatEvent(event)))
    }
}

struct OllamaComponent;

impl OllamaComponent {
    fn request(client: OllamaApi, request: OllamaChatRequest) -> ChatEvent {
        match client.chat(request) {
            Ok(response) => process_ollama_response(response),
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn streaming_request(
        client: OllamaApi,
        request: OllamaChatRequest,
    ) -> LlmChatStream<OllamaChatStream> {
        match client.stream_chat(request) {
            Ok(stream) => OllamaChatStream::new(client, request.config, request.messages),
            Err(err) => OllamaChatStream::failed(err),
        }
    }
}


impl Guest for OllamaComponent {
    type ChatStream = LlmChatStream<OllamaChatStream>;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        let client = OllamaApi::new();
        match messages_to_ollama_request(messages, config) {
            Ok(request) => Self::request(client, request),
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn continue_(
        messages: Vec<Message>,
        _tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());
        let client = OllamaApi::new();
        match messages_to_ollama_request(messages, config) {
            Ok(request) => Self::request(client, request),
            Err(err) => ChatEvent::Error(err),
        }
    }


    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        ChatStream::new(Self::unwrapped_stream(messages, config))
    }
}

impl ExtendedGuest for OllamaComponent {
    fn unwrapped_stream(
        messages: Vec<Message>,
        config: Config,
    ) -> LlmChatStream<OllamaChatStream> {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        let client = OllamaApi::new();
        match messages_to_ollama_request(messages, config) {
            Ok(request) => Self::streaming_request(client, request),
            Err(err) => OllamaChatStream::failed(err),
        }
    }

    fn retry_prompt(original_messages: &[Message], partial_result: &[StreamDelta]) -> Vec<Message> {
        let mut extended_messages = Vec::new();

        extended_messages.push(Message {
            role: Role::System,
            name: None,
            content: vec![ContentPart::Text(
                "You were asked the same question previously, but the response was interrupted before completion. \
                 Please continue your response from where you left off. \
                 Do not include the part of the response that was already seen."
                    .to_string(),
            )],
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

type DurableOllamaComponent = DurableLLM<OllamaComponent>;

golem_llm::export_llm!(DurableOllamaComponent with_types_in golem_llm);
