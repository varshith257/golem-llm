mod client;
mod conversions;

use crate::client::{MessagesApi, MessagesRequest};
use crate::conversions::{messages_to_request, process_response, tool_results_to_messages};
use golem_llm::config::with_config_key;
use golem_llm::durability::{DurableLLM, ExtendedGuest};
use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, Config, Error, Guest, GuestChatStream, Message, Pollable, StreamEvent,
    ToolCall, ToolResult,
};
use golem_llm::LOGGING_STATE;

struct AnthropicChatStream;

impl GuestChatStream for AnthropicChatStream {
    fn get_next(&self) -> Option<Vec<StreamEvent>> {
        todo!()
    }

    fn blocking_get_next(&self) -> Vec<StreamEvent> {
        todo!()
    }

    fn subscribe(&self) -> Pollable {
        todo!()
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
                    request.messages.extend(tool_results_to_messages(tool_results));
                    Self::request(client, request)
                },
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

        todo!()
    }
}

type DurableAnthropicComponent = DurableLLM<AnthropicComponent>;

golem_llm::export_llm!(DurableAnthropicComponent with_types_in golem_llm);
