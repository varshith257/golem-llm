mod client;
mod conversions;

use crate::client::{CompletionsApi, CompletionsRequest};
use crate::conversions::{messages_to_request, process_response, tool_results_to_messages};
use golem_llm::config::with_config_key;
use golem_llm::durability::{DurableLLM, ExtendedGuest};
use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, Config, Guest, GuestChatStream, Message, Pollable, StreamEvent,
    ToolCall, ToolResult,
};
use golem_llm::LOGGING_STATE;

struct GrokChatStream;

impl GuestChatStream for GrokChatStream {
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

struct GrokComponent;

impl GrokComponent {
    const ENV_VAR_NAME: &'static str = "XAI_API_KEY";

    fn request(client: CompletionsApi, request: CompletionsRequest) -> ChatEvent {
        match client.send_messages(request) {
            Ok(response) => process_response(response),
            Err(err) => ChatEvent::Error(err),
        }
    }

    fn streaming_request(client: CompletionsApi, mut request: CompletionsRequest) -> GrokChatStream {
        request.stream = Some(true);
        // match client.stream_send_messages(request) {
        //     Ok(stream) => GrokChatStream::new(stream),
        //     Err(err) => GrokChatStream::failed(err),
        // }
        todo!()
    }
}

impl Guest for GrokComponent {
    type ChatStream = GrokChatStream;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENV_VAR_NAME, ChatEvent::Error, |anthropic_api_key| {
            let client = CompletionsApi::new(anthropic_api_key);

            match messages_to_request(messages, config) {
                Ok(request) => Self::request(client, request),
                Err(err) => return ChatEvent::Error(err),
            }
        })    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        with_config_key(Self::ENV_VAR_NAME, ChatEvent::Error, |anthropic_api_key| {
            let client = CompletionsApi::new(anthropic_api_key);

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

impl ExtendedGuest for GrokComponent {
    fn unwrapped_stream(messages: Vec<Message>, config: Config) -> GrokChatStream {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        todo!()
        // with_config_key(
        //     Self::ENV_VAR_NAME,
        //     GrokChatStream::failed,
        //     |anthropic_api_key| {
        //         let client = CompletionsApi::new(anthropic_api_key);
        //
        //         match messages_to_request(messages, config) {
        //             Ok(request) => Self::streaming_request(client, request),
        //             Err(err) => GrokChatStream::failed(err),
        //         }
        //     },
        // )
    }
}

type DurableGrokComponent = DurableLLM<GrokComponent>;

golem_llm::export_llm!(DurableGrokComponent with_types_in golem_llm);
