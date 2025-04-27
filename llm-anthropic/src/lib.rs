mod client;

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
}

impl Guest for AnthropicComponent {
    type ChatStream = AnthropicChatStream;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        todo!()
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        LOGGING_STATE.with_borrow_mut(|state| state.init());

        todo!()
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
