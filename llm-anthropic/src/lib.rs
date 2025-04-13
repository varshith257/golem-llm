use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, Config, Error, Guest, GuestChatStream, Message, Pollable, StreamEvent,
    ToolCall, ToolResult,
};

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

impl Guest for AnthropicComponent {
    type ChatStream = AnthropicChatStream;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        todo!()
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        todo!()
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        todo!()
    }

    fn enable_debug_traces(_enable: bool) {}
}

golem_llm::export_llm!(AnthropicComponent with_types_in golem_llm);
