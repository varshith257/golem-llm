use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, Config, Error, Guest, GuestChatStream, Message, Pollable, StreamEvent,
    ToolCall, ToolResult,
};

struct OpenRouterChatStream;

impl GuestChatStream for OpenRouterChatStream {
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

struct OpenRouterComponent;

impl Guest for OpenRouterComponent {
    type ChatStream = OpenRouterChatStream;

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
}

golem_llm::export_llm!(OpenRouterComponent with_types_in golem_llm);
