
use golem_llm::golem::llm::llm::{ChatEvent, ChatStream, Config, Error, Guest, GuestChatStream, Message, StreamEvent, ToolResult};

struct OpenAIChatStream;

impl GuestChatStream for OpenAIChatStream {
    fn get_next(&self) -> Vec<StreamEvent> {
        todo!()
    }

    fn has_next(&self) -> bool {
        todo!()
    }
}

struct OpenAIComponent;

impl Guest for OpenAIComponent {
    type ChatStream = OpenAIChatStream;

    fn send(messages: Vec<Message>, config: Config) -> Result<ChatEvent, Error> {
        todo!()
    }

    fn continue_(messages: Vec<Message>, tool_results: Vec<ToolResult>, config: Config) -> Result<ChatEvent, Error> {
        todo!()
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        todo!()
    }
}

golem_llm::export_llm!(OpenAIComponent with_types_in golem_llm);