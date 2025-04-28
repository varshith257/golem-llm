use golem_llm::golem::llm::llm::{ChatEvent, Config, Error, Message, ToolCall, ToolResult};
use crate::client::{CompletionsRequest, CompletionsResponse};

pub fn messages_to_request(
    messages: Vec<Message>,
    config: Config,
) -> Result<CompletionsRequest, Error> {
    todo!()
}

pub fn process_response(response: CompletionsResponse) -> ChatEvent {
    todo!()
}

pub fn tool_results_to_messages(
    tool_results: Vec<(ToolCall, ToolResult)>,
) -> Vec<crate::client::Message> {
    todo!()
}