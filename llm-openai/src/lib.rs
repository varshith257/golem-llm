use std::cell::RefCell;
use std::task::Poll;
use crate::client::{CreateModelResponseRequest, Input, InputItem, ResponsesApi};
use crate::conversions::{
    messages_to_input_items, process_model_response, tool_defs_to_tools,
    tool_results_to_input_items,
};
use golem_llm::config::with_config_key;
use golem_llm::event_source::{Event, EventSource};
use golem_llm::golem::llm::llm::{ChatEvent, ChatStream, Config, Guest, GuestChatStream, Message, Pollable, StreamEvent, ToolCall, ToolResult};

mod client;
mod conversions;

struct OpenAIChatStream {
    stream: RefCell<EventSource>
}

impl GuestChatStream for OpenAIChatStream {
    fn get_next(&self) -> Option<Vec<StreamEvent>> {
        let mut stream = self.stream.borrow_mut();
        match stream.poll_next() {
            Poll::Ready(_) => Some(vec![]), // TODO
            Poll::Pending => None
        }
    }

    fn blocking_get_next(&self) -> Vec<StreamEvent> {
        let pollable = self.subscribe();
        pollable.block();
        self.get_next().unwrap()
    }

    fn subscribe(&self) -> Pollable {
        self.stream.borrow().subscribe()
    }
}

struct OpenAIComponent;

impl OpenAIComponent {
    const ENV_VAR_NAME: &'static str = "OPENAI_API_KEY";

    fn request(client: ResponsesApi, items: Vec<InputItem>, config: Config) -> ChatEvent {
        match tool_defs_to_tools(config.tools) {
            Ok(tools) => {
                let request = CreateModelResponseRequest {
                    input: Input::List(items),
                    model: config.model,
                    temperature: config.temperature,
                    max_output_tokens: config.max_tokens,
                    tools,
                    tool_choice: config.tool_choice,
                    stream: false
                };
                match client.create_model_response(request) {
                    Ok(response) => process_model_response(response),
                    Err(error) => ChatEvent::Error(error),
                }
            }
            Err(error) => ChatEvent::Error(error),
        }
    }

    fn streaming_request(client: ResponsesApi, items: Vec<InputItem>, config: Config) -> ChatEvent {
        match tool_defs_to_tools(config.tools) {
            Ok(tools) => {
                let request = CreateModelResponseRequest {
                    input: Input::List(items),
                    model: config.model,
                    temperature: config.temperature,
                    max_output_tokens: config.max_tokens,
                    tools,
                    tool_choice: config.tool_choice,
                    stream: true
                };
                match client.create_model_response(request) {
                    Ok(response) => process_model_response(response),
                    Err(error) => ChatEvent::Error(error),
                }
            }
            Err(error) => ChatEvent::Error(error),
        }
    }
}

impl Guest for OpenAIComponent {
    type ChatStream = OpenAIChatStream;

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        with_config_key(Self::ENV_VAR_NAME, ChatEvent::Error, |openai_api_key| {
            let client = ResponsesApi::new(openai_api_key);

            let items = messages_to_input_items(messages);
            Self::request(client, items, config)
        })
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<(ToolCall, ToolResult)>,
        config: Config,
    ) -> ChatEvent {
        with_config_key(Self::ENV_VAR_NAME, ChatEvent::Error, |openai_api_key| {
            let client = ResponsesApi::new(openai_api_key);

            let mut items = messages_to_input_items(messages);
            items.extend(tool_results_to_input_items(tool_results));
            Self::request(client, items, config)
        })
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        todo!()
    }
}

golem_llm::export_llm!(OpenAIComponent with_types_in golem_llm);
