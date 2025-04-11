use crate::client::{
    CreateModelResponseRequest, CreateModelResponseResponse, InnerInput, Input, InputItem,
    OutputItem, OutputMessageContent, ResponsesApi,
};
use crate::conversions::{content_part_to_inner_input_item, parse_error_code, to_openai_role_name};
use golem_llm::config::with_config_key;
use golem_llm::error::unsupported;
use golem_llm::golem::llm::llm::{
    ChatEvent, ChatStream, CompleteResponse, Config, ContentPart, Error, ErrorCode, Guest,
    GuestChatStream, Message, ResponseMetadata, StreamEvent, ToolCall, ToolResult, Usage,
};

mod client;
mod conversions;

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

    fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
        with_config_key("OPENAI_API_KEY", ChatEvent::Error, |openai_api_key| {
            let client = ResponsesApi::new(openai_api_key);

            let mut items = Vec::new();
            for message in messages {
                // TODO: what about message.name?

                let role = to_openai_role_name(message.role).to_string();
                let mut input_items = Vec::new();
                for content_part in message.content {
                    input_items.push(content_part_to_inner_input_item(content_part));
                }

                items.push(InputItem::InputMessage {
                    role,
                    content: InnerInput::List(input_items),
                });
            }

            let mut tools = Vec::new();

            for tool_def in config.tools {
                match serde_json::from_str(&tool_def.parameters_schema) {
                    Ok(value) => {
                        let tool = client::Tool::Function {
                            name: tool_def.name,
                            description: tool_def.description,
                            parameters: Some(value),
                        };
                        tools.push(tool);
                    }
                    Err(error) => {
                        return ChatEvent::Error(Error {
                            code: ErrorCode::InternalError,
                            message: format!(
                                "Failed to parse tool parameters for {}: {error}",
                                tool_def.name
                            ),
                            provider_error_json: None,
                        });
                    }
                }
            }

            let request = CreateModelResponseRequest {
                input: Input::List(items),
                model: config.model,
                temperature: config.temperature,
                max_output_tokens: config.max_tokens,
                tools,
                tool_choice: config.tool_choice,
            };
            match client.create_model_response(request) {
                Ok(response) => {
                    if let Some(error) = response.error {
                        ChatEvent::Error(Error {
                            code: parse_error_code(error.code),
                            message: error.message,
                            provider_error_json: None,
                        })
                    } else {
                        let mut contents = Vec::new();
                        let mut tool_calls = Vec::new();

                        for output_item in response.output {
                            match output_item {
                                OutputItem::Message { content, .. } => {
                                    for content in content {
                                        match content {
                                            OutputMessageContent::Text { text, .. } => {
                                                contents.push(ContentPart::Text(text));
                                            }
                                            OutputMessageContent::Refusal { refusal, .. } => {
                                                // TODO: ?
                                                contents.push(ContentPart::Text(format!(
                                                    "Refusal: {refusal}"
                                                )));
                                            }
                                        }
                                    }
                                }
                                OutputItem::ToolCall {
                                    arguments,
                                    call_id,
                                    name,
                                    ..
                                } => {
                                    let tool_call = ToolCall {
                                        id: call_id,
                                        name,
                                        arguments_json: arguments,
                                    };
                                    tool_calls.push(tool_call);
                                }
                            }
                        }

                        let metadata = ResponseMetadata {
                            finish_reason: None, // TODO
                            usage: Some(Usage {
                                input_tokens: Some(response.usage.input_tokens),
                                output_tokens: Some(response.usage.output_tokens),
                                total_tokens: Some(response.usage.total_tokens),
                            }),
                            provider_id: None,            // TODO
                            timestamp: None,              // TODO
                            provider_metadata_json: None, // TODO
                        };

                        if contents.is_empty() {
                            ChatEvent::ToolRequest(tool_calls)
                        } else {
                            ChatEvent::Message(CompleteResponse {
                                id: response.id,
                                content: contents,
                                tool_calls,
                                metadata,
                            })
                        }
                    }
                }
                Err(error) => ChatEvent::Error(error),
            }
        })
    }

    fn continue_(
        messages: Vec<Message>,
        tool_results: Vec<ToolResult>,
        config: Config,
    ) -> ChatEvent {
        ChatEvent::Error(unsupported("continue not implemented yet"))
    }

    fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
        todo!()
    }
}

golem_llm::export_llm!(OpenAIComponent with_types_in golem_llm);
