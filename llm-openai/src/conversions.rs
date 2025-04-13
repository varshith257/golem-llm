use crate::client::{
    CreateModelResponseResponse, Detail, InnerInput, InnerInputItem, InputItem, OutputItem,
    OutputMessageContent, Tool,
};
use golem_llm::golem::llm::llm::{
    ChatEvent, CompleteResponse, ContentPart, Error, ErrorCode, ImageDetail, Message,
    ResponseMetadata, Role, ToolCall, ToolDefinition, ToolResult, Usage,
};

pub fn messages_to_input_items(messages: Vec<Message>) -> Vec<InputItem> {
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
    items
}

pub fn tool_results_to_input_items(tool_results: Vec<(ToolCall, ToolResult)>) -> Vec<InputItem> {
    let mut items = Vec::new();
    for (tool_call, tool_result) in tool_results {
        let tool_call = InputItem::ToolCall {
            arguments: tool_call.arguments_json,
            call_id: tool_call.id,
            name: tool_call.name,
        };
        let tool_result = match tool_result {
            ToolResult::Success(success) => InputItem::ToolResult {
                call_id: success.id,
                output: format!(r#"{{ "success": {} }}"#, success.result_json),
            },
            ToolResult::Error(error) => InputItem::ToolResult {
                call_id: error.id,
                output: format!(
                    r#"{{ "error": {{ "code": {}, "message": {} }} }}"#,
                    error.error_code.unwrap_or_default(),
                    error.error_message
                ),
            },
        };
        items.push(tool_call);
        items.push(tool_result);
    }
    items
}

pub fn tool_defs_to_tools(tool_definitions: Vec<ToolDefinition>) -> Result<Vec<Tool>, Error> {
    let mut tools = Vec::new();
    for tool_def in tool_definitions {
        match serde_json::from_str(&tool_def.parameters_schema) {
            Ok(value) => {
                let tool = Tool::Function {
                    name: tool_def.name,
                    description: tool_def.description,
                    parameters: Some(value),
                    strict: true,
                };
                tools.push(tool);
            }
            Err(error) => {
                Err(Error {
                    code: ErrorCode::InternalError,
                    message: format!(
                        "Failed to parse tool parameters for {}: {error}",
                        tool_def.name
                    ),
                    provider_error_json: None,
                })?;
            }
        }
    }
    Ok(tools)
}

pub fn to_openai_role_name(role: Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
        Role::Tool => "tool",
    }
}

pub fn content_part_to_inner_input_item(content_part: ContentPart) -> InnerInputItem {
    match content_part {
        ContentPart::Text(msg) => InnerInputItem::TextInput { text: msg },
        ContentPart::Image(image_url) => InnerInputItem::ImageInput {
            image_url: image_url.url,
            detail: match image_url.detail {
                Some(ImageDetail::Auto) => Detail::Auto,
                Some(ImageDetail::Low) => Detail::Low,
                Some(ImageDetail::High) => Detail::High,
                None => Detail::default(),
            },
        },
    }
}

pub fn parse_error_code(_code: String) -> ErrorCode {
    // TODO: we don't know what `code` can be..
    ErrorCode::InternalError
}

pub fn process_model_response(response: CreateModelResponseResponse) -> ChatEvent {
    if let Some(error) = response.error {
        ChatEvent::Error(Error {
            code: parse_error_code(error.code),
            message: error.message,
            provider_error_json: None,
        })
    } else {
        let mut contents = Vec::new();
        let mut tool_calls = Vec::new();

        let metadata = create_response_metadata(&response);

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
                                contents.push(ContentPart::Text(format!("Refusal: {refusal}")));
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

pub fn create_response_metadata(response: &CreateModelResponseResponse) -> ResponseMetadata {
    ResponseMetadata {
        finish_reason: None, // TODO
        usage: response.usage.as_ref().map(|usage| Usage {
            input_tokens: Some(usage.input_tokens),
            output_tokens: Some(usage.output_tokens),
            total_tokens: Some(usage.total_tokens),
        }),
        provider_id: None, // TODO
        timestamp: Some(response.created_at.to_string()),
        provider_metadata_json: response.metadata.as_ref().map(|m| m.to_string()),
    }
}
