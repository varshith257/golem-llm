use crate::client::{
    BedrockModelRequest, BedrockModelResponse, BedrockMessage, BedrockContentBlock, 
    BedrockImageContentBlock, BedrockTextContentBlock, BedrockToolChoice, BedrockTool,
    BedrockToolFunction, BedrockToolDefinition, BedrockError, BedrockToolCall,
    BedrockErrorResponse, BedrockUsage
};
use golem_llm::error::error_code_from_status;
use golem_llm::golem::llm::llm::{
    ChatEvent, CompleteResponse, Config, ContentPart, Error, ErrorCode, ImageDetail, Message,
    ResponseMetadata, Role, ToolCall, ToolDefinition, ToolResult, Usage,
};
use reqwest::StatusCode;
use std::collections::HashMap;
use std::str::FromStr;
use serde_json::Value;

/// Create a Bedrock model request from Golem LLM types
pub fn create_request(
    messages: Vec<Message>,
    config: Config,
    tool_definitions: &[ToolDefinition],
) -> Result<BedrockModelRequest, Error> {
    let bedrock_messages = messages_to_bedrock_messages(messages)?;
    
    let options = config
        .provider_options
        .into_iter()
        .map(|kv| (kv.key, kv.value))
        .collect::<HashMap<_, _>>();
    
    let tools = if !tool_definitions.is_empty() {
        Some(tool_defs_to_bedrock_tools(tool_definitions)?)
    } else {
        None
    };
    
    let tool_choice = match config.tool_choice.as_deref() {
        Some("auto") => Some(BedrockToolChoice::Auto),
        Some("required") => Some(BedrockToolChoice::Required),
        Some(specific_tool) => Some(BedrockToolChoice::Tool {
            name: specific_tool.to_string(),
        }),
        None => None,
    };

    Ok(BedrockModelRequest {
        model: config.model,
        messages: bedrock_messages,
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        tools,
        tool_choice,
        top_p: options
            .get("top_p")
            .and_then(|top_p_s| top_p_s.parse::<f32>().ok()),
        stop_sequences: options
            .get("stop_sequences")
            .and_then(|s| serde_json::from_str(s).ok()),
        user: options
            .get("user")
            .map(|s| s.to_string()),
        stream: false,
    })
}

/// Convert Golem LLM messages to Bedrock messages
pub fn messages_to_bedrock_messages(messages: Vec<Message>) -> Result<Vec<BedrockMessage>, Error> {
    let mut bedrock_messages = Vec::new();
    
    for message in messages {
        let role = to_bedrock_role_name(message.role);
        let mut content_blocks = Vec::new();
        
        for content_part in message.content {
            match content_part {
                ContentPart::Text(text) => {
                    content_blocks.push(BedrockContentBlock::Text(BedrockTextContentBlock {
                        text,
                    }));
                }
                ContentPart::Image(image_url) => {
                    let detail = match image_url.detail {
                        Some(ImageDetail::Auto) => "auto",
                        Some(ImageDetail::Low) => "low",
                        Some(ImageDetail::High) => "high",
                        None => "auto",
                    };
                    
                    content_blocks.push(BedrockContentBlock::Image(BedrockImageContentBlock {
                        source: BedrockImageContentBlock::format_source_url(&image_url.url)?,
                        detail: detail.to_string(),
                    }));
                }
            }
        }
        
        bedrock_messages.push(BedrockMessage {
            role,
            content: content_blocks,
        });
    }
    
    Ok(bedrock_messages)
}

/// Convert Golem LLM tool definitions to Bedrock tool definitions
pub fn tool_defs_to_bedrock_tools(tool_definitions: &[ToolDefinition]) -> Result<Vec<BedrockTool>, Error> {
    let mut tools = Vec::new();
    
    for tool_def in tool_definitions {
        match serde_json::from_str::<Value>(&tool_def.parameters_schema) {
            Ok(parameters) => {
                let function = BedrockToolFunction {
                    name: tool_def.name.clone(),
                    description: tool_def.description.clone(),
                    parameters,
                };
                
                let tool = BedrockTool {
                    r#type: "function".to_string(),
                    function,
                };
                
                tools.push(tool);
            }
            Err(error) => {
                return Err(Error {
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
    
    Ok(tools)
}

/// Convert Golem LLM tool results to Bedrock messages
pub fn tool_results_to_bedrock_messages(tool_results: &[(ToolCall, ToolResult)]) -> Vec<BedrockMessage> {
    let mut messages = Vec::new();
    
    for (tool_call, tool_result) in tool_results {
        let result_content = match tool_result {
            ToolResult::Success(success) => success.result_json.clone(),
            ToolResult::Error(error) => {
                format!(
                    r#"{{"error": {{"code": "{}", "message": "{}"}}}}"#,
                    error.error_code.clone().unwrap_or_default(),
                    error.error_message
                )
            }
        };
        
        messages.push(BedrockMessage {
            role: "tool".to_string(),
            content: vec![BedrockContentBlock::Text(BedrockTextContentBlock {
                text: result_content,
            })],
        });
    }
    
    messages
}

/// Convert Bedrock role names to Golem LLM roles
pub fn to_bedrock_role_name(role: Role) -> String {
    match role {
        Role::User => "user".to_string(),
        Role::Assistant => "assistant".to_string(),
        Role::System => "system".to_string(),
        Role::Tool => "tool".to_string(),
    }
}

/// Parse Bedrock error response
pub fn parse_error_response(error: BedrockErrorResponse) -> Error {
    let code = if let Some(status) = error.status_code {
        if let Ok(status_code) = u16::from_str(&status.to_string()) {
            if let Ok(status) = StatusCode::from_u16(status_code) {
                error_code_from_status(status)
            } else {
                ErrorCode::InternalError
            }
        } else {
            ErrorCode::InternalError
        }
    } else {
        ErrorCode::InternalError
    };
    
    Error {
        code,
        message: error.message.unwrap_or_else(|| "Unknown Bedrock error".to_string()),
        provider_error_json: Some(serde_json::to_string(&error).unwrap_or_default()),
    }
}

/// Process Bedrock model response to Golem LLM ChatEvent
pub fn process_model_response(response: BedrockModelResponse) -> ChatEvent {
    if let Some(error) = response.error {
        return ChatEvent::Error(Error {
            code: ErrorCode::ProviderError,
            message: error.message.unwrap_or_else(|| "Unknown Bedrock error".to_string()),
            provider_error_json: Some(serde_json::to_string(&error).unwrap_or_default()),
        });
    }
    
    let mut contents = Vec::new();
    let mut tool_calls = Vec::new();
    
    // Extract content parts
    if let Some(message) = response.message {
        for content_block in message.content.unwrap_or_default() {
            match content_block {
                BedrockContentBlock::Text(text_block) => {
                    contents.push(ContentPart::Text(text_block.text));
                }
                BedrockContentBlock::Image(_) => {
                    // Bedrock typically doesn't return images, but we handle the case
                }
            }
        }
        
        // Extract tool calls if present
        if let Some(tool_calls_vec) = message.tool_calls {
            for bedrock_tool_call in tool_calls_vec {
                tool_calls.push(ToolCall {
                    id: bedrock_tool_call.id,
                    name: bedrock_tool_call.function.name,
                    arguments_json: bedrock_tool_call.function.arguments,
                });
            }
        }
    }
    
    let metadata = create_response_metadata(&response);
    
    if contents.is_empty() && !tool_calls.is_empty() {
        ChatEvent::ToolRequest(tool_calls)
    } else {
        ChatEvent::Message(CompleteResponse {
            id: response.id.unwrap_or_else(|| "unknown".to_string()),
            content: contents,
            tool_calls,
            metadata,
        })
    }
}

/// Create response metadata from Bedrock response
pub fn create_response_metadata(response: &BedrockModelResponse) -> ResponseMetadata {
    ResponseMetadata {
        finish_reason: response.stop_reason.clone(),
        usage: response.usage.as_ref().map(|usage| Usage {
            input_tokens: Some(usage.input_tokens),
            output_tokens: Some(usage.output_tokens),
            total_tokens: Some(usage.input_tokens + usage.output_tokens),
        }),
        provider_id: response.id.clone(),
        timestamp: None,
        provider_metadata_json: None,
    }
}