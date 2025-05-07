use crate::client::{
    BedrockContentBlock, BedrockImageContentBlock, BedrockMessage, BedrockTextContentBlock,
    BedrockTool, BedrockToolFunction, ErrorResponse, ErrorResponseDetails, InvokeModelRequest,
    InvokeModelResponse, InvokeResult, TextGenerationConfig,
};
use golem_llm::error::error_code_from_status;
use golem_llm::golem::llm::llm::{
    ChatEvent, CompleteResponse, Config, ContentPart, Error, ErrorCode, FinishReason, ImageDetail,
    Message, ResponseMetadata, Role, ToolCall, ToolDefinition, ToolResult, Usage,
};
use reqwest::StatusCode;
use serde_json::Value;
use std::collections::HashMap;
use std::str::FromStr;

/// Create a Bedrock model request from Golem LLM types
pub fn create_request(messages: Vec<Message>, config: Config) -> InvokeModelRequest {
    let prompt = messages_to_prompt(&messages);

    let options: HashMap<_, _> = config
        .provider_options
        .into_iter()
        .map(|kv| (kv.key, kv.value))
        .collect();

    let text_generation_config = TextGenerationConfig {
        temperature: config.temperature,
        top_p: options.get("top_p").and_then(|s| s.parse().ok()),
        top_k: options.get("top_k").and_then(|s| s.parse().ok()),
        max_token_count: config.max_tokens,
        stop_sequences: options
            .get("stop_sequences")
            .and_then(|s| serde_json::from_str(s).ok()),
    };

    let guardrail_identifier = options.get("guardrailIdentifier").cloned();
    let guardrail_version = options.get("guardrailVersion").cloned();

    InvokeModelRequest {
        input_text: prompt,
        text_generation_config,
        guardrail_identifier,
        guardrail_version,
    }
}

fn messages_to_prompt(messages: &[Message]) -> String {
    messages
        .iter()
        .map(|msg| {
            let prefix = match msg.role {
                Role::System => "[system] ",
                Role::User => "[user] ",
                Role::Assistant => "[assistant] ",
                Role::Tool => "[tool] ",
            };
            let body = msg
                .content
                .iter()
                .map(|cp| match cp {
                    ContentPart::Text(t) => t.clone(),
                    ContentPart::Image(i) => format!("[image:{}]", i.url),
                })
                .collect::<String>();
            format!("{prefix}{body}\n")
        })
        .collect()
}

pub fn tool_defs_to_tools(tool_definitions: &[ToolDefinition]) -> Result<Vec<BedrockTool>, Error> {
    let mut tools = Vec::new();
    for td in tool_definitions {
        let params: Value = serde_json::from_str(&td.parameters_schema).map_err(|e| Error {
            code: ErrorCode::InternalError,
            message: format!("Failed to parse tool schema for {}: {}", td.name, e),
            provider_error_json: None,
        })?;
        let function = BedrockToolFunction {
            name: td.name.clone(),
            description: td.description.clone().unwrap_or_default(),
            parameters: params,
        };
        tools.push(BedrockTool {
            r#type: "function".to_string(),
            function,
        });
    }
    Ok(tools)
}

pub fn tool_results_to_messages(tool_results: &[(ToolCall, ToolResult)]) -> Vec<Message> {
    let mut msgs = Vec::with_capacity(tool_results.len());
    for (call, result) in tool_results {
        msgs.push(Message {
            role: Role::Tool,
            name: Some(call.name.clone()),
            content: vec![ContentPart::Text(call.arguments_json.clone())],
        });
        let payload = match result {
            ToolResult::Success(s) => s.result_json.clone(),
            ToolResult::Error(err) => format!(
                r#"{{"error":{{"code":"{}","message":"{}"}}}}"#,
                err.error_code.clone().unwrap_or_default(),
                err.error_message
            ),
        };
        msgs.push(Message {
            role: Role::Tool,
            name: Some(call.name.clone()),
            content: vec![ContentPart::Text(payload)],
        });
    }
    msgs
}

pub fn to_bedrock_role_name(role: Role) -> String {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
        Role::Tool => "tool",
    }
    .to_string()
}

pub fn parse_error_code(code: String) -> ErrorCode {
    if let Some(code) = <u16 as FromStr>::from_str(&code)
        .ok()
        .and_then(|code| StatusCode::from_u16(code).ok())
    {
        error_code_from_status(code)
    } else {
        ErrorCode::InternalError
    }
}

pub fn process_model_response(resp: InvokeModelResponse) -> ChatEvent {
    let result = match resp.results.into_iter().next() {
        Some(r) => r,
        None => {
            return ChatEvent::Error(Error {
                code: ErrorCode::InternalError,
                message: "Bedrock returned zero results".into(),
                provider_error_json: None,
            })
        }
    };

    let text = result.output_text;
    let mut contents = vec![ContentPart::Text(text)];

    let metadata = ResponseMetadata {
        finish_reason: result
            .completion_reason
            .as_deref()
            .map(string_to_finish_reason),
        usage: None,
        provider_id: None,
        timestamp: None,
        provider_metadata_json: None,
    };

    ChatEvent::Message(CompleteResponse {
        id: "".into(),
        content: contents,
        tool_calls: vec![],
        metadata,
    })
}

fn string_to_finish_reason(s: &str) -> FinishReason {
    match s {
        "stopSequence" => FinishReason::Stop,
        "length" => FinishReason::Length,
        other => {
            log::warn!("Unknown Bedrock finishReason={}", other);
            FinishReason::Other
        }
    }
}

pub fn create_response_metadata(resp: &InvokeModelResponse) -> ResponseMetadata {
    let finish_reason = resp
        .results
        .get(0)
        .and_then(|r| r.completion_reason.as_deref())
        .map(string_to_finish_reason);

    ResponseMetadata {
        finish_reason,
        usage: None,
        provider_id: None,
        timestamp: None,
        provider_metadata_json: None,
    }
}
