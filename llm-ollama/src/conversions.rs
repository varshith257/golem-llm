use crate::client::{OllamaChatRequest, OllamaMessage, OllamaRole};
use golem_llm::error::error_code_from_status;
use golem_llm::golem::llm::llm::{
    ChatEvent, CompleteResponse, Config, ContentPart, Error, ErrorCode, ImageDetail, Message,
    ResponseMetadata, Role, ToolCall, ToolDefinition, ToolResult, Usage,
};
use reqwest::StatusCode;
use std::collections::HashMap;
use std::str::FromStr;

pub fn messages_to_ollama_request(
    messages: Vec<Message>,
    config: Config,
) -> Result<OllamaChatRequest, Error> {
    let options = config
        .provider_options
        .into_iter()
        .map(|kv| (kv.key, kv.value))
        .collect::<HashMap<_, _>>();

    let mut ollama_messages = Vec::new();
    for message in &messages {
        let ollama_role = match &message.role {
            Role::User => OllamaRole::User,
            Role::Assistant => OllamaRole::Assistant,
            Role::System => OllamaRole::System,
            Role::Tool => OllamaRole::User, // fallback: tools as user input
        };

        let combined_content = message
            .content
            .iter()
            .map(|part| match part {
                ContentPart::Text(text) => text.clone(),
                ContentPart::Image(_) => "[IMAGE OMITTED]".to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n");

        ollama_messages.push(OllamaMessage {
            role: ollama_role,
            content: combined_content,
        });
    }

    Ok(OllamaChatRequest {
        model: config.model,
        messages: ollama_messages,
        temperature: config.temperature,
        top_p: options.get("top_p").and_then(|v| v.parse::<f32>().ok()),
        stop: config.stop_sequences,
        stream: Some(false),
    })
}

pub fn process_ollama_response(response: OllamaChatResponse) -> ChatEvent {
    let content = vec![ContentPart::Text(response.message.content)];

    let metadata = ResponseMetadata {
        finish_reason: Some(FinishReason::Stop),
        usage: None,
        provider_id: Some(response.model),
        timestamp: None,
        provider_metadata_json: None,
    };

    ChatEvent::Message(CompleteResponse {
        id: response.created_at,
        content,
        tool_calls: Vec::new(),
        metadata,
    })
}
