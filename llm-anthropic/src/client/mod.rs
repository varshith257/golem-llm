use golem_llm::golem::llm::llm::{Error, ErrorCode};
use log::trace;
use reqwest::{Client, Method, Response, StatusCode};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Debug;

const BASE_URL: &str = "https://api.anthropic.com";

/// The Anthropic API client for creating model responses.
pub struct MessagesApi {
    api_key: String,
    client: Client,
}

impl MessagesApi {
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self { api_key, client }
    }

    pub fn send_messages(&self, request: MessagesRequest) -> Result<MessagesResponse, Error> {
        trace!("Sending request to Anthropic API: {request:?}");

        let response: Response = self
            .client
            .request(Method::POST, format!("{BASE_URL}/v1/messages"))
            .header("x-api-key", &self.api_key)
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesRequest {
    max_tokens: u32,
    messages: Vec<Message>,
    model: String,
    metadata: Option<MessagesRequestMetadata>,
    stop_sequences: Vec<String>,
    stream: bool,
    system: Vec<Content>, // can only be Text
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    // thinking
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesRequestMetadata {
    user_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    content: Vec<Content>,
    role: Role,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Content {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>, // citations
    },
    #[serde(rename = "image")]
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        input: Value,
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
        content: Vec<Content>, // can only be Text or Image
        is_error: bool,
    },
    // Document
    // Thinking
    // RedactedThinking
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheControl {
    #[serde(rename = "ephemeral")]
    Ephemeral,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ImageSource {
    #[serde(rename = "url")]
    Url { url: String },
    #[serde(rename = "base64")]
    Base64 { data: String, media_type: MediaType },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MediaType {
    #[serde(rename = "image/jpeg")]
    ImageJpeg,
    #[serde(rename = "image/png")]
    ImagePng,
    #[serde(rename = "image/svg+xml")]
    ImageGif,
    #[serde(rename = "image/webp")]
    ImageWebp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolChoice {
    #[serde(rename = "auto")]
    Auto {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    #[serde(rename = "any")]
    Any {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    #[serde(rename = "tool")]
    Tool {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    #[serde(rename = "none")]
    None {},
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Tool {
    #[serde(rename = "custom")]
    CustomTool {
        input_schema: Value,
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesResponse {
    content: Vec<Content>, // can only be Text or ToolUse (or Thinking / RedactedThinking)
    id: String,
    model: String,
    role: Role,
    stop_reason: Option<StopReason>,
    usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StopReason {
    #[serde(rename = "end_turn")]
    EndTurn,
    #[serde(rename = "max_tokens")]
    MaxTokens,
    #[serde(rename = "stop_sequence")]
    StopSequence,
    #[serde(rename = "tool_use")]
    ToolUse
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_read_input_tokens: Option<u32>,
    input_tokens: u32,
    output_tokens: u32
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Role {
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    error: ErrorResponseDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponseDetails {
    message: String,
    #[serde(rename = "type")]
    typ: String,
}

fn from_reqwest_error(details: impl AsRef<str>, err: reqwest::Error) -> Error {
    Error {
        code: ErrorCode::InternalError,
        message: format!("{}: {err}", details.as_ref()),
        provider_error_json: None,
    }
}

fn parse_response<T: DeserializeOwned + Debug>(response: Response) -> Result<T, Error> {
    let status = response.status();
    if status.is_success() {
        let body = response
            .json::<T>()
            .map_err(|err| from_reqwest_error("Failed to decode response body", err))?;

        trace!("Received response from Anthropic API: {body:?}");

        Ok(body)
    } else {
        let error_body = response
            .json::<ErrorResponse>()
            .map_err(|err| from_reqwest_error("Failed to receive error response body", err))?;

        trace!("Received {status} response from Anthropic API: {error_body:?}");

        Err(Error {
            code: error_code_from_status(status),
            message: format!("Request failed with {status}: {}", error_body.error.message),
            provider_error_json: Some(serde_json::to_string(&error_body).unwrap()),
        })
    }
}

fn error_code_from_status(status: StatusCode) -> ErrorCode {
    if status == StatusCode::TOO_MANY_REQUESTS {
        ErrorCode::RateLimitExceeded
    } else if status == StatusCode::UNAUTHORIZED
        || status == StatusCode::FORBIDDEN
        || status == StatusCode::PAYMENT_REQUIRED
    {
        ErrorCode::AuthenticationFailed
    } else if status.is_client_error() {
        ErrorCode::InvalidRequest
    } else {
        ErrorCode::InternalError
    }
}
