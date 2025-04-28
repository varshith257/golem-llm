use std::fmt::Debug;
use reqwest::{Client, Method, Response, StatusCode};
use reqwest::header::HeaderValue;
use serde::de::DeserializeOwned;
use golem_llm::event_source;
use golem_llm::event_source::EventSource;
use golem_llm::golem::llm::llm::{Error, ErrorCode};
use log::trace;
use serde::{Deserialize, Serialize};

const BASE_URL: &str = "https://api.x.ai";

/// The Completions API client for creating model responses.
pub struct CompletionsApi {
    api_key: String,
    client: Client,
}

impl CompletionsApi {
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self { api_key, client }
    }

    pub fn send_messages(&self, request: CompletionsRequest) -> Result<CompletionsResponse, Error> {
        trace!("Sending request to XAI API: {request:?}");

        let response: Response = self
            .client
            .request(Method::POST, format!("{BASE_URL}/v1/chat/completions"))
            .bearer_auth(self.api_key.clone())
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }

    pub fn stream_send_messages(&self, request: CompletionsRequest) -> Result<EventSource, Error> {
        trace!("Sending request to XAI API: {request:?}");

        let response: Response = self
            .client
            .request(Method::POST, format!("{BASE_URL}/v1/chat/completions"))
            .bearer_auth(self.api_key.clone())
            .header(
                reqwest::header::ACCEPT,
                HeaderValue::from_static("text/event-stream"),
            )
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        trace!("Initializing SSE stream");

        EventSource::new(response)
            .map_err(|err| from_event_source_error("Failed to create SSE stream", err))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionsRequest {
    pub messages: Vec<Message>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<Effort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Tool {
    #[serde(rename = "function")]
    Function {
        function: Function
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum Message {
    #[serde(rename = "system")]
    System {
        content: Vec<ContentPart>,
        name: Option<String>
    },
    #[serde(rename = "user")]
    User {
        content: Vec<ContentPart>,
        name: Option<String>
    },
    #[serde(rename = "assistant")]
    Assistant {
        content: Option<Vec<ContentPart>>,
        name: Option<String>,
        tool_calls: Option<Vec<ToolCall>>
    },
    #[serde(rename = "tool")]
    Tool {
        content: Vec<ContentPart>,
        name: Option<String>,
        tool_call_id: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    TextInput(String),
    List(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "input_text")]
    TextInput { text: String },
    #[serde(rename = "input_image")]
    ImageInput {
        image_url: ImageUrl,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum Detail {
    #[serde(rename = "auto")]
    #[default]
    Auto,
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "high")]
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Effort {
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "high")]
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    url: String,
    detail: Option<Detail>
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolCall {
    #[serde(rename = "function")]
    Function {
        function: FunctionCall,
        id: String,
        index: Option<u32>,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub arguments: String,
    pub name: String
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionsResponse {
    pub choices: Vec<Choice>,
    pub created: u64,
    pub id: String,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    finish_reason: Option<FinishReason>,
    index: u32,
    message: ResponseMessage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    #[serde(rename = "stop")]
    Stop,
    #[serde(rename = "length")]
    Length,
    #[serde(rename = "max_tokens")]
    MaxTokens,
    #[serde(rename = "end_turn")]
    EndTurn
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMessage {
    content: Option<String>,
    reasoning_content: Option<String>,
    refusal: Option<String>,
    tool_call: Vec<ToolCall>
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub completion_tokens: u32,
    pub completion_token_details: CompletionTokenDetails,
    pub prompt_tokens: u32,
    pub prompt_token_details: PromptTokenDetails,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionTokenDetails {
    pub accepted_prediction_tokens: u32,
    pub audio_tokens: u32,
    pub reasoning_tokens: u32,
    pub rejected_prediction_tokens: u32
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTokenDetails {
    pub audio_tokens: u32,
    pub cached_tokens: u32,
    pub image_tokens: u32,
    pub text_tokens: u32
}

// TODO: to shared lib
fn from_reqwest_error(details: impl AsRef<str>, err: reqwest::Error) -> Error {
    Error {
        code: ErrorCode::InternalError,
        message: format!("{}: {err}", details.as_ref()),
        provider_error_json: None,
    }
}

// TODO: to shared lib
fn from_event_source_error(details: impl AsRef<str>, err: event_source::error::Error) -> Error {
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
            .text()
            .map_err(|err| from_reqwest_error("Failed to receive error response body", err))?;

        trace!("Received {status} response from Anthropic API: {error_body:?}");

        Err(Error {
            code: error_code_from_status(status),
            message: format!("Request failed with {status}"),
            provider_error_json: Some(serde_json::to_string(&error_body).unwrap()),
        })
    }
}

// TODO: to shared lib
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
