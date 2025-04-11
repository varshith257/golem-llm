use golem_llm::golem::llm::llm::{Error, ErrorCode};
use reqwest::{Client, Method, Response, StatusCode};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

const BASE_URL: &'static str = "https://api.openai.com";

pub struct ResponsesApi {
    openai_api_key: String,
    client: Client,
}

impl ResponsesApi {
    pub fn new(openai_api_key: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self {
            openai_api_key,
            client,
        }
    }

    pub fn create_model_response(
        &self,
        request: CreateModelResponseRequest,
    ) -> Result<CreateModelResponseResponse, Error> {
        let response: Response = self
            .client
            .request(Method::POST, format!("{BASE_URL}/v1/responses"))
            .bearer_auth(&self.openai_api_key)
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateModelResponseRequest {
    pub input: Input,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    // TODO: stop-sequences ???
    // TODO: what to expose through provider-options ???
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateModelResponseResponse {
    pub id: String,
    pub created_at: u64,
    pub error: Option<ErrorObject>,
    pub incomplete_details: Option<IncompleteDetailsObject>,
    pub status: Status,
    pub output: Vec<OutputItem>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        content: Vec<OutputMessageContent>,
        role: String,
        status: Status,
    },
    #[serde(rename = "function_call")]
    ToolCall {
        arguments: String,
        call_id: String,
        name: String,
        id: String,
        status: Status,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputMessageContent {
    #[serde(rename = "output_text")]
    Text {
        text: String,
        // TODO: do we need annotations?
    },
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorObject {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Status {
    #[serde(rename = "completed")]
    Completed,
    #[serde(rename = "failed")]
    Failed,
    #[serde(rename = "in_progress")]
    InProgress,
    #[serde(rename = "incomplete")]
    Incomplete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncompleteDetailsObject {
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Input {
    TextInput(String),
    List(Vec<InputItem>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InputItem {
    #[serde(rename = "message")]
    InputMessage { content: InnerInput, role: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InnerInput {
    TextInput(String),
    List(Vec<InnerInputItem>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InnerInputItem {
    #[serde(rename = "input_text")]
    TextInput { text: String },
    #[serde(rename = "input_image")]
    ImageInput {
        image_url: String,
        #[serde(default)]
        detail: Detail,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Detail {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "high")]
    High,
}

impl Default for Detail {
    fn default() -> Self {
        Detail::Auto
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Tool {
    #[serde(rename = "function")]
    Function {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        parameters: Option<serde_json::Value>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub input_tokens_details: InputTokensDetails,
    pub output_tokens: u32,
    pub output_tokens_details: OutputTokensDetails,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputTokensDetails {
    pub cached_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: u32,
}

fn from_reqwest_error(details: impl AsRef<str>, err: reqwest::Error) -> Error {
    Error {
        code: ErrorCode::InternalError,
        message: format!("{}: {err}", details.as_ref()),
        provider_error_json: None,
    }
}

fn parse_response<T: DeserializeOwned>(response: Response) -> Result<T, Error> {
    let status = response.status();
    if status.is_success() {
        let body = response
            .json::<T>()
            .map_err(|err| from_reqwest_error("Failed to decode response body", err))?;
        Ok(body)
    } else {
        let body = response
            .text()
            .map_err(|err| from_reqwest_error("Failed to receive error response body", err))?;

        Err(Error {
            code: error_code_from_status(status),
            message: format!("Request failed with {status}"),
            provider_error_json: Some(body),
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
