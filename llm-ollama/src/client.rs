use golem_llm::error::{error_code_from_status, from_event_source_error, from_reqwest_error};
use golem_llm::event_source::EventSource;
use golem_llm::golem::llm::llm::Error;
use log::trace;
use reqwest::header::HeaderValue;
use reqwest::{Client, Method, Response};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

const BASE_URL: &str = "http://localhost:11434";

/// The Ollama API client for creating model responses.
pub struct OllamaApi {
    client: Client,
}

impl OllamaApi {
    pub fn new() -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self {
            client,
        }
    }

    pub fn generate(
        &self,
        request: OllamaGenerateRequest,
    ) -> Result<OllamaGenerateResponse, Error> {
        trace!("Sending generate request to Ollama API: {request:?}");

        let response: Response = self
            .client
            .request(Method::POST, format!("{BASE_URL}/api/generate"))
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }

    pub fn chat(
        &self,
        request: OllamaChatRequest,
    ) -> Result<OllamaChatResponse, Error> {
        trace!("Sending chat request to Ollama API: {request:?}");

        let response: Response = self
            .client
            .request(Method::POST, format!("{BASE_URL}/api/chat"))
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }

    pub fn stream_chat(&self, request: OllamaChatRequest) -> Result<EventSource, Error> {
        trace!("Sending streaming chat request to Ollama API: {request:?}");
    
        let response: Response = self.client
            .request(Method::POST, format!("{BASE_URL}/api/chat"))
            .header(reqwest::header::ACCEPT, HeaderValue::from_static("text/event-stream"))
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;
    
        trace!("Initializing SSE stream");
    
        EventSource::new(response)
            .map_err(|err| from_event_source_error("Failed to create SSE stream", err))
    }
    
    pub fn embeddings(
        &self,
        request: OllamaEmbeddingsRequest,
    ) -> Result<OllamaEmbeddingsResponse, Error> {
        trace!("Sending embeddings request to Ollama API: {request:?}");

        let response: Response = self
            .client
            .request(Method::POST, format!("{BASE_URL}/api/embeddings"))
            .json(&request)
            .send()
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response)
    }

    pub fn handle_unsupported(feature: &str) -> Result<(), Error> {
        Err(Error {
            code: "NotSupported".into(),
            message: format!("Ollama does not support {feature}"),
            provider_error_json: None,
        })
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaMessage {
    pub role: OllamaRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OllamaRole {
    User,
    Assistant,
    System,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputMessageContent {
    #[serde(rename = "output_text")]
    Text { text: String },
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorObject {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaErrorResponse {
    pub error: String,
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
pub struct OllamaGenerateRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaEmbeddingsRequest {
    pub model: String,
    pub prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatResponse {
    pub message: OllamaMessageResult,
    pub created_at: String,
    pub model: String,
    pub done: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaMessageResult {
    pub role: OllamaRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaGenerateResponse {
    pub response: String,
    pub created_at: String,
    pub model: String,
    pub done: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaEmbeddingsResponse {
    pub embeddings: Vec<f32>,
    pub created_at: String,
    pub model: String,
}

fn parse_response<T: DeserializeOwned + Debug>(response: Response) -> Result<T, Error> {
    let status = response.status();
    if status.is_success() {
        let body = response
            .json::<T>()
            .map_err(|err| from_reqwest_error("Failed to decode response body", err))?;

            trace!("Received response from Ollama API: {body:?}");

        Ok(body)
    } else {
        let error_body = response
            .json::<OllamaErrorResponse>()
            .map_err(|err| from_reqwest_error("Failed to receive error response body", err))?;

            trace!("Received {status} response from Ollama API: {error_body:?}");

        Err(Error {
            code: error_code_from_status(status),
            message: format!("Request failed with {status}: {}", error_body.error),
            provider_error_json: Some(serde_json::to_string(&error_body).unwrap()),
        })
    }
}