use chrono::Utc;
use golem_llm::error::{error_code_from_status, from_event_source_error, from_reqwest_error};
use golem_llm::event_source::EventSource;
use golem_llm::golem::llm::llm::Error;
use hex;
use hmac::{Hmac, Mac};
use log::trace;
use reqwest::header::HeaderValue;
use reqwest::{Client, Method, Response};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Debug;

type HmacSha256 = Hmac<Sha256>;

/// Configuration for AWS Bedrock runtime calls.
#[derive(Debug, Clone)]
pub struct BedrockRuntimeConfig {
    /// Your AWS access key ID.
    pub access_key_id: String,
    /// Your AWS secret access key.
    pub secret_access_key: String,
    /// Optional session token (if using STS/IAM role).
    pub session_token: Option<String>,
    /// AWS region, e.g. "us-east-1".
    pub region: String,
    /// The runtime endpoint host, e.g. "bedrock-runtime.us-east-1.amazonaws.com".
    pub endpoint: String,
}

/// The client for Bedrock runtime (inference) APIs.
pub struct BedrockRuntimeApi {
    config: BedrockRuntimeConfig,
    http_client: Client,
}

impl BedrockRuntimeApi {
    /// Construct a new Bedrock runtime client.
    pub fn new(config: BedrockRuntimeConfig) -> Self {
        let http_client = Client::builder()
            .build()
            .expect("Failed to build HTTP client");
        Self {
            config,
            http_client,
        }
    }

    /// InvokeModel: synchronous text (or embedding/image) generation.
    pub fn invoke_model(
        &self,
        model_id: &str,
        request: &InvokeModelRequest,
    ) -> Result<InvokeModelResponse, Error> {
        trace!("Bedrock InvokeModel request for model {model_id:?}: {request:?}");

        // 1) Serialize request body
        let body = serde_json::to_string(request).map_err(|e| Error {
            code: error_code_from_status(reqwest::StatusCode::INTERNAL_SERVER_ERROR),
            message: format!("Serialization error: {}", e),
            provider_error_json: None,
        })?;

        // 2) Prepare signing: include the contentType query param in the path
        let canonical_uri = format!("/model/{model_id}/invoke");
        let (amz_date, auth_header, body_sha256) = self.sign_request("POST", &canonical_uri, &body);

        // 3) Build full URL
        let url = format!(
            "https://{}{}?contentType=application/json",
            self.config.endpoint, canonical_uri
        );

        // 4) Assemble reqwest request
        let mut req = self
            .http_client
            .request(Method::POST, &url)
            .header("Host", &self.config.endpoint)
            .header("X-Amz-Date", &amz_date)
            .header("X-Amz-Content-Sha256", &body_sha256)
            .header("Authorization", auth_header)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json");

        if let Some(token) = &self.config.session_token {
            req = req.header("X-Amz-Security-Token", token);
        }

        // 5) Send
        let response = req
            .body(body)
            .send()
            .map_err(|e| from_reqwest_error("Bedrock request failed", e))?;

        // 6) Parse JSON or return typed Error
        parse_response(response)
    }

    /// InvokeModel with streaming SSE (if supported by the model).
    pub fn stream_invoke_model(
        &self,
        model_id: &str,
        request: &InvokeModelRequest,
    ) -> Result<EventSource, Error> {
        trace!("Bedrock InvokeModel (stream) for model {model_id:?}: {request:?}");

        let body = serde_json::to_string(request).map_err(|e| Error {
            code: error_code_from_status(reqwest::StatusCode::INTERNAL_SERVER_ERROR),
            message: format!("Serialization error: {}", e),
            provider_error_json: None,
        })?;

        let canonical_uri = format!("/model/{model_id}/invoke");
        let (amz_date, auth_header, body_sha256) = self.sign_request("POST", &canonical_uri, &body);

        let url = format!(
            "https://{}{}?contentType=application/json",
            self.config.endpoint, canonical_uri
        );

        let mut req = self
            .http_client
            .request(Method::POST, &url)
            .header("Host", &self.config.endpoint)
            .header("X-Amz-Date", &amz_date)
            .header("X-Amz-Content-Sha256", &body_sha256)
            .header("Authorization", auth_header)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream");

        if let Some(token) = &self.config.session_token {
            req = req.header("X-Amz-Security-Token", token);
        }

        let response = req
            .body(body)
            .send()
            .map_err(|e| from_reqwest_error("Bedrock streaming request failed", e))?;

        EventSource::new(response)
            .map_err(|e| from_event_source_error("Failed to initialize Bedrock SSE", e))
    }

    /// Builds AWS4-HMAC-SHA256 signature for a request.
    fn sign_request(
        &self,
        method: &str,
        canonical_uri: &str,
        body: &str,
    ) -> (
        String, /*amz-date*/
        String, /*Authorization*/
        String, /*body-sha256*/
    ) {
        let now = Utc::now();
        let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
        let date_stamp = now.format("%Y%m%d").to_string();

        // Payload hash
        let body_sha256 = hex::encode(Sha256::digest(body.as_bytes()));

        // Canonical request
        let canonical_querystring = "contentType=application/json";
        let host = &self.config.endpoint;
        let canonical_headers = format!(
            "host:{}\nx-amz-date:{}\nx-amz-content-sha256:{}\n",
            host, amz_date, body_sha256,
        );
        let signed_headers = "host;x-amz-date;x-amz-content-sha256";
        let hashed_request = hex::encode(Sha256::digest(
            format!(
                "{method}\n{uri}\n{qs}\n{headers}\n{signed}\n{hash}",
                method = method,
                uri = canonical_uri,
                qs = canonical_querystring,
                headers = canonical_headers,
                signed = signed_headers,
                hash = body_sha256,
            )
            .as_bytes(),
        ));

        // String to sign
        let credential_scope = format!(
            "{}/{}/bedrock-runtime/aws4_request",
            date_stamp, self.config.region
        );
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{hash}",
            amz_date = amz_date,
            scope = credential_scope,
            hash = hashed_request,
        );

        // Derive signing key
        let k_secret = format!("AWS4{}", self.config.secret_access_key);
        let k_date = hmac_sign(k_secret.as_bytes(), &date_stamp);
        let k_region = hmac_sign(&k_date, &self.config.region);
        let k_service = hmac_sign(&k_region, "bedrock-runtime");
        let k_signing = hmac_sign(&k_service, "aws4_request");

        // Signature
        let signature = hex::encode(hmac_sign(&k_signing, &string_to_sign));

        // Authorization header
        let auth_header = format!(
            "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
            self.config.access_key_id, credential_scope, signed_headers, signature,
        );

        (amz_date, auth_header, body_sha256)
    }
}

/// HMAC-SHA256 helper.
fn hmac_sign(key: &[u8], msg: &str) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC can take any key size");
    mac.update(msg.as_bytes());
    mac.finalize().into_bytes().to_vec()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TextGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    #[serde(rename = "topP", skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    #[serde(rename = "topK", skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    #[serde(rename = "maxTokenCount", skip_serializing_if = "Option::is_none")]
    pub max_token_count: Option<u32>,

    #[serde(rename = "stopSequences", skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

pub struct BedrockTool {
    pub r#type: String,
    pub function: BedrockToolFunction,
}

pub struct BedrockToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

pub struct BedrockMessage {
    pub role: String,
    pub content: Vec<BedrockContentBlock>,
}

pub enum BedrockContentBlock {
    Text(BedrockTextContentBlock),
    Image(BedrockImageContentBlock),
}

pub struct BedrockTextContentBlock {
    pub text: String,
}

pub struct BedrockImageContentBlock {
    pub source: String,
    pub detail: String,
}

/// Request body for InvokeModel.
#[derive(Debug, Serialize, Deserialize)]
pub struct InvokeModelRequest {
    /// The model input text (for text models) or JSON payload.
    #[serde(rename = "inputText")]
    pub input_text: String,

    /// Generation parameters.
    #[serde(rename = "textGenerationConfig")]
    pub text_generation_config: TextGenerationConfig,

    /// Optional guardrail settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "guardrailIdentifier")]
    pub guardrail_identifier: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "guardrailVersion")]
    pub guardrail_version: Option<String>,
}

/// A single result from InvokeModel.
#[derive(Debug, Serialize, Deserialize)]
pub struct InvokeModelResponse {
    #[serde(rename = "inputTextTokenCount")]
    pub input_text_token_count: u32,

    pub results: Vec<InvokeResult>,
}

/// One completion candidate.
#[derive(Debug, Serialize, Deserialize)]
pub struct InvokeResult {
    #[serde(rename = "tokenCount")]
    pub token_count: u32,

    #[serde(rename = "outputText")]
    pub output_text: String,

    #[serde(rename = "completionReason")]
    pub completion_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum OutputItem {
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorResponseDetails,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponseDetails {
    pub message: String,
    #[serde(rename = "type")]
    pub typ: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResponseOutputTextDelta {
    pub content_index: u32,
    pub delta: String,
    pub item_id: String,
    pub output_index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResponseOutputItemDone {
    pub item: OutputItem,
    pub output_index: u32,
}

/// Parse JSON success or produce a typed `Error`.
fn parse_response<T: DeserializeOwned + Debug>(response: Response) -> Result<T, Error> {
    let status = response.status();
    if status.is_success() {
        let body = response
            .json::<T>()
            .map_err(|err| from_reqwest_error("Failed to decode response body", err))?;
        trace!("Received response from Bedrock API: {body:?}");

        Ok(body)
    } else {
        let error_body = response
            .json::<ErrorResponse>()
            .map_err(|err| from_reqwest_error("Failed to receive error response body", err))?;
        trace!("Received {status} response from Bedrock API: {error_body:?}");

        Err(Error {
            code: error_code_from_status(status),
            message: format!("Request failed with {status}: {}", error_body.error.message),
            provider_error_json: Some(serde_json::to_string(&error_body).unwrap()),
        })
    }
}
