use golem_llm::error::{error_code_from_status, from_event_source_error, from_reqwest_error};
use golem_llm::event_source::EventSource;
use golem_llm::golem::llm::llm::Error;
use log::trace;
use reqwest::header::HeaderValue;
use reqwest::{Client, Method, Response};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::time::{SystemTime, UNIX_EPOCH};
use aws_sigv4::http_request::{sign, SigningParams, SigningSettings};
use aws_sigv4::sign::v4;
use http::Uri;

/// The AWS Bedrock API client for creating model responses.
///
/// Based on AWS Bedrock API documentation: https://docs.aws.amazon.com/bedrock/latest/APIReference/welcome.html
pub struct BedrockApi {
    aws_access_key: String,
    aws_secret_key: String,
    aws_region: String,
    client: Client,
}

impl BedrockApi {
    pub fn new(aws_access_key: String, aws_secret_key: String, aws_region: String) -> Self {
        let client = Client::builder()
            .build()
            .expect("Failed to initialize HTTP client");
        Self {
            aws_access_key,
            aws_secret_key,
            aws_region,
            client,
        }
    }

    fn get_endpoint_url(&self, model_id: &str) -> String {
        format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/invoke",
            self.aws_region, model_id
        )
    }

    fn get_endpoint_url_stream(&self, model_id: &str) -> String {
        format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/invoke-with-response-stream",
            self.aws_region, model_id
        )
    }

    async fn sign_request(&self, url: &str, payload: &[u8]) -> Result<reqwest::Request, Error> {
        let uri = url.parse::<Uri>().map_err(|err| Error {
            code: "InvalidUri".to_string(),
            message: format!("Invalid URI: {}", err),
            provider_error_json: None,
        })?;

        let mut request = http::Request::builder()
            .method("POST")
            .uri(uri)
            .header("Content-Type", "application/json")
            .body(payload.to_vec())
            .map_err(|err| Error {
                code: "RequestBuildError".to_string(),
                message: format!("Failed to build request: {}", err),
                provider_error_json: None,
            })?;

        // AWS SigV4 Signing
        let signing_settings = SigningSettings::default();
        let signing_params = SigningParams {
            access_key: &self.aws_access_key,
            secret_key: &self.aws_secret_key,
            region: &self.aws_region,
            service_name: "bedrock",
            time: SystemTime::now(),
            settings: &signing_settings,
        };

        sign(&mut request, &v4::SigningKey::new(
            signing_params.secret_key.as_bytes(),
            signing_params.region,
            signing_params.service_name,
            &signing_params.time.duration_since(UNIX_EPOCH).unwrap().as_secs().to_string(),
        ))
        .map_err(|err| Error {
            code: "SigningError".to_string(),
            message: format!("Failed to sign request: {}", err),
            provider_error_json: None,
        })?;

        // Convert http::Request to reqwest::Request
        let (parts, body) = request.into_parts();
        let mut req_builder = self.client.request(
            parts.method.try_into().unwrap(),
            parts.uri.to_string(),
        );

        for (name, value) in parts.headers.iter() {
            req_builder = req_builder.header(name, value);
        }

        req_builder.body(body).build().map_err(|err| Error {
            code: "RequestBuildError".to_string(),
            message: format!("Failed to build reqwest request: {}", err),
            provider_error_json: None,
        })
    }

    pub async fn create_model_response(
        &self,
        request: BedrockModelRequest,
    ) -> Result<BedrockModelResponse, Error> {
        let model_id = request.model_id.clone();
        let url = self.get_endpoint_url(&model_id);
        
        trace!("Sending request to AWS Bedrock API: {request:?}");
        
        let body = serde_json::to_vec(&request)
            .map_err(|err| Error {
                code: "SerializationError".to_string(),
                message: format!("Failed to serialize request: {}", err),
                provider_error_json: None,
            })?;
            
        let req = self.sign_request(&url, &body).await?;
        
        let response: Response = self.client
            .execute(req)
            .await
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        parse_response(response).await
    }

    pub async fn stream_model_response(
        &self,
        request: BedrockModelRequest,
    ) -> Result<EventSource, Error> {
        let model_id = request.model_id.clone();
        let streaming_request = BedrockStreamingRequest {
            model_id: model_id.clone(),
            body: request,
        };
        
        let url = self.get_endpoint_url_stream(&model_id);
        
        trace!("Sending streaming request to AWS Bedrock API: {streaming_request:?}");
        
        let body = serde_json::to_vec(&streaming_request.body)
            .map_err(|err| Error {
                code: "SerializationError".to_string(),
                message: format!("Failed to serialize request: {}", err),
                provider_error_json: None,
            })?;
            
        let req = self.sign_request(&url, &body).await?;
        
        let response: Response = self.client
            .execute(req)
            .await
            .map_err(|err| from_reqwest_error("Request failed", err))?;

        trace!("Initializing SSE stream");

        EventSource::new(response)
            .map_err(|err| from_event_source_error("Failed to create SSE stream", err))
    }
}

/// Model-agnostic request structure that will be mapped to provider-specific formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockModelRequest {
    /// The ID of the AWS Bedrock model to use (e.g., "anthropic.claude-v2", "amazon.titan-text-express-v1")
    #[serde(skip_serializing)]
    pub model_id: String,
    
    /// The prompt or messages for the model
    pub prompt: String,
    
    /// The maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    
    /// Temperature control for randomness (0.0 to 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    
    /// Top-p sampling control (0.0 to 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    
    /// Whether to stream the response
    #[serde(skip_serializing)]
    pub stream: bool,
    
    /// Provider-specific parameters that will be included directly in the request
    #[serde(flatten)]
    pub provider_parameters: serde_json::Value,
}

/// Structure for streaming requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockStreamingRequest {
    #[serde(skip_serializing)]
    pub model_id: String,
    pub body: BedrockModelRequest,
}

/// Response structure that contains the model's generated text and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockModelResponse {
    /// The model's generated completion
    pub completion: String,
    
    /// Usage statistics for the request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<BedrockUsage>,
    
    /// Provider-specific metadata that was included in the response
    #[serde(flatten)]
    pub provider_metadata: serde_json::Value,
}

/// Usage information for token counting and billing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

/// Stream event for token-by-token responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BedrockStreamEvent {
    /// The chunk of generated text
    pub completion: String,
    
    /// Flag indicating if this is the final chunk
    pub is_final: bool,
    
    /// Provider-specific metadata for this chunk
    #[serde(flatten)]
    pub provider_metadata: serde_json::Value,
}

/// Parses the response from the AWS Bedrock API
async fn parse_response<T: DeserializeOwned + Debug>(response: Response) -> Result<T, Error> {
    let status = response.status();
    if status.is_success() {
        let body = response
            .json::<T>()
            .await
            .map_err(|err| from_reqwest_error("Failed to decode response body", err))?;

        trace!("Received response from AWS Bedrock API: {body:?}");

        Ok(body)
    } else {
        let body = response
            .text()
            .await
            .map_err(|err| from_reqwest_error("Failed to receive error response body", err))?;

        trace!("Received {status} response from AWS Bedrock API: {body:?}");

        Err(Error {
            code: error_code_from_status(status),
            message: format!("Request failed with {status}"),
            provider_error_json: Some(body),
        })
    }
}

/// Model-specific adapters for different AWS Bedrock models
pub mod model_adapters {
    use super::*;
    
    /// Adapter for Anthropic Claude models on AWS Bedrock
    pub struct ClaudeAdapter;
    
    impl ClaudeAdapter {
        pub fn adapt_request(request: &BedrockModelRequest) -> serde_json::Value {
            let mut claude_request = serde_json::json!({
                "prompt": request.prompt,
                "max_tokens_to_sample": request.max_tokens.unwrap_or(500)
            });
            
            if let Some(temp) = request.temperature {
                claude_request["temperature"] = serde_json::json!(temp);
            }
            
            if let Some(top_p) = request.top_p {
                claude_request["top_p"] = serde_json::json!(top_p);
            }
            
            // Merge any provider-specific parameters
            if let serde_json::Value::Object(provider_params) = &request.provider_parameters {
                if let serde_json::Value::Object(claude_map) = claude_request {
                    let mut merged = claude_map;
                    for (key, value) in provider_params {
                        merged.insert(key.clone(), value.clone());
                    }
                    return serde_json::Value::Object(merged);
                }
            }
            
            claude_request
        }
        
        pub fn parse_response(response_json: serde_json::Value) -> Result<BedrockModelResponse, Error> {
            let completion = response_json["completion"]
                .as_str()
                .ok_or_else(|| Error {
                    code: "ParseError".to_string(),
                    message: "Missing 'completion' field in Claude response".to_string(),
                    provider_error_json: Some(response_json.to_string()),
                })?
                .to_string();
                
            Ok(BedrockModelResponse {
                completion,
                usage: None, // Claude on Bedrock doesn't include usage info in the response
                provider_metadata: response_json,
            })
        }
    }
    
    /// Adapter for Amazon Titan models on AWS Bedrock
    pub struct TitanAdapter;
    
    impl TitanAdapter {
        pub fn adapt_request(request: &BedrockModelRequest) -> serde_json::Value {
            let mut titan_request = serde_json::json!({
                "inputText": request.prompt,
                "textGenerationConfig": {
                    "maxTokenCount": request.max_tokens.unwrap_or(500)
                }
            });
            
            if let Some(temp) = request.temperature {
                if let Some(config) = titan_request["textGenerationConfig"].as_object_mut() {
                    config.insert("temperature".to_string(), serde_json::json!(temp));
                }
            }
            
            if let Some(top_p) = request.top_p {
                if let Some(config) = titan_request["textGenerationConfig"].as_object_mut() {
                    config.insert("topP".to_string(), serde_json::json!(top_p));
                }
            }
            
            // Merge any provider-specific parameters
            if let serde_json::Value::Object(provider_params) = &request.provider_parameters {
                if let Some(config) = titan_request["textGenerationConfig"].as_object_mut() {
                    for (key, value) in provider_params {
                        config.insert(key.clone(), value.clone());
                    }
                }
            }
            
            titan_request
        }
        
        pub fn parse_response(response_json: serde_json::Value) -> Result<BedrockModelResponse, Error> {
            let completion = response_json["results"]
                .as_array()
                .and_then(|results| results.first())
                .and_then(|first| first["outputText"].as_str())
                .ok_or_else(|| Error {
                    code: "ParseError".to_string(),
                    message: "Missing 'results[0].outputText' field in Titan response".to_string(),
                    provider_error_json: Some(response_json.to_string()),
                })?
                .to_string();
                
            Ok(BedrockModelResponse {
                completion,
                usage: None, // Titan models don't include usage info in the response
                provider_metadata: response_json,
            })
        }
    }
    
    /// Factory for creating the appropriate adapter based on model ID
    pub fn get_adapter(model_id: &str) -> Box<dyn Fn(&BedrockModelRequest) -> serde_json::Value> {
        if model_id.starts_with("anthropic.claude") {
            Box::new(ClaudeAdapter::adapt_request)
        } else if model_id.starts_with("amazon.titan") {
            Box::new(TitanAdapter::adapt_request)
        } else {
            // Default adapter that passes through the request as-is
            Box::new(|request| {
                serde_json::to_value(request).unwrap_or_else(|_| serde_json::json!({}))
            })
        }
    }
    
    /// Factory for creating the appropriate response parser based on model ID
    pub fn get_response_parser(model_id: &str) -> Box<dyn Fn(serde_json::Value) -> Result<BedrockModelResponse, Error>> {
        if model_id.starts_with("anthropic.claude") {
            Box::new(ClaudeAdapter::parse_response)
        } else if model_id.starts_with("amazon.titan") {
            Box::new(TitanAdapter::parse_response)
        } else {
            // Default parser that attempts to extract a standard completion field
            Box::new(|response_json| {
                // Try common response patterns
                let completion = response_json["completion"]
                    .as_str()
                    .or_else(|| response_json["output"].as_str())
                    .or_else(|| response_json["text"].as_str())
                    .or_else(|| response_json["generated_text"].as_str())
                    .ok_or_else(|| Error {
                        code: "ParseError".to_string(),
                        message: "Could not find completion text in response".to_string(),
                        provider_error_json: Some(response_json.to_string()),
                    })?
                    .to_string();
                    
                Ok(BedrockModelResponse {
                    completion,
                    usage: None,
                    provider_metadata: response_json,
                })
            })
        }
    }
}
