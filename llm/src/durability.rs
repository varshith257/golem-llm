use std::marker::PhantomData;

pub struct DurableOpenAI<Impl> {
    phantom: PhantomData<Impl>,
}

#[cfg(not(feature = "durability"))]
mod passthrough_impl {
    use crate::durability::DurableOpenAI;
    use crate::golem::llm::llm::{
        ChatEvent, ChatStream, Config, Guest, Message, ToolCall, ToolResult,
    };

    impl<Impl: Guest> Guest for DurableOpenAI<Impl> {
        type ChatStream = Impl::ChatStream;

        fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
            warn!("DEBUG: calling send() impl without durability");
            Impl::send(messages, config)
        }

        fn continue_(
            messages: Vec<Message>,
            tool_results: Vec<(ToolCall, ToolResult)>,
            config: Config,
        ) -> ChatEvent {
            Impl::continue_(messages, tool_results, config)
        }

        fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
            Impl::stream(messages, config)
        }
    }
}

#[cfg(feature = "durability")]
mod durable_impl {
    use crate::durability::DurableOpenAI;
    use crate::golem::llm::llm::{
        ChatEvent, ChatStream, CompleteResponse, Config, ContentPart, Error, ErrorCode,
        FinishReason, Guest, ImageDetail, ImageUrl, Kv, Message, ResponseMetadata, Role, ToolCall,
        ToolDefinition, ToolResult, Usage,
    };
    use golem_rust::bindings::golem::durability::durability::DurableFunctionType;
    use golem_rust::durability::Durability;
    use golem_rust::value_and_type::type_builder::TypeNodeBuilder;
    use golem_rust::value_and_type::{FromValueAndType, IntoValue};
    use golem_rust::wasm_rpc::{NodeBuilder, WitValueExtractor};
    use golem_rust::{with_persistence_level, PersistenceLevel};
    use std::fmt::{Display, Formatter};
    use log::warn;

    impl<Impl: Guest> Guest for DurableOpenAI<Impl> {
        type ChatStream = Impl::ChatStream;

        fn send(messages: Vec<Message>, config: Config) -> ChatEvent {
            warn!("DEBUG: wrapping send() impl with custom durability");

            let durability = Durability::<ChatEvent, UnusedError>::new(
                "golem_llm",
                "send",
                DurableFunctionType::WriteRemote,
            );
            if durability.is_live() {
                let result = with_persistence_level(PersistenceLevel::PersistNothing, || {
                    Impl::send(messages.clone(), config.clone())
                });
                durability.persist_infallible(SendInput { messages, config }, result)
            } else {
                durability.replay_infallible()
            }
        }

        fn continue_(
            messages: Vec<Message>,
            tool_results: Vec<(ToolCall, ToolResult)>,
            config: Config,
        ) -> ChatEvent {
            Impl::continue_(messages, tool_results, config)
        }

        fn stream(messages: Vec<Message>, config: Config) -> ChatStream {
            Impl::stream(messages, config)
        }
    }

    //   variant chat-event {
    //     message(complete-response),
    //     tool-request(list<tool-call>),
    //     error(error),
    //   }
    impl IntoValue for ChatEvent {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            match self {
                ChatEvent::Message(complete_response) => {
                    let builder = builder.variant(0);
                    complete_response.add_to_builder(builder).finish()
                }
                ChatEvent::ToolRequest(tool_calls) => {
                    let builder = builder.variant(1);
                    tool_calls.add_to_builder(builder).finish()
                }
                ChatEvent::Error(error) => {
                    let builder = builder.variant(2);
                    error.add_to_builder(builder).finish()
                }
            }
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.variant();
            builder = CompleteResponse::add_to_type_builder(builder.case("message"));
            builder = Vec::<ToolCall>::add_to_type_builder(builder.case("tool-request"));
            builder = Error::add_to_type_builder(builder.case("error"));
            builder.finish()
        }
    }

    impl FromValueAndType for ChatEvent {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            let (idx, inner) = extractor
                .variant()
                .ok_or_else(|| "ChatEvent should be variant".to_string())?;
            match idx {
                0 => Ok(ChatEvent::Message(CompleteResponse::from_extractor(
                    &inner.ok_or_else(|| "Missing message body".to_string())?,
                )?)),
                1 => Ok(ChatEvent::ToolRequest(Vec::<ToolCall>::from_extractor(
                    &inner.ok_or_else(|| "Missing tool request body".to_string())?,
                )?)),
                2 => Ok(ChatEvent::Error(Error::from_extractor(
                    &inner.ok_or_else(|| "Missing error body".to_string())?,
                )?)),
                _ => Err(format!("Invalid ChatEvent variant: {idx}")),
            }
        }
    }

    //   record complete-response {
    //     id: string,
    //     content: list<content-part>,
    //     tool-calls: list<tool-call>,
    //     metadata: response-metadata,
    //   }
    impl IntoValue for CompleteResponse {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.id.add_to_builder(builder.item());
            builder = self.content.add_to_builder(builder.item());
            builder = self.tool_calls.add_to_builder(builder.item());
            builder = self.metadata.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = builder.field("id").string();
            builder = Vec::<ContentPart>::add_to_type_builder(builder.field("content"));
            builder = Vec::<ToolCall>::add_to_type_builder(builder.field("tool-calls"));
            builder = ResponseMetadata::add_to_type_builder(builder.field("metadata"));
            builder.finish()
        }
    }

    impl FromValueAndType for CompleteResponse {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            Ok(Self {
                id: String::from_extractor(
                    &extractor
                        .field(0)
                        .ok_or_else(|| "Missing id field".to_string())?,
                )?,
                content: Vec::<ContentPart>::from_extractor(
                    &extractor
                        .field(1)
                        .ok_or_else(|| "Missing content field".to_string())?,
                )?,
                tool_calls: Vec::<ToolCall>::from_extractor(
                    &extractor
                        .field(2)
                        .ok_or_else(|| "Missing tool-calls field".to_string())?,
                )?,
                metadata: ResponseMetadata::from_extractor(
                    &extractor
                        .field(3)
                        .ok_or_else(|| "Missing metadata field".to_string())?,
                )?,
            })
        }
    }

    //   record tool-call {
    //     id: string,
    //     name: string,
    //     arguments-json: string,
    //   }
    impl IntoValue for ToolCall {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.id.add_to_builder(builder.item());
            builder = self.name.add_to_builder(builder.item());
            builder = self.arguments_json.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = builder.field("id").string();
            builder = builder.field("name").string();
            builder = builder.field("arguments-json").string();
            builder.finish()
        }
    }

    impl FromValueAndType for ToolCall {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            Ok(Self {
                id: String::from_extractor(
                    &extractor
                        .field(0)
                        .ok_or_else(|| "Missing id field".to_string())?,
                )?,
                name: String::from_extractor(
                    &extractor
                        .field(1)
                        .ok_or_else(|| "Missing name field".to_string())?,
                )?,
                arguments_json: String::from_extractor(
                    &extractor
                        .field(2)
                        .ok_or_else(|| "Missing arguments-json field".to_string())?,
                )?,
            })
        }
    }

    //   record response-metadata {
    //     finish-reason: option<finish-reason>,
    //     usage: option<usage>,
    //     provider-id: option<string>,
    //     timestamp: option<string>,
    //     provider-metadata-json: option<string>,
    //   }
    impl IntoValue for ResponseMetadata {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.finish_reason.add_to_builder(builder.item());
            builder = self.usage.add_to_builder(builder.item());
            builder = self.provider_id.add_to_builder(builder.item());
            builder = self.timestamp.add_to_builder(builder.item());
            builder = self.provider_metadata_json.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = Option::<FinishReason>::add_to_type_builder(builder.field("finish-reason"));
            builder = Option::<Usage>::add_to_type_builder(builder.field("usage"));
            builder = TypeNodeBuilder::finish(builder.field("provider-id").option().string());
            builder = TypeNodeBuilder::finish(builder.field("timestamp").option().string());
            builder =
                TypeNodeBuilder::finish(builder.field("provider-metadata-json").option().string());
            builder.finish()
        }
    }

    impl FromValueAndType for ResponseMetadata {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            Ok(Self {
                finish_reason: Option::<FinishReason>::from_extractor(
                    &extractor
                        .field(0)
                        .ok_or_else(|| "Missing finish-reason field".to_string())?,
                )?,
                usage: Option::<Usage>::from_extractor(
                    &extractor
                        .field(1)
                        .ok_or_else(|| "Missing usage field".to_string())?,
                )?,
                provider_id: Option::<String>::from_extractor(
                    &extractor
                        .field(2)
                        .ok_or_else(|| "Missing provider-id field".to_string())?,
                )?,
                timestamp: Option::<String>::from_extractor(
                    &extractor
                        .field(3)
                        .ok_or_else(|| "Missing timestamp field".to_string())?,
                )?,
                provider_metadata_json: Option::<String>::from_extractor(
                    &extractor
                        .field(4)
                        .ok_or_else(|| "Missing provider-metadata-json field".to_string())?,
                )?,
            })
        }
    }

    // record usage {
    //     input-tokens: option<u32>,
    //     output-tokens: option<u32>,
    //     total-tokens: option<u32>,
    //   }
    impl IntoValue for Usage {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = builder
                .item()
                .option_fn(self.input_tokens.is_some(), |inner| {
                    inner.u32(self.input_tokens.unwrap())
                });
            builder = builder
                .item()
                .option_fn(self.output_tokens.is_some(), |inner| {
                    inner.u32(self.output_tokens.unwrap())
                });
            builder = builder
                .item()
                .option_fn(self.total_tokens.is_some(), |inner| {
                    inner.u32(self.total_tokens.unwrap())
                });
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = TypeNodeBuilder::finish(builder.field("input-tokens").option().u32());
            builder = TypeNodeBuilder::finish(builder.field("output-tokens").option().u32());
            builder = TypeNodeBuilder::finish(builder.field("total-tokens").option().u32());
            builder.finish()
        }
    }

    impl FromValueAndType for Usage {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            Ok(Self {
                input_tokens: extractor
                    .field(0)
                    .ok_or_else(|| "Missing input-tokens field".to_string())?
                    .option()
                    .ok_or_else(|| "input-tokens is not an option".to_string())?
                    .map(|inner| {
                        inner
                            .u32()
                            .ok_or_else(|| "input-tokens is not u32".to_string())
                    })
                    .transpose()?,
                output_tokens: extractor
                    .field(1)
                    .ok_or_else(|| "Missing output-tokens field".to_string())?
                    .option()
                    .ok_or_else(|| "output-tokens is not an option".to_string())?
                    .map(|inner| {
                        inner
                            .u32()
                            .ok_or_else(|| "output-tokens is not u32".to_string())
                    })
                    .transpose()?,
                total_tokens: extractor
                    .field(2)
                    .ok_or_else(|| "Missing total-tokens field".to_string())?
                    .option()
                    .ok_or_else(|| "total-tokens is not an option".to_string())?
                    .map(|inner| {
                        inner
                            .u32()
                            .ok_or_else(|| "total-tokens is not u32".to_string())
                    })
                    .transpose()?,
            })
        }
    }

    // enum finish-reason {
    //     stop,
    //     length,
    //     tool-calls,
    //     content-filter,
    //     error,
    //     other,
    //   }
    impl IntoValue for FinishReason {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            match self {
                FinishReason::Stop => builder.enum_value(0),
                FinishReason::Length => builder.enum_value(1),
                FinishReason::ToolCalls => builder.enum_value(2),
                FinishReason::ContentFilter => builder.enum_value(3),
                FinishReason::Error => builder.enum_value(4),
                FinishReason::Other => builder.enum_value(5),
            }
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            builder.r#enum(&[
                "stop",
                "length",
                "tool-calls",
                "content-filter",
                "error",
                "other",
            ])
        }
    }

    impl FromValueAndType for FinishReason {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            match extractor.enum_value() {
                Some(0) => Ok(FinishReason::Stop),
                Some(1) => Ok(FinishReason::Length),
                Some(2) => Ok(FinishReason::ToolCalls),
                Some(3) => Ok(FinishReason::ContentFilter),
                Some(4) => Ok(FinishReason::Error),
                Some(5) => Ok(FinishReason::Other),
                _ => Err("Invalid finish reason".to_string()),
            }
        }
    }

    #[derive(Debug)]
    struct SendInput {
        messages: Vec<Message>,
        config: Config,
    }

    impl IntoValue for SendInput {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.messages.add_to_builder(builder.item());
            builder = self.config.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = Vec::<Message>::add_to_type_builder(builder.field("messages"));
            builder = Config::add_to_type_builder(builder.field("config"));
            builder.finish()
        }
    }

    //   record message {
    //     role: role,
    //     name: option<string>,
    //     content: list<content-part>,
    //   }
    impl IntoValue for Message {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.role.add_to_builder(builder.item());
            builder = self.name.add_to_builder(builder.item());
            builder = self.content.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = Role::add_to_type_builder(builder.field("role"));
            builder = TypeNodeBuilder::finish(builder.field("name").option().string());
            builder = Vec::<ContentPart>::add_to_type_builder(builder.field("content"));
            builder.finish()
        }
    }

    //   enum role {
    //     user,
    //     assistant,
    //     system,
    //     tool,
    //   }
    impl IntoValue for Role {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            match self {
                Role::User => builder.enum_value(0),
                Role::Assistant => builder.enum_value(1),
                Role::System => builder.enum_value(2),
                Role::Tool => builder.enum_value(3),
            }
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            builder.r#enum(&["user", "assistant", "system", "tool"])
        }
    }

    // variant content-part {
    //     text(string),
    //     image(image-url),
    //   }
    impl IntoValue for ContentPart {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            match self {
                ContentPart::Text(text) => builder.variant(0).string(&text).finish(),
                ContentPart::Image(image_url) => {
                    image_url.add_to_builder(builder.variant(1)).finish()
                }
            }
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.variant();
            builder = builder.case("text").string();
            builder = ImageUrl::add_to_type_builder(builder.case("image"));
            builder.finish()
        }
    }

    impl FromValueAndType for ContentPart {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            let (idx, inner) = extractor
                .variant()
                .ok_or_else(|| "ContentPart should be variant".to_string())?;
            match idx {
                0 => Ok(ContentPart::Text(
                    inner
                        .ok_or_else(|| "Missing text".to_string())?
                        .string()
                        .ok_or_else(|| "ContentPart::Text should be string".to_string())?
                        .to_string(),
                )),
                1 => Ok(ContentPart::Image(ImageUrl::from_extractor(
                    &inner.ok_or_else(|| "Missing image url".to_string())?,
                )?)),
                _ => Err(format!("Invalid ContentPart variant: {idx}")),
            }
        }
    }

    // record image-url {
    //     url: string,
    //     detail: option<image-detail>,
    //   }
    impl IntoValue for ImageUrl {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.url.add_to_builder(builder.item());
            builder = self.detail.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = builder.field("url").string();
            builder = Option::<ImageDetail>::add_to_type_builder(builder.field("detail"));
            builder.finish()
        }
    }

    impl FromValueAndType for ImageUrl {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            Ok(Self {
                url: String::from_extractor(
                    &extractor
                        .field(0)
                        .ok_or_else(|| "Missing url field".to_string())?,
                )?,
                detail: Option::<ImageDetail>::from_extractor(
                    &extractor
                        .field(1)
                        .ok_or_else(|| "Missing detail field".to_string())?,
                )?,
            })
        }
    }

    //   enum image-detail {
    //     low,
    //     high,
    //     auto,
    //   }
    impl IntoValue for ImageDetail {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            match self {
                ImageDetail::Low => builder.enum_value(0),
                ImageDetail::High => builder.enum_value(1),
                ImageDetail::Auto => builder.enum_value(2),
            }
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            builder.r#enum(&["low", "high", "auto"])
        }
    }

    impl FromValueAndType for ImageDetail {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            match extractor.enum_value() {
                Some(0) => Ok(ImageDetail::Low),
                Some(1) => Ok(ImageDetail::High),
                Some(2) => Ok(ImageDetail::Auto),
                _ => Err("Invalid image detail".to_string()),
            }
        }
    }

    //   record config {
    //     model: string,
    //     temperature: option<f32>,
    //     max-tokens: option<u32>,
    //     stop-sequences: option<list<string>>,
    //     tools: list<tool-definition>,
    //     tool-choice: option<string>,
    //     provider-options: list<kv>,
    //   }
    impl IntoValue for Config {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.model.add_to_builder(builder.item());
            builder = self.temperature.add_to_builder(builder.item());
            builder = self.max_tokens.add_to_builder(builder.item());
            builder = self.stop_sequences.add_to_builder(builder.item());
            builder = self.tools.add_to_builder(builder.item());
            builder = self.tool_choice.add_to_builder(builder.item());
            builder = self.provider_options.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = builder.field("model").string();
            builder = TypeNodeBuilder::finish(builder.field("temperature").option().f32());
            builder = TypeNodeBuilder::finish(builder.field("max-tokens").option().u32());
            builder = TypeNodeBuilder::finish(
                builder
                    .field("stop-sequences")
                    .option()
                    .list()
                    .string()
                    .finish(),
            );
            builder = Vec::<ToolDefinition>::add_to_type_builder(builder.field("tools"));
            builder = TypeNodeBuilder::finish(builder.field("tool-choice").option().string());
            builder = Vec::<Kv>::add_to_type_builder(builder.field("provider-options"));
            builder.finish()
        }
    }

    //   record tool-definition {
    //     name: string,
    //     description: option<string>,
    //     parameters-schema: string,
    //   }
    impl IntoValue for ToolDefinition {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.name.add_to_builder(builder.item());
            builder = self.description.add_to_builder(builder.item());
            builder = self.parameters_schema.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = builder.field("name").string();
            builder = TypeNodeBuilder::finish(builder.field("description").option().string());
            builder = builder.field("parameters-schema").string();
            builder.finish()
        }
    }

    //   record kv {
    //     key: string,
    //     value: string,
    //   }
    impl IntoValue for Kv {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.key.add_to_builder(builder.item());
            builder = self.value.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = builder.field("key").string();
            builder = builder.field("value").string();
            builder.finish()
        }
    }

    //   enum error-code {
    //     invalid-request,
    //     authentication-failed,
    //     rate-limit-exceeded,
    //     internal-error,
    //     unsupported,
    //     unknown,
    //   }
    impl IntoValue for ErrorCode {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            match self {
                ErrorCode::InvalidRequest => builder.enum_value(0),
                ErrorCode::AuthenticationFailed => builder.enum_value(1),
                ErrorCode::RateLimitExceeded => builder.enum_value(2),
                ErrorCode::InternalError => builder.enum_value(3),
                ErrorCode::Unsupported => builder.enum_value(4),
                ErrorCode::Unknown => builder.enum_value(5),
            }
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            builder.r#enum(&[
                "invalid-request",
                "authentication-failed",
                "rate-limit-exceeded",
                "internal-error",
                "unsupported",
                "unknown",
            ])
        }
    }

    impl FromValueAndType for ErrorCode {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            match extractor.enum_value() {
                Some(0) => Ok(ErrorCode::InvalidRequest),
                Some(1) => Ok(ErrorCode::AuthenticationFailed),
                Some(2) => Ok(ErrorCode::RateLimitExceeded),
                Some(3) => Ok(ErrorCode::InternalError),
                Some(4) => Ok(ErrorCode::Unsupported),
                Some(5) => Ok(ErrorCode::Unknown),
                _ => Err("Invalid error code".to_string()),
            }
        }
    }

    //   record error {
    //     code: error-code,
    //     message: string,
    //     provider-error-json: option<string>,
    //   }
    impl IntoValue for Error {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = self.code.add_to_builder(builder.item());
            builder = self.message.add_to_builder(builder.item());
            builder = self.provider_error_json.add_to_builder(builder.item());
            builder.finish()
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            let mut builder = builder.record();
            builder = ErrorCode::add_to_type_builder(builder.field("code"));
            builder = builder.field("message").string();
            builder =
                TypeNodeBuilder::finish(builder.field("provider-error-json").option().string());
            builder.finish()
        }
    }

    impl FromValueAndType for Error {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            Ok(Self {
                code: ErrorCode::from_extractor(
                    &extractor
                        .field(0)
                        .ok_or_else(|| "Missing code field".to_string())?,
                )?,
                message: String::from_extractor(
                    &extractor
                        .field(1)
                        .ok_or_else(|| "Missing message field".to_string())?,
                )?,
                provider_error_json: Option::<String>::from_extractor(
                    &extractor
                        .field(2)
                        .ok_or_else(|| "Missing provider-error-json field".to_string())?,
                )?,
            })
        }
    }

    #[derive(Debug)]
    struct UnusedError;

    impl Display for UnusedError {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "UnusedError")
        }
    }

    impl IntoValue for UnusedError {
        fn add_to_builder<T: NodeBuilder>(self, builder: T) -> T::Result {
            builder.variant_unit(0)
        }

        fn add_to_type_builder<T: TypeNodeBuilder>(builder: T) -> T::Result {
            builder.variant().unit_case("unused-error").finish()
        }
    }

    impl FromValueAndType for UnusedError {
        fn from_extractor<'a, 'b>(
            extractor: &'a impl WitValueExtractor<'a, 'b>,
        ) -> Result<Self, String> {
            let (idx, _inner) = extractor
                .variant()
                .ok_or_else(|| "UnusedError should be variant".to_string())?;
            if idx == 0 {
                Ok(UnusedError)
            } else {
                Err(format!("UnusedError should be variant 0, but got {idx}"))
            }
        }
    }
}
