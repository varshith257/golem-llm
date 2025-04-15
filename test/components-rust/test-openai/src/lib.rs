#[allow(static_mut_refs)]
mod bindings;

use crate::bindings::exports::test::openai_exports::test_openai_api::*;
use crate::bindings::golem::llm::llm;
use crate::bindings::golem::llm::llm::StreamEvent;

struct Component;

impl Guest for Component {
    fn test1() -> String {
        let config = llm::Config {
            model: "gpt-3.5-turbo".to_string(),
            temperature: Some(0.2),
            max_tokens: None,
            stop_sequences: None,
            tools: vec![],
            tool_choice: None,
            provider_options: vec![],
        };

        println!("Sending request to LLM...");
        let response = llm::send(
            &[llm::Message {
                role: llm::Role::User,
                name: Some("vigoo".to_string()),
                content: vec![llm::ContentPart::Text(
                    "What is the usual weather on the Vršič pass in the beginning of May?"
                        .to_string(),
                )],
            }],
            &config,
        );
        println!("Response: {:?}", response);

        match response {
            llm::ChatEvent::Message(msg) => {
                format!(
                    "{}",
                    msg.content
                        .into_iter()
                        .map(|content| match content {
                            llm::ContentPart::Text(txt) => txt,
                            llm::ContentPart::Image(img) => format!("[IMAGE: {}]", img.url),
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            llm::ChatEvent::ToolRequest(request) => {
                format!("Tool request: {:?}", request)
            }
            llm::ChatEvent::Error(error) => {
                format!(
                    "ERROR: {:?} {} ({})",
                    error.code,
                    error.message,
                    error.provider_error_json.unwrap_or_default()
                )
            }
        }
    }

    fn test2() -> String {
        let config = llm::Config {
            model: "gpt-3.5-turbo".to_string(),
            temperature: Some(0.2),
            max_tokens: None,
            stop_sequences: None,
            tools: vec![llm::ToolDefinition {
                name: "test-tool".to_string(),
                description: Some("Test tool for generating test values".to_string()),
                parameters_schema: r#"{
                        "type": "object",
                        "properties": {
                            "maximum": {
                                "type": "number",
                                "description": "Upper bound for the test value"
                            }
                        },
                        "required": [
                            "maximum"
                        ],
                        "additionalProperties": false
                    }"#
                .to_string(),
            }],
            tool_choice: Some("auto".to_string()),
            provider_options: vec![],
        };

        let input = vec![
            llm::ContentPart::Text("Generate a random number between 1 and 10".to_string()),
            llm::ContentPart::Text(
                "then translate this number to German and output it as a text message.".to_string(),
            ),
        ];

        println!("Sending request to LLM...");
        let response1 = llm::send(
            &[llm::Message {
                role: llm::Role::User,
                name: Some("vigoo".to_string()),
                content: input.clone(),
            }],
            &config,
        );
        match response1 {
            llm::ChatEvent::Message(msg) => {
                format!("Message 1: {:?}", msg)
            }
            llm::ChatEvent::ToolRequest(request) => {
                println!("Tool request: {:?}", request);
                let mut calls = Vec::new();
                for call in request {
                    calls.push((
                        call.clone(),
                        llm::ToolResult::Success(llm::ToolSuccess {
                            id: call.id,
                            name: call.name,
                            result_json: r#"{ value: 6 }"#.to_string(),
                            execution_time_ms: None,
                        }),
                    ));
                }

                let response2 = llm::continue_(
                    &[llm::Message {
                        role: llm::Role::User,
                        name: Some("vigoo".to_string()),
                        content: input.clone(),
                    }],
                    &calls,
                    &config,
                );

                match response2 {
                    llm::ChatEvent::Message(msg) => {
                        format!("Message 2: {:?}", msg)
                    }
                    llm::ChatEvent::ToolRequest(request) => {
                        format!("Tool request 2: {:?}", request)
                    }
                    llm::ChatEvent::Error(error) => {
                        format!(
                            "ERROR 2: {:?} {} ({})",
                            error.code,
                            error.message,
                            error.provider_error_json.unwrap_or_default()
                        )
                    }
                }
            }
            llm::ChatEvent::Error(error) => {
                format!(
                    "ERROR 1: {:?} {} ({})",
                    error.code,
                    error.message,
                    error.provider_error_json.unwrap_or_default()
                )
            }
        }
    }

    fn test3() -> String {
        let config = llm::Config {
            model: "gpt-3.5-turbo".to_string(),
            temperature: Some(0.2),
            max_tokens: None,
            stop_sequences: None,
            tools: vec![],
            tool_choice: None,
            provider_options: vec![],
        };

        println!("Starting streaming request to LLM...");
        let stream = llm::stream(
            &[llm::Message {
                role: llm::Role::User,
                name: Some("vigoo".to_string()),
                content: vec![llm::ContentPart::Text(
                    "What is the usual weather on the Vršič pass in the beginning of May?"
                        .to_string(),
                )],
            }],
            &config,
        );

        let mut result = String::new();

        loop {
            let events = stream.blocking_get_next();
            if events.is_empty() {
                break;
            }

            for event in events {
                println!("Received {event:?}");

                match event {
                    StreamEvent::Delta(delta) => {
                        result.push_str(&format!("DELTA: {:?}\n", delta,));
                    }
                    StreamEvent::Finish(finish) => {
                        result.push_str(&format!("FINISH: {:?}\n", finish,));
                    }
                    StreamEvent::Error(error) => {
                        result.push_str(&format!(
                            "ERROR: {:?} {} ({})\n",
                            error.code,
                            error.message,
                            error.provider_error_json.unwrap_or_default()
                        ));
                    }
                }
            }
        }

        result
    }

    fn test4() -> String {
        let config = llm::Config {
            model: "gpt-3.5-turbo".to_string(),
            temperature: Some(0.2),
            max_tokens: None,
            stop_sequences: None,
            tools: vec![llm::ToolDefinition {
                name: "test-tool".to_string(),
                description: Some("Test tool for generating test values".to_string()),
                parameters_schema: r#"{
                        "type": "object",
                        "properties": {
                            "maximum": {
                                "type": "number",
                                "description": "Upper bound for the test value"
                            }
                        },
                        "required": [
                            "maximum"
                        ],
                        "additionalProperties": false
                    }"#
                .to_string(),
            }],
            tool_choice: Some("auto".to_string()),
            provider_options: vec![],
        };

        let input = vec![
            llm::ContentPart::Text("Generate a random number between 1 and 10".to_string()),
            llm::ContentPart::Text(
                "then translate this number to German and output it as a text message.".to_string(),
            ),
        ];

        println!("Starting streaming request to LLM...");
        let stream = llm::stream(
            &[llm::Message {
                role: llm::Role::User,
                name: Some("vigoo".to_string()),
                content: input,
            }],
            &config,
        );

        let mut result = String::new();

        loop {
            let events = stream.blocking_get_next();
            if events.is_empty() {
                break;
            }

            for event in events {
                println!("Received {event:?}");

                match event {
                    StreamEvent::Delta(delta) => {
                        result.push_str(&format!("DELTA: {:?}\n", delta,));
                    }
                    StreamEvent::Finish(finish) => {
                        result.push_str(&format!("FINISH: {:?}\n", finish,));
                    }
                    StreamEvent::Error(error) => {
                        result.push_str(&format!(
                            "ERROR: {:?} {} ({})\n",
                            error.code,
                            error.message,
                            error.provider_error_json.unwrap_or_default()
                        ));
                    }
                }
            }
        }

        result
    }

    fn test5() {
        let config = llm::Config {
            model: "gpt-4o-mini".to_string(),
            temperature: None,
            max_tokens: None,
            stop_sequences: None,
            tools: vec![],
            tool_choice: None,
            provider_options: vec![],
        };

        println!("Sending request to LLM...");
        let response = llm::send(
            &[
                llm::Message {
                    role: llm::Role::User,
                    name: None,
                    content: vec![
                        llm::ContentPart::Text("What is on this image?".to_string()),
                        llm::ContentPart::Image(llm::ImageUrl {
                            url: "https://blog.vigoo.dev/images/blog-zio-kafka-debugging-3.png"
                                .to_string(),
                            detail: Some(llm::ImageDetail::High),
                        }),
                    ],
                },
                llm::Message {
                    role: llm::Role::System,
                    name: None,
                    content: vec![llm::ContentPart::Text(
                        "Produce the output in both English and Hungarian".to_string(),
                    )],
                },
            ],
            &config,
        );
        println!("Response: {:?}", response);
    }

    fn test6() {}
}

bindings::export!(Component with_types_in bindings);
