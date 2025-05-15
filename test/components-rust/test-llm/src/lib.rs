#[allow(static_mut_refs)]
mod bindings;

use golem_rust::atomically;
use crate::bindings::exports::test::llm_exports::test_llm_api::*;
use crate::bindings::golem::llm::llm;
use crate::bindings::golem::llm::llm::StreamEvent;
use crate::bindings::test::helper_client::test_helper_client::TestHelperApi;

struct Component;

#[cfg(feature = "openai")]
const MODEL: &'static str = "gpt-3.5-turbo";
#[cfg(feature = "anthropic")]
const MODEL: &'static str = "claude-3-7-sonnet-20250219";
#[cfg(feature = "grok")]
const MODEL: &'static str = "grok-3-beta";
#[cfg(feature = "openrouter")]
const MODEL: &'static str = "openrouter/auto";
#[cfg(feature = "ollama")]
const MODEL: &'static str = "llama3";

#[cfg(feature = "openai")]
const IMAGE_MODEL: &'static str = "gpt-4o-mini";
#[cfg(feature = "anthropic")]
const IMAGE_MODEL: &'static str = "claude-3-7-sonnet-20250219";
#[cfg(feature = "grok")]
const IMAGE_MODEL: &'static str = "grok-2-vision-latest";
#[cfg(feature = "openrouter")]
const IMAGE_MODEL: &'static str = "openrouter/auto";
#[cfg(feature = "ollama")]
const IMAGE_MODEL: &'static str = "llava";

impl Guest for Component {
    /// test1 demonstrates a simple, non-streaming text question-answer interaction with the LLM.
    fn test1() -> String {
        let config = llm::Config {
            model: MODEL.to_string(),
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

    /// test2 demonstrates how to use tools with the LLM, including generating a tool response
    /// and continuing the conversation with it.
    fn test2() -> String {
        let config = llm::Config {
            model: MODEL.to_string(),
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
        let tool_request = match response1 {
            llm::ChatEvent::Message(msg) => {
                println!("Message 1: {:?}", msg);
                msg.tool_calls
            }
            llm::ChatEvent::ToolRequest(request) => {
                println!("Tool request: {:?}", request);
                request
            }
            llm::ChatEvent::Error(error) => {
                println!(
                    "ERROR 1: {:?} {} ({})",
                    error.code,
                    error.message,
                    error.provider_error_json.unwrap_or_default()
                );
                vec![]
            }
        };
        
        if !tool_request.is_empty() {
            let mut calls = Vec::new();
            for call in tool_request {
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
        } else {
            "No tool request".to_string()
        }
    }

    /// test3 is a streaming version of test1, a single turn question-answer interaction
    fn test3() -> String {
        let config = llm::Config {
            model: MODEL.to_string(),
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

    /// test4 shows how streaming works together with using tools
    fn test4() -> String {
        let config = llm::Config {
            model: MODEL.to_string(),
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

    /// test5 demonstrates how to send image urls to the LLM
    fn test5() {
        let config = llm::Config {
            model: IMAGE_MODEL.to_string(),
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

    /// test6 simulates a crash during a streaming LLM response, but only first time. 
    /// after the automatic recovery it will continue and finish the request successfully.
    fn test6() -> String {
        let config = llm::Config {
            model: MODEL.to_string(),
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

        let name = std::env::var("GOLEM_WORKER_NAME").unwrap();
        let mut round = 0;

        loop {
            let events = stream.blocking_get_next();
            if events.is_empty() {
                break;
            }

            for event in events {
                println!("Received {event:?}");

                match event {
                    StreamEvent::Delta(delta) => {
                        for content in delta.content.unwrap_or_default() {
                            match content {
                                llm::ContentPart::Text(txt) => {
                                    result.push_str(&txt);
                                }
                                llm::ContentPart::Image(img) => {
                                    result.push_str(&format!("IMAGE: {} ({:?})\n", img.url, img.detail));
                                }
                            }
                        }
                    }
                    StreamEvent::Finish(finish) => {
                        result.push_str(&format!("\nFINISH: {:?}\n", finish,));
                    }
                    StreamEvent::Error(error) => {
                        result.push_str(&format!(
                            "\nERROR: {:?} {} ({})\n",
                            error.code,
                            error.message,
                            error.provider_error_json.unwrap_or_default()
                        ));
                    }
                }
            }

            if round == 2 {
                atomically(|| {
                    let client = TestHelperApi::new(&name);
                    let answer = client.blocking_inc_and_get();
                    if answer == 1 {
                        panic!("Simulating crash")
                    }
                });
            }

            round += 1;
        }

        result
    }
}

bindings::export!(Component with_types_in bindings);
