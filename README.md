# golem-llm

Known issues:

- The "magical retry prompt" does not seem to skip the existing messages with Anthropic (also with auto model in OpenRouter, probably because it selects Claude)
  - the reason can be that the system prompt is not interleaved with the other roles 
- test3 => test6 fails with unexpected oplog entry (both Anthropic and OpenAI)
 
