use crate::client::{Detail, InnerInput, InnerInputItem, Input};
use golem_llm::golem::llm::llm::{ContentPart, ErrorCode, ImageDetail, Role};

pub fn to_openai_role_name(role: Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
        Role::Tool => "tool",
    }
}

pub fn content_part_to_inner_input_item(content_part: ContentPart) -> InnerInputItem {
    match content_part {
        ContentPart::Text(msg) => InnerInputItem::TextInput { text: msg },
        ContentPart::Image(image_url) => InnerInputItem::ImageInput {
            image_url: image_url.url,
            detail: match image_url.detail {
                Some(ImageDetail::Auto) => Detail::Auto,
                Some(ImageDetail::Low) => Detail::Low,
                Some(ImageDetail::High) => Detail::High,
                None => Detail::default(),
            },
        },
    }
}

pub fn parse_error_code(code: String) -> ErrorCode {
    // TODO: we don't know what `code` can be..
    ErrorCode::InternalError
}
