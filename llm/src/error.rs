use crate::golem::llm::llm::{Error, ErrorCode};

pub fn unsupported(what: impl AsRef<str>) -> Error {
    Error {
        code: ErrorCode::Unsupported,
        message: format!("Unsupported: {}", what.as_ref()),
        provider_error_json: None,
    }
}