pub mod config;
pub mod error;

use wit_bindgen::generate;

generate!({
    path: "../wit",
    world: "llm-library",
    generate_all,
    generate_unused_types: true,
    pub_export_macro: true
});

pub use crate::exports::golem;
pub use __export_llm_library_impl as export_llm;
