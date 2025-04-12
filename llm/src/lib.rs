pub mod config;
pub mod error;

#[allow(dead_code)]
pub mod event_source;

wit_bindgen::generate!({
    path: "../wit",
    world: "llm-library",
    generate_all,
    generate_unused_types: true,
    pub_export_macro: true,
    with: {
        "wasi:io/poll@0.2.0": golem_rust::wasm_rpc::wasi::io::poll,
    }
});

pub use crate::exports::golem;
pub use __export_llm_library_impl as export_llm;
