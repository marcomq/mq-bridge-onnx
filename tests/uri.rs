//  mq-bridge-onnx
//  © Copyright 2026, by Marco Mengelkoch
//  Licensed under MIT OR Apache-2.0, see LICENSE file for more details
//  git clone https://github.com/marcomq/mq-bridge-onnx

// Its own test binary: it registers the endpoint, which `plugin.rs` loads.

use serde_json::json;

// The task is the part after `+`; the model path keeps its leading `/`.
#[test]
fn a_uri_names_the_task_the_model_path_and_the_options() {
    mq_bridge_onnx::register().unwrap();
    let schema = mq_bridge::plugin::endpoint_uri_schema(mq_bridge_onnx::NAME);

    let config = schema
        .config_from_uri(
            "onnx+embeddings:///models/minilm/model.onnx?input=${payload:text}&max_len=256",
        )
        .unwrap();
    assert_eq!(config["task"], "embeddings");
    assert_eq!(config["model"], "/models/minilm/model.onnx");
    assert_eq!(config["max_len"], 256);

    let config = schema
        .config_from_uri("onnx+score:models/churn.onnx?features=${payload:a},${payload:b}")
        .unwrap();
    assert_eq!(config["task"], "score");
    assert_eq!(config["model"], "models/churn.onnx");
    assert_eq!(config["features"], json!(["${payload:a}", "${payload:b}"]));
}
