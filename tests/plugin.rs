//  mq-bridge-onnx
//  © Copyright 2026, by Marco Mengelkoch
//  Licensed under MIT OR Apache-2.0, see LICENSE file for more details
//  git clone https://github.com/marcomq/mq-bridge-onnx

//! The same implementation, reached two ways: through the factory linked into
//! this test binary, and through the factory the host builds from the compiled
//! plugin library. Identical results mean the ABI round trip kept the
//! endpoint's behaviour, and that the Python and npm packages get the same one.

use mq_bridge::plugin::load_endpoint_plugins;
use mq_bridge::plugin::test_support::build_plugin_cdylib;
use mq_bridge::traits::{CustomEndpointFactory, CustomMiddlewareFactory};
use mq_bridge::{CanonicalMessage, SentBatch};
use mq_bridge_onnx::{OnnxFactory, OnnxMiddlewareFactory};
use serde_json::json;

const CLASSIFIER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/tiny_classifier.onnx"
);
const TENSOR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/tiny_tensor.onnx"
);

/// Sends JSON tensors through the `onnx` middleware to task `tensor`.
async fn tensors(
    factory: &dyn CustomEndpointFactory,
    middleware: &dyn CustomMiddlewareFactory,
) -> Vec<String> {
    let publisher = factory
        .create_publisher("route", &json!({ "task": "tensor", "model": TENSOR }))
        .await
        .expect("create the publisher");
    let publisher = middleware
        .apply_publisher(publisher, "route", &json!({ "decode": "json" }))
        .await
        .expect("apply the middleware");
    let batch = vec![
        CanonicalMessage::from("[1, 2]"),
        CanonicalMessage::from("[1, 2, 3]"),
    ];
    let SentBatch::Partial { responses, failed } = publisher.send_batch(batch).await.unwrap()
    else {
        panic!("expected answers");
    };
    assert_eq!(failed.len(), 1, "the wrong shape fails its message alone");
    let lookups = publisher
        .lookup_each(&[CanonicalMessage::from("[1, 2]")])
        .await
        .expect("a lookup batch passes through the middleware");
    assert!(lookups[0].as_ref().unwrap().is_some());
    responses
        .unwrap()
        .iter()
        .map(|m| String::from_utf8(m.payload.to_vec()).unwrap())
        .collect()
}

async fn scores(factory: &dyn CustomEndpointFactory) -> Vec<String> {
    let config = json!({ "task": "score", "model": CLASSIFIER, "features": ["${payload:a}", "${payload:b}", "${payload:c}"] });
    let publisher = factory
        .create_publisher("route", &config)
        .await
        .expect("create the publisher");
    let batch = vec![
        CanonicalMessage::from(r#"{"a": 8, "b": 1, "c": 2}"#),
        CanonicalMessage::from(r#"{"a": 1, "b": 1, "c": 9}"#),
    ];
    let SentBatch::Partial { responses, failed } = publisher.send_batch(batch).await.unwrap()
    else {
        panic!("expected answers");
    };
    assert!(failed.is_empty());
    responses
        .unwrap()
        .iter()
        .map(|m| String::from_utf8(m.payload.to_vec()).unwrap())
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn the_endpoint_behaves_the_same_linked_directly_and_loaded_as_a_plugin() {
    let direct = scores(&OnnxFactory).await;
    assert_eq!(direct.len(), 2);

    let library = build_plugin_cdylib(".", "mq-bridge-onnx").expect("build the plugin");
    let names: Vec<String> = load_endpoint_plugins(&library)
        .expect("load the plugin")
        .into_iter()
        .map(|info| info.name)
        .collect();
    assert_eq!(names, [mq_bridge_onnx::NAME]);
    let factory = mq_bridge::extensions::get_endpoint_factory("onnx")
        .expect("loading a plugin registers its endpoint");

    assert_eq!(scores(factory.as_ref()).await, direct);

    let config = json!({ "task": "score", "model": CLASSIFIER, "features": ["${payload:a}", "${payload:b}", "${payload:c}"] });
    let loaded = factory.create_publisher("route", &config).await.unwrap();
    let lookups = loaded
        .lookup_each(&[CanonicalMessage::from(r#"{"a": 8, "b": 1, "c": 2}"#)])
        .await
        .expect("a lookup batch is one plugin call");
    let answer = lookups[0].as_ref().unwrap().as_ref().unwrap().to_string();
    assert_eq!(answer, direct[0]);

    let middleware = mq_bridge::extensions::get_middleware_factory("onnx")
        .expect("loading the plugin registers its middleware too");
    let direct = tensors(&OnnxFactory, &OnnxMiddlewareFactory).await;
    assert_eq!(direct.len(), 1);
    assert_eq!(tensors(factory.as_ref(), middleware.as_ref()).await, direct);
}
