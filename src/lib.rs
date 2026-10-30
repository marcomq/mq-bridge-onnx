//  mq-bridge-onnx
//  © Copyright 2026, by Marco Mengelkoch
//  Licensed under MIT OR Apache-2.0, see LICENSE file for more details
//  git clone https://github.com/marcomq/mq-bridge-onnx

//! Local ONNX inference for `mq-bridge`, as one output-only endpoint, `onnx`, that
//! answers every message, so it works as a route output and inside `lookup`. Its
//! `task` picks what it answers with:
//!
//! * `embeddings`: text (plus optional numbers) to a vector, a drop-in for
//!   `openai_embeddings`;
//! * `score`: numeric features to the model's outputs, e.g. a CatBoost, XGBoost or
//!   LightGBM export;
//! * `tensor`: a tensor per message to the model's outputs or to one vector, e.g.
//!   an image embedding.
//!
//! The `onnx` middleware makes those tensors: it decodes a PNG or JPEG image, a
//! JSON array (or an object of named arrays), a `.npy` file or, with feature
//! `audio`, PCM or WAV audio, optionally from base64, into the payload, with the
//! element type and shape in the metadata.
//!
//! ONNX Runtime is linked statically, so nothing has to be installed. The same
//! implementation is used three ways:
//!
//! * linked directly by a Rust program, which calls [`register`];
//! * loaded from the compiled `cdylib` by any mq-bridge host through
//!   `mq_bridge::plugin::load_endpoint_plugin`, or found by plugin discovery;
//! * from Python or Node.js, whose `mq-bridge-onnx` packages ship that same
//!   library and call the host's generic loader.

#[cfg(feature = "audio")]
mod audio;
mod config;
mod decode;
mod embeddings;
mod preprocess;
mod runtime;
mod scoring;
mod tensor;
mod tensor_data;

use std::sync::Arc;

use anyhow::anyhow;
use async_trait::async_trait;
use mq_bridge::errors::{InvalidConfig, PublisherError};
use mq_bridge::traits::{CustomEndpointFactory, MessagePublisher};
use mq_bridge::{CanonicalMessage, SentBatch};
use ort::value::{DynValue, Tensor, TensorElementType};
use serde_json::Value;

use config::TaskConfig;
pub use config::{OnnxConfig, OnnxDevice, OnnxInput, OnnxPooling, OnnxTask};
pub use decode::{OnnxDecode, OnnxDecodeConfig, OnnxMiddlewareFactory, OnnxPcmFormat};
pub use preprocess::OnnxExtent;
pub use tensor_data::{DTYPE_KEY, ERROR_KEY, NAMES_KEY, OnnxDtype, SHAPE_KEY};

/// The endpoint and middleware name, which is also the plugin's discovery name.
pub const NAME: &str = "onnx";

/// Factory of the `onnx` endpoint.
#[derive(Debug, Default)]
pub struct OnnxFactory;

// Exports the same factory as a loadable plugin. `register()` below covers the
// directly linked case; this covers every host that loads the compiled library.
#[cfg(feature = "plugin")]
mq_bridge::export_endpoint_plugin! {
    name: "onnx",
    factory: OnnxFactory,
    middleware: OnnxMiddlewareFactory,
    capabilities: mq_bridge::plugin::sdk::CAPABILITIES_OUTPUT_ONLY
        | mq_bridge::plugin::sdk::CAPABILITIES_MIDDLEWARE_ONLY,
}

/// Registers this crate's endpoint and middleware under `onnx`. Only needed when
/// linking this crate directly; a host that loads the plugin registers both.
pub fn register() -> anyhow::Result<()> {
    mq_bridge::extensions::register_endpoint_factory(NAME, Arc::new(OnnxFactory))?;
    mq_bridge::extensions::register_middleware_factory(NAME, Arc::new(OnnxMiddlewareFactory))
}

#[async_trait]
impl CustomEndpointFactory for OnnxFactory {
    fn config_schema(&self) -> Option<Value> {
        serde_json::to_value(schemars::schema_for!(OnnxConfig)).ok()
    }

    async fn create_publisher(
        &self,
        _route_name: &str,
        value: &Value,
    ) -> anyhow::Result<Box<dyn MessagePublisher>> {
        let config: OnnxConfig = serde_json::from_value(value.clone())
            .map_err(|e| InvalidConfig(anyhow!("onnx: {e}")))?;
        let task = config
            .into_task()
            .map_err(|e| InvalidConfig(anyhow!("onnx: {e}")))?;
        Ok(match task {
            TaskConfig::Embeddings(config) => {
                Box::new(embeddings::OnnxEmbeddingsPublisher::new(config).await?)
            }
            TaskConfig::Score(config) => Box::new(scoring::OnnxPublisher::new(config).await?),
            TaskConfig::Tensor(config) => Box::new(tensor::OnnxTensorPublisher::new(config).await?),
        })
    }
}

/// Answers in message order become responses; failed ones fail alone.
fn sent_batch(
    messages: Vec<CanonicalMessage>,
    answers: Vec<Result<Value, PublisherError>>,
) -> SentBatch {
    let mut responses = Vec::with_capacity(messages.len());
    let mut failed = Vec::new();
    for (message, answer) in messages.into_iter().zip(answers) {
        match answer {
            Ok(value) => responses.push(CanonicalMessage::new(
                value.to_string().into_bytes(),
                Some(message.message_id),
            )),
            Err(e) => failed.push((message, e)),
        }
    }
    SentBatch::Partial {
        responses: Some(responses),
        failed,
    }
}

/// A JSON array of `values`, each written as short as `f32` allows; NaN becomes null.
fn float_json(values: &[f32]) -> Value {
    Value::Array(values.iter().map(|x| float_value(*x)).collect())
}

fn float_value(x: f32) -> Value {
    // Display gives the shortest text that reads back as the same f32.
    x.to_string()
        .parse::<f64>()
        .ok()
        .and_then(serde_json::Number::from_f64)
        .map_or(Value::Null, Value::Number)
}

fn int_tensor(
    ty: TensorElementType,
    shape: [usize; 2],
    values: Vec<i64>,
) -> anyhow::Result<DynValue> {
    let shape = [shape[0] as i64, shape[1] as i64];
    let tensor = match ty {
        TensorElementType::Int32 => {
            let values = values.into_iter().map(|v| v as i32).collect::<Vec<_>>();
            Tensor::from_array((shape, values)).map(Tensor::into_dyn)
        }
        _ => Tensor::from_array((shape, values)).map(Tensor::into_dyn),
    };
    tensor.map_err(|e| anyhow!("{e}"))
}

fn model_name(model: &runtime::OnnxModel) -> &str {
    &model.path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_are_written_short() {
        assert_eq!(
            float_json(&[0.1, -2.5, f32::NAN]).to_string(),
            "[0.1,-2.5,null]"
        );
    }

    #[test]
    fn a_failed_answer_fails_its_message_alone() {
        let messages = vec![CanonicalMessage::from("a"), CanonicalMessage::from("b")];
        let first = messages[0].message_id;
        let answers = vec![
            Ok(Value::from(1)),
            Err(PublisherError::NonRetryable(anyhow!("bad"))),
        ];
        let SentBatch::Partial { responses, failed } = sent_batch(messages, answers) else {
            panic!("expected a partial batch");
        };
        let responses = responses.unwrap();
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].message_id, first);
        assert_eq!(failed.len(), 1);
    }

    #[test]
    fn the_schema_takes_the_task_and_model_path_from_a_uri() {
        let schema = OnnxFactory.config_schema().unwrap();
        assert_eq!(schema["properties"]["task"]["x-mqb-uri"], "subscheme");
        assert_eq!(schema["properties"]["model"]["x-mqb-uri"], "filepath");
    }
}
