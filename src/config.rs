//  mq-bridge-onnx
//  © Copyright 2026, by Marco Mengelkoch
//  Licensed under MIT OR Apache-2.0, see LICENSE file for more details
//  git clone https://github.com/marcomq/mq-bridge-onnx

use anyhow::bail;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The `onnx` endpoint: runs a local ONNX model on every message.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "ONNX model")]
#[serde(deny_unknown_fields)]
pub struct OnnxConfig {
    /// What the model answers with; the `embeddings` in `onnx+embeddings://`.
    #[schemars(extend("x-mqb-uri" = "subscheme"))]
    pub task: OnnxTask,
    /// Path of the `.onnx` model file, e.g. `/models/all-MiniLM-L6-v2/model.onnx`.
    #[schemars(extend("x-mqb-uri" = "filepath"))]
    pub model: String,
    /// Embeddings: template of the text to embed, e.g. `${payload:body}`.
    pub input: Option<String>,
    /// Embeddings: path of `tokenizer.json`. Defaults to the one next to the model.
    pub tokenizer: Option<String>,
    /// Embeddings: templates of numbers appended to the vector as given; missing = null.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub numbers: Vec<String>,
    /// Embeddings: tokens per text; longer texts are cut. Defaults to 128. Memory grows with its square.
    pub max_len: Option<usize>,
    /// Embeddings and tensor `vector`: how token vectors become one vector. Defaults to mean.
    pub pooling: Option<OnnxPooling>,
    /// Embeddings and tensor `vector`: scale each vector to length 1, before `numbers` are appended. Defaults to true.
    pub normalize: Option<bool>,
    /// Score: template of each feature in model order, e.g. `${payload:orders_30d}`; missing = NaN.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub features: Vec<String>,
    /// Score and tensor: model outputs in the response. Defaults to all of them.
    pub outputs: Option<Vec<String>>,
    /// Score and tensor: model input the features or tensors go to. Defaults to the only one.
    pub input_name: Option<String>,
    /// Tensor: answer with this output as one flat vector instead of an object of outputs.
    pub vector: Option<String>,
    /// Messages per inference call. Defaults to 32 for embeddings and tensor, 256 for score.
    pub max_batch: Option<usize>,
    /// CPU threads of one inference call. Defaults to 2.
    pub threads: Option<usize>,
    /// Where the model runs.
    #[serde(default)]
    pub device: OnnxDevice,
}

/// What the `onnx` endpoint answers with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum OnnxTask {
    /// Text, plus optional numbers, to a vector, like `openai_embeddings`.
    Embeddings,
    /// Numeric features to the model's outputs, e.g. a CatBoost, XGBoost or LightGBM export.
    Score,
    /// A tensor per message, as the `onnx` middleware decodes it, to the model's outputs.
    Tensor,
}

/// How token vectors are pooled into one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum OnnxPooling {
    /// Average of the tokens, as sentence-transformers models expect.
    #[default]
    Mean,
    /// The first token, as BGE models expect.
    Cls,
}

/// Where an ONNX model runs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum OnnxDevice {
    /// The CPU only, so memory stays predictable.
    #[default]
    Cpu,
    /// Whatever ONNX Runtime picks, e.g. CoreML on macOS; memory is then not predictable.
    Auto,
}

#[derive(Debug, Clone)]
pub(crate) struct EmbeddingsConfig {
    pub model: String,
    pub tokenizer: Option<String>,
    pub input: String,
    pub numbers: Vec<String>,
    pub max_len: Option<usize>,
    pub max_batch: Option<usize>,
    pub threads: Option<usize>,
    pub pooling: OnnxPooling,
    pub normalize: Option<bool>,
    pub device: OnnxDevice,
}

#[derive(Debug, Clone)]
pub(crate) struct ScoreConfig {
    pub model: String,
    pub features: Vec<String>,
    pub outputs: Option<Vec<String>>,
    pub input_name: Option<String>,
    pub max_batch: Option<usize>,
    pub threads: Option<usize>,
    pub device: OnnxDevice,
}

#[derive(Debug, Clone)]
pub(crate) struct TensorConfig {
    pub model: String,
    pub input_name: Option<String>,
    pub outputs: Option<Vec<String>>,
    pub vector: Option<String>,
    pub pooling: Option<OnnxPooling>,
    pub normalize: Option<bool>,
    pub max_batch: Option<usize>,
    pub threads: Option<usize>,
    pub device: OnnxDevice,
}

pub(crate) enum TaskConfig {
    Embeddings(EmbeddingsConfig),
    Score(ScoreConfig),
    Tensor(TensorConfig),
}

impl OnnxConfig {
    /// Splits the flat configuration by task, refusing a field the task does not read.
    pub(crate) fn into_task(self) -> anyhow::Result<TaskConfig> {
        let foreign: &[(&str, bool)] = match self.task {
            OnnxTask::Embeddings => &[
                ("features", !self.features.is_empty()),
                ("outputs", self.outputs.is_some()),
                ("input_name", self.input_name.is_some()),
                ("vector", self.vector.is_some()),
            ],
            OnnxTask::Score => &[
                ("input", self.input.is_some()),
                ("tokenizer", self.tokenizer.is_some()),
                ("numbers", !self.numbers.is_empty()),
                ("max_len", self.max_len.is_some()),
                ("pooling", self.pooling.is_some()),
                ("normalize", self.normalize.is_some()),
                ("vector", self.vector.is_some()),
            ],
            OnnxTask::Tensor => &[
                ("input", self.input.is_some()),
                ("tokenizer", self.tokenizer.is_some()),
                ("numbers", !self.numbers.is_empty()),
                ("max_len", self.max_len.is_some()),
                ("features", !self.features.is_empty()),
            ],
        };
        if let Some((field, _)) = foreign.iter().find(|(_, set)| *set) {
            let task = serde_json::to_value(self.task)?;
            bail!("`{field}` is not read by task {task}");
        }
        if self.task == OnnxTask::Tensor {
            if self.vector.is_some() && self.outputs.is_some() {
                bail!("give `vector` or `outputs`, not both");
            }
            if self.vector.is_none() && (self.pooling.is_some() || self.normalize.is_some()) {
                bail!(
                    "`pooling` and `normalize` shape the `vector` answer; name its output in `vector`"
                );
            }
        }
        Ok(match self.task {
            OnnxTask::Embeddings => {
                let Some(input) = self.input else {
                    bail!("task \"embeddings\" needs `input`, the template of the text to embed");
                };
                TaskConfig::Embeddings(EmbeddingsConfig {
                    model: self.model,
                    tokenizer: self.tokenizer,
                    input,
                    numbers: self.numbers,
                    max_len: self.max_len,
                    max_batch: self.max_batch,
                    threads: self.threads,
                    pooling: self.pooling.unwrap_or_default(),
                    normalize: self.normalize,
                    device: self.device,
                })
            }
            OnnxTask::Tensor => TaskConfig::Tensor(TensorConfig {
                model: self.model,
                input_name: self.input_name,
                outputs: self.outputs,
                vector: self.vector,
                pooling: self.pooling,
                normalize: self.normalize,
                max_batch: self.max_batch,
                threads: self.threads,
                device: self.device,
            }),
            OnnxTask::Score => TaskConfig::Score(ScoreConfig {
                model: self.model,
                features: self.features,
                outputs: self.outputs,
                input_name: self.input_name,
                max_batch: self.max_batch,
                threads: self.threads,
                device: self.device,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn task(value: serde_json::Value) -> anyhow::Result<TaskConfig> {
        serde_json::from_value::<OnnxConfig>(value)?.into_task()
    }

    #[test]
    fn a_field_of_the_other_task_is_refused() {
        let error =
            task(json!({ "task": "score", "model": "m.onnx", "features": ["1"], "input": "x" }))
                .err()
                .unwrap();
        assert_eq!(error.to_string(), "`input` is not read by task \"score\"");
        assert!(task(json!({ "task": "embeddings", "model": "m.onnx" })).is_err());
        assert!(matches!(
            task(json!({ "task": "embeddings", "model": "m.onnx", "input": "x" })).unwrap(),
            TaskConfig::Embeddings(_)
        ));
    }

    #[test]
    fn a_tensor_answers_with_outputs_or_one_vector() {
        let tensor = |extra: serde_json::Value| {
            let mut config = json!({ "task": "tensor", "model": "m.onnx" });
            config
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            task(config).map(|_| ()).map_err(|e| e.to_string())
        };
        assert!(tensor(json!({})).is_ok());
        assert!(tensor(json!({ "vector": "image_embeds", "pooling": "cls" })).is_ok());
        assert!(
            tensor(json!({ "vector": "a", "outputs": ["b"] }))
                .unwrap_err()
                .contains("not both")
        );
        assert!(
            tensor(json!({ "normalize": false }))
                .unwrap_err()
                .contains("`vector`")
        );
        assert!(
            tensor(json!({ "features": ["1"] }))
                .unwrap_err()
                .contains("not read by task \"tensor\"")
        );
    }
}
