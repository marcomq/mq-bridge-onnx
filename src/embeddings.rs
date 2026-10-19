//  mq-bridge-onnx
//  © Copyright 2026, by Marco Mengelkoch
//  Licensed under MIT OR Apache-2.0, see LICENSE file for more details
//  git clone https://github.com/marcomq/mq-bridge-onnx

//! Task `embeddings`: text to one vector per message with a local model.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, anyhow, bail};
use async_trait::async_trait;
use ort::value::{DynValue, TensorElementType};
use serde_json::Value;
use tokenizers::{Encoding, Tokenizer, TruncationParams};

use crate::config::{EmbeddingsConfig, OnnxPooling};
use crate::runtime::{self as onnx, OnnxModel};
use crate::{float_json, int_tensor, model_name};
use mq_bridge::CanonicalMessage;
use mq_bridge::SentBatch;
use mq_bridge::errors::PublisherError;
use mq_bridge::support::interpolation::CompiledTemplate;
use mq_bridge::traits::MessagePublisher;

const DEFAULT_MAX_LEN: usize = 128;
const DEFAULT_MAX_BATCH: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Feed {
    Ids,
    Mask,
    TypeIds,
}

/// What a batch needs besides the model: fixed at startup.
struct Plan {
    /// Each model input, in model order: what it gets and its integer type.
    inputs: Vec<(String, Feed, TensorElementType)>,
    output: String,
    max_batch: usize,
    pooling: OnnxPooling,
    normalize: bool,
}

pub(crate) struct OnnxEmbeddingsPublisher {
    model: Arc<OnnxModel>,
    tokenizer: Arc<Tokenizer>,
    plan: Arc<Plan>,
    input: CompiledTemplate,
    numbers: Vec<CompiledTemplate>,
}

impl OnnxEmbeddingsPublisher {
    pub(crate) async fn new(config: EmbeddingsConfig) -> anyhow::Result<Self> {
        let input = CompiledTemplate::compile(&config.input, None).context("input")?;
        let numbers = config
            .numbers
            .iter()
            .enumerate()
            .map(|(i, n)| {
                CompiledTemplate::compile(n, None).with_context(|| format!("numbers[{i}]"))
            })
            .collect::<anyhow::Result<_>>()?;
        let max_len = config.max_len.unwrap_or(DEFAULT_MAX_LEN).max(1);
        let max_batch = config.max_batch.unwrap_or(DEFAULT_MAX_BATCH).max(1);
        let tokenizer_path = config.tokenizer.clone().unwrap_or_else(|| {
            Path::new(&config.model)
                .with_file_name("tokenizer.json")
                .to_string_lossy()
                .into_owned()
        });
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("cannot read tokenizer '{tokenizer_path}': {e}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: max_len,
                ..Default::default()
            }))
            .map_err(|e| anyhow!("tokenizer '{tokenizer_path}': {e}"))?;
        tokenizer.with_padding(None);
        let tokenizer = Arc::new(tokenizer);

        let pooling = config.pooling;
        let normalize = config.normalize.unwrap_or(true);
        let warm_tokenizer = tokenizer.clone();
        let (model, plan) = onnx::open_and_warm_up(
            "onnx embeddings",
            config.model.clone(),
            config.threads.unwrap_or(onnx::DEFAULT_THREADS),
            config.device,
            format!("{max_batch} texts × {max_len} tokens"),
            move |model| {
                let plan = plan_for(model, max_batch, pooling, normalize)?;
                let longest = warm_tokenizer
                    .encode_fast("a ".repeat(max_len), true)
                    .map_err(|e| anyhow!("{e}"))?;
                let batch: Vec<&Encoding> = (0..max_batch).map(|_| &longest).collect();
                run_batch(model, &plan, &batch)?;
                Ok(plan)
            },
        )
        .await?;
        Ok(Self {
            model,
            tokenizer,
            plan: Arc::new(plan),
            input,
            numbers,
        })
    }

    async fn embed(&self, messages: &[CanonicalMessage]) -> Vec<Result<Value, PublisherError>> {
        let texts: Vec<String> = messages
            .iter()
            .map(|m| String::from_utf8_lossy(&self.input.render(Some(m))).into_owned())
            .collect();
        let count = texts.len();
        let tokenizer = self.tokenizer.clone();
        let plan = self.plan.clone();
        let embedded = onnx::infer(self.model.clone(), move |model| {
            Ok(embed_texts(model, &tokenizer, &plan, texts))
        })
        .await;
        match embedded {
            Ok(vectors) => vectors
                .into_iter()
                .zip(messages)
                .map(|(vector, message)| {
                    vector
                        .and_then(|v| append_numbers(&v, &self.numbers, message))
                        .map_err(PublisherError::NonRetryable)
                })
                .collect(),
            Err(e) => {
                let reason = e.to_string();
                (0..count)
                    .map(|_| Err(PublisherError::NonRetryable(anyhow!("{reason}"))))
                    .collect()
            }
        }
    }
}

/// Appends the message's `numbers` to its vector; a missing one is null.
fn append_numbers(
    vector: &[f32],
    numbers: &[CompiledTemplate],
    message: &CanonicalMessage,
) -> anyhow::Result<Value> {
    let mut values = float_json(vector);
    if let Value::Array(values) = &mut values {
        let numbers = crate::scoring::features_of("numbers", numbers, message)?;
        values.extend(
            numbers
                .into_iter()
                .map(|n| serde_json::Number::from_f64(n).map_or(Value::Null, Value::Number)),
        );
    }
    Ok(values)
}

/// Works out which tensor each model input gets, and which output holds the vectors.
fn plan_for(
    model: &OnnxModel,
    max_batch: usize,
    pooling: OnnxPooling,
    normalize: bool,
) -> anyhow::Result<Plan> {
    let mut inputs = Vec::new();
    for (name, dtype) in model.inputs() {
        let feed = match name.as_str() {
            n if n.contains("input_ids") => Feed::Ids,
            n if n.contains("attention_mask") => Feed::Mask,
            n if n.contains("token_type_ids") => Feed::TypeIds,
            other => bail!(
                "ONNX model '{}' has input '{other}'; task embeddings feeds only input_ids, attention_mask and token_type_ids",
                model.path
            ),
        };
        let ty = dtype
            .tensor_type()
            .filter(|ty| matches!(ty, TensorElementType::Int64 | TensorElementType::Int32))
            .ok_or_else(|| {
                anyhow!("ONNX model input '{name}' is {dtype}, not an integer tensor")
            })?;
        inputs.push((name, feed, ty));
    }
    if !inputs.iter().any(|(_, feed, _)| *feed == Feed::Ids) {
        bail!("ONNX model '{}' has no input_ids input", model.path);
    }
    let outputs = model.outputs();
    let output = outputs
        .iter()
        .find(|(name, _)| name == "sentence_embedding")
        .or_else(|| outputs.first())
        .map(|(name, _)| name.clone())
        .ok_or_else(|| anyhow!("ONNX model '{}' has no output", model.path))?;
    Ok(Plan {
        inputs,
        output,
        max_batch,
        pooling,
        normalize,
    })
}

/// Embeds every text, shortest first so a batch pads little; results in input order.
fn embed_texts(
    model: &OnnxModel,
    tokenizer: &Tokenizer,
    plan: &Plan,
    texts: Vec<String>,
) -> Vec<anyhow::Result<Vec<f32>>> {
    let encodings: Vec<anyhow::Result<Encoding>> = texts
        .into_iter()
        .map(|text| {
            tokenizer
                .encode_fast(text, true)
                .map_err(|e| anyhow!("{e}"))
        })
        .collect();
    let mut results: Vec<Option<anyhow::Result<Vec<f32>>>> =
        encodings.iter().map(|_| None).collect();
    let mut order = Vec::new();
    for (index, encoding) in encodings.iter().enumerate() {
        match encoding {
            Ok(encoding) => order.push((encoding.len(), index)),
            Err(e) => results[index] = Some(Err(anyhow!("cannot tokenize the input: {e}"))),
        }
    }
    order.sort_unstable();
    for chunk in order.chunks(plan.max_batch) {
        let batch: Vec<&Encoding> = chunk
            .iter()
            .filter_map(|(_, index)| encodings[*index].as_ref().ok())
            .collect();
        match run_batch(model, plan, &batch) {
            Ok(vectors) => {
                for ((_, index), vector) in chunk.iter().zip(vectors) {
                    results[*index] = Some(Ok(vector));
                }
            }
            Err(e) => {
                let reason = format!("{e:#}");
                for (_, index) in chunk {
                    results[*index] = Some(Err(anyhow!("{reason}")));
                }
            }
        }
    }
    results
        .into_iter()
        .map(|r| r.unwrap_or_else(|| Err(anyhow!("text was not embedded"))))
        .collect()
}

fn run_batch(model: &OnnxModel, plan: &Plan, batch: &[&Encoding]) -> anyhow::Result<Vec<Vec<f32>>> {
    let rows = batch.len();
    let len = batch.iter().map(|e| e.len()).max().unwrap_or(0).max(1);
    let padded = |pick: fn(&Encoding) -> &[u32]| -> Vec<i64> {
        let mut flat = vec![0i64; rows * len];
        for (row, encoding) in batch.iter().enumerate() {
            for (slot, value) in flat[row * len..].iter_mut().zip(pick(encoding)) {
                *slot = i64::from(*value);
            }
        }
        flat
    };
    let mask = padded(Encoding::get_attention_mask);
    let mut inputs: Vec<(String, DynValue)> = Vec::with_capacity(plan.inputs.len());
    for (name, feed, ty) in &plan.inputs {
        let values = match feed {
            Feed::Ids => padded(Encoding::get_ids),
            Feed::Mask => mask.clone(),
            Feed::TypeIds => padded(Encoding::get_type_ids),
        };
        inputs.push((name.clone(), int_tensor(*ty, [rows, len], values)?));
    }
    model.with_session(|session| {
        let outputs = session.run(inputs).map_err(|e| anyhow!("{e}"))?;
        let (shape, data) = outputs[plan.output.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("output '{}': {e}", plan.output))?;
        let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
        let mut vectors = match dims.as_slice() {
            [r, d] if *r == rows => data.chunks(*d).map(<[f32]>::to_vec).collect(),
            [r, l, d] if *r == rows => pool(data, *l, *d, &mask, len, plan.pooling),
            _ => bail!(
                "output '{}' has shape {dims:?}; expected [{rows}, dim] or [{rows}, tokens, dim]",
                plan.output
            ),
        };
        if plan.normalize {
            vectors.iter_mut().for_each(|v| normalize(v));
        }
        Ok(vectors)
    })
}

/// One vector per row of a `[rows, tokens, dim]` output; `mask` is `[rows, mask_len]`.
fn pool(
    data: &[f32],
    tokens: usize,
    dim: usize,
    mask: &[i64],
    mask_len: usize,
    pooling: OnnxPooling,
) -> Vec<Vec<f32>> {
    data.chunks(tokens * dim)
        .enumerate()
        .map(|(row, hidden)| match pooling {
            OnnxPooling::Cls => hidden[..dim].to_vec(),
            OnnxPooling::Mean => {
                let mut sum = vec![0f32; dim];
                let mut counted = 0f32;
                let row_mask = &mask[row * mask_len..(row + 1) * mask_len];
                for (token, weight) in hidden.chunks(dim).zip(row_mask) {
                    if *weight != 0 {
                        sum.iter_mut().zip(token).for_each(|(s, t)| *s += t);
                        counted += 1.0;
                    }
                }
                let counted = counted.max(1e-9);
                sum.iter_mut().for_each(|s| *s /= counted);
                sum
            }
        })
        .collect()
}

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    vector.iter_mut().for_each(|x| *x /= norm);
}

#[async_trait]
impl MessagePublisher for OnnxEmbeddingsPublisher {
    async fn send_batch(
        &self,
        messages: Vec<CanonicalMessage>,
    ) -> Result<SentBatch, PublisherError> {
        if messages.is_empty() {
            return Ok(SentBatch::Ack);
        }
        let answers = self.embed(&messages).await;
        Ok(crate::sent_batch(messages, answers))
    }

    async fn lookup_each(
        &self,
        requests: &[CanonicalMessage],
    ) -> Option<Vec<Result<Option<Value>, PublisherError>>> {
        let answers = self.embed(requests).await;
        Some(answers.into_iter().map(|a| a.map(Some)).collect())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl std::fmt::Debug for OnnxEmbeddingsPublisher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxEmbeddingsPublisher")
            .field("model", &model_name(&self.model))
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pooling_skips_padding() {
        // Two rows, two tokens, dim 2; row 1's second token is padding.
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0, 100.0];
        let mask = [1, 1, 1, 0];
        let pooled = pool(&data, 2, 2, &mask, 2, OnnxPooling::Mean);
        assert_eq!(pooled, vec![vec![2.0, 3.0], vec![5.0, 6.0]]);
    }

    #[test]
    fn cls_pooling_takes_the_first_token() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let pooled = pool(&data, 2, 2, &[1, 1], 2, OnnxPooling::Cls);
        assert_eq!(pooled, vec![vec![1.0, 2.0]]);
    }

    #[test]
    fn numbers_follow_the_vector_and_a_missing_one_is_null() {
        let numbers: Vec<_> = ["${payload:age}", "${payload:score}"]
            .iter()
            .map(|t| CompiledTemplate::compile(t, None).unwrap())
            .collect();
        let message = CanonicalMessage::from(r#"{"age": 41}"#);
        let value = append_numbers(&[0.5], &numbers, &message).unwrap();
        assert_eq!(value.to_string(), "[0.5,41.0,null]");
        let text = CanonicalMessage::from(r#"{"age": "old"}"#);
        let error = append_numbers(&[0.5], &numbers, &text).unwrap_err();
        assert_eq!(error.to_string(), "numbers 0 is not a number");
    }

    #[test]
    fn normalized_vectors_have_length_one() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert_eq!(v, vec![0.6, 0.8]);
        let mut zero = vec![0.0, 0.0];
        normalize(&mut zero);
        assert_eq!(zero, vec![0.0, 0.0]);
    }
}
