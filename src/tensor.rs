//  mq-bridge-onnx
//  © Copyright 2026, by Marco Mengelkoch
//  Licensed under MIT OR Apache-2.0, see LICENSE file for more details
//  git clone https://github.com/marcomq/mq-bridge-onnx

//! Task `tensor`: one tensor per message, as the `onnx` middleware writes it,
//! stacked into a batch; the answer is the model's outputs, or one flat vector.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{anyhow, bail};
use async_trait::async_trait;
use ort::value::{DynValue, Tensor};
use serde_json::Value;

use crate::config::{OnnxPooling, TensorConfig};
use crate::runtime::{self as onnx, OnnxModel};
use crate::tensor_data::{Element, OnnxDtype, TensorData};
use crate::{embeddings, float_json, model_name, scoring};
use mq_bridge::CanonicalMessage;
use mq_bridge::SentBatch;
use mq_bridge::errors::PublisherError;
use mq_bridge::traits::MessagePublisher;

const DEFAULT_MAX_BATCH: usize = 32;

enum Answer {
    Outputs(Vec<String>),
    Vector {
        output: String,
        pooling: OnnxPooling,
        normalize: bool,
    },
}

struct Plan {
    input: String,
    dtype: OnnxDtype,
    /// The input's dimensions after the batch one; -1 or 0 where any size goes.
    dims: Vec<i64>,
    answer: Answer,
    max_batch: usize,
}

pub(crate) struct OnnxTensorPublisher {
    model: Arc<OnnxModel>,
    plan: Arc<Plan>,
}

impl OnnxTensorPublisher {
    pub(crate) async fn new(config: TensorConfig) -> anyhow::Result<Self> {
        let max_batch = config.max_batch.unwrap_or(DEFAULT_MAX_BATCH).max(1);
        let vector = config.vector.map(|output| Answer::Vector {
            output,
            pooling: config.pooling.unwrap_or_default(),
            normalize: config.normalize.unwrap_or(true),
        });
        let answer = vector.ok_or(config.outputs);
        let input_name = config.input_name;
        let (model, plan) = onnx::open_and_warm_up(
            "onnx tensor",
            config.model,
            config.threads.unwrap_or(onnx::DEFAULT_THREADS),
            config.device,
            move |model| {
                let plan = plan_for(model, input_name, answer, max_batch)?;
                let fixed: Option<Vec<usize>> = plan
                    .dims
                    .iter()
                    .map(|d| usize::try_from(*d).ok().filter(|d| *d > 0))
                    .collect();
                let Some(dims) = fixed else {
                    return Ok((plan, None));
                };
                let bytes = vec![0; dims.iter().product::<usize>() * plan.dtype.size()];
                let zeros = TensorData::new(plan.dtype, dims.clone(), bytes)?;
                run_group(model, &plan, &vec![&zeros; max_batch])?;
                Ok((plan, Some(format!("{max_batch} × {dims:?} tensors"))))
            },
        )
        .await?;
        Ok(Self {
            model,
            plan: Arc::new(plan),
        })
    }

    async fn answer(&self, messages: &[CanonicalMessage]) -> Vec<Result<Value, PublisherError>> {
        let mut answers: Vec<Option<Result<Value, PublisherError>>> =
            messages.iter().map(|_| None).collect();
        let mut samples = Vec::new();
        for (index, message) in messages.iter().enumerate() {
            match TensorData::of_message(message).and_then(|t| fits(&self.plan, t)) {
                Ok(tensor) => samples.push((index, tensor)),
                Err(e) => answers[index] = Some(Err(PublisherError::NonRetryable(e))),
            }
        }
        let indices: Vec<usize> = samples.iter().map(|(index, _)| *index).collect();
        if !samples.is_empty() {
            let plan = self.plan.clone();
            let answered = onnx::infer(self.model.clone(), move |model| {
                Ok(run_all(model, &plan, samples))
            })
            .await;
            match answered {
                Ok(results) => {
                    for (index, result) in results {
                        answers[index] =
                            Some(result.map_err(|e| PublisherError::NonRetryable(anyhow!("{e}"))));
                    }
                }
                Err(e) => {
                    let reason = e.to_string();
                    for index in indices {
                        answers[index] =
                            Some(Err(PublisherError::NonRetryable(anyhow!("{reason}"))));
                    }
                }
            }
        }
        answers
            .into_iter()
            .map(|a| {
                a.unwrap_or_else(|| Err(PublisherError::NonRetryable(anyhow!("not answered"))))
            })
            .collect()
    }
}

/// Checks the input, the batch size and the outputs once, when the model opens.
fn plan_for(
    model: &OnnxModel,
    input_name: Option<String>,
    answer: Result<Answer, Option<Vec<String>>>,
    max_batch: usize,
) -> anyhow::Result<Plan> {
    let (name, dtype) = scoring::input_of(model, input_name.as_deref())?;
    let element = dtype
        .tensor_type()
        .and_then(OnnxDtype::of)
        .ok_or_else(|| anyhow!("ONNX model input '{name}' is {dtype}; onnx feeds float, double, int32, int64 or uint8 tensors"))?;
    let shape = dtype.tensor_shape().map(|s| s.to_vec()).unwrap_or_default();
    let Some((batch, dims)) = shape.split_first() else {
        bail!("ONNX model input '{name}' has no batch dimension");
    };
    scoring::check_batch(&name, *batch, max_batch)?;
    let answer = match answer {
        Ok(vector) => {
            if let Answer::Vector { output, .. } = &vector {
                scoring::outputs_of(model, Some(vec![output.clone()]))?;
            }
            vector
        }
        Err(outputs) => Answer::Outputs(scoring::outputs_of(model, outputs)?),
    };
    Ok(Plan {
        input: name,
        dtype: element,
        dims: dims.to_vec(),
        answer,
        max_batch,
    })
}

fn fits(plan: &Plan, tensor: TensorData) -> anyhow::Result<TensorData> {
    let matches = tensor.shape.len() == plan.dims.len()
        && tensor
            .shape
            .iter()
            .zip(&plan.dims)
            .all(|(size, dim)| *dim <= 0 || *size as i64 == *dim);
    if !matches {
        bail!(
            "ONNX model input '{}' takes samples of shape {:?} (-1 is any size); the message has {:?}",
            plan.input,
            plan.dims,
            tensor.shape
        );
    }
    Ok(tensor)
}

/// Runs samples of the same shape together, `max_batch` at a time; a failed
/// call fails only its own samples.
fn run_all(
    model: &OnnxModel,
    plan: &Plan,
    samples: Vec<(usize, TensorData)>,
) -> Vec<(usize, Result<Value, String>)> {
    let mut groups: BTreeMap<Vec<usize>, Vec<(usize, TensorData)>> = BTreeMap::new();
    for (index, tensor) in samples {
        groups
            .entry(tensor.shape.clone())
            .or_default()
            .push((index, tensor));
    }
    let mut results = Vec::new();
    for chunk in groups
        .values()
        .flat_map(|group| group.chunks(plan.max_batch))
    {
        let tensors: Vec<&TensorData> = chunk.iter().map(|(_, t)| t).collect();
        let indices = chunk.iter().map(|(index, _)| *index);
        match run_group(model, plan, &tensors) {
            Ok(values) => results.extend(indices.zip(values.into_iter().map(Ok))),
            Err(e) => {
                let reason = format!("{e:#}");
                results.extend(indices.map(|index| (index, Err(reason.clone()))));
            }
        }
    }
    results
}

/// One answer per tensor; all tensors have the same shape.
fn run_group(
    model: &OnnxModel,
    plan: &Plan,
    tensors: &[&TensorData],
) -> anyhow::Result<Vec<Value>> {
    let rows = tensors.len();
    let mut shape = vec![rows];
    shape.extend(&tensors[0].shape);
    let input = match plan.dtype {
        OnnxDtype::Float32 => stack::<f32>(shape, tensors),
        OnnxDtype::Float64 => stack::<f64>(shape, tensors),
        OnnxDtype::Int32 => stack::<i32>(shape, tensors),
        OnnxDtype::Int64 => stack::<i64>(shape, tensors),
        OnnxDtype::Uint8 => stack::<u8>(shape, tensors),
    }?;
    model.with_session(|session| {
        let outputs = session
            .run(vec![(plan.input.as_str(), input)])
            .map_err(|e| anyhow!("{e}"))?;
        match &plan.answer {
            Answer::Outputs(names) => scoring::objects_of(&outputs, names, rows),
            Answer::Vector { output, pooling, normalize } => {
                let (shape, data) = outputs[output.as_str()]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| anyhow!("output '{output}': {e}"))?;
                let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                let mut vectors = match dims.as_slice() {
                    [r, d] if *r == rows => data.chunks(*d).map(<[f32]>::to_vec).collect(),
                    [r, tokens, d] if *r == rows => {
                        let every = vec![1; rows * tokens];
                        embeddings::pool(data, *tokens, *d, &every, *tokens, *pooling)
                    }
                    _ => bail!(
                        "output '{output}' has shape {dims:?}; expected [{rows}, dim] or [{rows}, tokens, dim]"
                    ),
                };
                if *normalize {
                    vectors.iter_mut().for_each(|v| embeddings::normalize(v));
                }
                Ok(vectors.iter().map(|v| float_json(v)).collect())
            }
        }
    })
}

fn stack<T: Element>(shape: Vec<usize>, tensors: &[&TensorData]) -> anyhow::Result<DynValue> {
    let mut values = Vec::with_capacity(shape.iter().product());
    tensors.iter().for_each(|t| t.extend_into::<T>(&mut values));
    Tensor::from_array((shape, values))
        .map(Tensor::into_dyn)
        .map_err(|e| anyhow!("{e}"))
}

#[async_trait]
impl MessagePublisher for OnnxTensorPublisher {
    async fn send_batch(
        &self,
        messages: Vec<CanonicalMessage>,
    ) -> Result<SentBatch, PublisherError> {
        if messages.is_empty() {
            return Ok(SentBatch::Ack);
        }
        let answers = self.answer(&messages).await;
        Ok(crate::sent_batch(messages, answers))
    }

    async fn lookup_each(
        &self,
        requests: &[CanonicalMessage],
    ) -> Option<Vec<Result<Option<Value>, PublisherError>>> {
        let answers = self.answer(requests).await;
        Some(answers.into_iter().map(|a| a.map(Some)).collect())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl std::fmt::Debug for OnnxTensorPublisher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxTensorPublisher")
            .field("model", &model_name(&self.model))
            .finish_non_exhaustive()
    }
}
