//  mq-bridge-onnx
//  © Copyright 2026, by Marco Mengelkoch
//  Licensed under MIT OR Apache-2.0, see LICENSE file for more details
//  git clone https://github.com/marcomq/mq-bridge-onnx

//! Task `tensor`: the tensors of each message, as the `onnx` middleware writes them,
//! stacked into a batch per model input; the answer is the model's outputs, or one
//! flat vector.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, anyhow, bail};
use async_trait::async_trait;
use ort::value::{DynValue, Tensor};
use serde_json::Value;

use crate::config::{OnnxInput, OnnxPooling, TensorConfig};
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

/// One model input and where its tensor comes from.
struct Feed {
    input: String,
    dtype: OnnxDtype,
    source: Source,
}

enum Source {
    Message {
        /// The message tensor's name; `None` takes a message's one unnamed tensor.
        tensor: Option<String>,
        /// The input's dimensions, after the batch one when `batched`; -1 or 0 where any size goes.
        dims: Vec<i64>,
        batched: bool,
    },
    Constant(TensorData),
}

struct Plan {
    feeds: Vec<Feed>,
    answer: Answer,
    max_batch: usize,
    /// One message per call with a tensor passed whole: outputs are answered whole too.
    whole: bool,
}

impl Plan {
    fn message_feeds(&self) -> impl Iterator<Item = (&Feed, &Option<String>, &[i64])> {
        self.feeds.iter().filter_map(|feed| match &feed.source {
            Source::Message { tensor, dims, .. } => Some((feed, tensor, dims.as_slice())),
            Source::Constant(_) => None,
        })
    }
}

/// A message's tensors, one per message feed, in feed order.
type Sample = Vec<TensorData>;

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
        let wanted = if config.inputs.is_empty() {
            vec![(config.input_name, None)]
        } else {
            config
                .inputs
                .into_iter()
                .map(|(name, input)| (Some(name), Some(input)))
                .collect()
        };
        let (model, plan) = onnx::open_and_warm_up(
            "onnx tensor",
            config.model,
            config.threads.unwrap_or(onnx::DEFAULT_THREADS),
            config.device,
            move |model| {
                let plan = plan_for(model, wanted, answer, max_batch)?;
                let zeros: Option<Sample> = plan
                    .message_feeds()
                    .map(|(feed, _, dims)| {
                        let dims: Vec<usize> = dims
                            .iter()
                            .map(|d| usize::try_from(*d).ok().filter(|d| *d > 0))
                            .collect::<Option<_>>()?;
                        let bytes = vec![0; dims.iter().product::<usize>() * feed.dtype.size()];
                        TensorData::new(feed.dtype, dims, bytes).ok()
                    })
                    .collect();
                let Some(zeros) = zeros else {
                    return Ok((plan, None));
                };
                run_group(model, &plan, &vec![&zeros; max_batch])?;
                let shapes: Vec<&[usize]> = zeros.iter().map(|t| t.shape.as_slice()).collect();
                Ok((plan, Some(format!("{max_batch} × {shapes:?} tensors"))))
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
            match sample_of(&self.plan, message) {
                Ok(sample) => samples.push((index, sample)),
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

/// Checks the inputs, the batch size and the outputs once, when the model opens.
/// `wanted` names each model input and its source; `None` is the shorthand
/// `input_name` (or the only input), fed by the message's one tensor.
fn plan_for(
    model: &OnnxModel,
    wanted: Vec<(Option<String>, Option<OnnxInput>)>,
    answer: Result<Answer, Option<Vec<String>>>,
    max_batch: usize,
) -> anyhow::Result<Plan> {
    let named = wanted.iter().all(|(_, input)| input.is_some());
    let mut feeds = Vec::with_capacity(wanted.len());
    for (name, input) in wanted {
        let (name, dtype) = scoring::input_of(model, name.as_deref())?;
        let element = dtype
            .tensor_type()
            .and_then(OnnxDtype::of)
            .ok_or_else(|| anyhow!("ONNX model input '{name}' is {dtype}; onnx feeds float, double, int32, int64 or uint8 tensors"))?;
        let shape = dtype.tensor_shape().map(|s| s.to_vec()).unwrap_or_default();
        let tensor = input
            .as_ref()
            .map(|i| i.from.clone().unwrap_or_else(|| name.clone()));
        let input = input.unwrap_or_default();
        let source = if let Some(value) = &input.value {
            let constant = TensorData::from_json_value(value, element)
                .with_context(|| format!("`inputs.{name}.value`"))?;
            fits(&name, &shape, &constant)?;
            if max_batch != 1 && shape.first().is_some_and(|d| *d <= 0) {
                bail!(
                    "ONNX model input '{name}' has a dynamic first dimension; its constant `value` needs `max_batch: 1`"
                );
            }
            Source::Constant(constant)
        } else if input.batch == Some(false) {
            if max_batch != 1 {
                bail!(
                    "ONNX model input '{name}' takes its tensor whole (`batch: false`); set `max_batch: 1`"
                );
            }
            Source::Message {
                tensor,
                dims: shape,
                batched: false,
            }
        } else {
            let Some((batch, dims)) = shape.split_first() else {
                bail!("ONNX model input '{name}' has no batch dimension; give it `batch: false`");
            };
            scoring::check_batch(&name, *batch, max_batch)?;
            Source::Message {
                tensor,
                dims: dims.to_vec(),
                batched: true,
            }
        };
        feeds.push(Feed {
            input: name,
            dtype: element,
            source,
        });
    }
    if named {
        for (input, _) in model.inputs() {
            if !feeds.iter().any(|feed| feed.input == input) {
                bail!("ONNX model input '{input}' is missing from `inputs`");
            }
        }
    }
    let whole = feeds
        .iter()
        .any(|feed| matches!(feed.source, Source::Message { batched: false, .. }));
    let answer = match answer {
        Ok(vector) => {
            if whole {
                bail!("`vector` needs batched inputs; drop `batch: false` or answer `outputs`");
            }
            if let Answer::Vector { output, .. } = &vector {
                scoring::outputs_of(model, Some(vec![output.clone()]))?;
            }
            vector
        }
        Err(outputs) => Answer::Outputs(scoring::outputs_of(model, outputs)?),
    };
    let plan = Plan {
        feeds,
        answer,
        max_batch,
        whole,
    };
    if plan.message_feeds().next().is_none() {
        bail!("every model input is a constant `value`; one must take the message's tensor");
    }
    Ok(plan)
}

/// Picks each message feed's tensor out of `message` and checks its shape.
fn sample_of(plan: &Plan, message: &CanonicalMessage) -> anyhow::Result<Sample> {
    let tensors = TensorData::all_of_message(message)?;
    let only_one = plan.message_feeds().count() == 1;
    let mut sample = Vec::with_capacity(tensors.len());
    for (feed, wanted, dims) in plan.message_feeds() {
        let found = match (tensors.as_slice(), wanted) {
            ([(None, _)], _) | ([_], None) if only_one => Some(0),
            (_, Some(wanted)) => tensors
                .iter()
                .position(|(name, _)| name.as_deref() == Some(wanted.as_str())),
            (_, None) => None,
        };
        let Some(found) = found else {
            let names: Vec<&str> = tensors.iter().filter_map(|(n, _)| n.as_deref()).collect();
            bail!(
                "ONNX model input '{}' finds no tensor {} in the message; it has {names:?}",
                feed.input,
                wanted
                    .as_deref()
                    .map_or("without a name".into(), |w| format!("'{w}'"))
            );
        };
        let tensor = tensors[found].1.clone();
        fits(&feed.input, dims, &tensor)?;
        sample.push(tensor);
    }
    Ok(sample)
}

fn fits(input: &str, dims: &[i64], tensor: &TensorData) -> anyhow::Result<()> {
    let matches = tensor.shape.len() == dims.len()
        && tensor
            .shape
            .iter()
            .zip(dims)
            .all(|(size, dim)| *dim <= 0 || *size as i64 == *dim);
    if !matches {
        bail!(
            "ONNX model input '{input}' takes tensors of shape {dims:?} (-1 is any size); this one has {:?}",
            tensor.shape
        );
    }
    Ok(())
}

/// Runs samples of the same shapes together, `max_batch` at a time; a failed
/// call fails only its own samples.
fn run_all(
    model: &OnnxModel,
    plan: &Plan,
    samples: Vec<(usize, Sample)>,
) -> Vec<(usize, Result<Value, String>)> {
    let mut groups: BTreeMap<Vec<Vec<usize>>, Vec<(usize, Sample)>> = BTreeMap::new();
    for (index, sample) in samples {
        let shapes = sample.iter().map(|t| t.shape.clone()).collect();
        groups.entry(shapes).or_default().push((index, sample));
    }
    let mut results = Vec::new();
    for chunk in groups
        .values()
        .flat_map(|group| group.chunks(plan.max_batch))
    {
        let samples: Vec<&Sample> = chunk.iter().map(|(_, s)| s).collect();
        let indices = chunk.iter().map(|(index, _)| *index);
        match run_group(model, plan, &samples) {
            Ok(values) => results.extend(indices.zip(values.into_iter().map(Ok))),
            Err(e) => {
                let reason = format!("{e:#}");
                results.extend(indices.map(|index| (index, Err(reason.clone()))));
            }
        }
    }
    results
}

/// One answer per sample; all samples have the same shapes.
fn run_group(model: &OnnxModel, plan: &Plan, samples: &[&Sample]) -> anyhow::Result<Vec<Value>> {
    let rows = samples.len();
    let mut inputs = Vec::with_capacity(plan.feeds.len());
    let mut next = 0;
    for feed in &plan.feeds {
        let value = match &feed.source {
            Source::Constant(tensor) => tensor_value(feed.dtype, tensor.shape.clone(), &[tensor]),
            Source::Message { batched, .. } => {
                let tensors: Vec<&TensorData> = samples.iter().map(|s| &s[next]).collect();
                next += 1;
                let mut shape = if *batched { vec![rows] } else { Vec::new() };
                shape.extend(&tensors[0].shape);
                tensor_value(feed.dtype, shape, &tensors)
            }
        }?;
        inputs.push((feed.input.as_str(), value));
    }
    model.with_session(|session| {
        let outputs = session.run(inputs).map_err(|e| anyhow!("{e}"))?;
        match &plan.answer {
            Answer::Outputs(names) => scoring::objects_of(&outputs, names, rows, plan.whole),
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

/// `tensors` one after another as one model input of `shape`, cast to `dtype`.
fn tensor_value(
    dtype: OnnxDtype,
    shape: Vec<usize>,
    tensors: &[&TensorData],
) -> anyhow::Result<DynValue> {
    match dtype {
        OnnxDtype::Float32 => stack::<f32>(shape, tensors),
        OnnxDtype::Float64 => stack::<f64>(shape, tensors),
        OnnxDtype::Int32 => stack::<i32>(shape, tensors),
        OnnxDtype::Int64 => stack::<i64>(shape, tensors),
        OnnxDtype::Uint8 => stack::<u8>(shape, tensors),
    }
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
