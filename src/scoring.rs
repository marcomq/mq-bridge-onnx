//  mq-bridge-onnx
//  © Copyright 2026, by Marco Mengelkoch
//  Licensed under MIT OR Apache-2.0, see LICENSE file for more details
//  git clone https://github.com/marcomq/mq-bridge-onnx

//! Task `score`: one row of numeric features per message in, the model's outputs out,
//! e.g. a CatBoost, XGBoost or LightGBM model exported to ONNX.

use std::sync::Arc;

use anyhow::{Context, anyhow, bail};
use async_trait::async_trait;
use ort::session::SessionOutputs;
use ort::value::{DynMapValueType, DynValue, Tensor, TensorElementType, ValueType};
use serde_json::{Map, Value};

use crate::config::ScoreConfig;
use crate::runtime::{self as onnx, OnnxModel};
use crate::{float_value, model_name};
use mq_bridge::CanonicalMessage;
use mq_bridge::SentBatch;
use mq_bridge::errors::PublisherError;
use mq_bridge::support::interpolation::CompiledTemplate;
use mq_bridge::traits::MessagePublisher;

const DEFAULT_MAX_BATCH: usize = 256;

struct Plan {
    input: String,
    double: bool,
    width: usize,
    outputs: Vec<String>,
    max_batch: usize,
}

pub(crate) struct OnnxPublisher {
    model: Arc<OnnxModel>,
    plan: Arc<Plan>,
    features: Vec<CompiledTemplate>,
}

impl OnnxPublisher {
    pub(crate) async fn new(config: ScoreConfig) -> anyhow::Result<Self> {
        if config.features.is_empty() {
            bail!("onnx needs at least one entry in 'features'");
        }
        let features = config
            .features
            .iter()
            .enumerate()
            .map(|(i, f)| {
                CompiledTemplate::compile(f, None).with_context(|| format!("features[{i}]"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let width = features.len();
        let max_batch = config.max_batch.unwrap_or(DEFAULT_MAX_BATCH).max(1);
        let (input, outputs) = (config.input_name.clone(), config.outputs.clone());
        let (model, plan) = onnx::open_and_warm_up(
            "onnx score",
            config.model.clone(),
            config.threads.unwrap_or(onnx::DEFAULT_THREADS),
            config.device,
            move |model| {
                let plan = plan_for(model, input, outputs, width, max_batch)?;
                run_batch(model, &plan, vec![0.0; max_batch * width])?;
                Ok((plan, Some(format!("{max_batch} rows × {width} features"))))
            },
        )
        .await?;
        Ok(Self {
            model,
            plan: Arc::new(plan),
            features,
        })
    }

    async fn score(&self, messages: &[CanonicalMessage]) -> Vec<Result<Value, PublisherError>> {
        let mut answers: Vec<Option<Result<Value, PublisherError>>> =
            messages.iter().map(|_| None).collect();
        let mut rows = Vec::new();
        let mut values = Vec::with_capacity(messages.len() * self.plan.width);
        for (index, message) in messages.iter().enumerate() {
            match features_of("feature", &self.features, message) {
                Ok(row) => {
                    rows.push(index);
                    values.extend(row);
                }
                Err(e) => answers[index] = Some(Err(PublisherError::NonRetryable(e))),
            }
        }
        if !rows.is_empty() {
            let plan = self.plan.clone();
            let scored = onnx::infer(self.model.clone(), move |model| {
                let chunk = plan.max_batch * plan.width;
                Ok(values
                    .chunks(chunk)
                    .map(|part| run_batch(model, &plan, part.to_vec()))
                    .collect::<Vec<_>>())
            })
            .await;
            match scored {
                Ok(chunks) => {
                    for (indices, chunk) in rows.chunks(self.plan.max_batch).zip(chunks) {
                        match chunk {
                            Ok(scores) => {
                                for (index, score) in indices.iter().zip(scores) {
                                    answers[*index] = Some(Ok(score));
                                }
                            }
                            Err(e) => {
                                let reason = format!("{e:#}");
                                for index in indices {
                                    answers[*index] = Some(Err(PublisherError::NonRetryable(
                                        anyhow!("{reason}"),
                                    )));
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    let reason = e.to_string();
                    for index in rows {
                        answers[index] =
                            Some(Err(PublisherError::NonRetryable(anyhow!("{reason}"))));
                    }
                }
            }
        }
        answers
            .into_iter()
            .map(|a| a.unwrap_or_else(|| Err(PublisherError::NonRetryable(anyhow!("not scored")))))
            .collect()
    }
}

/// Each template's number for `message`; `label` names them in errors. A missing,
/// null or empty value is NaN, which tree models treat as missing.
pub(crate) fn features_of(
    label: &str,
    features: &[CompiledTemplate],
    message: &CanonicalMessage,
) -> anyhow::Result<Vec<f64>> {
    features
        .iter()
        .enumerate()
        .map(|(i, template)| {
            let Some(raw) = template.render_resolved(Some(message)) else {
                return Ok(f64::NAN);
            };
            parse_feature(&String::from_utf8_lossy(&raw))
                .ok_or_else(|| anyhow!("{label} {i} is not a number"))
        })
        .collect()
}

fn parse_feature(text: &str) -> Option<f64> {
    match text.trim() {
        "" | "null" => Some(f64::NAN),
        "true" => Some(1.0),
        "false" => Some(0.0),
        number => number.parse().ok(),
    }
}

fn plan_for(
    model: &OnnxModel,
    input: Option<String>,
    outputs: Option<Vec<String>>,
    width: usize,
    max_batch: usize,
) -> anyhow::Result<Plan> {
    let (name, dtype) = input_of(model, input.as_deref())?;
    let double = match dtype.tensor_type() {
        Some(TensorElementType::Float32) => false,
        Some(TensorElementType::Float64) => true,
        _ => bail!("ONNX model input '{name}' is {dtype}; onnx feeds float or double tensors"),
    };
    if let Some(shape) = dtype.tensor_shape() {
        if shape.len() != 2 {
            bail!(
                "ONNX model input '{name}' has rank {}; onnx feeds [rows, features]",
                shape.len()
            );
        }
        let (batch, features) = (shape[0], shape[1]);
        if features > 0 && features as usize != width {
            bail!(
                "ONNX model input '{name}' takes {features} features, but {width} are configured"
            );
        }
        check_batch(&name, batch, max_batch)?;
    }
    let outputs = outputs_of(model, outputs)?;
    Ok(Plan {
        input: name,
        double,
        width,
        outputs,
        max_batch,
    })
}

/// The model input named `wanted`, or its only one.
pub(crate) fn input_of(
    model: &OnnxModel,
    wanted: Option<&str>,
) -> anyhow::Result<(String, ValueType)> {
    let inputs = model.inputs();
    match wanted {
        Some(wanted) => inputs
            .into_iter()
            .find(|(name, _)| name == wanted)
            .ok_or_else(|| anyhow!("ONNX model '{}' has no input '{wanted}'", model.path)),
        None => match <[_; 1]>::try_from(inputs) {
            Ok([only]) => Ok(only),
            Err(inputs) => bail!(
                "ONNX model '{}' has {} inputs; name one in 'input_name'",
                model.path,
                inputs.len()
            ),
        },
    }
}

/// Refuses a fixed batch dimension that batches of up to `max_batch` cannot fill.
pub(crate) fn check_batch(name: &str, batch: i64, max_batch: usize) -> anyhow::Result<()> {
    match batch {
        1 if max_batch != 1 => {
            bail!("ONNX model input '{name}' takes one row per call; set `max_batch: 1`")
        }
        rows if rows > 1 => bail!(
            "ONNX model input '{name}' takes exactly {rows} rows per call; export it with a dynamic batch dimension"
        ),
        _ => Ok(()),
    }
}

/// The `wanted` model outputs, all of them by default.
pub(crate) fn outputs_of(
    model: &OnnxModel,
    wanted: Option<Vec<String>>,
) -> anyhow::Result<Vec<String>> {
    let available: Vec<String> = model.outputs().into_iter().map(|(n, _)| n).collect();
    Ok(match wanted {
        Some(wanted) if wanted.is_empty() => {
            bail!("`outputs` names no output; leave it out to get all of them")
        }
        Some(wanted) => {
            if let Some(missing) = wanted.iter().find(|w| !available.contains(w)) {
                bail!(
                    "ONNX model '{}' has no output '{missing}'; it has {}",
                    model.path,
                    available.join(", ")
                );
            }
            wanted
        }
        None => available,
    })
}

/// Scores `values`, `width` per row, into one JSON object of outputs per row.
fn run_batch(model: &OnnxModel, plan: &Plan, values: Vec<f64>) -> anyhow::Result<Vec<Value>> {
    let rows = values.len() / plan.width;
    let shape = [rows as i64, plan.width as i64];
    let tensor: DynValue = if plan.double {
        Tensor::from_array((shape, values)).map(Tensor::into_dyn)
    } else {
        let values: Vec<f32> = values.into_iter().map(|v| v as f32).collect();
        Tensor::from_array((shape, values)).map(Tensor::into_dyn)
    }
    .map_err(|e| anyhow!("{e}"))?;
    model.with_session(|session| {
        let outputs = session
            .run(vec![(plan.input.as_str(), tensor)])
            .map_err(|e| anyhow!("{e}"))?;
        objects_of(&outputs, &plan.outputs, rows, false)
    })
}

/// One JSON object of the `names` outputs per row; `whole` answers each output
/// whole, for one row whose outputs need not have a batch dimension.
pub(crate) fn objects_of(
    outputs: &SessionOutputs<'_>,
    names: &[String],
    rows: usize,
    whole: bool,
) -> anyhow::Result<Vec<Value>> {
    let mut objects: Vec<Map<String, Value>> = (0..rows).map(|_| Map::new()).collect();
    for name in names {
        let per_row = rows_of(&outputs[name.as_str()], rows, whole)
            .with_context(|| format!("output '{name}'"))?;
        for (object, value) in objects.iter_mut().zip(per_row) {
            object.insert(name.clone(), value);
        }
    }
    Ok(objects.into_iter().map(Value::Object).collect())
}

/// Splits an output into one JSON value per row: a number, a list, a string,
/// or an object of class to probability for a classifier's map output.
fn rows_of(value: &DynValue, rows: usize, whole: bool) -> anyhow::Result<Vec<Value>> {
    let failed = |e: ort::Error| anyhow!("{e}");
    match value.dtype() {
        ValueType::Tensor { ty, .. } => {
            let (shape, cells) = match ty {
                TensorElementType::String => {
                    let (shape, text) = value.try_extract_strings().map_err(failed)?;
                    (
                        shape.to_vec(),
                        text.into_iter().map(Value::String).collect(),
                    )
                }
                TensorElementType::Float32 => {
                    cells(value.try_extract_tensor::<f32>(), |x| float_value(*x))?
                }
                TensorElementType::Float64 => cells(value.try_extract_tensor::<f64>(), |x| {
                    serde_json::Number::from_f64(*x).map_or(Value::Null, Value::Number)
                })?,
                TensorElementType::Int64 => {
                    cells(value.try_extract_tensor::<i64>(), |x| Value::from(*x))?
                }
                TensorElementType::Int32 => {
                    cells(value.try_extract_tensor::<i32>(), |x| Value::from(*x))?
                }
                TensorElementType::Bool => {
                    cells(value.try_extract_tensor::<bool>(), |x| Value::from(*x))?
                }
                other => bail!("{other} tensors are not supported"),
            };
            if whole {
                let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
                return Ok(vec![nest(&mut cells.into_iter(), &dims)]);
            }
            split_rows(&shape, cells, rows)
        }
        ValueType::Sequence(_) => {
            let maps = value
                .try_extract_sequence::<DynMapValueType>()
                .map_err(failed)?;
            if maps.len() != rows {
                bail!("has {} entries for {rows} rows", maps.len());
            }
            maps.iter().map(map_value).collect()
        }
        other => bail!("{other} outputs are not supported"),
    }
}

type Cells = (Vec<i64>, Vec<Value>);

fn cells<T>(
    extracted: ort::Result<(&ort::value::Shape, &[T])>,
    convert: impl Fn(&T) -> Value,
) -> anyhow::Result<Cells> {
    let (shape, data) = extracted.map_err(|e| anyhow!("{e}"))?;
    Ok((shape.to_vec(), data.iter().map(convert).collect()))
}

/// `[rows]` and `[rows, 1]` give a value per row, `[rows, k]` a list of `k`, and
/// `[rows, k, m]` a list of `k` lists of `m`.
fn split_rows(shape: &[i64], cells: Vec<Value>, rows: usize) -> anyhow::Result<Vec<Value>> {
    if shape.first().map(|r| *r as usize) != Some(rows) {
        bail!("has shape {shape:?} for {rows} rows");
    }
    let inner: Vec<usize> = shape[1..].iter().map(|d| *d as usize).collect();
    if inner.iter().product::<usize>() <= 1 {
        return Ok(cells);
    }
    let mut cells = cells.into_iter();
    Ok((0..rows).map(|_| nest(&mut cells, &inner)).collect())
}

fn nest(cells: &mut impl Iterator<Item = Value>, dims: &[usize]) -> Value {
    match dims {
        [] => cells.next().unwrap_or(Value::Null),
        [len, inner @ ..] => Value::Array((0..*len).map(|_| nest(cells, inner)).collect()),
    }
}

/// A map output as an object, keys in label order: ONNX Runtime returns them unordered.
fn map_value(map: &ort::value::Value<DynMapValueType>) -> anyhow::Result<Value> {
    let entries: Vec<(String, f32)> = match map.try_extract_map::<i64, f32>() {
        Ok(entries) => {
            let mut entries: Vec<(i64, f32)> = entries.into_iter().collect();
            entries.sort_unstable_by_key(|(k, _)| *k);
            entries
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect()
        }
        Err(_) => {
            let mut entries: Vec<(String, f32)> = map
                .try_extract_map::<String, f32>()
                .map_err(|e| anyhow!("{e}"))?
                .into_iter()
                .collect();
            entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            entries
        }
    };
    Ok(Value::Object(
        entries
            .into_iter()
            .map(|(k, v)| (k, float_value(v)))
            .collect(),
    ))
}

#[async_trait]
impl MessagePublisher for OnnxPublisher {
    async fn send_batch(
        &self,
        messages: Vec<CanonicalMessage>,
    ) -> Result<SentBatch, PublisherError> {
        if messages.is_empty() {
            return Ok(SentBatch::Ack);
        }
        let answers = self.score(&messages).await;
        Ok(crate::sent_batch(messages, answers))
    }

    async fn lookup_each(
        &self,
        requests: &[CanonicalMessage],
    ) -> Option<Vec<Result<Option<Value>, PublisherError>>> {
        let answers = self.score(requests).await;
        Some(answers.into_iter().map(|a| a.map(Some)).collect())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl std::fmt::Debug for OnnxPublisher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxPublisher")
            .field("model", &model_name(&self.model))
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn features_read_numbers_and_missing_values() {
        let templates = [
            "${payload:a}",
            "${payload:b}",
            "${payload:c}",
            "${payload:d}",
        ]
        .map(|t| CompiledTemplate::compile(t, None).unwrap());
        let message = CanonicalMessage::from(r#"{"a": 3, "b": "1.5", "c": null}"#);
        let row = features_of("feature", &templates, &message).unwrap();
        assert_eq!(&row[..2], &[3.0, 1.5]);
        assert!(row[2].is_nan() && row[3].is_nan());

        let text = CanonicalMessage::from(r#"{"a": "high"}"#);
        assert!(features_of("feature", &templates[..1], &text).is_err());
    }

    #[test]
    fn outputs_split_into_rows() {
        let flat = vec![json!(1), json!(2)];
        assert_eq!(split_rows(&[2], flat.clone(), 2).unwrap(), flat);
        assert_eq!(split_rows(&[2, 1], flat.clone(), 2).unwrap(), flat);
        let probabilities = vec![json!(0.9), json!(0.1), json!(0.2), json!(0.8)];
        assert_eq!(
            split_rows(&[2, 2], probabilities, 2).unwrap(),
            vec![json!([0.9, 0.1]), json!([0.2, 0.8])]
        );
        assert!(split_rows(&[3], vec![json!(1); 3], 2).is_err());
        let cube: Vec<Value> = (0..8).map(|i| json!(i)).collect();
        assert_eq!(
            split_rows(&[2, 2, 2], cube, 2).unwrap(),
            vec![json!([[0, 1], [2, 3]]), json!([[4, 5], [6, 7]])]
        );
    }
}
