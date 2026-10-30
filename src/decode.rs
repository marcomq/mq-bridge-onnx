//  mq-bridge-onnx
//  © Copyright 2026, by Marco Mengelkoch
//  Licensed under MIT OR Apache-2.0, see LICENSE file for more details
//  git clone https://github.com/marcomq/mq-bridge-onnx

//! The `onnx` middleware: turns an image, a JSON array or a `.npy` file into the
//! tensor task `tensor` reads. A message it cannot decode keeps its payload and
//! gets `tensor_error`, so the endpoint fails that message alone.

use std::borrow::Cow;
use std::sync::Arc;

use anyhow::{Context, anyhow, bail};
use async_trait::async_trait;
use base64::Engine as _;
use mq_bridge::errors::{ConsumerError, InvalidConfig, PublisherError};
use mq_bridge::support::interpolation::CompiledTemplate;
use mq_bridge::traits::{BoxFuture, CustomMiddlewareFactory, MessageConsumer, MessagePublisher};
use mq_bridge::{CanonicalMessage, Received, ReceivedBatch, SentBatch};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::preprocess::{OnnxExtent, Preprocessing};
use crate::tensor_data::{ERROR_KEY, OnnxDtype, TensorData};

/// The `onnx` middleware: decodes each message into a tensor for task `tensor`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[schemars(title = "ONNX tensor decoder")]
#[serde(deny_unknown_fields)]
pub struct OnnxDecodeConfig {
    /// What the input is.
    pub decode: OnnxDecode,
    /// Template of the input, e.g. `${payload:photo}`. Defaults to the whole payload.
    pub from: Option<String>,
    /// Decode the input from base64 first; a `data:` URL prefix is skipped.
    #[serde(default)]
    pub base64: bool,
    /// json: element type of the tensor. Defaults to float32.
    pub dtype: Option<OnnxDtype>,
    /// image: path of a Hugging Face `preprocessor_config.json`; the fields below override it.
    pub preprocessor: Option<String>,
    /// image: resize to this shortest edge, or to `[height, width]`.
    pub size: Option<OnnxExtent>,
    /// image: centre crop to this square, or to `[height, width]`.
    pub crop: Option<OnnxExtent>,
    /// image: factor of each 0–255 pixel value. Defaults to 1/255.
    pub rescale: Option<f32>,
    /// image: per-channel mean subtracted after rescaling.
    pub mean: Option<[f32; 3]>,
    /// image: per-channel standard deviation divided by after the mean.
    pub std: Option<[f32; 3]>,
    /// pcm: sample format. Defaults to s16le.
    pub format: Option<OnnxPcmFormat>,
    /// pcm: interleaved channels, averaged to mono. Defaults to 1.
    pub channels: Option<usize>,
    /// pcm: sample rate in Hz; needed by `resample_to`.
    pub sample_rate: Option<u32>,
    /// pcm and wav: resample to this rate in Hz, e.g. 16000.
    pub resample_to: Option<u32>,
}

/// What the `onnx` middleware decodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum OnnxDecode {
    /// A PNG or JPEG image, to a `[3, height, width]` float32 tensor.
    Image,
    /// A (nested) JSON array of numbers, to a tensor of its shape; an object of
    /// them, to one named tensor per field.
    Json,
    /// A NumPy `.npy` file.
    Npy,
    /// Raw little-endian PCM samples, to a mono `[samples]` float32 tensor. Feature `audio`.
    Pcm,
    /// A WAV file, to a mono `[samples]` float32 tensor. Feature `audio`.
    Wav,
}

/// Sample format of decode `pcm`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum OnnxPcmFormat {
    /// 16-bit signed integers.
    #[default]
    S16le,
    /// 32-bit floats in -1..1.
    F32le,
}

enum Kind {
    Image(Preprocessing),
    Json(OnnxDtype),
    Npy,
    #[cfg(feature = "audio")]
    Pcm {
        format: OnnxPcmFormat,
        channels: usize,
        rate: Option<u32>,
        resample_to: Option<u32>,
    },
    #[cfg(feature = "audio")]
    Wav {
        resample_to: Option<u32>,
    },
}

pub(crate) struct Decoder {
    from: Option<CompiledTemplate>,
    base64: bool,
    kind: Kind,
}

impl Decoder {
    pub(crate) fn new(config: OnnxDecodeConfig) -> anyhow::Result<Self> {
        use OnnxDecode::{Image, Json, Pcm, Wav};
        let fields: [(&str, bool, &[OnnxDecode]); 11] = [
            ("dtype", config.dtype.is_some(), &[Json]),
            ("preprocessor", config.preprocessor.is_some(), &[Image]),
            ("size", config.size.is_some(), &[Image]),
            ("crop", config.crop.is_some(), &[Image]),
            ("rescale", config.rescale.is_some(), &[Image]),
            ("mean", config.mean.is_some(), &[Image]),
            ("std", config.std.is_some(), &[Image]),
            ("format", config.format.is_some(), &[Pcm]),
            ("channels", config.channels.is_some(), &[Pcm]),
            ("sample_rate", config.sample_rate.is_some(), &[Pcm]),
            ("resample_to", config.resample_to.is_some(), &[Pcm, Wav]),
        ];
        if config.decode == OnnxDecode::Npy && config.dtype.is_some() {
            bail!("`dtype` is not read by decode \"npy\"; the file names its own");
        }
        let decode = serde_json::to_value(config.decode)?;
        if let Some((field, ..)) = fields
            .iter()
            .find(|(_, set, readers)| *set && !readers.contains(&config.decode))
        {
            bail!("`{field}` is not read by decode {decode}");
        }
        if config.decode == Pcm && config.resample_to.is_some() && config.sample_rate.is_none() {
            bail!("decode \"pcm\" needs `sample_rate` to resample");
        }
        let zero = [
            ("channels", config.channels == Some(0)),
            ("sample_rate", config.sample_rate == Some(0)),
            ("resample_to", config.resample_to == Some(0)),
        ];
        if let Some((field, _)) = zero.iter().find(|(_, is_zero)| *is_zero) {
            bail!("`{field}` must be greater than 0");
        }
        let kind = match config.decode {
            OnnxDecode::Image => {
                let mut settings = match &config.preprocessor {
                    Some(path) => Preprocessing::from_file(path)?,
                    None => Preprocessing::default(),
                };
                settings.resize = config.size.or(settings.resize);
                settings.crop = config.crop.or(settings.crop);
                settings.rescale = config.rescale.unwrap_or(settings.rescale);
                settings.mean = config.mean.or(settings.mean);
                settings.std = config.std.or(settings.std);
                Kind::Image(settings)
            }
            OnnxDecode::Json => Kind::Json(config.dtype.unwrap_or_default()),
            OnnxDecode::Npy => Kind::Npy,
            #[cfg(feature = "audio")]
            OnnxDecode::Pcm => Kind::Pcm {
                format: config.format.unwrap_or_default(),
                channels: config.channels.unwrap_or(1),
                rate: config.sample_rate,
                resample_to: config.resample_to,
            },
            #[cfg(feature = "audio")]
            OnnxDecode::Wav => Kind::Wav {
                resample_to: config.resample_to,
            },
            #[cfg(not(feature = "audio"))]
            OnnxDecode::Pcm | OnnxDecode::Wav => {
                bail!("decode {decode} needs mq-bridge-onnx built with feature `audio`")
            }
        };
        let from = config
            .from
            .as_deref()
            .map(|t| CompiledTemplate::compile(t, None))
            .transpose()
            .map_err(|e| anyhow!("from: {e}"))?;
        Ok(Self {
            from,
            base64: config.base64,
            kind,
        })
    }

    fn from_value(config: &Value) -> anyhow::Result<Self> {
        let config: OnnxDecodeConfig = serde_json::from_value(config.clone())
            .map_err(|e| InvalidConfig(anyhow!("onnx middleware: {e}")))?;
        Ok(Self::new(config).map_err(|e| InvalidConfig(anyhow!("onnx middleware: {e:#}")))?)
    }

    /// Replaces the message's payload by its tensor, or marks why it has none.
    fn decode(&self, message: &mut CanonicalMessage) {
        if let Err(e) = self.try_decode(message) {
            message.set_metadata(ERROR_KEY, format!("{e:#}"));
        }
    }

    fn try_decode(&self, message: &mut CanonicalMessage) -> anyhow::Result<()> {
        let mut input: Cow<[u8]> = match &self.from {
            Some(template) => Cow::Owned(
                template
                    .render_resolved(Some(message))
                    .ok_or_else(|| anyhow!("`from` has no value in this message"))?,
            ),
            None => Cow::Borrowed(&message.payload),
        };
        if self.base64 {
            input = Cow::Owned(from_base64(&input)?);
        }
        let mut image_size = None;
        let (tensor, audio_rate): (TensorData, Option<u32>) = match &self.kind {
            Kind::Image(settings) => {
                let (tensor, width, height) = settings.run(&input)?;
                image_size = Some((width, height, serde_json::to_string(settings)?));
                (tensor, None)
            }
            Kind::Json(dtype) => {
                let value: Value =
                    serde_json::from_slice(&input).map_err(|e| anyhow!("not JSON: {e}"))?;
                if let Value::Object(fields) = value {
                    if fields.is_empty() {
                        bail!("the JSON object has no fields");
                    }
                    if let Some(name) = fields.keys().find(|name| name.contains(',')) {
                        bail!("field '{name}': tensor names cannot contain ','");
                    }
                    let named = fields
                        .into_iter()
                        .map(|(name, v)| {
                            let tensor = TensorData::from_json_value(&v, *dtype)
                                .with_context(|| format!("field '{name}'"))?;
                            Ok((name, tensor))
                        })
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    drop(input);
                    TensorData::all_into_message(named, message);
                    return Ok(());
                }
                (TensorData::from_json_value(&value, *dtype)?, None)
            }
            Kind::Npy => (TensorData::from_npy(&input)?, None),
            #[cfg(feature = "audio")]
            Kind::Pcm {
                format,
                channels,
                rate,
                resample_to,
            } => {
                let mono = crate::audio::from_pcm(&input, *format, *channels)?;
                audio_tensor(mono, *rate, *resample_to)?
            }
            #[cfg(feature = "audio")]
            Kind::Wav { resample_to } => {
                let (mono, rate) = crate::audio::from_wav(&input)?;
                audio_tensor(mono, Some(rate), *resample_to)?
            }
        };
        drop(input);
        tensor.into_message(message);
        if let Some((width, height, settings)) = image_size {
            message.set_metadata("image_width", width.to_string());
            message.set_metadata("image_height", height.to_string());
            message.set_metadata("image_preprocessing", settings);
        }
        if let Some(rate) = audio_rate {
            message.set_metadata("audio_sample_rate", rate.to_string());
        }
        Ok(())
    }

    /// Decodes every message off the async threads; image decoding is CPU work.
    async fn decode_all(
        self: &Arc<Self>,
        mut messages: Vec<CanonicalMessage>,
    ) -> Vec<CanonicalMessage> {
        let decoder = Arc::clone(self);
        let handle = tokio::task::spawn_blocking(move || {
            messages.iter_mut().for_each(|m| decoder.decode(m));
            messages
        });
        match handle.await {
            Ok(messages) => messages,
            Err(e) => std::panic::resume_unwind(e.into_panic()),
        }
    }
}

/// Mono samples as a `[samples]` tensor, resampled when asked, and their rate if known.
#[cfg(feature = "audio")]
fn audio_tensor(
    mono: Vec<f32>,
    rate: Option<u32>,
    resample_to: Option<u32>,
) -> anyhow::Result<(TensorData, Option<u32>)> {
    let mono = match (rate, resample_to) {
        (Some(from), Some(to)) => crate::audio::resample(mono, from, to)?,
        _ => mono,
    };
    let tensor = TensorData::from_values(vec![mono.len()], &mono);
    Ok((tensor, resample_to.or(rate)))
}

fn from_base64(text: &[u8]) -> anyhow::Result<Vec<u8>> {
    let text = text.trim_ascii();
    let text = match text.strip_prefix(b"data:") {
        Some(url) => {
            let start = url
                .windows(8)
                .position(|w| w == b";base64,")
                .ok_or_else(|| anyhow!("the data URL is not base64"))?;
            &url[start + 8..]
        }
        None => text,
    };
    base64::engine::general_purpose::STANDARD
        .decode(text)
        .map_err(|e| anyhow!("not base64: {e}"))
}

/// Factory of the `onnx` middleware.
#[derive(Debug, Default)]
pub struct OnnxMiddlewareFactory;

fn schema() -> Option<Value> {
    serde_json::to_value(schemars::schema_for!(OnnxDecodeConfig)).ok()
}

#[async_trait]
impl CustomMiddlewareFactory for OnnxMiddlewareFactory {
    async fn apply_consumer(
        &self,
        consumer: Box<dyn MessageConsumer>,
        _route_name: &str,
        config: &Value,
    ) -> anyhow::Result<Box<dyn MessageConsumer>> {
        Ok(Box::new(DecodeConsumer {
            inner: consumer,
            decoder: Arc::new(Decoder::from_value(config)?),
        }))
    }

    async fn apply_publisher(
        &self,
        publisher: Box<dyn MessagePublisher>,
        _route_name: &str,
        config: &Value,
    ) -> anyhow::Result<Box<dyn MessagePublisher>> {
        Ok(Box::new(DecodePublisher {
            inner: publisher,
            decoder: Arc::new(Decoder::from_value(config)?),
        }))
    }

    fn config_schema(&self) -> Option<Value> {
        schema()
    }
}

// What a host that loads the plugin wraps around its own publisher or consumer.
#[cfg(feature = "plugin")]
mod plugin {
    use super::*;
    use mq_bridge::plugin::sdk::{BatchFilter, MiddlewareFactory};

    struct DecodeFilter(Arc<Decoder>);

    #[async_trait]
    impl BatchFilter for DecodeFilter {
        async fn on_receive(
            &self,
            messages: Vec<CanonicalMessage>,
        ) -> anyhow::Result<Vec<Option<CanonicalMessage>>> {
            Ok(self
                .0
                .decode_all(messages)
                .await
                .into_iter()
                .map(Some)
                .collect())
        }

        async fn on_send(
            &self,
            messages: Vec<CanonicalMessage>,
        ) -> anyhow::Result<Vec<Option<CanonicalMessage>>> {
            self.on_receive(messages).await
        }
    }

    #[async_trait]
    impl MiddlewareFactory for OnnxMiddlewareFactory {
        async fn create(
            &self,
            _route_name: &str,
            config: &Value,
        ) -> anyhow::Result<Box<dyn BatchFilter>> {
            Ok(Box::new(DecodeFilter(Arc::new(Decoder::from_value(
                config,
            )?))))
        }

        fn config_schema(&self) -> Option<Value> {
            schema()
        }
    }
}

struct DecodePublisher {
    inner: Box<dyn MessagePublisher>,
    decoder: Arc<Decoder>,
}

#[async_trait]
impl MessagePublisher for DecodePublisher {
    fn on_connect_hook(&self) -> Option<BoxFuture<'_, anyhow::Result<()>>> {
        self.inner.on_connect_hook()
    }

    fn on_disconnect_hook(&self) -> Option<BoxFuture<'_, anyhow::Result<()>>> {
        self.inner.on_disconnect_hook()
    }

    async fn send_batch(
        &self,
        messages: Vec<CanonicalMessage>,
    ) -> Result<SentBatch, PublisherError> {
        let decoded = self.decoder.decode_all(messages).await;
        self.inner.send_batch(decoded).await
    }

    async fn lookup_each(
        &self,
        requests: &[CanonicalMessage],
    ) -> Option<Vec<Result<Option<Value>, PublisherError>>> {
        let decoded = self.decoder.decode_all(requests.to_vec()).await;
        self.inner.lookup_each(&decoded).await
    }

    async fn flush(&self) -> anyhow::Result<()> {
        self.inner.flush().await
    }

    fn requires_ordered_publish(&self) -> bool {
        self.inner.requires_ordered_publish()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

struct DecodeConsumer {
    inner: Box<dyn MessageConsumer>,
    decoder: Arc<Decoder>,
}

#[async_trait]
impl MessageConsumer for DecodeConsumer {
    fn set_exit_on_empty(&mut self, exit_on_empty: bool) {
        self.inner.set_exit_on_empty(exit_on_empty);
    }

    fn commit_requires_order(&self) -> bool {
        self.inner.commit_requires_order()
    }

    fn on_connect_hook(&self) -> Option<BoxFuture<'_, anyhow::Result<()>>> {
        self.inner.on_connect_hook()
    }

    fn on_disconnect_hook(&self) -> Option<BoxFuture<'_, anyhow::Result<()>>> {
        self.inner.on_disconnect_hook()
    }

    async fn receive(&mut self) -> Result<Received, ConsumerError> {
        let Received {
            mut message,
            commit,
        } = self.inner.receive().await?;
        self.decoder.decode(&mut message);
        Ok(Received { message, commit })
    }

    async fn receive_batch(&mut self, max_messages: usize) -> Result<ReceivedBatch, ConsumerError> {
        let ReceivedBatch { messages, commit } = self.inner.receive_batch(max_messages).await?;
        let messages = self.decoder.decode_all(messages).await;
        Ok(ReceivedBatch { messages, commit })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_data::{DTYPE_KEY, SHAPE_KEY};
    use serde_json::json;

    fn decoder(config: Value) -> anyhow::Result<Decoder> {
        Decoder::new(serde_json::from_value(config)?)
    }

    #[test]
    fn a_base64_json_field_becomes_a_tensor() {
        let decoder =
            decoder(json!({ "decode": "json", "from": "${payload:x}", "base64": true })).unwrap();
        let encoded = base64::engine::general_purpose::STANDARD.encode("[[1, 2], [3, 4]]");
        let mut message = CanonicalMessage::from(json!({ "x": encoded }).to_string());
        decoder.decode(&mut message);
        assert_eq!(message.get_metadata(ERROR_KEY), None);
        assert_eq!(message.get_metadata(DTYPE_KEY), Some("float32"));
        assert_eq!(message.get_metadata(SHAPE_KEY), Some("2,2"));
        assert_eq!(message.payload.len(), 16);
    }

    #[test]
    fn a_message_that_does_not_decode_is_marked_and_kept() {
        let decoder = decoder(json!({ "decode": "image" })).unwrap();
        let mut message = CanonicalMessage::from("not an image");
        decoder.decode(&mut message);
        assert!(
            message
                .get_metadata(ERROR_KEY)
                .unwrap()
                .contains("not a PNG or JPEG")
        );
        assert_eq!(&message.payload[..], b"not an image");
    }

    #[test]
    fn a_data_url_is_read_as_base64() {
        assert_eq!(
            from_base64(b" data:text/plain;base64,aGk= ").unwrap(),
            b"hi"
        );
        assert!(from_base64(b"data:text/plain,hi").is_err());
    }

    #[test]
    fn a_field_of_another_decode_is_refused() {
        let error = decoder(json!({ "decode": "json", "size": 224 }))
            .err()
            .unwrap();
        assert_eq!(error.to_string(), "`size` is not read by decode \"json\"");
        assert!(decoder(json!({ "decode": "image", "dtype": "int64" })).is_err());
        assert!(decoder(json!({ "decode": "npy", "dtype": "int64" })).is_err());
        assert!(decoder(json!({ "decode": "wav", "channels": 2 })).is_err());
        assert!(
            decoder(json!({ "decode": "pcm", "resample_to": 16000 }))
                .err()
                .unwrap()
                .to_string()
                .contains("`sample_rate`")
        );
    }

    #[test]
    fn zero_audio_sizes_are_refused() {
        for field in ["channels", "sample_rate", "resample_to"] {
            let error = decoder(json!({ "decode": "pcm", "sample_rate": 16000, field: 0 }))
                .err()
                .unwrap();
            assert_eq!(
                error.to_string(),
                format!("`{field}` must be greater than 0")
            );
        }
    }

    #[test]
    fn a_json_object_needs_fields_with_plain_names() {
        let decoder = decoder(json!({ "decode": "json" })).unwrap();
        for (body, reason) in [
            (json!({}), "no fields"),
            (json!({ "a,b": [1] }), "cannot contain ','"),
        ] {
            let mut message = CanonicalMessage::from(body.to_string());
            decoder.decode(&mut message);
            assert!(message.get_metadata(ERROR_KEY).unwrap().contains(reason));
        }
    }

    #[test]
    fn a_json_object_becomes_named_tensors() {
        let decoder = decoder(json!({ "decode": "json" })).unwrap();
        let mut message =
            CanonicalMessage::from(json!({ "input": [1, 2], "state": [[0], [0]] }).to_string());
        decoder.decode(&mut message);
        assert_eq!(message.get_metadata(ERROR_KEY), None);
        assert_eq!(
            message.get_metadata(crate::tensor_data::NAMES_KEY),
            Some("input,state")
        );
        assert_eq!(message.get_metadata(SHAPE_KEY), Some("2;2,1"));
        assert_eq!(message.payload.len(), 16);
    }

    #[cfg(feature = "audio")]
    #[test]
    fn pcm_is_resampled_to_the_asked_rate() {
        let decoder = decoder(json!({
            "decode": "pcm", "format": "f32le", "sample_rate": 32000, "resample_to": 16000
        }))
        .unwrap();
        let bytes: Vec<u8> = (0..3200).flat_map(|_| 0.5f32.to_le_bytes()).collect();
        let mut message = CanonicalMessage::from(bytes);
        decoder.decode(&mut message);
        assert_eq!(message.get_metadata(ERROR_KEY), None);
        assert_eq!(message.get_metadata(SHAPE_KEY), Some("1600"));
        assert_eq!(message.get_metadata("audio_sample_rate"), Some("16000"));
    }

    #[cfg(not(feature = "audio"))]
    #[test]
    fn audio_needs_its_feature() {
        let error = decoder(json!({ "decode": "wav" })).err().unwrap();
        assert!(error.to_string().contains("feature `audio`"), "{error}");
    }
}
