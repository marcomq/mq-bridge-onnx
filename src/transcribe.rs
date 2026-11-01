//  mq-bridge-onnx
//  © Copyright 2026, by Marco Mengelkoch
//  Licensed under MIT OR Apache-2.0, see LICENSE file for more details
//  git clone https://github.com/marcomq/mq-bridge-onnx

//! Task `transcribe`: speech to text with Parakeet TDT. The model decodes token by
//! token with `&mut self`, so it lives on one thread of its own that takes the
//! messages one after another.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc;

use anyhow::{anyhow, bail};
use async_trait::async_trait;
use mq_bridge::errors::PublisherError;
use mq_bridge::traits::MessagePublisher;
use mq_bridge::{CanonicalMessage, SentBatch};
use parakeet_rs::{ExecutionConfig, ParakeetTDT, TimestampMode, Transcriber};
use serde_json::{Value, json};
use tokio::sync::oneshot;

use crate::config::TranscribeConfig;
use crate::float_value;
use crate::runtime::{self, peak_memory_bytes};
use crate::tensor_data::{DTYPE_KEY, ERROR_KEY, OnnxDtype, TensorData};

/// The sample rate Parakeet listens at.
const RATE: u32 = 16_000;
/// Longer audio is refused: the encoder's memory grows fast with length.
const MAX_SECONDS: usize = 600;
/// Shorter audio is too short to hold a word, and too short for the encoder.
const MIN_SAMPLES: usize = RATE as usize / 10;

type Answers = Vec<anyhow::Result<Value>>;
type Job = (Vec<CanonicalMessage>, oneshot::Sender<Answers>);

pub(crate) struct OnnxTranscribePublisher {
    jobs: mpsc::Sender<Job>,
    model: String,
}

impl OnnxTranscribePublisher {
    pub(crate) async fn new(config: TranscribeConfig) -> anyhow::Result<Self> {
        let threads = config.threads.unwrap_or(runtime::DEFAULT_THREADS).max(1);
        let (jobs, inbox) = mpsc::channel::<Job>();
        let (ready, loaded) = oneshot::channel();
        let dir = config.model.clone();
        std::thread::Builder::new()
            .name("onnx-transcribe".into())
            .spawn(move || {
                let mut parakeet = match open(&dir, threads) {
                    Ok(parakeet) => parakeet,
                    Err(e) => {
                        let _ = ready.send(Err(e));
                        return;
                    }
                };
                let _ = ready.send(Ok(()));
                // Ends once the publisher, and with it the sender, is dropped.
                for (messages, reply) in inbox {
                    if reply.is_closed() {
                        continue;
                    }
                    let answers = messages
                        .iter()
                        .map(|m| {
                            // A panic fails its message alone instead of ending the thread.
                            catch_unwind(AssertUnwindSafe(|| transcribe(&mut parakeet, m)))
                                .unwrap_or_else(|_| Err(anyhow!("transcription panicked")))
                        })
                        .collect();
                    let _ = reply.send(answers);
                }
            })?;
        loaded
            .await
            .map_err(|_| anyhow!("loading Parakeet model '{}' stopped", config.model))??;
        Ok(Self {
            jobs,
            model: config.model,
        })
    }

    async fn run(&self, messages: Vec<CanonicalMessage>) -> Vec<Result<Value, PublisherError>> {
        let count = messages.len();
        let (reply, answers) = oneshot::channel();
        let answers = match self.jobs.send((messages, reply)) {
            Ok(()) => answers.await.ok(),
            Err(_) => None,
        };
        match answers {
            Some(answers) => answers
                .into_iter()
                .map(|a| a.map_err(PublisherError::NonRetryable))
                .collect(),
            None => (0..count)
                .map(|_| {
                    Err(PublisherError::NonRetryable(anyhow!(
                        "the transcribe thread of '{}' stopped",
                        self.model
                    )))
                })
                .collect(),
        }
    }
}

/// Opens the model folder in the shared ONNX Runtime environment, on the CPU.
fn open(dir: &str, threads: usize) -> anyhow::Result<ParakeetTDT> {
    let before = peak_memory_bytes();
    let config = ExecutionConfig::new()
        .with_environment(runtime::environment()?.clone())
        .with_intra_threads(threads);
    let parakeet = ParakeetTDT::from_pretrained(dir, Some(config))
        .map_err(|e| anyhow!("cannot open Parakeet TDT model '{dir}': {e}"))?;
    let mib = |bytes: u64| bytes / (1024 * 1024);
    match (before, peak_memory_bytes()) {
        (Some(before), Some(after)) => tracing::info!(
            endpoint = "onnx transcribe",
            model = %dir,
            threads,
            "ONNX model ready; peak memory {} MiB ({} MiB before loading)",
            mib(after),
            mib(before),
        ),
        _ => tracing::info!(endpoint = "onnx transcribe", model = %dir, threads, "ONNX model ready"),
    }
    Ok(parakeet)
}

/// `{ "text": ..., "words": [{ "text", "start", "end" }] }`, times in seconds.
fn transcribe(parakeet: &mut ParakeetTDT, message: &CanonicalMessage) -> anyhow::Result<Value> {
    let (samples, rate) = samples_of(message)?;
    let Some(samples) = at_16_khz(samples, rate)? else {
        return Ok(json!({ "text": "", "words": [] }));
    };
    let result = parakeet
        .transcribe_samples(samples, RATE, 1, Some(TimestampMode::Words))
        .map_err(|e| anyhow!("{e}"))?;
    let words: Vec<Value> = result
        .tokens
        .iter()
        .map(|w| {
            json!({
                "text": w.text.trim(),
                "start": float_value(w.start),
                "end": float_value(w.end),
            })
        })
        .collect();
    Ok(json!({ "text": result.text.trim(), "words": words }))
}

/// `samples` resampled to [`RATE`], `None` when too short to hold speech.
fn at_16_khz(samples: Vec<f32>, rate: u32) -> anyhow::Result<Option<Vec<f32>>> {
    if rate == 0 {
        bail!("the audio's sample rate is 0");
    }
    let seconds = samples.len() / rate as usize;
    if seconds > MAX_SECONDS {
        bail!("{seconds} s of audio; task transcribe takes up to {MAX_SECONDS} s");
    }
    let samples = crate::audio::resample(samples, rate, RATE)?;
    Ok((samples.len() >= MIN_SAMPLES).then_some(samples))
}

/// Mono samples and their rate: from a WAV payload, or from the tensor the
/// `onnx` middleware decoded.
fn samples_of(message: &CanonicalMessage) -> anyhow::Result<(Vec<f32>, u32)> {
    if message.get_metadata(DTYPE_KEY).is_none() && message.get_metadata(ERROR_KEY).is_none() {
        return crate::audio::from_wav(&message.payload);
    }
    let mut tensors = TensorData::all_of_message(message)?;
    let tensor = match tensors.pop() {
        Some((_, tensor)) if tensors.is_empty() => tensor,
        _ => bail!("task transcribe takes one audio tensor"),
    };
    if tensor.dtype != OnnxDtype::Float32 || tensor.shape.len() != 1 {
        bail!("task transcribe takes a float32 `[samples]` tensor, as decode pcm or wav gives");
    }
    let rate = message
        .get_metadata("audio_sample_rate")
        .ok_or_else(|| anyhow!("the audio's rate is unknown; give decode pcm a `sample_rate`"))?
        .parse()
        .map_err(|_| anyhow!("audio_sample_rate is not a number"))?;
    let mut samples = Vec::with_capacity(tensor.shape[0]);
    tensor.extend_into(&mut samples);
    Ok((samples, rate))
}

#[async_trait]
impl MessagePublisher for OnnxTranscribePublisher {
    async fn send_batch(
        &self,
        messages: Vec<CanonicalMessage>,
    ) -> Result<SentBatch, PublisherError> {
        if messages.is_empty() {
            return Ok(SentBatch::Ack);
        }
        let answers = self.run(messages.clone()).await;
        Ok(crate::sent_batch(messages, answers))
    }

    async fn lookup_each(
        &self,
        requests: &[CanonicalMessage],
    ) -> Option<Vec<Result<Option<Value>, PublisherError>>> {
        let answers = self.run(requests.to_vec()).await;
        Some(answers.into_iter().map(|a| a.map(Some)).collect())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl std::fmt::Debug for OnnxTranscribePublisher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxTranscribePublisher")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_comes_as_wav_or_as_a_decoded_tensor() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 8000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut file = std::io::Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(&mut file, spec).unwrap();
        writer.write_sample(16384i16).unwrap();
        writer.finalize().unwrap();
        let wav = CanonicalMessage::new(file.into_inner(), None);
        assert_eq!(samples_of(&wav).unwrap(), (vec![0.5], 8000));

        let mut decoded = CanonicalMessage::from("");
        TensorData::from_values(vec![2], &[0.25f32, -0.5]).into_message(&mut decoded);
        assert!(samples_of(&decoded).unwrap_err().to_string().contains("rate"));
        decoded.set_metadata("audio_sample_rate", "16000");
        assert_eq!(samples_of(&decoded).unwrap(), (vec![0.25, -0.5], 16000));

        decoded.set_metadata(ERROR_KEY, "not a WAV file");
        assert_eq!(samples_of(&decoded).unwrap_err().to_string(), "not a WAV file");
    }

    #[test]
    fn too_short_audio_is_silence_and_too_long_audio_is_refused() {
        assert_eq!(at_16_khz(vec![0.1; 100], 16_000).unwrap(), None);
        assert_eq!(at_16_khz(vec![0.0; 3200], 16_000).unwrap().unwrap().len(), 3200);
        assert_eq!(at_16_khz(vec![0.0; 9600], 48_000).unwrap().unwrap().len(), 3200);
        let long = vec![0.0; 8000 * (MAX_SECONDS + 1)];
        assert!(at_16_khz(long, 8000).unwrap_err().to_string().contains("up to 600 s"));
        assert!(at_16_khz(vec![0.0; 10], 0).is_err());
    }
}
