//  mq-bridge-onnx
//  © Copyright 2026, by Marco Mengelkoch
//  Licensed under MIT OR Apache-2.0, see LICENSE file for more details
//  git clone https://github.com/marcomq/mq-bridge-onnx

//! Decode `pcm` and `wav`: audio to mono float32 samples in -1..1, optionally resampled.

use anyhow::{anyhow, bail};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

use crate::decode::OnnxPcmFormat;

/// Raw little-endian interleaved samples, averaged to mono.
pub(crate) fn from_pcm(
    bytes: &[u8],
    format: OnnxPcmFormat,
    channels: usize,
) -> anyhow::Result<Vec<f32>> {
    let width = match format {
        OnnxPcmFormat::S16le => 2,
        OnnxPcmFormat::F32le => 4,
    };
    if channels == 0 || !bytes.len().is_multiple_of(width * channels) {
        bail!(
            "{} bytes are not whole frames of {channels} {width}-byte samples",
            bytes.len()
        );
    }
    let samples = bytes
        .chunks_exact(width)
        .map(|b| match format {
            OnnxPcmFormat::S16le => f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0,
            OnnxPcmFormat::F32le => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        })
        .collect();
    Ok(downmix(samples, channels))
}

/// A WAV file's samples, averaged to mono, and its sample rate.
pub(crate) fn from_wav(bytes: &[u8]) -> anyhow::Result<(Vec<f32>, u32)> {
    let reader = hound::WavReader::new(std::io::Cursor::new(bytes))
        .map_err(|e| anyhow!("not a WAV file: {e}"))?;
    let spec = reader.spec();
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader.into_samples::<f32>().collect::<Result<_, _>>(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|s| s as f32 * scale))
                .collect()
        }
    }
    .map_err(|e| anyhow!("WAV samples: {e}"))?;
    Ok((
        downmix(samples, usize::from(spec.channels)),
        spec.sample_rate,
    ))
}

fn downmix(samples: Vec<f32>, channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples;
    }
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// `mono` at `to` Hz instead of `from` Hz.
pub(crate) fn resample(mono: Vec<f32>, from: u32, to: u32) -> anyhow::Result<Vec<f32>> {
    if from == to || mono.is_empty() {
        return Ok(mono);
    }
    let failed = |e: &dyn std::fmt::Display| anyhow!("cannot resample {from} Hz to {to} Hz: {e}");
    let mut resampler = Fft::<f32>::new(from as usize, to as usize, 1024, 1, FixedSync::Input)
        .map_err(|e| failed(&e))?;
    let input = InterleavedSlice::new(&mono[..], 1, mono.len()).map_err(|e| failed(&e))?;
    let output = resampler
        .process_all(&input, mono.len(), None)
        .map_err(|e| failed(&e))?;
    Ok(output.take_data())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stereo_pcm_is_averaged_to_mono() {
        let bytes: Vec<u8> = [16384i16, 0, -32768, -32768]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        assert_eq!(
            from_pcm(&bytes, OnnxPcmFormat::S16le, 2).unwrap(),
            [0.25, -1.0]
        );
        assert!(from_pcm(&bytes[..3], OnnxPcmFormat::S16le, 1).is_err());
    }

    #[test]
    fn wav_files_give_their_rate() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 8000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut file = std::io::Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(&mut file, spec).unwrap();
        [16384i16, -16384]
            .iter()
            .for_each(|s| writer.write_sample(*s).unwrap());
        writer.finalize().unwrap();
        let (samples, rate) = from_wav(file.get_ref()).unwrap();
        assert_eq!((samples, rate), (vec![0.5, -0.5], 8000));
        assert!(from_wav(b"RIFF nonsense").is_err());
    }

    #[test]
    fn resampling_keeps_the_duration_and_the_tone() {
        let tone: Vec<f32> = (0..32000)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 32000.0).sin())
            .collect();
        let half = resample(tone, 32000, 16000).unwrap();
        assert_eq!(half.len(), 16000);
        let peak = half[1000..15000].iter().fold(0f32, |m, s| m.max(s.abs()));
        assert!((peak - 1.0).abs() < 0.05, "peak {peak}");
    }
}
