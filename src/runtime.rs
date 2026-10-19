//  mq-bridge-onnx
//  © Copyright 2026, by Marco Mengelkoch
//  Licensed under MIT OR Apache-2.0, see LICENSE file for more details
//  git clone https://github.com/marcomq/mq-bridge-onnx

//! What both tasks share: opening a model on the CPU and
//! running it off the async threads. ONNX Runtime is linked into this library.

use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, anyhow};
use mq_bridge::errors::PublisherError;
use ort::session::Session;
use ort::session::builder::{AutoDevicePolicy, GraphOptimizationLevel};
use ort::value::ValueType;

use crate::config::OnnxDevice;

/// Default CPU threads of one inference call.
pub(crate) const DEFAULT_THREADS: usize = 2;

static ENVIRONMENT: OnceLock<Result<ort::environment::Environment, String>> = OnceLock::new();

fn environment() -> anyhow::Result<&'static ort::environment::Environment> {
    let started = ENVIRONMENT.get_or_init(|| {
        ort::init()
            .with_name("mq-bridge")
            .build()
            .map_err(|e| format!("ONNX Runtime did not start: {e}"))
    });
    started.as_ref().map_err(|e| anyhow!("{e}"))
}

/// An opened model. ONNX Runtime runs one call per session at a time.
pub(crate) struct OnnxModel {
    session: Mutex<Session>,
    pub(crate) path: String,
}

/// Name and type of each model input or output, in model order.
pub(crate) type Outlets = Vec<(String, ValueType)>;

impl OnnxModel {
    /// Opens `path` with `threads` CPU threads. Blocks while the model loads.
    pub(crate) fn open(path: &str, threads: usize, device: OnnxDevice) -> anyhow::Result<Self> {
        let environment = environment()?;
        let failed = |e: ort::Error<_>| anyhow!("{e}");
        let mut builder = Session::builder(environment)
            .map_err(|e| anyhow!("{e}"))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(failed)?
            .with_intra_threads(threads.max(1))
            .map_err(failed)?;
        // ONNX Runtime's own default picks CoreML on macOS, whose memory is unbounded.
        if device == OnnxDevice::Cpu {
            builder = builder
                .with_auto_device(AutoDevicePolicy::PreferCPU)
                .map_err(failed)?;
        }
        let session = builder
            .commit_from_file(path)
            .map_err(|e| anyhow!("{e}"))
            .with_context(|| format!("cannot open ONNX model '{path}'"))?;
        Ok(Self {
            session: Mutex::new(session),
            path: path.to_string(),
        })
    }

    pub(crate) fn inputs(&self) -> Outlets {
        let session = self.lock();
        session
            .inputs()
            .iter()
            .map(|o| (o.name().to_string(), o.dtype().clone()))
            .collect()
    }

    pub(crate) fn outputs(&self) -> Outlets {
        let session = self.lock();
        session
            .outputs()
            .iter()
            .map(|o| (o.name().to_string(), o.dtype().clone()))
            .collect()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Session> {
        self.session.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Runs `call` on the session. Blocks; see [`infer`].
    pub(crate) fn with_session<T>(
        &self,
        call: impl FnOnce(&mut Session) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        call(&mut self.lock())
    }
}

/// Runs `call` on a blocking thread, so inference never stalls the async workers.
/// A model rejects the same input every time, so a failure is not retried.
pub(crate) async fn infer<T: Send + 'static>(
    model: Arc<OnnxModel>,
    call: impl FnOnce(&OnnxModel) -> anyhow::Result<T> + Send + 'static,
) -> Result<T, PublisherError> {
    tokio::task::spawn_blocking(move || call(&model))
        .await
        .map_err(|e| PublisherError::NonRetryable(anyhow!("ONNX inference stopped: {e}")))?
        .map_err(PublisherError::NonRetryable)
}

/// The process's peak resident memory so far, in bytes.
pub(crate) fn peak_memory_bytes() -> Option<u64> {
    #[cfg(unix)]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        // SAFETY: getrusage only writes into the struct it is given.
        let usage = unsafe {
            if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) != 0 {
                return None;
            }
            usage.assume_init()
        };
        let peak = u64::try_from(usage.ru_maxrss).ok()?;
        // macOS reports bytes, Linux kibibytes.
        Some(if cfg!(target_os = "macos") {
            peak
        } else {
            peak * 1024
        })
    }
    #[cfg(not(unix))]
    None
}

/// Opens a model, runs `warm_up` once at the largest shape it will see, and logs
/// what that cost, so a pod's memory can be sized from the log.
pub(crate) async fn open_and_warm_up<P: Send + 'static>(
    endpoint: &'static str,
    path: String,
    threads: usize,
    device: OnnxDevice,
    shape: String,
    warm_up: impl FnOnce(&OnnxModel) -> anyhow::Result<P> + Send + 'static,
) -> anyhow::Result<(Arc<OnnxModel>, P)> {
    let (model, plan) = tokio::task::spawn_blocking(move || -> anyhow::Result<(OnnxModel, P)> {
        let before = peak_memory_bytes();
        let model = OnnxModel::open(&path, threads, device)?;
        let plan = warm_up(&model).with_context(|| format!("warm-up of ONNX model '{path}' failed"))?;
        let mib = |bytes: u64| bytes / (1024 * 1024);
        match (before, peak_memory_bytes()) {
            (Some(before), Some(after)) => tracing::info!(
                endpoint,
                model = %path,
                threads,
                "ONNX model ready; peak memory {} MiB after a {shape} warm-up ({} MiB before loading)",
                mib(after),
                mib(before),
            ),
            _ => tracing::info!(endpoint, model = %path, threads, "ONNX model ready"),
        }
        Ok((model, plan))
    })
    .await
    .map_err(|e| anyhow!("loading the ONNX model stopped: {e}"))??;
    Ok((Arc::new(model), plan))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_memory_is_known_on_unix() {
        if cfg!(unix) {
            assert!(peak_memory_bytes().unwrap() > 1024 * 1024);
        }
    }
}
