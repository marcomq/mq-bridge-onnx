//  mq-bridge-onnx
//  © Copyright 2026, by Marco Mengelkoch
//  Licensed under MIT OR Apache-2.0, see LICENSE file for more details
//  git clone https://github.com/marcomq/mq-bridge-onnx

//! The tensor a message carries between the `onnx` middleware and task `tensor`:
//! the payload holds the raw little-endian elements of one sample, and metadata
//! holds their type and shape.

use std::fmt::Debug;

use anyhow::{anyhow, bail};
use mq_bridge::CanonicalMessage;
use ort::value::{PrimitiveTensorElementType, TensorElementType};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Metadata key of the element type, e.g. `float32`.
pub const DTYPE_KEY: &str = "tensor_dtype";
/// Metadata key of the shape of one sample, e.g. `3,224,224`.
pub const SHAPE_KEY: &str = "tensor_shape";
/// Metadata key of why the middleware could not decode a message.
pub const ERROR_KEY: &str = "tensor_error";

/// Element type of a tensor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum OnnxDtype {
    #[default]
    Float32,
    Float64,
    Int32,
    Int64,
    Uint8,
}

impl OnnxDtype {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float64 => "float64",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::Uint8 => "uint8",
        }
    }

    fn parse(name: &str) -> Option<Self> {
        [
            Self::Float32,
            Self::Float64,
            Self::Int32,
            Self::Int64,
            Self::Uint8,
        ]
        .into_iter()
        .find(|d| d.name() == name)
    }

    pub(crate) fn size(self) -> usize {
        match self {
            Self::Uint8 => 1,
            Self::Float32 | Self::Int32 => 4,
            Self::Float64 | Self::Int64 => 8,
        }
    }

    pub(crate) fn of(ty: TensorElementType) -> Option<Self> {
        Some(match ty {
            TensorElementType::Float32 => Self::Float32,
            TensorElementType::Float64 => Self::Float64,
            TensorElementType::Int32 => Self::Int32,
            TensorElementType::Int64 => Self::Int64,
            TensorElementType::Uint8 => Self::Uint8,
            _ => return None,
        })
    }
}

/// A number type a tensor can hold; conversions between them are numeric casts.
pub(crate) trait Element: PrimitiveTensorElementType + Copy + Debug + 'static {
    const DTYPE: OnnxDtype;
    fn read(bytes: &[u8]) -> Self;
    fn write(self, out: &mut Vec<u8>);
    fn to_f64(self) -> f64;
    fn from_f64(x: f64) -> Self;
}

macro_rules! element {
    ($($t:ty => $dtype:ident),+) => {$(
        impl Element for $t {
            const DTYPE: OnnxDtype = OnnxDtype::$dtype;
            fn read(bytes: &[u8]) -> Self {
                let mut le = [0u8; size_of::<$t>()];
                le.copy_from_slice(bytes);
                <$t>::from_le_bytes(le)
            }
            fn write(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn from_f64(x: f64) -> Self {
                x as $t
            }
        }
    )+};
}

element!(f32 => Float32, f64 => Float64, i32 => Int32, i64 => Int64, u8 => Uint8);

/// One sample: `shape` elements of `dtype`, little-endian in `bytes`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TensorData {
    pub dtype: OnnxDtype,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl TensorData {
    pub(crate) fn new(dtype: OnnxDtype, shape: Vec<usize>, bytes: Vec<u8>) -> anyhow::Result<Self> {
        let expected = shape.iter().product::<usize>() * dtype.size();
        if bytes.len() != expected {
            bail!(
                "a {} tensor of shape {shape:?} has {expected} bytes, not {}",
                dtype.name(),
                bytes.len()
            );
        }
        Ok(Self {
            dtype,
            shape,
            bytes,
        })
    }

    pub(crate) fn from_values<T: Element>(shape: Vec<usize>, values: &[T]) -> Self {
        let mut bytes = Vec::with_capacity(values.len() * T::DTYPE.size());
        values.iter().for_each(|v| v.write(&mut bytes));
        Self {
            dtype: T::DTYPE,
            shape,
            bytes,
        }
    }

    /// The tensor `message` carries, or why it has none.
    pub(crate) fn of_message(message: &CanonicalMessage) -> anyhow::Result<Self> {
        if let Some(error) = message.get_metadata(ERROR_KEY) {
            bail!("{error}");
        }
        let (Some(dtype), Some(shape)) = (
            message.get_metadata(DTYPE_KEY),
            message.get_metadata(SHAPE_KEY),
        ) else {
            bail!(
                "the message has no `{DTYPE_KEY}` and `{SHAPE_KEY}` metadata; decode it with the onnx middleware"
            );
        };
        let dtype =
            OnnxDtype::parse(dtype).ok_or_else(|| anyhow!("unknown {DTYPE_KEY} '{dtype}'"))?;
        let shape = shape
            .split(',')
            .filter(|d| !d.trim().is_empty())
            .map(|d| d.trim().parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| anyhow!("{SHAPE_KEY} '{shape}' is not a list of sizes"))?;
        Self::new(dtype, shape, message.payload.to_vec())
    }

    /// Writes this tensor into `message`, replacing its payload.
    pub(crate) fn into_message(self, message: &mut CanonicalMessage) {
        let shape: Vec<String> = self.shape.iter().map(usize::to_string).collect();
        message.set_metadata(DTYPE_KEY, self.dtype.name());
        message.set_metadata(SHAPE_KEY, shape.join(","));
        message.remove_metadata(ERROR_KEY);
        message.payload = self.bytes.into();
    }

    /// Appends the elements as `T`, cast when the types differ.
    pub(crate) fn extend_into<T: Element>(&self, out: &mut Vec<T>) {
        fn cast<S: Element, T: Element>(bytes: &[u8], out: &mut Vec<T>) {
            out.extend(
                bytes
                    .chunks_exact(S::DTYPE.size())
                    .map(|b| T::from_f64(S::read(b).to_f64())),
            );
        }
        if self.dtype == T::DTYPE {
            out.extend(self.bytes.chunks_exact(T::DTYPE.size()).map(T::read));
            return;
        }
        match self.dtype {
            OnnxDtype::Float32 => cast::<f32, T>(&self.bytes, out),
            OnnxDtype::Float64 => cast::<f64, T>(&self.bytes, out),
            OnnxDtype::Int32 => cast::<i32, T>(&self.bytes, out),
            OnnxDtype::Int64 => cast::<i64, T>(&self.bytes, out),
            OnnxDtype::Uint8 => cast::<u8, T>(&self.bytes, out),
        }
    }

    /// A rectangular, possibly nested JSON array of numbers.
    pub(crate) fn from_json(text: &[u8], dtype: OnnxDtype) -> anyhow::Result<Self> {
        let value: Value = serde_json::from_slice(text).map_err(|e| anyhow!("not JSON: {e}"))?;
        let mut shape = Vec::new();
        let mut level = &value;
        while let Value::Array(items) = level {
            shape.push(items.len());
            let Some(first) = items.first() else { break };
            level = first;
        }
        let mut numbers = Vec::with_capacity(shape.iter().product());
        flatten(&value, &shape, &mut numbers)?;
        Ok(match dtype {
            OnnxDtype::Float32 => Self::cast_from(shape, &numbers, f32::from_f64),
            OnnxDtype::Float64 => Self::cast_from(shape, &numbers, f64::from_f64),
            OnnxDtype::Int32 => Self::cast_from(shape, &numbers, i32::from_f64),
            OnnxDtype::Int64 => Self::cast_from(shape, &numbers, i64::from_f64),
            OnnxDtype::Uint8 => Self::cast_from(shape, &numbers, u8::from_f64),
        })
    }

    fn cast_from<T: Element>(shape: Vec<usize>, numbers: &[f64], cast: fn(f64) -> T) -> Self {
        let values: Vec<T> = numbers.iter().map(|x| cast(*x)).collect();
        Self::from_values(shape, &values)
    }

    /// A NumPy `.npy` file of a C-ordered little-endian array.
    pub(crate) fn from_npy(file: &[u8]) -> anyhow::Result<Self> {
        let Some(rest) = file.strip_prefix(b"\x93NUMPY") else {
            bail!("not a .npy file");
        };
        let (header_len, rest) = match rest {
            [1, _, a, b, rest @ ..] => (usize::from(u16::from_le_bytes([*a, *b])), rest),
            [2 | 3, _, a, b, c, d, rest @ ..] => {
                (u32::from_le_bytes([*a, *b, *c, *d]) as usize, rest)
            }
            _ => bail!("unsupported .npy version"),
        };
        if rest.len() < header_len {
            bail!(".npy header is cut off");
        }
        let (header, data) = rest.split_at(header_len);
        let header = String::from_utf8_lossy(header);
        let field = |name: &str| {
            let start = header
                .find(&format!("'{name}':"))
                .ok_or_else(|| anyhow!(".npy header has no '{name}'"))?;
            Ok::<_, anyhow::Error>(header[start + name.len() + 3..].trim_start())
        };
        let descr = field("descr")?;
        let dtype = match descr.split('\'').nth(1) {
            Some("<f4") => OnnxDtype::Float32,
            Some("<f8") => OnnxDtype::Float64,
            Some("<i4") => OnnxDtype::Int32,
            Some("<i8") => OnnxDtype::Int64,
            Some("|u1" | "<u1") => OnnxDtype::Uint8,
            other => bail!(".npy dtype {other:?} is not supported"),
        };
        if field("fortran_order")?.starts_with("True") {
            bail!(".npy arrays in Fortran order are not supported");
        }
        let shape = field("shape")?;
        let shape = shape
            .strip_prefix('(')
            .and_then(|s| s.split(')').next())
            .ok_or_else(|| anyhow!(".npy shape is not a tuple"))?
            .split(',')
            .filter(|d| !d.trim().is_empty())
            .map(|d| d.trim().parse::<usize>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| anyhow!(".npy shape is not a list of sizes"))?;
        Self::new(dtype, shape, data.to_vec())
    }
}

fn flatten(value: &Value, shape: &[usize], out: &mut Vec<f64>) -> anyhow::Result<()> {
    match (value, shape) {
        (Value::Array(items), [len, inner @ ..]) if items.len() == *len => {
            items.iter().try_for_each(|item| flatten(item, inner, out))
        }
        (Value::Array(_), _) => bail!("the JSON array is not rectangular"),
        (Value::Number(n), []) => {
            out.push(n.as_f64().ok_or_else(|| anyhow!("{n} is not a number"))?);
            Ok(())
        }
        (Value::Bool(b), []) => {
            out.push(f64::from(u8::from(*b)));
            Ok(())
        }
        (other, _) => bail!("{other} is not a number or a rectangular array"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_arrays_keep_their_shape() {
        let tensor = TensorData::from_json(b"[[1, 2, 3], [4, 5, 6]]", OnnxDtype::Int64).unwrap();
        assert_eq!(tensor.shape, [2, 3]);
        let mut values = Vec::<i64>::new();
        tensor.extend_into(&mut values);
        assert_eq!(values, [1, 2, 3, 4, 5, 6]);
        assert!(TensorData::from_json(b"[[1, 2], [3]]", OnnxDtype::Float32).is_err());
        assert_eq!(
            TensorData::from_json(b"7", OnnxDtype::Float32)
                .unwrap()
                .shape,
            Vec::<usize>::new()
        );
    }

    #[test]
    fn a_tensor_round_trips_through_a_message_and_casts() {
        let mut message = CanonicalMessage::from("x");
        TensorData::from_values(vec![3], &[1.5f64, -2.0, 7.0]).into_message(&mut message);
        assert_eq!(message.get_metadata(SHAPE_KEY), Some("3"));
        let tensor = TensorData::of_message(&message).unwrap();
        let mut values = Vec::<f32>::new();
        tensor.extend_into(&mut values);
        assert_eq!(values, [1.5, -2.0, 7.0]);

        message.set_metadata(SHAPE_KEY, "4");
        assert!(TensorData::of_message(&message).is_err());
    }

    #[test]
    fn npy_files_are_read() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 1), }";
        let mut file = b"\x93NUMPY\x01\x00".to_vec();
        file.extend_from_slice(&(header.len() as u16).to_le_bytes());
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(&1.0f32.to_le_bytes());
        file.extend_from_slice(&2.0f32.to_le_bytes());
        let tensor = TensorData::from_npy(&file).unwrap();
        assert_eq!(
            (tensor.dtype, tensor.shape.as_slice()),
            (OnnxDtype::Float32, &[2, 1][..])
        );
        assert!(TensorData::from_npy(b"not numpy").is_err());
    }
}
