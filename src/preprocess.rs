//  mq-bridge-onnx
//  © Copyright 2026, by Marco Mengelkoch
//  Licensed under MIT OR Apache-2.0, see LICENSE file for more details
//  git clone https://github.com/marcomq/mq-bridge-onnx

//! Image to `[3, height, width]` float32 tensor, as a Hugging Face image processor
//! prepares it: resize, centre crop, rescale, normalise. Resampling is the `image`
//! crate's, so pixels can differ slightly from Pillow's.

use anyhow::{Context, anyhow, bail};
use image::imageops::{self, FilterType};
use serde::Serialize;
use serde_json::Value;

use crate::decode::OnnxExtent;
use crate::tensor_data::TensorData;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct Preprocessing {
    pub resize: Option<OnnxExtent>,
    #[serde(skip)]
    pub filter: FilterType,
    pub crop: Option<OnnxExtent>,
    pub rescale: f32,
    pub mean: Option<[f32; 3]>,
    pub std: Option<[f32; 3]>,
}

impl Default for Preprocessing {
    fn default() -> Self {
        Self {
            resize: None,
            filter: FilterType::Triangle,
            crop: None,
            rescale: 1.0 / 255.0,
            mean: None,
            std: None,
        }
    }
}

impl Preprocessing {
    /// Reads the settings of a Hugging Face `preprocessor_config.json`.
    pub(crate) fn from_file(path: &str) -> anyhow::Result<Self> {
        let text = std::fs::read(path).with_context(|| format!("cannot read '{path}'"))?;
        let config: Value =
            serde_json::from_slice(&text).with_context(|| format!("'{path}' is not JSON"))?;
        Self::from_hugging_face(&config).with_context(|| format!("'{path}'"))
    }

    fn from_hugging_face(config: &Value) -> anyhow::Result<Self> {
        let flag =
            |key: &str, default: bool| config.get(key).and_then(Value::as_bool).unwrap_or(default);
        let triple = |key: &str| -> anyhow::Result<Option<[f32; 3]>> {
            match config.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(value) => serde_json::from_value(value.clone())
                    .map(Some)
                    .map_err(|_| anyhow!("`{key}` is not three numbers")),
            }
        };
        let mut settings = Self::default();
        if flag("do_resize", true) {
            settings.resize = extent(config.get("size"), "shortest_edge")?;
        }
        if flag("do_center_crop", false) {
            settings.crop = extent(config.get("crop_size"), "")?;
        }
        settings.filter = match config.get("resample").and_then(Value::as_u64) {
            Some(0) => FilterType::Nearest,
            Some(1) => FilterType::Lanczos3,
            Some(3) => FilterType::CatmullRom,
            _ => FilterType::Triangle,
        };
        settings.rescale = match flag("do_rescale", true) {
            true => config
                .get("rescale_factor")
                .and_then(Value::as_f64)
                .map_or(settings.rescale, |f| f as f32),
            false => 1.0,
        };
        if flag("do_normalize", true) {
            settings.mean = triple("image_mean")?;
            settings.std = triple("image_std")?;
        }
        Ok(settings)
    }

    /// The image in `bytes` as a tensor, with its original width and height.
    pub(crate) fn run(&self, bytes: &[u8]) -> anyhow::Result<(TensorData, u32, u32)> {
        let image = image::load_from_memory(bytes)
            .map_err(|e| anyhow!("not a PNG or JPEG image: {e}"))?
            .to_rgb8();
        let (width, height) = image.dimensions();
        let image = match self.resize {
            None => image,
            Some(OnnxExtent::Exact([h, w])) => imageops::resize(&image, w, h, self.filter),
            Some(OnnxExtent::Edge(edge)) => {
                let scale = |side: u32| {
                    (u64::from(edge) * u64::from(side) / u64::from(width.min(height))) as u32
                };
                imageops::resize(&image, scale(width), scale(height), self.filter)
            }
        };
        let image = match self.crop {
            None => image,
            Some(crop) => {
                let [h, w] = match crop {
                    OnnxExtent::Edge(side) => [side, side],
                    OnnxExtent::Exact(size) => size,
                };
                let (iw, ih) = image.dimensions();
                if w > iw || h > ih {
                    bail!("the image is {iw}×{ih} after resizing, smaller than the {w}×{h} crop");
                }
                imageops::crop_imm(&image, (iw - w) / 2, (ih - h) / 2, w, h).to_image()
            }
        };
        let (w, h) = image.dimensions();
        let plane = (w * h) as usize;
        let mean = self.mean.unwrap_or([0.0; 3]);
        let std = self.std.unwrap_or([1.0; 3]);
        let mut values = vec![0f32; 3 * plane];
        for (i, pixel) in image.pixels().enumerate() {
            for c in 0..3 {
                values[c * plane + i] = (f32::from(pixel[c]) * self.rescale - mean[c]) / std[c];
            }
        }
        let tensor = TensorData::from_values(vec![3, h as usize, w as usize], &values);
        Ok((tensor, width, height))
    }
}

/// A Hugging Face size: a number, `{"height", "width"}`, or `{edge_key: n}`.
fn extent(value: Option<&Value>, edge_key: &str) -> anyhow::Result<Option<OnnxExtent>> {
    let number = |v: &Value| v.as_u64().and_then(|n| u32::try_from(n).ok());
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    if let Some(n) = number(value) {
        return Ok(Some(OnnxExtent::Edge(n)));
    }
    if let (Some(h), Some(w)) = (
        value.get("height").and_then(number),
        value.get("width").and_then(number),
    ) {
        return Ok(Some(OnnxExtent::Exact([h, w])));
    }
    match value.get(edge_key).and_then(number) {
        Some(n) if !edge_key.is_empty() => Ok(Some(OnnxExtent::Edge(n))),
        _ => bail!("size {value} is not supported; give `size` and `crop` instead"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn png(width: u32, height: u32) -> Vec<u8> {
        let image = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([x as u8 * 10, y as u8 * 20, 30])
        });
        let mut bytes = std::io::Cursor::new(Vec::new());
        image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
        bytes.into_inner()
    }

    fn values(tensor: &TensorData) -> Vec<f32> {
        let mut values = Vec::new();
        tensor.extend_into(&mut values);
        values
    }

    #[test]
    fn a_centre_crop_is_rescaled_and_normalised_per_channel() {
        let settings = Preprocessing {
            crop: Some(OnnxExtent::Edge(2)),
            rescale: 1.0,
            mean: Some([10.0, 20.0, 30.0]),
            std: Some([10.0, 20.0, 1.0]),
            ..Preprocessing::default()
        };
        let (tensor, width, height) = settings.run(&png(4, 6)).unwrap();
        assert_eq!((width, height), (4, 6));
        assert_eq!(tensor.shape, [3, 2, 2]);
        // Crop starts at x 1, y 2: red 10·x, green 20·y, blue 30.
        assert_eq!(
            values(&tensor),
            [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn the_shortest_edge_is_resized_keeping_the_aspect() {
        let settings = Preprocessing {
            resize: Some(OnnxExtent::Edge(3)),
            ..Preprocessing::default()
        };
        let (tensor, _, _) = settings.run(&png(4, 8)).unwrap();
        assert_eq!(tensor.shape, [3, 6, 3]);
        assert!(settings.run(b"not an image").is_err());
    }

    #[test]
    fn hugging_face_settings_are_read() {
        let clip = json!({
            "do_resize": true, "size": {"shortest_edge": 224}, "resample": 3,
            "do_center_crop": true, "crop_size": {"height": 224, "width": 224},
            "do_rescale": true, "rescale_factor": 0.00392156862745098,
            "do_normalize": true, "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711]
        });
        let settings = Preprocessing::from_hugging_face(&clip).unwrap();
        assert_eq!(settings.resize, Some(OnnxExtent::Edge(224)));
        assert_eq!(settings.crop, Some(OnnxExtent::Exact([224, 224])));
        assert_eq!(settings.filter, FilterType::CatmullRom);
        assert_eq!(settings.mean.unwrap()[0], 0.48145466);
        let vit = json!({ "size": {"height": 384, "width": 384}, "do_normalize": false });
        let settings = Preprocessing::from_hugging_face(&vit).unwrap();
        assert_eq!(
            (settings.resize, settings.mean),
            (Some(OnnxExtent::Exact([384, 384])), None)
        );
    }
}
