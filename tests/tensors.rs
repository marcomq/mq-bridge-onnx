//  mq-bridge-onnx
//  © Copyright 2026, by Marco Mengelkoch
//  Licensed under MIT OR Apache-2.0, see LICENSE file for more details
//  git clone https://github.com/marcomq/mq-bridge-onnx

// The `onnx` middleware feeding task `tensor`, on the generated fixtures from
// make_tiny_tensor_models.py.

use mq_bridge::traits::{CustomEndpointFactory, CustomMiddlewareFactory, MessagePublisher};
use mq_bridge::{CanonicalMessage, SentBatch};
use mq_bridge_onnx::{ERROR_KEY, OnnxFactory, OnnxMiddlewareFactory};
use serde_json::{Value, json};

const TENSOR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/tiny_tensor.onnx"
);
const IMAGE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/tiny_image.onnx"
);
const STATEFUL: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/tiny_stateful.onnx"
);

async fn decoding(decode: Value, task: Value) -> Box<dyn MessagePublisher> {
    let publisher = OnnxFactory.create_publisher("onnx", &task).await.unwrap();
    OnnxMiddlewareFactory
        .apply_publisher(publisher, "onnx", &decode)
        .await
        .unwrap()
}

/// The answers, with `None` where a message failed.
async fn answers(
    publisher: &dyn MessagePublisher,
    messages: Vec<CanonicalMessage>,
) -> Vec<Option<Value>> {
    let lookups = publisher.lookup_each(&messages).await.unwrap();
    lookups
        .into_iter()
        .map(|answer| answer.ok().map(Option::unwrap))
        .collect()
}

/// A 4×4 PNG with red 10·x, green 20·y and blue 30.
fn png() -> Vec<u8> {
    let image = image::RgbImage::from_fn(4, 4, |x, y| image::Rgb([x as u8 * 10, y as u8 * 20, 30]));
    let mut bytes = std::io::Cursor::new(Vec::new());
    image.write_to(&mut bytes, image::ImageFormat::Png).unwrap();
    bytes.into_inner()
}

fn npy_f64(values: &[f64]) -> Vec<u8> {
    let mut header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({},), }}",
        values.len()
    );
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend((header.len() as u16).to_le_bytes());
    bytes.extend(header.as_bytes());
    values.iter().for_each(|v| bytes.extend(v.to_le_bytes()));
    bytes
}

fn close(actual: &Value, expected: &[f64]) {
    let actual: Vec<f64> = serde_json::from_value(actual.clone()).unwrap();
    assert_eq!(actual.len(), expected.len(), "{actual:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-5, "{actual:?} is not {expected:?}");
    }
}

#[tokio::test]
async fn json_tensors_get_every_output_and_bad_ones_fail_alone() {
    let publisher = decoding(
        json!({ "decode": "json", "from": "${payload:x}" }),
        json!({ "task": "tensor", "model": TENSOR }),
    )
    .await;
    let messages = [
        json!({ "x": [1, 2] }),
        json!({ "x": [1, 2, 3] }),
        json!({ "x": "not numbers" }),
        json!({ "x": [0.5, -1] }),
    ]
    .map(|m| CanonicalMessage::from(m.to_string()))
    .to_vec();
    let answers = answers(publisher.as_ref(), messages).await;

    assert!(answers[1].is_none(), "a sample of the wrong shape fails");
    assert!(
        answers[2].is_none(),
        "a message the middleware cannot decode fails"
    );
    let first = answers[0].as_ref().unwrap();
    close(&first["y"], &[9.0, 12.0, 15.0]);
    assert_eq!(first["label"], 2);
    let last = answers[3].as_ref().unwrap();
    close(&last["y"], &[-3.5, -4.0, -4.5]);
    assert_eq!(last["label"], 0);
}

#[tokio::test]
async fn an_image_embedding_is_one_vector() {
    let channel_means = [15.0 / 255.0, 30.0 / 255.0, 30.0 / 255.0];
    for (output, pooling) in [("image_embeds", "cls"), ("last_hidden_state", "mean")] {
        let publisher = decoding(
            json!({ "decode": "image" }),
            json!({ "task": "tensor", "model": IMAGE, "vector": output, "pooling": pooling, "normalize": false }),
        )
        .await;
        let answers = answers(publisher.as_ref(), vec![CanonicalMessage::from(png())]).await;
        close(answers[0].as_ref().unwrap(), &channel_means);
    }
}

#[tokio::test]
async fn a_base64_image_is_normalised_and_bad_base64_fails_alone() {
    let publisher = decoding(
        json!({ "decode": "image", "size": [4, 4], "base64": true }),
        json!({ "task": "tensor", "model": IMAGE, "vector": "image_embeds" }),
    )
    .await;
    use base64::Engine as _;
    let encoded = base64::engine::general_purpose::STANDARD.encode(png());
    let messages = vec![
        CanonicalMessage::from(encoded),
        CanonicalMessage::from("not base64!"),
    ];
    let SentBatch::Partial { responses, failed } = publisher.send_batch(messages).await.unwrap()
    else {
        panic!("expected answers");
    };
    assert_eq!(failed.len(), 1);
    let vector: Vec<f64> = serde_json::from_slice(&responses.unwrap()[0].payload).unwrap();
    let length: f64 = vector.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        (length - 1.0).abs() < 1e-5,
        "normalised by default: {vector:?}"
    );
}

#[tokio::test]
async fn npy_doubles_are_cast_to_the_float_input() {
    let publisher = decoding(
        json!({ "decode": "npy" }),
        json!({ "task": "tensor", "model": TENSOR, "outputs": ["y"] }),
    )
    .await;
    let answers = answers(
        publisher.as_ref(),
        vec![CanonicalMessage::from(npy_f64(&[2.0, 0.0]))],
    )
    .await;
    let answer = answers[0].as_ref().unwrap();
    close(&answer["y"], &[2.0, 4.0, 6.0]);
    assert!(answer.get("label").is_none());
}

fn stateful_task(extra: Value) -> Value {
    let mut task = json!({
        "task": "tensor",
        "model": STATEFUL,
        "max_batch": 1,
        "inputs": { "input": {}, "state": { "batch": false }, "sr": { "value": 10 } },
    });
    task.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    task
}

#[tokio::test]
async fn the_caller_carries_the_state_from_call_to_call() {
    let publisher = decoding(json!({ "decode": "json" }), stateful_task(json!({}))).await;
    let mut state = json!([[[0, 0]], [[0, 0]]]);
    for (chunk, sum, carried) in [([1, 2], 30.0, [1.0, 2.0]), ([3, 4], 70.0, [4.0, 6.0])] {
        let message = json!({ "input": chunk, "state": state });
        let answers = answers(
            publisher.as_ref(),
            vec![CanonicalMessage::from(message.to_string())],
        )
        .await;
        let answer = answers[0].as_ref().expect("answered");
        assert_eq!(answer["output"], json!([[sum]]), "output is answered whole");
        let next: Vec<Vec<Vec<f64>>> = serde_json::from_value(answer["stateN"].clone()).unwrap();
        assert_eq!(next.len(), 2, "stateN is answered whole: {next:?}");
        for layer in &next {
            close(&json!(layer[0]), &carried);
        }
        state = answer["stateN"].clone();
    }
}

#[tokio::test]
async fn a_missing_tensor_fails_its_message_alone() {
    let publisher = decoding(json!({ "decode": "json" }), stateful_task(json!({}))).await;
    let lookups = publisher
        .lookup_each(&[CanonicalMessage::from(
            json!({ "input": [1, 2] }).to_string(),
        )])
        .await
        .unwrap();
    let error = lookups[0].as_ref().unwrap_err().to_string();
    assert!(error.contains("'state'"), "{error}");
}

#[tokio::test]
async fn a_whole_tensor_needs_one_message_per_call() {
    let error = OnnxFactory
        .create_publisher("onnx", &stateful_task(json!({ "max_batch": 4 })))
        .await
        .err()
        .unwrap();
    assert!(format!("{error:#}").contains("`max_batch: 1`"), "{error:#}");

    let error = OnnxFactory
        .create_publisher(
            "onnx",
            &json!({ "task": "tensor", "model": STATEFUL, "max_batch": 1 }),
        )
        .await
        .err()
        .unwrap();
    assert!(format!("{error:#}").contains("3 inputs"), "{error:#}");

    let error = OnnxFactory
        .create_publisher("onnx", &stateful_task(json!({ "vector": "output" })))
        .await
        .err()
        .unwrap();
    assert!(
        format!("{error:#}").contains("`vector` needs batched inputs"),
        "{error:#}"
    );
}

#[tokio::test]
async fn inputs_name_every_model_input() {
    let mut task = stateful_task(json!({}));
    task["inputs"].as_object_mut().unwrap().remove("sr");
    let error = OnnxFactory
        .create_publisher("onnx", &task)
        .await
        .err()
        .unwrap();
    assert!(
        format!("{error:#}").contains("'sr' is missing"),
        "{error:#}"
    );
}

#[tokio::test]
async fn a_constant_with_a_dynamic_batch_needs_one_message_per_call() {
    let task = json!({
        "task": "tensor",
        "model": STATEFUL,
        "max_batch": 4,
        "inputs": { "input": { "value": [[1, 2]] }, "state": {}, "sr": { "value": 10 } },
    });
    let error = OnnxFactory
        .create_publisher("onnx", &task)
        .await
        .err()
        .unwrap();
    assert!(
        format!("{error:#}").contains("dynamic first dimension"),
        "{error:#}"
    );
}

#[tokio::test]
async fn one_named_tensor_feeds_the_only_input() {
    let publisher = decoding(
        json!({ "decode": "json" }),
        json!({ "task": "tensor", "model": TENSOR, "outputs": ["y"] }),
    )
    .await;
    let message = CanonicalMessage::from(json!({ "anything": [1, 2] }).to_string());
    let answers = answers(publisher.as_ref(), vec![message]).await;
    close(&answers[0].as_ref().unwrap()["y"], &[9.0, 12.0, 15.0]);
}

#[cfg(feature = "audio")]
#[tokio::test]
async fn a_wav_file_becomes_the_sample_tensor() {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut file = std::io::Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut file, spec).unwrap();
    [16384i16, 16384, -16384, -16384]
        .iter()
        .for_each(|s| writer.write_sample(*s).unwrap());
    writer.finalize().unwrap();

    let publisher = decoding(
        json!({ "decode": "wav" }),
        json!({ "task": "tensor", "model": TENSOR, "outputs": ["y"] }),
    )
    .await;
    let answers = answers(
        publisher.as_ref(),
        vec![CanonicalMessage::from(file.into_inner())],
    )
    .await;
    close(&answers[0].as_ref().unwrap()["y"], &[-1.5, -1.5, -1.5]);
}

#[tokio::test]
async fn a_message_marked_by_the_middleware_is_not_run() {
    let publisher = OnnxFactory
        .create_publisher("onnx", &json!({ "task": "tensor", "model": TENSOR }))
        .await
        .unwrap();
    let mut message = CanonicalMessage::from("[1, 2]");
    message.set_metadata(ERROR_KEY, "it went wrong upstream");
    let lookups = publisher.lookup_each(&[message]).await.unwrap();
    let error = lookups[0].as_ref().unwrap_err().to_string();
    assert!(error.contains("it went wrong upstream"), "{error}");
}
