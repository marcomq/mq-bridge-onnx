//  mq-bridge-onnx
//  © Copyright 2026, by Marco Mengelkoch
//  Licensed under MIT OR Apache-2.0, see LICENSE file for more details
//  git clone https://github.com/marcomq/mq-bridge-onnx

// Runs real models. The embedding test also needs MQB_ONNX_TEST_MODEL_DIR: a folder
// with all-MiniLM-L6-v2's model.onnx and tokenizer.json; it skips itself without one.

use mq_bridge::traits::{CustomEndpointFactory, MessagePublisher};
use mq_bridge::{CanonicalMessage, SentBatch};
use mq_bridge_onnx::OnnxFactory;
use serde_json::{Value, json};

const CLASSIFIER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/tiny_classifier.onnx"
);

fn responses(sent: SentBatch) -> (Vec<Value>, usize) {
    let SentBatch::Partial { responses, failed } = sent else {
        panic!("expected answers");
    };
    let values = responses
        .unwrap_or_default()
        .iter()
        .map(|m| serde_json::from_slice(&m.payload).unwrap())
        .collect();
    (values, failed.len())
}

async fn classifier() -> Box<dyn MessagePublisher> {
    let features: Vec<String> = ["a", "b", "c"]
        .map(|f| format!("${{payload:{f}}}"))
        .to_vec();
    OnnxFactory
        .create_publisher(
            "onnx",
            &json!({ "task": "score", "model": CLASSIFIER, "features": features }),
        )
        .await
        .unwrap()
}

#[tokio::test]
async fn a_catboost_classifier_scores_each_message() {
    let messages = [
        r#"{"a": 8, "b": 1, "c": 2}"#,
        r#"{"a": 1, "b": 1, "c": 9}"#,
        r#"{"a": 1, "c": 9}"#,
        r#"{"a": "high", "b": 1, "c": 9}"#,
    ]
    .map(CanonicalMessage::from)
    .to_vec();
    let (scores, failed) = responses(classifier().await.send_batch(messages).await.unwrap());

    assert_eq!(failed, 1, "the non-numeric feature fails its message alone");
    assert_eq!(scores.len(), 3, "a missing feature is scored as missing");
    assert_eq!(scores[0]["label"], 1);
    assert_eq!(scores[1]["label"], 0);
    // CatBoost's own predict_proba for the first row.
    let probability = scores[0]["probabilities"]["1"].as_f64().unwrap();
    assert!((probability - 0.983104).abs() < 1e-4, "{}", scores[0]);
}

#[tokio::test]
async fn a_feature_count_the_model_does_not_take_is_refused() {
    let error = OnnxFactory
        .create_publisher(
            "onnx",
            &json!({ "task": "score", "model": CLASSIFIER, "features": ["${payload:a}"] }),
        )
        .await
        .err()
        .unwrap();
    assert!(
        format!("{error:#}").contains("takes 3 features"),
        "{error:#}"
    );
}

#[tokio::test]
async fn minilm_vectors_match_sentence_transformers_pooling() {
    let Ok(dir) = std::env::var("MQB_ONNX_TEST_MODEL_DIR") else {
        eprintln!("skipped: MQB_ONNX_TEST_MODEL_DIR is not set");
        return;
    };
    let config = json!({
        "task": "embeddings",
        "model": format!("{dir}/model.onnx"),
        "input": "${payload:text}",
        "numbers": ["${payload:age}"],
        "max_batch": 2,
    });
    let publisher = OnnxFactory
        .create_publisher("embed", &config)
        .await
        .unwrap();

    // First dimensions of the masked-mean, normalised vectors, from Python onnxruntime.
    let expected: [(&str, [f64; 4]); 3] = [
        (
            "The cat sits on the mat.",
            [0.134177, -0.0329, -0.024804, 0.041439],
        ),
        (
            "A kitten rests on a rug.",
            [0.074453, 0.014573, 0.019018, 0.066072],
        ),
        (
            "Quarterly revenue grew by four percent.",
            [0.119614, 0.009014, -0.006429, -0.042733],
        ),
    ];
    let messages = expected
        .iter()
        .map(|(text, _)| CanonicalMessage::from(json!({ "text": text, "age": 7 }).to_string()))
        .collect();
    let (vectors, failed) = responses(publisher.send_batch(messages).await.unwrap());

    assert_eq!(failed, 0);
    for ((text, head), vector) in expected.iter().zip(&vectors) {
        let vector: Vec<f64> = vector
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        assert_eq!(vector.len(), 385, "384 dimensions and one number");
        assert_eq!(vector[384], 7.0);
        for (got, want) in vector.iter().zip(head) {
            assert!((got - want).abs() < 1e-4, "{text}: {got} vs {want}");
        }
    }
}
