# mq-bridge-onnx

Local ONNX inference for [mq-bridge](https://github.com/marcomq/mq-bridge), as a
native plugin. It adds one output-only endpoint, `onnx`, that answers every
message, so it works as a route output and inside `lookup`. Its `task` picks the
answer:

| `task` | Input | Answer |
| --- | --- | --- |
| `embeddings` | a text template, plus optional numbers | a JSON array of floats, like `openai_embeddings` |
| `score` | numeric feature templates | a JSON object of the model's outputs |

`score` runs CatBoost, XGBoost and LightGBM exports and other ONNX models with one
numeric input. ONNX Runtime is linked into the library, so nothing else has to
be installed. Nothing is trained here: bring your own model files.

## Install

Installed into a directory on mq-bridge's plugin search path (Homebrew does
this), the plugin is found when a route or URI names `onnx`, with no
configuration. Otherwise list it:

```yaml
plugins:
  - "/opt/mq-bridge/plugins/libmq_bridge_onnx.so"
```

From Python or Node.js, install `mq-bridge-onnx` and call `register()` once:

```python
import mq_bridge_onnx
mq_bridge_onnx.register()
```

```js
import { register } from "mq-bridge-onnx";
register();
```

Prebuilt for Linux x86_64 and arm64 (glibc), macOS arm64 and Windows x64. There
is no build for musl or Intel macOS, since ONNX Runtime has no prebuilt static
library for them.

## Embeddings

```yaml
routes:
  embed:
    input:
      file: { path: "articles.jsonl" }
    output:
      custom:
        name: onnx
        config:
          task: embeddings
          model: "/models/all-MiniLM-L6-v2/model.onnx"
          input: "${payload:title} ${payload:body}"
          numbers: ["${payload:age_days}"]
```

| Field | Default | Meaning |
| --- | --- | --- |
| `task` | required | `embeddings` |
| `model` | required | path of the `.onnx` file |
| `tokenizer` | `tokenizer.json` next to the model | Hugging Face tokenizer file |
| `input` | required | template of the text to embed |
| `numbers` | `[]` | templates of numbers appended to the vector as given; missing = `null` |
| `max_len` | `128` | tokens per text; longer texts are cut |
| `max_batch` | `32` | texts per inference call |
| `threads` | `2` | CPU threads of one inference call |
| `pooling` | `mean` | `mean` (sentence-transformers) or `cls` (BGE), for models that return one vector per token |
| `normalize` | `true` | scale the text vector to length 1, before `numbers` are appended |
| `device` | `cpu` | `auto` lets ONNX Runtime choose, e.g. CoreML on macOS, whose memory is not bounded |

`numbers` are not scaled: scale them with `transform` first if their range
differs from the vector's.

## Scoring

```yaml
middlewares:
  - lookup:
      from:
        custom:
          name: onnx
          config:
            task: score
            model: "/models/churn.onnx"
            features: ["${payload:orders_30d}", "${payload:days_since_login}", "${payload:tickets}"]
      into: churn
```

The answer for a CatBoost classifier is `{"label": 1, "probabilities": {"0": 0.017, "1": 0.983}}`.

| Field | Default | Meaning |
| --- | --- | --- |
| `task` | required | `score` |
| `model` | required | path of the `.onnx` file |
| `features` | required | one template per model feature, in model order |
| `outputs` | all | model outputs in the answer |
| `input_name` | the only input | model input the features go to |
| `max_batch` | `256` | rows per inference call |
| `threads` | `2` | CPU threads of one inference call |
| `device` | `cpu` | as above |

- A missing feature, `""` or `null` is NaN, which tree models treat as missing.
  `true` / `false` are 1 / 0. Anything else that is not a number fails that
  message alone.
- The feature count must match the model, or the endpoint does not start; so
  does a field the task does not read, such as `input` with `score`.
- An output of shape `[rows]` or `[rows, 1]` becomes a number per message, a
  wider one an array. A classifier's probability map (ONNX `ZipMap`) becomes an
  object keyed by class label, in label order.
- CatBoost exports only numeric features to ONNX; encode categorical ones as
  numbers before training.

## URIs

```sh
mqb copy articles.jsonl "onnx+embeddings:///models/minilm/model.onnx?input=\${payload:body}&max_len=256"
```

The task follows the `+`, and the path is the model file: `onnx+score:///models/a.onnx`
is absolute, `onnx+score:models/a.onnx` relative. Every other field is a query
parameter; a list is comma separated.

## Memory

Peak resident memory is set mostly by `max_batch` × `max_len`, and grows with
the square of `max_len` (attention), not by the model's size. The endpoint runs
one batch of the largest shape at startup and logs the peak, so a pod can be
sized from that log line. Measured with all-MiniLM-L6-v2 (fp32, 90 MB) on the
CPU, 2 threads, Apple M1:

| `max_batch` × `max_len` | Peak |
| --- | --- |
| model loaded, no batch | 250 MB |
| 8 × 128 | 298 MB |
| 32 × 128 | 344 MB |
| 32 × 256 | 886 MB |

The int8 model loads in 89 MB, but its vectors differ from the fp32 ones
(cosine 0.97–0.99), so do not mix the two in one index.

## Development

```sh
cargo test
```

`tests/models.rs` also checks MiniLM against reference vectors when
`MQB_ONNX_TEST_MODEL_DIR` names a folder with all-MiniLM-L6-v2's `model.onnx`
and `tokenizer.json`; it skips that test otherwise. `tests/fixtures/tiny_classifier.onnx`
is a 20 KB CatBoost model made by `tests/fixtures/make_tiny_classifier.py`.

The build downloads a prebuilt ONNX Runtime from pyke's CDN (checked by sha256).
Set `ORT_LIB_LOCATION` to a local ONNX Runtime build to build offline.

To build against an unreleased mq-bridge checkout, add `.cargo/config.toml`:

```toml
[patch.crates-io]
mq-bridge = { path = "../mq-bridge" }
```

To release, run `python3 scripts/set_version.py X.Y.Z`, commit, and push the tag
`X.Y.Z`. The publishing prerequisites (trusted publishers, tap deploy key,
Anaconda token) are listed in mq-bridge's `examples/plugin-template/README.md`.
