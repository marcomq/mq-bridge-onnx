# mq-bridge-onnx

Local ONNX inference endpoints for [mq-bridge](https://github.com/marcomq/mq-bridge), shipped as a native plugin.

```js
import { register } from "mq-bridge-onnx";

register();
```

`register()` adds the `onnx` endpoint; routes then use e.g.
`{ custom: { name: "onnx", config: { task: "embeddings", model: "/models/minilm/model.onnx", input: "${payload:text}" } } }`.
See the [repository README](https://github.com/marcomq/mq-bridge-onnx) for every field.
