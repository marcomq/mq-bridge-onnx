# Regenerates the two tiny models of the `tensor` task tests:
#   tiny_image.onnx   pixel_values [N,3,4,4] -> image_embeds [N,3] (channel means),
#                     last_hidden_state [N,16,3] (one token per pixel)
#   tiny_tensor.onnx  x [N,2] -> y = x @ [[1,2,3],[4,5,6]] [N,3], label = argmax(y) [N]
# uvx --with onnx python make_tiny_tensor_models.py
import sys
from pathlib import Path
from onnx import TensorProto, helper, save_model

here = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)

image = helper.make_graph(
    [
        helper.make_node("ReduceMean", ["pixel_values"], ["image_embeds"], axes=[2, 3], keepdims=0),
        helper.make_node("Reshape", ["pixel_values", "tokens_shape"], ["channels_by_pixel"]),
        helper.make_node("Transpose", ["channels_by_pixel"], ["last_hidden_state"], perm=[0, 2, 1]),
    ],
    "tiny_image",
    [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["N", 3, 4, 4])],
    [
        helper.make_tensor_value_info("image_embeds", TensorProto.FLOAT, ["N", 3]),
        helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["N", 16, 3]),
    ],
    [helper.make_tensor("tokens_shape", TensorProto.INT64, [3], [0, 3, 16])],
)
tensor = helper.make_graph(
    [
        helper.make_node("MatMul", ["x", "w"], ["y"]),
        helper.make_node("ArgMax", ["y"], ["label"], axis=1, keepdims=0),
    ],
    "tiny_tensor",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2])],
    [
        helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 3]),
        helper.make_tensor_value_info("label", TensorProto.INT64, ["N"]),
    ],
    [helper.make_tensor("w", TensorProto.FLOAT, [2, 3], [1, 2, 3, 4, 5, 6])],
)
for graph in (image, tensor):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    save_model(model, here / f"{graph.name}.onnx")
