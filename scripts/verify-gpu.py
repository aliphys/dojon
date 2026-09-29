"""Small GPU checks; run with NVIDIA Container Runtime on the Jetson."""

import os
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import torch
import tensorrt as trt
import numpy as np
import onnx
import onnxruntime as ort

assert torch.cuda.is_available(), "GPU unavailable; check the NVIDIA container runtime"
print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
x = torch.ones((64, 64), device="cuda")
assert (x @ x)[0, 0].item() == 64
conv = torch.nn.Conv2d(3, 8, 3).cuda()
assert conv(torch.ones(1, 3, 32, 32, device="cuda")).shape == (1, 8, 30, 30)
with torch.autocast("cuda", dtype=torch.float16):
    assert conv(torch.ones(1, 3, 32, 32, device="cuda")).dtype == torch.float16
q = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
assert torch.nn.functional.scaled_dot_product_attention(q, q, q).shape == q.shape
compiled = torch.compile(lambda t: t.sin() + t.cos(), fullgraph=True)
sample = torch.randn(128, device="cuda")
torch.testing.assert_close(compiled(sample), sample.sin() + sample.cos())
torch.cuda.synchronize()
assert torch.backends.cudnn.version() == 92000
print("PASS PyTorch matmul, convolution, FP16, attention and torch.compile", flush=True)

# Exercise the complete PyTorch -> ONNX -> ONNX Runtime / TensorRT route.
class OnnxModel(torch.nn.Module):
    def forward(self, value):
        return torch.relu(value * 2.0 + 1.0)


onnx_path = "/tmp/smoke.onnx"
onnx_input = torch.tensor([[-1.0, 0.0, 2.0, 4.0]], dtype=torch.float32)
expected = OnnxModel()(onnx_input).numpy()
torch.onnx.export(
    OnnxModel(), onnx_input, onnx_path, input_names=["input"],
    output_names=["output"], opset_version=17, dynamo=False,
)
model = onnx.load(onnx_path)
onnx.checker.check_model(model)
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
actual = session.run(["output"], {"input": onnx_input.numpy()})[0]
np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
print("PASS PyTorch ONNX export, ONNX checker and ARM64 ONNX Runtime CPU inference", flush=True)

logger = trt.Logger(trt.Logger.WARNING)
with trt.Builder(logger) as builder:
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError("TensorRT ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    serialized = builder.build_serialized_network(network, builder.create_builder_config())
    assert serialized is not None
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(serialized)
        context = engine.create_execution_context()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            source = onnx_input.cuda()
            target = torch.empty_like(source)
            context.set_tensor_address("input", source.data_ptr())
            context.set_tensor_address("output", target.data_ptr())
            assert context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        torch.testing.assert_close(target.cpu(), torch.from_numpy(expected))
print("PASS TensorRT ONNX parse, engine build and GPU inference", flush=True)

loaded = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
          if any(name in line for name in ("libcudnn", "libcudart", "libnvinfer"))}
for path in sorted(loaded):
    assert path.startswith(("/usr/lib/aarch64-linux-gnu/", "/usr/local/cuda-13.2/")), path
    print("BASE LIBRARY:", path)
print("All GPU checks passed.")
