"""Check a torch.cond early exit with PyTorch and TensorRT on CUDA."""

import onnx
import tensorrt as trt
import torch
import torch.nn.functional as F

WIDTH = 64
MODEL_FILE = "/tmp/torch-cond.onnx"


class EarlyExit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.small = torch.nn.Linear(8, WIDTH, bias=False)
        self.large = torch.nn.Linear(WIDTH, WIDTH, bias=False)
        with torch.no_grad():
            self.small.weight.fill_(1 / 8)
            self.large.weight.copy_(2 * torch.eye(WIDTH))

    def forward(self, x):
        hidden = self.small(x)
        return torch.cond(
            hidden.mean() > 0,
            lambda value, _: value.clone(),
            lambda value, weight: F.linear(value, weight).relu(),
            (hidden, self.large.weight),
        )


class Profiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.times = {}

    def report_layer_time(self, name, milliseconds):
        self.times[name] = milliseconds


def infer(context, sample):
    source = sample.cuda()
    result = torch.empty((1, WIDTH), device="cuda")
    profiler = Profiler()
    context.profiler = profiler
    context.set_tensor_address("sample", source.data_ptr())
    context.set_tensor_address("result", result.data_ptr())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        assert context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return result, profiler.times


assert torch.cuda.is_available(), "CUDA is unavailable"
model = EarlyExit().cuda().eval()
early = torch.ones(1, 8, device="cuda")
full = -early
with torch.no_grad():
    expected_early, expected_full = model(early), model(full)
torch.testing.assert_close(expected_early, torch.ones(1, WIDTH, device="cuda"))
torch.testing.assert_close(expected_full, torch.zeros(1, WIDTH, device="cuda"))
print("PASS torch.cond selects both branches on CUDA")

model.cpu()
torch.onnx.export(
    model, (early.cpu(),), MODEL_FILE, input_names=["sample"],
    output_names=["result"], opset_version=18, dynamo=True, external_data=False,
)
graph = onnx.load(MODEL_FILE)
onnx.checker.check_model(graph)
assert sum(node.op_type == "If" for node in graph.graph.node) == 1

logger = trt.Logger(trt.Logger.WARNING)
with trt.Builder(logger) as builder:
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    assert parser.parse_from_file(MODEL_FILE), [
        str(parser.get_error(i)) for i in range(parser.num_errors)
    ]
    plan = builder.build_serialized_network(network, builder.create_builder_config())
    assert plan is not None

with trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    actual_early, early_times = infer(context, early)
    actual_full, full_times = infer(context, full)

torch.testing.assert_close(actual_early, expected_early)
torch.testing.assert_close(actual_full, expected_full)
lazy_layers = [
    name for name, time in full_times.items()
    if time > 0 and early_times.get(name) == 0
]
assert len(lazy_layers) >= 2, (early_times, full_times)
print("PASS TensorRT runs both outputs on CUDA and skips the large branch on early exit")
print("Lazy full-path layers:", lazy_layers)
