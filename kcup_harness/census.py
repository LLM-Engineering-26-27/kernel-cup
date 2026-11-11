"""What one call of the submission launches, and the rules it must keep (§3, §6.4)."""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

# Allocation, fills and views. A wrapper operator that may hide compute (aten::to, aten::contiguous)
# records that compute as a child (aten::_to_copy, aten::copy_), which is not allowed.
# LIBRARY_KERNELS holds parts of the kernel names the libraries use for softmax, log-softmax, NLL and
# cross-entropy, as UC3's H100 recorded them (measurement of 2026-11-11), never a generic word such as
# "softmax", so a pair's own kernel name cannot match by accident. It is the only check that sees a C++
# extension calling ATen kernels below the dispatcher, where no operator is recorded.
# LIBRARY_KERNEL_NAMES holds Liger's Triton kernels by their whole names. Triton names a kernel after its
# Python function, so a part of such a name would also catch a pair's own ce_softmax_multi_block_kernel.
ALLOWED_OPS = frozenset({
    "aten::empty", "aten::empty_like", "aten::empty_strided", "aten::new_empty", "aten::new_empty_strided",
    "aten::zeros", "aten::zeros_like", "aten::new_zeros", "aten::ones", "aten::ones_like", "aten::full",
    "aten::full_like", "aten::new_full", "aten::fill_", "aten::zero_", "aten::view", "aten::reshape",
    "aten::_reshape_alias", "aten::_unsafe_view", "aten::as_strided", "aten::expand", "aten::select",
    "aten::slice", "aten::narrow", "aten::split", "aten::chunk", "aten::unbind", "aten::unsqueeze",
    "aten::squeeze", "aten::permute", "aten::transpose", "aten::t", "aten::detach", "aten::alias",
    "aten::lift_fresh", "aten::resize_", "aten::contiguous", "aten::view_as", "aten::expand_as", "aten::to",
    "aten::flatten", "aten::split_with_sizes",
})
INDUCTOR_PATTERNS = ("triton_poi_", "triton_red_", "triton_per_", "triton_for_", "triton_tem_", "triton_spl_")
INDUCTOR_NAME = "triton_"  # every Inductor kernel's whole name under TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=0
LIBRARY_KERNELS = (
    # PyTorch: softmax and log-softmax forward and backward (rows up to 1,024, longer rows, over a channel
    # dimension), NLL with reduction and without
    "cunn_SoftMaxForward", "cunn_SoftMaxBackward", "softmax_warp_forward", "softmax_warp_backward",
    "cunn_SpatialSoftMax", "nll_loss_forward_reduce_cuda_kernel", "nll_loss_backward_reduce_cuda_kernel",
    "nll_loss_forward_no_reduce_cuda_kernel", "nll_loss_backward_no_reduce_cuda_kernel",
    # QuACK (CuTe DSL): the DSL names a kernel method after its module, kernel_cutlass_kernel_quackcross_entropy...
    "kernel_cutlass_kernel_quack",
)
LIBRARY_KERNEL_NAMES = frozenset({
    # Liger 0.8.5: liger_kernel/ops/cross_entropy.py and ops/softmax.py (the multi-block pair from the source,
    # since LigerSoftmax refused the one V that would have launched it)
    "liger_cross_entropy_kernel", "_softmax_single_block_forward_kernel", "_softmax_single_block_backward_kernel",
    "_softmax_multi_block_forward_kernel", "_softmax_multi_block_backward_kernel",
})
COMPILED_REGION = "Torch-Compiled Region"
FORBIDDEN_MODULES = ("liger_kernel", "quack")


def ops_from_trace(trace: dict) -> list[dict]:
    """PyTorch operators with their input shapes. The trace, unlike FunctionEvent.input_shapes, keeps
    the shapes inside a TensorList input (aten::cat, aten::stack) as a list of shapes."""
    return [{"name": e["name"], "shapes": e.get("args", {}).get("Input Dims", []),
             "types": e.get("args", {}).get("Input type", [])}
            for e in trace.get("traceEvents", [])
            if e.get("cat") == "cpu_op" and e.get("name", "").startswith(("aten::", COMPILED_REGION))]


def kernels_from_trace(trace: dict) -> list[dict]:
    events = [e for e in trace.get("traceEvents", []) if e.get("cat") == "kernel"]
    events.sort(key=lambda e: e["ts"])
    return [{"name": e["name"], "stream": e.get("args", {}).get("stream")} for e in events]


def profile_call(fn, logits, targets, *, device: torch.device) -> dict:
    """Profile one warm call. A one-element fill on the current stream comes first, as the
    stream reference."""
    on_gpu = device.type == "cuda"
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if on_gpu else [])
    marker = torch.empty(1, device=device)
    if on_gpu:
        torch.cuda.synchronize(device)
    # Without profile_all_threads the profiler records only this thread's operators, not a worker thread's.
    threads = torch._C._profiler._ExperimentalConfig(profile_all_threads=True)
    with profile(activities=activities, record_shapes=True, experimental_config=threads) as prof:
        marker.fill_(0)
        fn(logits, targets)
        if on_gpu:
            torch.cuda.synchronize(device)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.json"
        prof.export_chrome_trace(str(path))
        trace = json.loads(path.read_text())
    return {"ops": ops_from_trace(trace), "kernels": kernels_from_trace(trace)}


def violations(record: dict, n: int, *, device: torch.device) -> list[str]:
    """The rules a profiled call broke. device is the one it was profiled on."""
    problems: list[str] = []
    for op in record["ops"]:
        if op["name"].startswith(COMPILED_REGION):
            problems.append(f"{op['name']}: torch.compile output on the logits is not allowed")
            continue
        if op["name"] in ALLOWED_OPS:
            continue
        if any(t == "TensorList" and not s for t, s in zip(op.get("types", []), op["shapes"])):
            problems.append(f"PyTorch operator {op['name']} reads a list of more than 30 tensors, which the "
                            "profiler cannot size. Work on the logits must be your own kernel")
        big = [s for s in op["shapes"] if _elements(s) > n]
        if big:
            problems.append(f"PyTorch operator {op['name']} reads {_describe(big[0])}, more than N = {n} "
                            "elements: work on the logits must be your own kernel")
    kernels = record["kernels"]
    if not kernels and device.type == "cuda":  # on CPU no kernel is recorded at all
        problems.append("the trace holds no kernel, not even the stream marker (a one-element fill), so the "
                        "profiler saw none of the call's work on the GPU")
    elif kernels and "FillFunctor" not in kernels[0]["name"]:
        problems.append("the stream marker (a one-element fill) is missing from the trace, so the "
                        f"harness's stream is unknown: the first kernel is {kernels[0]['name']}")
    elif kernels:
        stream = kernels[0]["stream"]
        for kernel in kernels[1:]:
            if kernel["name"].startswith(INDUCTOR_PATTERNS) or kernel["name"] == INDUCTOR_NAME:
                problems.append(f"kernel {kernel['name']} was generated by torch.compile")
            if kernel["name"] in LIBRARY_KERNEL_NAMES or any(name in kernel["name"] for name in LIBRARY_KERNELS):
                problems.append(f"kernel {kernel['name']} is a library kernel")
            if kernel["stream"] != stream:
                problems.append(f"kernel {kernel['name']} runs on stream {kernel['stream']}, not on the "
                                f"harness's stream {stream}: work on another stream escapes the timing")
    return list(dict.fromkeys(problems))


def _elements(dims: list) -> int:
    """Elements of one input: a tensor's shape, or a TensorList's shapes summed."""
    if dims and isinstance(dims[0], list):
        return sum(math.prod(shape) for shape in dims)
    return math.prod(dims)


def _describe(dims: list) -> str:
    if dims and isinstance(dims[0], list):
        return f"a list of {len(dims)} tensors with {_elements(dims)} elements"
    return f"a tensor of shape {dims}"


def kernels_per_call(record: dict) -> int:
    return max(0, len(record["kernels"]) - 1)


def forbidden_modules() -> list[str]:
    return [name for name in FORBIDDEN_MODULES if name in sys.modules]
