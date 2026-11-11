"""The Kernel Cup starter: fused cross-entropy forward and backward in the CuTe DSL.

Adapted from QuACK (Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao, Apache License 2.0,
see LICENSES/QUACK.txt) through Bonus 3's softmax_cute.py, and modified.

This file is yours to change. The harness calls ``cross_entropy_fwd_bwd(logits, targets)``
and checks what it returns (see the README):

    logits:  (N, V) bfloat16, contiguous, on CUDA.
    targets: (N,) int64, on CUDA. -100 marks an ignored row. At least one row is valid.
    returns: loss, an fp32 scalar: the mean over the valid rows of logsumexp(row) - row[target].
             grad, (N, V) bfloat16: d loss / d logits. Ignored rows get a zero gradient.

The starter is correct on every test case, and it is slow. It beats the eager PyTorch
reference on every benchmark shape, so every improvement you make shows on the board.
PROFILING.md shows how to find where its time goes.

How it works
------------
One call runs four CuTe kernels, one after the other on the caller's stream. Each kernel
gives one CTA (thread block) of 1,024 threads to one row. The CTA walks along its row in
chunks of 1,024 columns, one bf16 value per thread per chunk, so consecutive threads read
consecutive values (Bonus 3's ``strided`` mapping). A tail predicate (``if col < v``) lets
the last chunk stop at V, so any V works.

1. ``row_max_kernel`` finds the largest logit of each row.
2. ``row_lse_kernel`` sums the exponentials of the row, shifted by its max, and writes
   the log-sum-exp and the row's loss (0 for an ignored row).
3. ``grad_kernel`` writes the fp32 gradient: softmax(row), minus 1 at the target, times
   1 / n_valid, and 0 on an ignored row.
4. ``cast_kernel`` narrows the fp32 gradient to bf16.

Around them, ``cross_entropy_fwd_bwd`` allocates the buffers, counts the valid rows and
averages the per-row losses with PyTorch. PyTorch may allocate, fill and take views. It may
compute only on tensors of at most N elements, so every other pass over an N x V tensor
must be one of your own kernels (the README lists the rules).

The two reductions over a row are Bonus 3's three tiers: each thread folds its own values
while it walks the row, then a warp combines its 32 lanes with register shuffles, then the
warps combine their partials through shared memory.

The kernels are compiled once per process. N and V are runtime values, so one compiled
program serves every shape. The first call takes a while (compilation). Later calls do not.

CuTe DSL API provenance
-----------------------
Checked against CUTLASS 4.8.0 (``nvidia-cutlass-dsl`` 4.8.0, the package the kit's job
runs) and QuACK 0.6.5:

- Host and kernel split, ``@cute.jit`` / ``@cute.kernel``, ``.launch(grid=, block=,
  stream=)``, ``cute.compile``: CUTLASS ``examples/python/CuTeDSL/notebooks/
  elementwise_add.ipynb``, as in Bonus 3's ``scripts/kernels/softmax_cute.py``. One
  ``@cute.jit`` host function launches all four kernels, so one compiled call does the
  whole launch sequence.
- ``from_dlpack(t, assumed_align=...).mark_layout_dynamic()``: ``cutlass/cute/runtime.py``
  (``from_dlpack``, ``_Tensor.mark_layout_dynamic``). The shapes and the row stride are
  dynamic, the unit stride of the last mode stays static.
- ``stream: cuda.CUstream`` with ``cuda.CUstream(torch.cuda.current_stream(device).
  cuda_stream)``: QuACK ``quack/softmax.py``, as in Bonus 3.
- ``cutlass.range(start, stop, step)``: a dynamic loop, ``cutlass/base_dsl/ast_helpers.py``
  (``range``). Variables assigned in its body and inside a dynamic ``if`` are carried
  through by the AST preprocessor (``cutlass/base_dsl/ast_preprocessor.py``), which also
  turns ``a if cond else b`` on dynamic values into a select (``visit_IfExp``).
- Scalar loads and stores ``tensor[row, col]`` and ``tensor[row] = value``:
  ``cutlass/cute/tensor.py`` (``_Tensor.__getitem__``, ``__setitem__``: a store widens
  implicitly, never narrows, so the bf16 store converts first). ``Numeric.to(dtype)`` and
  ``Int32(int64_value)``: ``cutlass/base_dsl/typing.py``, the target read as in QuACK
  ``quack/cross_entropy.py``.
- ``cute.math.exp2(..., fastmath=True)`` with the log2(e) scaling, and ``cute.math.log``:
  ``cutlass/cute/math.py`` over ``cutlass/_mlir_helpers/math.py``.
- ``row_reduce``: Bonus 3's three-tier reduction (after QuACK ``quack/reduce.py``),
  started here from a per-thread scalar: ``cute.arch.shuffle_sync_bfly``,
  ``cute.arch.fmax``, ``cute.arch.lane_idx`` / ``warp_idx`` / ``barrier``
  (``cutlass/cute/arch/nvvm_wrappers.py``), ``cutlass.range_constexpr``,
  ``cute.ReductionOp``, ``cute.arch.WARP_SIZE`` (``cutlass/cute/arch/constants.py``) and
  ``Float32.inf``.
- ``cutlass.memory.SmemAllocator().allocate_tensor(dtype, layout, byte_alignment=)``:
  ``cutlass/memory/smem.py``. ``cute.make_layout``, ``cute.size(t, mode=[0])``:
  ``cutlass/cute/core.py``. ``cute.arch.thread_idx`` / ``block_idx``:
  ``cutlass/cute/arch/nvvm_wrappers.py``.

The module has no ``from __future__ import annotations``: the DSL reads JIT
signatures with ``inspect.signature(..., eval_str=True)``, which cannot resolve string
annotations that name the imports local to ``_build``. The DSL is imported inside
``_build`` and ``_launch``, so this file byte-compiles without it.
"""

import math
import operator
from functools import cache

import torch

IGNORE_INDEX = -100
THREADS = 1024  # threads per CTA, and columns per chunk: one value per thread
LOG2_E = math.log2(math.e)  # e^y computed as 2^(y * log2 e) on the special-function unit


@cache
def _build():
    """Define the four kernels and the host function that launches them."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32

    num_warps = THREADS // cute.arch.WARP_SIZE

    @cute.jit
    def row_reduce(val: Float32, op: cutlass.Constexpr, buffer: cute.Tensor) -> Float32:
        """Combine one value per thread into one value per CTA. Every thread gets the result."""
        fold = {cute.ReductionOp.MAX: cute.arch.fmax, cute.ReductionOp.ADD: operator.add}[op]
        # Across the warp's 32 lanes: a butterfly of five register shuffles.
        for i in cutlass.range_constexpr(5):
            val = fold(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
        # Across warps: lane 0 of each warp writes its partial to shared memory.
        # Both indices are read before the branch: warp_idx() is a warp-wide shuffle.
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        if lane == 0:
            buffer[warp] = val
        cute.arch.barrier()
        acc = buffer[0]
        for w in cutlass.range_constexpr(1, num_warps):
            acc = fold(acc, buffer[w])
        # Every thread must finish reading before anyone writes the buffer again.
        cute.arch.barrier()
        return acc

    @cute.kernel
    def row_max_kernel(mX: cute.Tensor, mRowMax: cute.Tensor, v: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        buffer = cutlass.memory.SmemAllocator().allocate_tensor(
            Float32, cute.make_layout(num_warps), byte_alignment=4)
        thread_max = Float32(-Float32.inf)
        for start in cutlass.range(0, v, THREADS):
            col = start + tidx
            if col < v:  # the last chunk may stop short of THREADS columns
                thread_max = cute.arch.fmax(thread_max, mX[row, col].to(Float32))
        row_max = row_reduce(thread_max, cute.ReductionOp.MAX, buffer)
        if tidx == 0:
            mRowMax[row] = row_max

    @cute.kernel
    def row_lse_kernel(mX: cute.Tensor, mTargets: cute.Tensor, mRowMax: cute.Tensor, mLse: cute.Tensor,
                       mRowLoss: cute.Tensor, v: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        buffer = cutlass.memory.SmemAllocator().allocate_tensor(
            Float32, cute.make_layout(num_warps), byte_alignment=4)
        row_max = mRowMax[row]
        thread_sum = Float32(0.0)
        for start in cutlass.range(0, v, THREADS):
            col = start + tidx
            if col < v:
                x = mX[row, col].to(Float32)
                # Subtracting the row's max keeps every exponent at most 0, so nothing overflows.
                thread_sum = thread_sum + cute.math.exp2((x - row_max) * LOG2_E, fastmath=True)
        row_sum = row_reduce(thread_sum, cute.ReductionOp.ADD, buffer)
        if tidx == 0:
            lse = row_max + cute.math.log(row_sum)
            mLse[row] = lse
            t = Int32(mTargets[row])
            row_loss = Float32(0.0)
            if t != IGNORE_INDEX:  # an ignored row adds nothing to the loss
                row_loss = lse - mX[row, t].to(Float32)
            mRowLoss[row] = row_loss

    @cute.kernel
    def grad_kernel(mX: cute.Tensor, mTargets: cute.Tensor, mLse: cute.Tensor, mInvN: cute.Tensor,
                    mGrad32: cute.Tensor, v: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        t = Int32(mTargets[row])
        lse = mLse[row]
        inv_n = mInvN[0]
        if t == IGNORE_INDEX:
            inv_n = Float32(0.0)  # an ignored row gets a zero gradient
        for start in cutlass.range(0, v, THREADS):
            col = start + tidx
            if col < v:
                x = mX[row, col].to(Float32)
                p = cute.math.exp2((x - lse) * LOG2_E, fastmath=True)  # softmax, from the log-sum-exp
                target = Float32(1.0) if col == t else Float32(0.0)
                mGrad32[row, col] = (p - target) * inv_n

    @cute.kernel
    def cast_kernel(mGrad32: cute.Tensor, mGrad: cute.Tensor, v: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        for start in cutlass.range(0, v, THREADS):
            col = start + tidx
            if col < v:
                mGrad[row, col] = mGrad32[row, col].to(BFloat16)

    @cute.jit
    def cross_entropy_host(mX: cute.Tensor, mTargets: cute.Tensor, mRowMax: cute.Tensor, mLse: cute.Tensor,
                           mRowLoss: cute.Tensor, mInvN: cute.Tensor, mGrad32: cute.Tensor, mGrad: cute.Tensor,
                           v: Int32, stream: cuda.CUstream):
        grid = [cute.size(mX, mode=[0]), 1, 1]  # one CTA per row
        block = [THREADS, 1, 1]
        row_max_kernel(mX, mRowMax, v).launch(grid=grid, block=block, stream=stream)
        row_lse_kernel(mX, mTargets, mRowMax, mLse, mRowLoss, v).launch(grid=grid, block=block, stream=stream)
        grad_kernel(mX, mTargets, mLse, mInvN, mGrad32, v).launch(grid=grid, block=block, stream=stream)
        cast_kernel(mGrad32, mGrad, v).launch(grid=grid, block=block, stream=stream)

    return cross_entropy_host


_COMPILED: dict[int, object] = {}


def _launch(logits, targets, row_max, lse, row_loss, inv_n, grad32, grad):
    """Hand the tensors to the DSL and run the compiled host function on the current stream."""
    import cuda.bindings.driver as cuda
    from cutlass import Int32
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    # Only the element size is promised as alignment: with V = 50,257 a bf16 row can start at
    # an odd multiple of 2 bytes. Dynamic layouts let one compiled program serve every N and V.
    tensors = [from_dlpack(t, assumed_align=t.element_size()).mark_layout_dynamic()
               for t in (logits, targets, row_max, lse, row_loss, inv_n, grad32, grad)]
    v = Int32(logits.shape[1])
    stream = cuda.CUstream(torch.cuda.current_stream(logits.device).cuda_stream)
    key = logits.device.index
    if key not in _COMPILED:
        _COMPILED[key] = cute.compile(_build(), *tensors, v, stream)
    _COMPILED[key](*tensors, v, stream)


def cross_entropy_fwd_bwd(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean cross-entropy over the valid rows, and its gradient with respect to the logits."""
    n, v = logits.shape
    row_max = torch.empty(n, device=logits.device, dtype=torch.float32)
    lse = torch.empty_like(row_max)
    row_loss = torch.empty_like(row_max)
    grad32 = torch.zeros(n, v, device=logits.device, dtype=torch.float32)
    grad = torch.empty(n, v, device=logits.device, dtype=torch.bfloat16)
    valid = targets != IGNORE_INDEX
    n_valid = valid.sum()  # N elements, so PyTorch may do it
    inv_n = (1.0 / n_valid.float()).reshape(1)  # stays on the GPU: no wait for the host
    _launch(logits, targets, row_max, lse, row_loss, inv_n, grad32, grad)
    loss = row_loss.sum() / n_valid
    return loss, grad
