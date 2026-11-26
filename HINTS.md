# Questions worth asking

These are things that **could** matter for a kernel that moves N·V values and does little
arithmetic on each. Some of these questions matter for your kernel and some do not. The answer can change each time you change your kernel. The list is in no particular order, and it is not a to-do list. Each
question names a measurement that tells you whether it matters for you (`PROFILING.md` shows
where each number is).

Change one thing at a time, and measure before and after.

**How much of a call is host time at small shapes?** Compare the `kcup-call` range in `nsys`
to the kernels inside it, at `--shape 256 8192`.

**How wide is each load?** A thread can load 2, 4, 8 or 16 bytes per instruction. What width
does your kernel use? How many instructions per row does that imply? Check the `LDG` widths in the SASS. Check `Access Size` on the `ncu` Source page. Also check the sectors per request.

**How many times do the logits cross DRAM?** The ideal is one read of the logits and one write
of the gradient, 4·N·V bytes. Over all kernels of one call, add up `dram__bytes_read.sum` and `dram__bytes_write.sum`. Then divide the total by 4·N·V, the ideal bytes.

**Are the numerics safe?** In what precision do you accumulate the max and the sum of
exponentials? What happens with a +1000 offset on every logit? With `-inf` entries? With a single huge logit? Does a faster `exp` stay within the tolerances (`TOLERANCES` in
`kcup_harness/checks.py`)? Run `sbatch job.sh test` and see which suite cases fail. Each
case's ID names its input distribution.

**Is there enough parallel work for 132 SMs at every shape?** The grid has N = 256 as well as
N = 8,192. Look at `Grid Size`, the `Block Limit …` rows, and `Waves Per SM` in `ncu` at
each N.

**Which N × V buffers does a call touch, and how many bytes does each cost?** Look at the operators and their input shapes. You find them in torch.profiler and in `census.calls` of your results JSON. Also check the bytes written per kernel in `ncu`.

**What runs outside your own kernels?** Fills, copies, casts, and reductions that PyTorch runs
for you are kernels too, each with its own launch. Check the kernel list in
torch.profiler and `census.kernels_per_call` in your results JSON, at a small and a large
shape.

**A 262,144-wide bf16 row is 512 KB. What fits on one SM, and what follows for your
design?** Check the shared-memory and register limits in `ncu`'s Launch Statistics and
Occupancy sections. Compute bytes moved / ideal bytes at V = 262,144 and at V = 8,192, and compare the two.

**How many registers does a thread use, and does occupancy matter here?** What hides memory
latency in your kernel? Check `Registers Per Thread`, `Achieved Occupancy`, and the warp
stall reasons in `ncu`. Also check `REG:` from `cuobjdump --dump-resource-usage`.

**What happens at the end of a row?** V is not always a multiple of your chunk. At V = 50,257 a row has an odd number of elements. So consecutive rows start at different alignments. What does
your kernel assume about a row's length and where it starts? What does that cost? Check the
suite's odd-V cases and the predicated instructions (`@P…`) in the SASS.

**Is a difference real?** Each timing in the results JSON has a `spread` across five loops,
and nodes differ a little. To check: run the same kernel twice. Compare speedups within one
job, or compare the same kernel across two jobs. Never compare milliseconds across jobs.

**Does an `ncu` rule point at something that matters?** Its `Est. Speedup` is a heuristic.
Check how many of the call's bytes, or how much of its time, the rule's subject accounts for.
