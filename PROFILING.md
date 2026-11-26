# Profiling your kernel on UC3

Three tools, from coarse to fine, then the compiled instructions:

1. **torch.profiler** (you know it from Week 2): which kernels a call launches, and how long each one takes.
2. **Nsight Systems** (`nsys`): the timeline of a call, with the gaps between kernels and the host's share.
3. **Nsight Compute** (`ncu`): what happens inside one kernel: bytes moved, throughput, waves, load widths, stalls.
4. **SASS**: the machine instructions your kernel compiled to.

This guide shows where each number is and what it means. It does not tell you what your kernel's numbers will be. Run the tools on the starter first, then on each version of your kernel, and draw your own conclusions.

## How a profile job works

```bash
sbatch job.sh profile torch --shape 2048 128256
```

A profile job loads your submission and makes one batch of random inputs at the shape you give. It calls your function once to compile and warm up. Then it makes three more calls, each inside an NVTX range named `kcup-call`, under the profiler. It runs no tests and writes no results JSON. The reports go to `results/profiles/<job id>-…`, which git ignores. The log goes to `kcup-<job id>.out`.

**Pick the shape for the question.** One shape does not show everything:

| Shape | What it shows well |
|---|---|
| `--shape 256 8192` | Little work per call: the host side and launch costs |
| `--shape 256 262144` | Few, very long rows: is there enough parallel work? |
| `--shape 2048 50304` | A middle shape, small enough for `ncu` |
| `--shape 8192 128256` | Much work per call: bandwidth. Its `ncu` report is too large to copy and open comfortably (see below) |

Profile more than one shape. The score is a geometric mean over all 12.

## Before you open a report: the ideal bytes

A call must read every logit once and write every gradient entry once. Both tensors are in bf16. This is the **ideal traffic**. A bf16 value is 2 bytes, so one read and one write cost 4 bytes per logit: 4·N·V bytes in total. The targets add 8·N bytes (N int64 values), which is small. At 2,048 × 128,256 that is 4 · 2,048 · 128,256 = 1,050,673,152 bytes, about 1.05 GB.

Two numbers follow from the ideal traffic:

- **bytes moved / ideal bytes**: how many times a call's DRAM traffic exceeds the minimum. `ncu` gives the bytes moved (section 3).
- **the time floor**: ideal bytes divided by the copy bandwidth. Your results JSON lists the shapes in `shapes`. Each entry holds the copy's time, `impls.copy.median_s`, and your `sol_percent`, which is your time measured against that floor.

## Copying a report to your laptop

The reports open in free GUIs on your laptop. First, on UC3, print the folder's path:

```bash
cd "$(ws_find lme)/kcup"
pwd
```

Then, on your laptop, copy the profiles using that path. Here `uc3` is the short login from the course's `CLUSTER.md`, part B3. Without it, write `tu_zxabc12@uc3.scc.kit.edu` with your own user name:

```bash
rsync -av uc3:/path/printed/above/results/profiles/ kcup-profiles/
```

On Windows, run `rsync` in WSL, or use `scp -r` with the same two paths.

The GUIs are free downloads from NVIDIA's developer site, for Windows, macOS and Linux: Nsight Systems and Nsight Compute. The site may ask you to log in with a free NVIDIA developer account. The course image writes reports with Nsight Systems 2026.3.1 and Nsight Compute 2026.2.1. For each tool, install that version or a newer one. An older GUI may refuse a report written by a newer version. The torch.profiler trace opens in any browser at <https://ui.perfetto.dev>.

## 1. torch.profiler: which kernels, how long

```bash
sbatch job.sh profile torch --shape 2048 128256
```

The job log holds a table of every PyTorch operator and every kernel across the three calls, sorted by `CUDA total`. The trace `results/profiles/<job id>-torch.json` shows the same calls on a timeline.

What the columns mean:

- **Rows named `aten::…`** are PyTorch operators: what your Python code asked PyTorch to do. Their `CUDA total` includes the kernels they launched.
- **The other rows are kernels.** Your own CuTe kernels are named `kernel_cutlass_<function name>_<argument types>__<index>`. PyTorch's are named after their C++ templates, for example `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, …>`.
- **`# of Calls`** counts over all three calls. Divide by 3 for one call.
- **`Self CUDA`** is the GPU time of a kernel itself. `Self CUDA %` is its share of the total.
- **`Self CPU time total` and `Self CUDA time total`** appear under the table. Compare them to see whether the host or the GPU took more time. The CPU figure includes the profiler's own overhead.
- **`cudaDeviceSynchronize`**: the job waits for the GPU once after the three calls. If this row is a large share of the CPU time, the host was waiting for the GPU. If it is tiny, the GPU had already finished when the host asked.
- **`Activity Buffer Request`** is the profiler's own work, not your code.

Questions to answer here: How many kernels does one call launch? Which of them touch an N × V tensor? Which PyTorch operators run around your kernels, and what do they cost at this shape?

You do not need a profile job for the kernel list alone. Every full job runs the census. The results JSON holds what it recorded: `census.calls["N,V"]` lists every operator with its input shapes and every kernel with its stream. `census.kernels_per_call["N,V"]` is the number of kernels one call launches at that shape.

## 2. Nsight Systems: the timeline

```bash
sbatch job.sh profile nsys --shape 256 8192
```

The report is `results/profiles/<job id>-nsys.nsys-rep`, about a megabyte. Open it in the Nsight Systems GUI (File, Open).

**The rows.** Under the process you see the CUDA API row (the host launching kernels), the GPU's row with the kernels as they ran on the H100, and the NVTX row with the three `kcup-call` ranges. Zoom into one `kcup-call` range. Hover over a kernel to see its duration, grid, block and registers per thread.

**Is the GPU busy for most of the call?** Compare the length of the `kcup-call` range with the kernels inside it. If the kernels fill the range, the GPU is the limit. If the range is much longer than the kernels added up, with wide gaps between them, the host is the limit: Python, argument handling and launches. Look at where the largest gap is and what the host does during it.

**The same numbers as tables.** The GUI's Stats System View (in the bottom pane) shows them. So does `nsys stats --report <name>`, which you can run on UC3 in a short CPU job from your `kcup` folder. It writes to `slurm-<job id>.out`:

```bash
sbatch --partition=cpu --time=00:05:00 --wrap 'apptainer exec ${TMPDIR:+--bind "${TMPDIR}"} /pfs/work9/workspace/scratch/tu_iiojo01-lme-course/pytorch-26.07-py3.sif nsys stats --report nvtx_kern_sum results/profiles/<job id>-nsys.nsys-rep'
```

The reports worth reading:

| Report | What it holds |
|---|---|
| `nvtx_sum` | each `kcup-call` range as the host saw it (host time, not kernel time) |
| `nvtx_gpu_proj_sum` | the same ranges projected onto the GPU: from the start of their first kernel to the end of their last |
| `nvtx_kern_sum` | the kernels inside each range, with how often each ran |
| `cuda_gpu_trace` | every kernel launch: start, duration, grid, block, registers |
| `cuda_kern_exec_trace` | per launch: when the host called the launch, how long the kernel waited in the queue, when it ran |
| `cuda_gpu_kern_sum` | every kernel of the whole job. This includes the warm-up call and the kernels that generate the inputs, so prefer `nvtx_kern_sum` |

A useful check: compare a range's GPU projection, its host time and the sum of its kernel times. If the GPU projection is close to the host time but far above the sum of the kernel times, the call is host-bound.

**Times under a profiler are larger** than in the benchmark. `nsys` adds a little to every API call, and the job runs only three calls. Compare ratios within one report, not absolute times across tools.

## 3. Nsight Compute: inside one kernel

```bash
sbatch job.sh profile ncu --shape 2048 50304
```

`ncu --set full` profiles every kernel of the three calls. To read all the counters, `ncu` replays each kernel dozens of times. That is why the job takes minutes. A kernel with many launches per call may need more than the default 15 minutes: add `--time` as the README shows. The report is `results/profiles/<job id>-ncu.ncu-rep`.

Staff ran `ncu` with its DRAM counters as a normal UC3 user in the course image. If your job log shows `ERR_NVGPUCTRPERM`, the counters are blocked: tell the staff.

**Keep the shape small.** At the largest shapes the report is hundreds of megabytes: too large to copy and open comfortably. Use a middle or small shape, for example 2,048 × 50,304 or 256 × 262,144.

Open the report in the Nsight Compute GUI. The Summary page lists every profiled kernel in launch order, all three calls. Look at one call only. Its kernels come first in the list, as many as one call launches. Double-click a kernel for its Details page. The Source page shows its SASS line by line. The Raw page lists every metric by its ID, with a search box.

### Where each number is

| Number | Section (Details page) | Display name | Metric ID (Raw page) |
|---|---|---|---|
| Kernel time | GPU Speed Of Light Throughput | `Duration` | `gpu__time_duration.sum` |
| Share of DRAM peak | GPU Speed Of Light Throughput | `DRAM Throughput` (%) | |
| Share of the busiest memory unit's peak | GPU Speed Of Light Throughput | `Memory Throughput` (%) | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM bytes read and written | Memory Workload Analysis chart, Device Memory box | | `dram__bytes_read.sum`, `dram__bytes_write.sum` |
| Global load instructions | | | `smsp__sass_inst_executed_op_global_ld.sum` |
| Sectors and requests of those loads | | | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`, `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` |
| Grid, block, registers | Launch Statistics | `Grid Size`, `Block Size`, `Registers Per Thread` | `launch__grid_size`, `launch__block_size`, `launch__registers_per_thread` |
| Waves | Launch Statistics | `Waves Per SM` | `launch__waves_per_multiprocessor` |
| CTAs per SM | Occupancy | the `Block Limit …` rows: `Barriers`, `SM`, `Registers`, `Shared Mem`, `Warps` | `launch__occupancy_limit_blocks` (SM), `launch__occupancy_limit_registers`, `launch__occupancy_limit_shared_mem`, `launch__occupancy_limit_warps` |
| Achieved occupancy | Occupancy | `Achieved Occupancy` (%) | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Cycles between issued instructions, and why warps wait | Warp State Statistics | `Warp Cycles Per Issued Instruction`, the stall reasons | |
| Load width per instruction | Source page | `Access Size`, `L2 Theoretical Sectors Global`, `… Ideal`, `… Excessive` | |

An empty cell means that name was not checked against a report. To see the metric ID behind every display name in a report, print its Details page with IDs in a short CPU job:

```bash
sbatch --partition=cpu --time=00:05:00 --wrap 'apptainer exec ${TMPDIR:+--bind "${TMPDIR}"} /pfs/work9/workspace/scratch/tu_iiojo01-lme-course/pytorch-26.07-py3.sif ncu --import results/profiles/<job id>-ncu.ncu-rep --page details --print-metric-name name'
```

### Method: bytes against the ideal

Add `dram__bytes_read.sum` and `dram__bytes_write.sum` over **every** kernel of one call, including PyTorch's small kernels. Divide by 4·N·V. Then look at which kernels contribute which bytes. Divide each kernel's bytes by N·V to get bytes per logit: a bf16 read is 2, a bf16 write is 2, an fp32 value is 4.

Two things to know about these counters:

- `ncu` measures each kernel alone, with the caches flushed between its replays.
- The write counter misses bytes that are still in the L2 cache when a kernel ends. So `dram__bytes_write.sum` can be lower than the bytes the kernel wrote. This is most visible at small shapes.

`DRAM Throughput` is a property of one kernel. The time of a call is the sum over its kernels. Read the percentages together with the bytes per call.

### Method: waves

The H100 on UC3 has 132 SMs. The Occupancy section has five `Block Limit …` rows: barriers, SM, registers, shared memory and warps. The smallest of them is how many CTAs one SM can hold at once. Then:

waves = CTAs in the grid / (132 × CTAs per SM).

`ncu` prints this as `Waves Per SM`. Below 1, some SMs have no work at all. A fractional last wave leaves SMs idle while it runs: 600 CTAs at 2 per SM make 2.27 waves, and the third wave keeps only 72 of 264 slots busy. Work this out for each N of the grid, since the grid usually grows with N.

`Achieved Occupancy` is averaged over the cycles in which an SM is active. Cycles in which an SM has no work are left out. So high occupancy does not mean every SM had work.

### Method: load width

A warp's load instruction moves 32 threads × the bytes each thread loads. The L2 works in 32-byte sectors, so sectors per request is about:

- 2 for 2-byte loads,
- 4 for 4-byte loads,
- 8 for 8-byte loads,
- 16 for 16-byte loads,

when consecutive threads read consecutive addresses. For each kernel, divide `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` by `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`. The result is the sectors per request. On the Source page, `Access Size` gives the bits per thread of each load line, and `Instructions Executed` shows how often it ran. The load instruction count times the bytes per instruction should roughly match the bytes the kernel reads.

Sector utilization (`… Excessive`, and the "uncoalesced" rules) measures coalescing. Load width is a separate property. To find it, read the load instructions in the SASS.

### Rules and their estimated speedups

Under each section, `ncu` prints rules: `OPT` lines with an `Est. Speedup`, and `INF` and `WRN` notes. These are heuristics. Each one looks at a single counter and estimates what you would gain if that counter were perfect. They can point at things that do not matter for your kernel. Before you act on one, check it against the bytes and the time: a rule about one kind of access can save at most what that access costs.

**Warp stalls.** In a memory-bound kernel, most warps spend most cycles waiting for memory. A large stall share on the L1TEX scoreboard is normal in that case. It tells you the kernel waits for loads. It does not tell you why.

### Kernel names differ by tool

PyTorch's kernels print differently in each tool. torch.profiler writes `void at::native::vectorized_elementwise_kernel<4, …>`. `nsys` adds the template argument types (`<(int)4, …>`). `ncu` drops `native::`. Your CuTe kernels have the same name in all three: `kernel_cutlass_<function name>_<argument types>__<index>`.

## 4. SASS: what your kernel compiled to

The quickest route is the `ncu` Source page: choose the SASS view, and every instruction of the kernel appears with its counters.

To read the whole program, as Bonus 3's `check_softmax.py --print-layouts` does, keep the cubin that the CuTe DSL compiles and disassemble it. Every job mode compiles your kernel. `profile nsys` is a short one: it runs no tests and makes four calls. Create the folder first, because the DSL does not create it:

```bash
mkdir -p results/profiles/cubin
sbatch --export=ALL,CUTE_DSL_KEEP=cubin,CUTE_DSL_DUMP_DIR=$PWD/results/profiles/cubin job.sh profile nsys --shape 256 8192
```

`cuobjdump` is in the course image. Run it in a short CPU job, which writes to `slurm-<job id>.out`:

```bash
sbatch --partition=cpu --time=00:05:00 --wrap 'apptainer exec ${TMPDIR:+--bind "${TMPDIR}"} /pfs/work9/workspace/scratch/tu_iiojo01-lme-course/pytorch-26.07-py3.sif cuobjdump -sass results/profiles/cubin/*.cubin'
```

With `--dump-resource-usage` in place of `-sass` it prints each kernel's registers per thread (`REG:`) and shared memory (`SHARED:`).

What to look for:

| SASS | Meaning |
|---|---|
| `LDG.E.U16`, `LDG.E`, `LDG.E.64`, `LDG.E.128` | a global load of 2, 4, 8 or 16 bytes per thread |
| `STG.E.U16`, `STG.E`, `STG.E.64`, `STG.E.128` | a global store of the same widths |
| `@P0 …`, `@!P0 …` | an instruction that runs only where predicate P0 holds (or does not): a tail guard |
| `ISETP` | an integer compare that sets a predicate |
| `SHFL.BFLY` | a warp shuffle, the butterfly step of a warp reduction |
| `BAR.SYNC` | a barrier across the CTA |
| `MUFU.EX2` | the hardware exp2 |
| `FMNMX` | a float max or min |

Find the main loop over a row and count its loads. Then compare: how many bytes does one iteration of the loop load per thread, and how many instructions does it take?
