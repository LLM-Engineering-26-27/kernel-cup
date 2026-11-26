# Christmas Kernel Cup

Write the fastest correct fused cross-entropy kernel for UC3's H100: the forward and the
backward pass over bf16 logits, in one call. You start from a working kernel that is correct
and slow, find out where its time goes, and make it faster.

The cup is optional and ungraded. It is not an official course competition and has no effect
on your grade or on the exercise gate. The leaderboard stays open until Monday 2027-03-15, so you
can tinker whenever you like. On Tuesday 2026-12-15 the instructor takes an informal look at the
board, and the best entries at that point get a small prize from Australia.

The cup builds on Bonus Chapter 3 (GPU kernels and the CuTe DSL). The starter is a CuTe kernel in
the style of that chapter's softmax.

## The task

Your entry is one Python function in `submission/kernel.py`:

```python
def cross_entropy_fwd_bwd(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    """Fused cross-entropy, forward and backward, in one call.

    logits:  (N, V) bfloat16, contiguous, on CUDA.
    targets: (N,) int64, on CUDA. -100 marks an ignored row. At least one row is valid.
    Returns:
      loss: fp32 scalar, the mean over non-ignored rows of logsumexp(row) - row[target].
      grad: (N, V) bfloat16, d loss / d logits. Ignored rows get a zero gradient.
    """
```

- One call returns both outputs, so your kernel may write the gradient while it computes the
  loss. You may also use two or more kernels.
- The inputs must be unchanged after the call. The outputs must be new plain `torch.Tensor`s
  with the shapes, dtypes and device above, and `grad` must be contiguous.
- You may add files under `submission/`. Your kernel must not import anything from the kit
  outside `submission/`: the suite refuses such imports.
- Compiled programs, CUDA graphs, cached `from_dlpack` wrappers and scratch buffers may persist across calls.
  Every call must still recompute its outputs from its own inputs.
  Outputs may not persist across calls. Neither may anything computed from an earlier call's inputs. Relying on
  what a scratch buffer still holds from an earlier call is not allowed either.

The correctness suite (`kcup_harness/test_kernel.py`) runs about 1,300 cases before any timing.
It covers N from 1 to 8,192, V from 2 to 262,144, scaled, shifted and spiked logits, `-inf`
entries, targets in the first and last column, and ignored rows. The benchmark runs only after
every test has passed for your current files.

## Timeline

All times are Berlin time.

| Date | What happens |
|---|---|
| Thu 2026-11-26 | The kit and the leaderboard go public |
| Fri 2026-12-04 | UC3 is expected back. GPU work starts |
| Tue 2026-12-15, 23:59:59 | Informal check: the instructor looks at the leaderboard |
| Mon 2027-03-15, 23:59:59 | The leaderboard closes |

UC3 is down from 2026-11-23 to about 2026-12-04. You can read the starter code and plan your
kernel before then, but no jobs will run.

**The check on 2026-12-15** is a look at the leaderboard, not a re-run. Whatever you pushed by
23:59:59 counts. The instructor reads the kernels of the best provisional entries
in each division and picks who gets a prize. If UC3 is not back by Monday 2026-12-07, the check
moves to Tue 2027-01-05, 23:59:59. After the check, nothing changes: keep tinkering until the
leaderboard closes.

## Rules

### Divisions

Each pair declares a division in `submission/team.toml`:

- **Unassisted.** The course's exercise rule applies. AI assistants (chat models, coding
  agents) may explain things. They may also help with the harness, the plots and profiler
  questions. Your pair writes the kernel.
- **Assisted.** AI assistants may write code too.

You may move from Unassisted to Assisted at any time, but never back. Once any commit on
`kcup` declares Assisted, the entry stays Assisted. Nobody can check the declaration, so it is
taken on trust.

### Languages

The CuTe DSL is the supported language: the kit ships a CuTe starter. Triton and CUDA C++
(through `torch.utils.cpp_extension.load_inline`) are allowed too, but without a starter and
without staff support. The harness only calls your Python function. Set `language` in
`team.toml` to `cute`, `triton` or `cuda`.

### What PyTorch may do

PyTorch may do three things in your submission:

- allocate tensors, and fill them with zeros or a constant
- take views
- compute on tensors of at most N elements, such as the N per-row losses.

Every other operation that reads or writes an N × V tensor must be one of your own kernels.
The same rule applies to an N × k tensor of per-row partials (with k > 1) and to a tensor of
length V.
PyTorch may allocate them, but not compute on them.

Your submission never calls PyTorch's softmax, log-softmax or cross-entropy, Liger, QuACK, or
a function produced by `torch.compile`. It does not import `liger_kernel` or `quack`.

The **census** in the correctness suite checks this as far as a profiler can see. It profiles
one call on every benchmark shape and fails your entry if:

- a PyTorch operator other than allocation, a fill or a view has an input with more than N
  elements
- a kernel has a name from a library (the list below)
- a kernel runs on a stream other than the one the harness called on.

Staff also review the code by hand.

### Kernel names you must not use

The census recognizes library kernels by name. Your own kernel names must not:

- contain `cunn_SoftMaxForward`, `cunn_SoftMaxBackward`, `softmax_warp_forward`,
  `softmax_warp_backward`, `cunn_SpatialSoftMax`, `nll_loss_forward_reduce_cuda_kernel`,
  `nll_loss_backward_reduce_cuda_kernel`, `nll_loss_forward_no_reduce_cuda_kernel`,
  `nll_loss_backward_no_reduce_cuda_kernel` or `kernel_cutlass_kernel_quack`
- be one of Liger's kernel names: `liger_cross_entropy_kernel`,
  `_softmax_single_block_forward_kernel`, `_softmax_single_block_backward_kernel`,
  `_softmax_multi_block_forward_kernel` or `_softmax_multi_block_backward_kernel`. Triton
  names a kernel after its Python function, so do not give a function one of these names
- start with `triton_poi_`, `triton_red_`, `triton_per_`, `triton_for_`, `triton_tem_` or
  `triton_spl_`, or be exactly `triton_`. These are the kernels `torch.compile` generates.

The CuTe DSL names a kernel `kernel_cutlass_`, then its function name, then its arguments. A method's first argument is the object the method belongs to. So the kernel's name also contains
the names of that object's module and class. For
example, a CuTe kernel method named `kernel` on a class in a module whose name starts with
`quack` gets a name starting with `kernel_cutlass_kernel_quack`. Do not combine those names.

### Sources and copying

In both divisions you may read anything, including QuACK's cross-entropy, Liger and the
Dao-AILab note on memory-bound kernels. You may reuse:

- the starter, all of it, including its lines adapted from QuACK
- Bonus 3's code
- small utilities from elsewhere, such as QuACK's reduction helpers in `quack/reduce.py`.

The library kernels are QuACK's kernels in `quack/cross_entropy.py` and `quack/softmax.py`,
Liger's kernels, and any other library's softmax or cross-entropy kernel. Copying any part of
them disqualifies the entry. If you reuse code, say where it comes from in a comment.

The starter is adapted from Bonus 3's softmax kernel, which is adapted from QuACK's
`quack/softmax.py` (Apache License 2.0, see `LICENSES/QUACK.txt`).

### Nicknames

The leaderboard shows your `nickname` from `team.toml`, not your names. Nicknames are unique,
compared without regard to case. If two pairs pick the same one, the pair that committed it
first keeps it. The other pair is shown with a suffix such as `-2` until it renames.

`team.toml` is not part of your kernel's hash, so editing it never makes a measurement stale.

### Publishing

Staff may share the code of entries with `publish_code = true` in `team.toml` in a private
repository that course members can read. Nothing is published in the public kit repository.

### Fair play

There is no staff re-run: the leaderboard shows your own measurements, and the check trusts them.
At the check, the instructor reads the kernels of the best entries. An entry that caches outputs
across calls, or tampers with the harness from inside the process, gets no prize. Tampering
includes importing `kcup_harness`, patching `torch.cuda.Event`, `sys.modules` or torch functions,
and starting threads or subprocesses. Compilers' own processes are fine. Your own code must not
read environment variables or files outside `submission/`. Compilers' own caches are fine. The
same applies to a library softmax or cross-entropy call, copied library kernel code, or a break of
the PyTorch rule above.

### When an entry is unranked

The leaderboard marks an entry unranked if:

- its harness was changed (`kcup_harness/`, `job.sh`, `pytest.ini`)
- the census failed
- its `team.toml` needs a fix, for example a missing nickname
- its summary % of speed of light is above 105% (not plausible).

## Quick start

You need the UC3 setup from the course's `CLUSTER.md`: an account, the workspace `lme` and
your pair's repository cloned into it.

**1. Create the `kcup` branch** (one partner, once). This branch lives in your pair repository
but has no history in common with `main`. A second folder beside your clone keeps `main`
checked out for the weekly exercises. Log in to UC3 and run:

```bash
cd "$(ws_find lme)/<repository name>"
git fetch https://github.com/LLM-Engineering-26-27/kernel-cup main
git worktree add -b kcup ../kcup FETCH_HEAD
cd ../kcup
git push -u origin kcup
```

UC3 is down until 2026-12-04. Until then, the first partner can create the branch from a clone
of the pair repository on a laptop: the same commands, with a `cd` into that clone in place of
the first line.

The other partner, and anyone who sets up the branch on UC3 later, then runs in their own
clone:

```bash
cd "$(ws_find lme)/<repository name>"
git fetch origin
git worktree add ../kcup kcup
cd ../kcup
```

If you prefer a second clone to a worktree, the first partner clones the repository into a
new folder and creates the branch there:

```bash
cd "$(ws_find lme)"
git clone git@github.com:LLM-Engineering-26-27/<repository name>.git kcup
cd kcup
git fetch https://github.com/LLM-Engineering-26-27/kernel-cup main
git switch --orphan kcup
git reset --hard FETCH_HEAD
git push -u origin kcup
```

The other partner then clones the branch alone:

```bash
cd "$(ws_find lme)"
git clone -b kcup git@github.com:LLM-Engineering-26-27/<repository name>.git kcup
cd kcup
```

**2. Name your entry.** One partner edits `submission/team.toml`: a nickname, your division
and your language. Then commit and push:

```bash
git add submission/team.toml
git commit -m "kcup: team"
git push
```

**Working as a pair.** Both partners push to `kcup`. Run `git pull --no-rebase --no-edit`
before every push, so your partner's commits are merged in. Result files never conflict,
because each job writes a new file named after its time and job id. Let only one partner edit
`team.toml`, so the two of you never change it at the same time.

**3. Run the tests.** From the `kcup` folder:

```bash
sbatch job.sh test
squeue --me
```

The log is `kcup-<job id>.out` in the same folder (`tail -f kcup-<job id>.out` follows it).
Every test passes for the starter. A job on a node that has not run the kit before loads the
environment from disk and takes up to a minute longer.

**4. Run the full job.**

```bash
sbatch job.sh
```

This runs the tests, then the benchmark on all 12 shapes, then the plot. It prints a table of
times and your score, and writes `results/<UTC time>-<job id>.json` and a `.png` with the
same name.

**5. Submit the result.** Commit the JSON (and the plot too, if you like) and push:

```bash
git add results/
git commit -m "kcup: starter measured"
git pull --no-rebase --no-edit
git push
```

Within about 30 minutes the leaderboard shows your entry.

**Then improve the kernel.** Edit `submission/kernel.py`, check one shape with
`sbatch job.sh --shape 2048 128256`, and profile it (`PROFILING.md`). `HINTS.md` lists
questions worth asking. When a change works, commit it, run the full job, then commit its JSON
and push.

Two things make a measurement not count:

- **Uncommitted files.** The job records whether `submission/` matches your last commit. If
  it does not, or if `submission/` holds an untracked file, the measurement is uncommitted and
  the leaderboard ignores it. Commit before you submit the full job.
- **Edits during the job.** A job reads your files when it starts, not when you submit it,
  and it checks them again while it runs. Do not edit `submission/` until the job has
  finished.

## Job modes

`job.sh` asks for one H100 on `gpu_h100_short` for 15 minutes.

| Command | What it does |
|---|---|
| `sbatch job.sh` | The tests, the benchmark on the whole grid, the plot. The only mode the leaderboard counts |
| `sbatch job.sh --shape 2048 128256` | The tests and the benchmark for one shape, about 2 minutes. Its JSON is marked partial and the leaderboard ignores it |
| `sbatch job.sh test` | The tests only. No benchmark, no JSON |
| `sbatch job.sh test --shape 2048 128256` | The tests for one shape only |
| `sbatch job.sh profile torch --shape 2048 128256` | A torch.profiler table and trace (`PROFILING.md`) |
| `sbatch job.sh profile nsys --shape 256 8192` | An Nsight Systems report |
| `sbatch job.sh profile ncu --shape 2048 50304` | An Nsight Compute report |
| `sbatch --partition=gpu_a100_short job.sh test` | The tests on an A100, when the H100 queue is long |

**Time.** A job that asks for less time starts sooner, so the default is 15 minutes. The
starter's full job takes a few minutes. If a job needs longer (for example because your kernel
compiles once per shape), ask for more on the command line: `sbatch --time=00:25:00 job.sh`.
`gpu_h100_short` allows at most 30 minutes.

**Never edit `job.sh`.** It is part of the harness, and a changed harness unranks the entry on the
leaderboard.
Pass SLURM options on the `sbatch` command line instead.

**A100s** are for `test` only. They lack the H100's sm_90 features, so a kernel that uses them
runs only on the H100. Only H100 results count for the leaderboard.

## The score

The benchmark grid has 12 shapes:

| | V = 8,192 | V = 50,304 | V = 128,256 | V = 262,144 |
|---|---|---|---|---|
| N = 256 | ✓ | ✓ | ✓ | ✓ |
| N = 2,048 | ✓ | ✓ | ✓ | ✓ |
| N = 8,192 | ✓ | ✓ | ✓ | ✓ |

N is the number of tokens. V = 8,192 is the course model's vocabulary. The others are
GPT-2 (padded), Llama 3 and Gemma 3. The benchmark logits are standard normal, with random
targets and no ignored rows.

- **Speedup per shape**: the eager reference's time divided by your kernel's time, both
  measured in the same job on the same GPU. The eager reference is plain PyTorch: the loss in
  fp32 and its gradient through autograd (`kcup_harness/reference.py`). A slower node
  therefore does not change your score.
- **Score**: the geometric mean of the 12 speedups. Doubling your speed on any one shape
  counts the same.
- **% of speed of light per shape**: how close you are to the hardware limit. A call must read the logits once (2·N·V bytes in bf16), write the gradient once (2·N·V bytes in
  bf16) and read the targets (8·N bytes in int64). The total is 4·N·V + 8·N bytes. Your
  % of speed of light is these ideal bytes divided by your time, expressed as a share of
  the bandwidth of the copy (a plain `out.copy_(logits)`, timed in the same job). The
  leaderboard's summary column is the geometric mean over the shapes.

Each timing is the median of five loops of back-to-back calls. The input buffers rotate so
they do not stay in the L2 cache. After the last loop, the outputs on fresh inputs are
checked again.

The job also times comparators that never rank: PyTorch's bf16 `F.cross_entropy` with its
backward, `torch.compile` of the reference, QuACK's cross-entropy, the starter, and the copy.
They all appear in the printed table and the JSON.

**The plot** (`results/<UTC time>-<job id>.png`) has one panel per N. The x-axis is V, the
y-axis is % of speed of light, and 100% is drawn as the ceiling. Your kernel is the
highlighted line. The muted lines are the starter, the eager reference and QuACK.
`python -m kcup_harness.plot --all results/<file>.json` adds `F.cross_entropy` and
`torch.compile`. This command also runs on a laptop: from a checkout of your `kcup` branch
(because it imports `kcup_harness`), with PyTorch and matplotlib installed.

## The leaderboard

<https://kernelcup.strueber.dev>

Every 30 minutes the leaderboard reads the head of each pair's `kcup` branch. It takes the newest results JSON that covers the whole grid, that ran on an H100, and that measured
committed files. Partial, A100 and uncommitted runs are skipped. It reads the nickname and the language
from `team.toml` at the head.

| Status | Meaning |
|---|---|
| provisional | the newest counted result matches your committed kernel and an unmodified harness |
| stale | your kernel changed after its newest counted result. The page shows that older score |
| unranked | the harness was modified, the census failed, `team.toml` needs a fix, no result counts yet, or the summary % of speed of light is above 105% (not plausible) |

Every score on the page is self-measured on UC3. When the leaderboard closes on 2027-03-15, its
banner says "Closing" for ten minutes. Then it says "Closed", and each row keeps the status its
branch head had at 23:59:59, as GitHub recorded your pushes. A push after that does not count, whatever
its commit dates say.

## Branch rules

- `kcup` lives in your pair repository and shares no history with `main`. Pushing to it starts
  no grading.
- Never force-push or delete `kcup`: the leaderboard reads its history.
- Change only `submission/` and `results/`. Editing `kcup_harness/`, `job.sh` or `pytest.ini`
  unranks the entry on the leaderboard.
- If staff publish a new kit version, merge it into `kcup`, push, and run the full job again:
  `git pull --no-rebase --no-edit https://github.com/LLM-Engineering-26-27/kernel-cup main`.

## The files

| Path | What it is |
|---|---|
| `submission/kernel.py` | The CuTe starter. Yours to change |
| `submission/team.toml` | Nickname, division, language, `publish_code` |
| `results/` | The result JSONs and plots you commit |
| `job.sh` | The SLURM job (harness) |
| `kcup_harness/` | Reference, tests, benchmark, census, plot (harness) |
| `PROFILING.md` | How to profile your kernel on UC3 |
| `HINTS.md` | Questions worth asking about your kernel |
| `LICENSES/QUACK.txt` | QuACK's licence, for the code the starter adapts |
