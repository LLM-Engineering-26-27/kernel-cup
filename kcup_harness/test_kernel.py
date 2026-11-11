"""The Kernel Cup correctness suite (design §5). Run it with: sbatch job.sh test"""

import sys

from kcup_harness import cases, census, checks, loading, reference


def test_case(case, submission, device, secret):
    logits, targets = cases.make_inputs(case, device, secret)
    logits_before, targets_before = logits.clone(), targets.clone()
    out = submission(logits, targets)
    problems = checks.contract_problems(logits, targets, out, logits_before=logits_before,
                                        targets_before=targets_before)
    assert not problems, "\n".join(problems)
    problems = checks.value_problems(out[0], out[1], *reference.oracle(logits, targets), targets)
    assert not problems, "\n".join(problems)


def test_census(bench_shape, submission, device, secret):
    n, v = bench_shape
    logits, targets = cases.make_benchmark_buffers(n, v, 1, device, seed=cases.mix_seed(1, secret))[0]
    submission(logits, targets)  # warm-up: compilation stays outside the profile
    problems = census.violations(census.profile_call(submission, logits, targets, device=device), n,
                                  device=device)
    assert not problems, "\n".join(problems)


def test_no_library_and_nothing_outside_submission_was_imported(submission, modules_before_kernel):
    """The last test, so it sees what the calls imported too. The fixture loads the kernel if no test has yet."""
    loaded = census.forbidden_modules()
    assert not loaded, f"the submission imported {', '.join(loaded)}: library kernels are not allowed (README, Rules)"
    loading.refuse_kit_imports(set(sys.modules) - modules_before_kernel, loading.kit_root() / "submission")
