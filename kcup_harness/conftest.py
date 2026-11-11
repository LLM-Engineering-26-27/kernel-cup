"""Parametrizes the suite and writes the correctness marker that bench.py requires."""

import os
import sys

import pytest

from kcup_harness import cases, loading


def pytest_addoption(parser):
    parser.addoption("--kcup-shape", nargs=2, type=int, default=None, metavar=("N", "V"),
                     help="run only the cases of one shape")


def _scope(max_elements: int, shape: tuple[int, int] | None) -> str:
    if max_elements < cases.MAX_ELEMENTS:
        return "capped"
    return f"{shape[0]},{shape[1]}" if shape else "full"


# What the marker records, read in pytest_configure before the kernel is imported.
_kernel_sha256 = _harness_sha256 = _shape = _secret = None
_max_elements = cases.MAX_ELEMENTS
_expected_items = 0  # what the suite's functions generate, before any selection
_passed_calls = 0


def pytest_configure(config):
    """Read before the kernel is imported: the kernel and harness hashes, so an edit during the suite
    leaves a marker that no longer matches the files, and the cap, the shape and the secret seed, so a
    kernel that changes the environment at import changes nothing. Pytest also calls this hook when it
    imports this conftest during collection (pytest .), after the session started."""
    global _kernel_sha256, _harness_sha256, _shape, _secret, _max_elements
    try:
        identity = loading.kernel_identity()
    except (RuntimeError, SystemExit) as err:  # no manifest or no kernel: pytest prints the message only
        raise pytest.UsageError(str(err)) from None
    _kernel_sha256, _harness_sha256 = identity["kernel"]["sha256"], identity["harness"]["sha256"]
    shape = config.getoption("--kcup-shape")
    _shape = tuple(shape) if shape else None
    _max_elements = int(os.environ.get("KCUP_MAX_ELEMENTS", cases.MAX_ELEMENTS))
    _secret = loading.secret_seed()


def pytest_generate_tests(metafunc):
    """Pytest collects the whole module even when a node ID selects one test, so this counts every
    item of the suite."""
    global _expected_items
    if "case" in metafunc.fixturenames:
        selected = cases.suite_cases(_shape, _max_elements)
        if not selected:
            raise pytest.UsageError(f"no suite cases for shape {_shape}")
        metafunc.parametrize("case", selected, ids=[c.id for c in selected])
        _expected_items += len(selected)
    elif "bench_shape" in metafunc.fixturenames:
        shapes = [s for s in cases.benchmark_shapes()
                  if (_shape is None or s == _shape) and s[0] * s[1] <= _max_elements]
        metafunc.parametrize("bench_shape", shapes, ids=[f"N{n}-V{v}" for n, v in shapes])
        _expected_items += len(shapes)
    else:
        _expected_items += 1


def pytest_collection_modifyitems(config, items):
    """A scope without a benchmark shape has no census test. Drop pytest's skipped placeholder
    for the empty parameter set, so every remaining item must pass. A skip raised during a
    test's call (a kernel calling pytest.skip) still counts as not passed."""
    def placeholder(item):
        mark = item.get_closest_marker("skip")
        return mark is not None and str(mark.kwargs.get("reason", "")).startswith("got empty parameter set")

    items[:] = [item for item in items if not placeholder(item)]


@pytest.fixture(scope="session")
def modules_before_kernel():
    """The modules loaded before the kernel, so the last test can tell which ones the kernel imported."""
    return set(sys.modules)


@pytest.fixture(scope="session")
def submission(modules_before_kernel):
    return loading.load_submission()


@pytest.fixture(scope="session")
def device():
    return loading.device()


@pytest.fixture(scope="session")
def secret():
    return _secret


def pytest_runtest_logreport(report):
    global _passed_calls
    if report.when == "call" and report.passed:
        _passed_calls += 1


def _filtered(config) -> bool:
    """A marker may only come from a run of every case of its scope."""
    option = config.option
    return bool(option.collectonly or option.keyword or option.markexpr or getattr(option, "lf", False)
                or getattr(option, "last_failed", False) or option.deselect)


def _why_no_marker(config, items: int) -> str | None:
    """Why a run without a failure earns no marker, or None if it earns one."""
    if _filtered(config):
        return "the run used -k, -m, --lf, --deselect or --collect-only"
    if items == 0:
        return "this process ran no tests (pytest-xdist runs them in other processes)"
    if items != _expected_items:
        return f"the run collected {items} of the suite's {_expected_items} tests"
    if _passed_calls != items:
        return f"{items - _passed_calls} of {items} tests did not pass (skipped, xfailed or not run)"
    return None


def pytest_sessionfinish(session, exitstatus):
    items = len(getattr(session, "items", []))
    reason = _why_no_marker(session.config, items)
    loading.write_marker(passed=exitstatus == 0 and reason is None, kernel_sha256=_kernel_sha256,
                         harness_sha256=_harness_sha256, scope=_scope(_max_elements, _shape), cases=items,
                         seed_mode="secret" if _secret is not None else "public")
    if exitstatus == 0 and reason is not None:
        print(f"\nNo correctness marker: {reason}. The benchmark needs a run of the whole suite where every "
              "test passes.")
