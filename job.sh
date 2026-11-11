#!/usr/bin/env bash
#SBATCH --job-name=kcup
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=kcup-%j.out
# The Kernel Cup job. Submit it from your kcup checkout:
#   sbatch job.sh                                    tests, the full benchmark, the plot
#   sbatch job.sh --shape 2048 128256                one shape, about 2 minutes
#   sbatch job.sh test                               the tests only
#   sbatch job.sh profile nsys --shape 2048 128256   (or torch, ncu)
#   sbatch --partition=gpu_a100_short job.sh test    the tests on an A100 (its timings do not count)
# Pass SLURM options on the command line, for example sbatch --time=00:25:00 job.sh.
# This file belongs to the harness: editing it unranks your entry.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this with sbatch from your kcup checkout}"

COURSE_DIR=/pfs/work9/workspace/scratch/tu_iiojo01-lme-course
IMAGE="${COURSE_DIR}/pytorch-26.07-py3.sif"
PY="${COURSE_DIR}/kcup/pylibs-kcup/bin/python"
export KCUP_STATE_DIR="${KCUP_STATE_DIR:-${PWD}/.kcup}"
export KCUP_RESULTS_DIR="${KCUP_RESULTS_DIR:-${PWD}/results}"

mode=full
tool=""
shape=()
while (($#)); do
    case "$1" in
        full|test) mode="$1"; shift ;;
        profile) mode=profile; tool="${2:?profile needs torch, nsys or ncu}"; shift 2 ;;
        --shape) shape=("${2:?--shape needs N and V}" "${3:?--shape needs N and V}"); shift 3 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -f "${IMAGE}" ]] || { echo "the course image is missing: ${IMAGE}. Tell the staff." >&2; exit 1; }
[[ -x "${PY}" ]] || { echo "the competition environment is missing: ${PY}. Tell the staff." >&2; exit 1; }

# Git runs here on the host: inside the container a git worktree's .git file points nowhere.
command -v python3 >/dev/null || module load devel/python/3.12.3-gnu-14.2
python3 kcup_harness/manifest.py write --state-dir "${KCUP_STATE_DIR}" --job-id "${SLURM_JOB_ID}"

in_container() {
    apptainer exec --nv --bind "${COURSE_DIR}:${COURSE_DIR}:ro" ${TMPDIR:+--bind "${TMPDIR}"} \
        --env PYTHONNOUSERSITE=1 --env LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
        --env TRITON_LIBCUDA_PATH=/.singularity.d/libs \
        --env "KCUP_STATE_DIR=${KCUP_STATE_DIR}" --env "KCUP_RESULTS_DIR=${KCUP_RESULTS_DIR}" \
        --env "KCUP_JOB_ID=${SLURM_JOB_ID}" "${IMAGE}" "$@"
}

echo "kcup job ${SLURM_JOB_ID} on $(hostname), commit $(git rev-parse --short HEAD), mode ${mode}"
nvidia-smi -L
suite_shape=()
bench_args=(--full)
if ((${#shape[@]})); then
    suite_shape=(--kcup-shape "${shape[0]}" "${shape[1]}")
    bench_args=(--shape "${shape[0]}" "${shape[1]}")
fi

case "${mode}" in
    test)
        in_container "${PY}" -m pytest -c pytest.ini kcup_harness/test_kernel.py "${suite_shape[@]}" ;;
    full)
        in_container "${PY}" -m pytest -c pytest.ini -q kcup_harness/test_kernel.py "${suite_shape[@]}"
        in_container "${PY}" -m kcup_harness.bench "${bench_args[@]}"
        in_container "${PY}" -m kcup_harness.plot --latest ;;
    profile)
        ((${#shape[@]})) || { echo "profile needs --shape N V" >&2; exit 2; }
        mkdir -p "${KCUP_RESULTS_DIR}/profiles"
        out="${KCUP_RESULTS_DIR}/profiles/${SLURM_JOB_ID}"
        call=("${PY}" -m kcup_harness.bench --profile-call "${shape[0]}" "${shape[1]}")
        case "${tool}" in
            torch) in_container "${call[@]}" --torch-trace "${out}-torch.json" ;;
            nsys) in_container nsys profile --trace=cuda,nvtx --force-overwrite true -o "${out}-nsys" "${call[@]}" ;;
            ncu) in_container ncu --set full --nvtx --nvtx-include "kcup-call/" -f -o "${out}-ncu" "${call[@]}" ;;
            *) echo "unknown profiler: ${tool}" >&2; exit 2 ;;
        esac ;;
esac
