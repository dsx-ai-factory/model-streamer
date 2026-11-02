#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the real-GPU tests: the environment of tests/gpu/compose.yaml (a clean CUDA container, and the services that the
# selected suites need), on a machine of your own or in CI. The workflow runs this too, with every setting explicit.
#
#   tests/gpu/run.sh (--wheel PATH ... | --wheel-dir DIR | --streamer-spec SPEC) [options]
#
#   --wheel PATH            a wheel of the streamer to test: a file, or a glob that matches exactly one. Repeat it for
#                           more than one wheel (the object_storage suite also wants the one of the S3 plugin). They are
#                           copied into a temporary directory that the container mounts
#   --wheel-dir DIR         a directory that holds the wheels, mounted as it is (what CI downloads the artifact into)
#   --streamer-spec SPEC    a requirement specifier of the streamer instead, installed with pip (a released version:
#                           'runai-model-streamer==0.17.0', 'runai-model-streamer>=0.16', or just the name)
#   --suites NAME[,NAME...] the suites to run (tests/gpu/suites/, default streaming). A suite that needs services has a
#                           compose profile of its own name, which starts them (compose.yaml)
#   --gpus N                1 or 2 (default: the GPUs that the machine has, at most 2). With a number given, the machine
#                           must have that many, or this fails. One GPU runs what needs one rank: a suite leaves out what
#                           needs two, and the summary says how much
#   --compose-file PATH     the environment, another compose file than tests/gpu/compose.yaml (see the README)
#   --project NAME          the compose project (default runai-gpu-tests): what to `docker compose -p NAME down` afterwards
#   --torch-spec SPEC, --numpy-spec SPEC   the PyTorch and numpy to install (default: the newest torch, any numpy)
#   --s3-spec SPEC          the S3 plugin to install when there is no wheel of it (default: the newest on PyPI)
#   --seed N                the seed of the models (default: the time)
#   --image IMAGE           the CUDA image (default nvidia/cuda:12.6.3-base-ubuntu24.04, Python 3.12)
#   --no-chown              do not give what the container wrote back to you (CI does not need it)
#   --dry-run               print what would run and stop
#
# Of the streaming suite:
#   --matrix core|full|smoke   which rows (default core, a quick set: see tests/gpu/suites/streaming/README.md)
#   --rows PREFIX[,PREFIX...]  only the rows whose id starts with one of these
#   --self-check            also prove that the checks of the test fire
#
# Needs Docker with the compose plugin and the NVIDIA container toolkit. The results are in test-results/gpu/ of the
# checkout, the models of the streaming suite in .gpu-fixtures/ (about 9 GB, on a disk that serves O_DIRECT: not a tmpfs).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

die() { echo "run.sh: $*" >&2; exit 2; }

wheels=() wheel_dir="" spec="" suites="streaming" gpus="" matrix="core" rows="" torch_spec="" numpy_spec=""
s3_spec="" seed="" self_check="false" image="" compose_file="$here/compose.yaml" project=""
chown_back=1 dry_run=0
while [ $# -gt 0 ]; do
    case "$1" in
        --wheel) [ $# -ge 2 ] || die "--wheel needs a path"; wheels+=("$2"); shift 2 ;;
        --wheel-dir) [ $# -ge 2 ] || die "--wheel-dir needs a directory"; wheel_dir=$2; shift 2 ;;
        --streamer-spec) [ $# -ge 2 ] || die "--streamer-spec needs a specifier"; spec=$2; shift 2 ;;
        --suites) [ $# -ge 2 ] || die "--suites needs names"; suites=$2; shift 2 ;;
        --gpus) [ $# -ge 2 ] || die "--gpus needs 1 or 2"; gpus=$2; shift 2 ;;
        --compose-file) [ $# -ge 2 ] || die "--compose-file needs a path"; compose_file=$2; shift 2 ;;
        --project) [ $# -ge 2 ] || die "--project needs a name"; project=$2; shift 2 ;;
        --matrix) [ $# -ge 2 ] || die "--matrix needs core, full or smoke"; matrix=$2; shift 2 ;;
        --rows) [ $# -ge 2 ] || die "--rows needs prefixes"; rows=$2; shift 2 ;;
        --torch-spec) [ $# -ge 2 ] || die "--torch-spec needs a specifier"; torch_spec=$2; shift 2 ;;
        --numpy-spec) [ $# -ge 2 ] || die "--numpy-spec needs a specifier"; numpy_spec=$2; shift 2 ;;
        --s3-spec) [ $# -ge 2 ] || die "--s3-spec needs a specifier"; s3_spec=$2; shift 2 ;;
        --seed) [ $# -ge 2 ] || die "--seed needs a number"; seed=$2; shift 2 ;;
        --self-check) self_check="true"; shift ;;
        --image) [ $# -ge 2 ] || die "--image needs an image"; image=$2; shift 2 ;;
        --no-chown) chown_back=0; shift ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) awk 'NR >= 5 { if (/^#/) print; else exit }' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option $1 (see --help)" ;;
    esac
done

# What an option sets is what a run gets: a variable of the same name that happens to be exported in your shell (a spec of
# the streamer from another day, a row filter) must not override the option that was not given. (The variables that have no
# option, GPU_REAL_SMOKE, GPU_DEVICE, GPU_BUDGET_MINUTES, GPU_EXPECT_NEWEST_TORCH and GPU_ALLOW_EMPTY_ROWS, pass through.)
unset GPU_STREAMER_SPEC GPU_S3_SPEC GPU_ROWS GPU_TORCH_SPEC GPU_NUMPY_SPEC GPU_TEST_IMAGE

# The streamer comes from one place: wheels (listed, or in a directory), or a specifier.
sources=0
[ "${#wheels[@]}" -gt 0 ] && sources=$((sources + 1))
[ -n "$wheel_dir" ] && sources=$((sources + 1))
[ -n "$spec" ] && sources=$((sources + 1))
if [ "$sources" -gt 1 ]; then die "give one of --wheel, --wheel-dir and --streamer-spec, not more"; fi
if [ "$sources" -eq 0 ]; then die "give --wheel PATH, --wheel-dir DIR or --streamer-spec SPEC: the streamer to test"; fi
if [ -n "$wheel_dir" ] && [ ! -d "$wheel_dir" ]; then die "--wheel-dir $wheel_dir is not a directory"; fi
case "$matrix" in core|full|smoke) ;; *) die "--matrix must be core, full or smoke, not '$matrix'" ;; esac
[ -f "$compose_file" ] || die "the compose file $compose_file was not found"

# The suites. A suite that needs services has a compose profile of its own name: that profile starts them.
profiles=()
IFS=',' read -r -a suite_list <<< "$suites"
for suite in "${suite_list[@]}"; do
    suite=${suite// /}
    [ -f "$here/suites/$suite/suite.sh" ] || die "there is no suite '$suite' (the suites are: $(ls "$here/suites" | tr '\n' ' '))"
    profiles+=("$suite")
done

# The GPUs. GPU_LOCAL_GPU_COUNT is for the tests of this script, which run without a GPU.
if [ -n "${GPU_LOCAL_GPU_COUNT:-}" ]; then
    available=$GPU_LOCAL_GPU_COUNT
else
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi was not found: this needs a machine with NVIDIA GPUs"
    available=$(nvidia-smi -L | grep -c '^GPU ' || true)
fi
[ "$available" -ge 1 ] || die "no GPU is visible"
if [ -z "$gpus" ]; then gpus=$(( available < 2 ? available : 2 )); fi
case "$gpus" in 1|2) ;; *) die "--gpus must be 1 or 2, not '$gpus'" ;; esac
[ "$gpus" -le "$available" ] || die "--gpus $gpus was asked for, and the machine has $available"
if [ "$gpus" -eq 1 ]; then devices="0"; else devices="0,1"; fi

# The directory of wheels that the container mounts: the wheels that were given, the directory that was given, or an
# empty directory for a specifier.
tmp_dir=""
cleanup() { [ -z "$tmp_dir" ] || rm -rf "$tmp_dir"; }
trap cleanup EXIT
streamer=""
if [ -n "$wheel_dir" ]; then
    mounted="$(cd "$wheel_dir" && pwd)"
    streamer="the wheels in $mounted"
else
    tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/gpu-wheel.XXXXXX")"
    mounted="$tmp_dir"
    for wheel in ${wheels[@]+"${wheels[@]}"}; do
        # shellcheck disable=SC2206  # a glob on purpose
        matches=( $wheel )
        [ "${#matches[@]}" -eq 1 ] && [ -f "${matches[0]}" ] || die "--wheel must name exactly one file, '$wheel' matches ${#matches[@]}"
        case "${matches[0]}" in *.whl) ;; *) die "--wheel must be a .whl file: ${matches[0]}" ;; esac
        cp "${matches[0]}" "$tmp_dir/"
        streamer="${streamer:+$streamer, }the wheel $(basename "${matches[0]}")"
    done
    [ -n "$spec" ] && streamer="pip install '$spec'"
fi

export GPU_WHEEL_HOST_DIR="$mounted" GPU_SUITES="$suites"
export GPU_WORLD="$gpus" CUDA_VISIBLE_DEVICES="$devices" GPU_MATRIX="$matrix" GPU_SELF_CHECK="$self_check"
export GPU_SEED="${seed:-$(date +%s)}"
[ -n "$rows" ] && export GPU_ROWS="$rows"
[ -n "$spec" ] && export GPU_STREAMER_SPEC="$spec"
[ -n "$s3_spec" ] && export GPU_S3_SPEC="$s3_spec"
[ -n "$torch_spec" ] && export GPU_TORCH_SPEC="$torch_spec"
[ -n "$numpy_spec" ] && export GPU_NUMPY_SPEC="$numpy_spec"
[ -n "$image" ] && export GPU_TEST_IMAGE="$image"
export GPU_REPO="$repo"

compose=(docker compose -f "$compose_file")
[ -n "$project" ] && compose=(docker compose -p "$project" -f "$compose_file")
for profile in "${profiles[@]}"; do
    compose+=(--profile "$profile")
done
echo "run.sh: suites $suites, ${gpus} GPU(s) (CUDA_VISIBLE_DEVICES=$devices), seed $GPU_SEED, streamer: $streamer"
if [ "$gpus" -eq 1 ]; then
    echo "run.sh: one GPU: what needs two is not run, and the summary says how much"
fi
if [ "$dry_run" -eq 1 ]; then
    echo "run.sh: would run: ${compose[*]} run --rm -T tests"
    env | grep -E '^(GPU_|CUDA_VISIBLE_DEVICES)' | sort | sed 's/^/  /'
    exit 0
fi

command -v docker >/dev/null 2>&1 || die "docker was not found"
docker compose version >/dev/null 2>&1 || die "the docker compose plugin was not found"
# compose.yaml needs docker compose 2.20 or later (depends_on with required: false).
compose_version=$(docker compose version --short 2>/dev/null | tr -d 'v' || true)
compose_major=${compose_version%%.*}
compose_rest=${compose_version#*.}
compose_minor=${compose_rest%%.*}
case "$compose_major$compose_minor" in
    *[!0-9]*|"") die "could not read the version of docker compose ('$compose_version'): 2.20 or later is needed" ;;
esac
if [ "$compose_major" -lt 2 ] || { [ "$compose_major" -eq 2 ] && [ "$compose_minor" -lt 20 ]; }; then
    die "docker compose $compose_version is too old: 2.20 or later is needed (depends_on with required: false)"
fi

# The services of the suites (object storage) go with the run, and so does the network that they made: also when the run is
# interrupted.
stop_services() {
    "${compose[@]}" down --volumes --remove-orphans >/dev/null 2>&1 || true
}
# The container runs as root, so what it wrote into the checkout is root's: give it back (best effort).
give_back_files() {
    [ "$chown_back" -eq 1 ] && [ "$(id -u)" -ne 0 ] || return 0
    for path in "$repo/test-results" "$repo/.gpu-fixtures"; do
        [ -e "$path" ] || continue
        docker run --rm -v "$path:/data" --entrypoint chown "${GPU_TEST_IMAGE:-nvidia/cuda:12.6.3-base-ubuntu24.04}" \
            -R "$(id -u):$(id -g)" /data >/dev/null 2>&1 || true
    done
}
runner=""
on_signal() {
    # bash runs a trap only when the foreground command ends: the run is in the background, and waited for, so that a
    # signal to this script stops the run at once and not when the tests end by themselves.
    [ -z "$runner" ] || kill -TERM "$runner" 2>/dev/null || true
    stop_services
    give_back_files
    exit 130
}
trap on_signal INT TERM HUP

status=0
# -T: no pseudo-terminal. CI has none (compose would refuse), and the output is a log, not a screen.
"${compose[@]}" run --rm -T tests &
runner=$!
wait "$runner" || status=$?
runner=""

stop_services
give_back_files
echo "run.sh: results are in $repo/test-results/gpu"
exit "$status"
