#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run the real-GPU tests on a machine of your own, in the same clean CUDA container as CI (tests/gpu/compose.yaml).
#
#   tests/gpu/run_local.sh (--wheel PATH ... | --streamer-spec SPEC) [options]
#
#   --wheel PATH            a wheel of the streamer to test: a file, or a glob that matches exactly one. Repeat it for
#                           more than one wheel (the object-storage suite also wants the one of the S3 plugin)
#   --streamer-spec SPEC    a requirement specifier of the streamer instead, installed with pip (a released version:
#                           'runai-model-streamer==0.17.0', 'runai-model-streamer>=0.16', or just the name)
#   --suites NAME[,NAME...] the suites to run (tests/gpu/suites/, default streaming). The services that they need start
#                           with the environment (suite_services of their suite.sh: compose profiles)
#   --gpus N                1 or 2 (default: the GPUs that the machine has, at most 2). One GPU runs what needs one rank:
#                           a suite leaves out what needs two, and the summary says how much
#   --compose-file PATH     the environment, another compose file than tests/gpu/compose.yaml (see the README)
#   --torch-spec SPEC, --numpy-spec SPEC   the PyTorch and numpy to install (default: the newest torch, any numpy)
#   --s3-spec SPEC          the S3 plugin to install when there is no wheel of it (default: the newest on PyPI)
#   --seed N                the seed of the models (default: the time)
#   --image IMAGE           the CUDA image (default nvidia/cuda:12.6.3-base-ubuntu24.04, Python 3.12)
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

die() { echo "run_local.sh: $*" >&2; exit 2; }

wheels=() spec="" suites="${GPU_SUITES:-streaming}" gpus="" matrix="core" rows="" torch_spec="" numpy_spec="" s3_spec=""
seed="" self_check="false" image="" compose_file="${GPU_COMPOSE_FILE:-$here/compose.yaml}" dry_run=0
while [ $# -gt 0 ]; do
    case "$1" in
        --wheel) [ $# -ge 2 ] || die "--wheel needs a path"; wheels+=("$2"); shift 2 ;;
        --streamer-spec) [ $# -ge 2 ] || die "--streamer-spec needs a specifier"; spec=$2; shift 2 ;;
        --suites) [ $# -ge 2 ] || die "--suites needs names"; suites=$2; shift 2 ;;
        --gpus) [ $# -ge 2 ] || die "--gpus needs 1 or 2"; gpus=$2; shift 2 ;;
        --compose-file) [ $# -ge 2 ] || die "--compose-file needs a path"; compose_file=$2; shift 2 ;;
        --matrix) [ $# -ge 2 ] || die "--matrix needs core, full or smoke"; matrix=$2; shift 2 ;;
        --rows) [ $# -ge 2 ] || die "--rows needs prefixes"; rows=$2; shift 2 ;;
        --torch-spec) [ $# -ge 2 ] || die "--torch-spec needs a specifier"; torch_spec=$2; shift 2 ;;
        --numpy-spec) [ $# -ge 2 ] || die "--numpy-spec needs a specifier"; numpy_spec=$2; shift 2 ;;
        --s3-spec) [ $# -ge 2 ] || die "--s3-spec needs a specifier"; s3_spec=$2; shift 2 ;;
        --seed) [ $# -ge 2 ] || die "--seed needs a number"; seed=$2; shift 2 ;;
        --self-check) self_check="true"; shift ;;
        --image) [ $# -ge 2 ] || die "--image needs an image"; image=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) sed -n '5,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option $1 (see --help)" ;;
    esac
done

# The streamer comes from one place: wheels, or a specifier.
if [ "${#wheels[@]}" -gt 0 ] && [ -n "$spec" ]; then die "give --wheel or --streamer-spec, not both"; fi
if [ "${#wheels[@]}" -eq 0 ] && [ -z "$spec" ]; then die "give --wheel PATH or --streamer-spec SPEC: the streamer to test"; fi
case "$matrix" in core|full|smoke) ;; *) die "--matrix must be core, full or smoke, not '$matrix'" ;; esac
[ -f "$compose_file" ] || die "the compose file $compose_file was not found"

# The suites, and the services (compose profiles) that they need: what a suite.sh says, which is only definitions.
profiles=()
IFS=',' read -r -a suite_list <<< "$suites"
for suite in "${suite_list[@]}"; do
    suite=${suite// /}
    [ -f "$here/suites/$suite/suite.sh" ] || die "there is no suite '$suite' (the suites are: $(ls "$here/suites" | tr '\n' ' '))"
    while IFS= read -r profile; do
        [ -n "$profile" ] || continue
        case " ${profiles[*]-} " in *" $profile "*) ;; *) profiles+=("$profile") ;; esac
    done < <(bash -c 'source "$1"; printf "%s\n" ${suite_services[@]+"${suite_services[@]}"}' _ "$here/suites/$suite/suite.sh")
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

# The wheel directory that the container mounts: the wheels, or an empty directory for a specifier.
wheel_dir="$(mktemp -d "${TMPDIR:-/tmp}/gpu-wheel.XXXXXX")"
cleanup() { rm -rf "$wheel_dir"; }
trap cleanup EXIT
streamer=""
for wheel in ${wheels[@]+"${wheels[@]}"}; do
    # shellcheck disable=SC2206  # a glob on purpose
    matches=( $wheel )
    [ "${#matches[@]}" -eq 1 ] && [ -f "${matches[0]}" ] || die "--wheel must name exactly one file, '$wheel' matches ${#matches[@]}"
    case "${matches[0]}" in *.whl) ;; *) die "--wheel must be a .whl file: ${matches[0]}" ;; esac
    cp "${matches[0]}" "$wheel_dir/"
    streamer="${streamer:+$streamer, }the wheel $(basename "${matches[0]}")"
done
[ -n "$spec" ] && streamer="pip install '$spec'"

export GPU_WHEEL_HOST_DIR="$wheel_dir" GPU_SUITES="$suites"
export GPU_WORLD="$gpus" CUDA_VISIBLE_DEVICES="$devices" GPU_MATRIX="$matrix" GPU_SELF_CHECK="$self_check"
export GPU_SEED="${seed:-$(date +%s)}"
[ -n "$rows" ] && export GPU_ROWS="$rows"
[ -n "$spec" ] && export GPU_STREAMER_SPEC="$spec"
[ -n "$s3_spec" ] && export GPU_S3_SPEC="$s3_spec"
[ -n "$torch_spec" ] && export GPU_TORCH_SPEC="$torch_spec"
[ -n "$numpy_spec" ] && export GPU_NUMPY_SPEC="$numpy_spec"
[ -n "$image" ] && export GPU_TEST_IMAGE="$image"
export GPU_REPO="$repo"

profile_args=()
for profile in ${profiles[@]+"${profiles[@]}"}; do
    profile_args+=(--profile "$profile")
done
compose=(docker compose -f "$compose_file" ${profile_args[@]+"${profile_args[@]}"})
echo "run_local.sh: suites $suites, ${gpus} GPU(s) (CUDA_VISIBLE_DEVICES=$devices), seed $GPU_SEED, streamer: $streamer"
if [ "${#profiles[@]}" -gt 0 ]; then
    echo "run_local.sh: services of the suites: ${profiles[*]}"
fi
if [ "$gpus" -eq 1 ]; then
    echo "run_local.sh: one GPU: what needs two is not run, and the summary says how much"
fi
if [ "$dry_run" -eq 1 ]; then
    echo "run_local.sh: would run: ${compose[*]} run --rm tests"
    env | grep -E '^(GPU_|CUDA_VISIBLE_DEVICES)' | sort | sed 's/^/  /'
    exit 0
fi

command -v docker >/dev/null 2>&1 || die "docker was not found"
docker compose version >/dev/null 2>&1 || die "the docker compose plugin was not found"

status=0
"${compose[@]}" run --rm tests || status=$?

# The services of the suites (object storage) go with the run, and so does the network that they made.
if [ "${#profiles[@]}" -gt 0 ]; then
    "${compose[@]}" down --volumes --remove-orphans >/dev/null 2>&1 || true
fi

# The container runs as root, so what it wrote into the checkout is root's: give it back (best effort).
if [ "$(id -u)" -ne 0 ]; then
    for path in "$repo/test-results" "$repo/.gpu-fixtures"; do
        [ -e "$path" ] || continue
        docker run --rm -v "$path:/data" --entrypoint chown "${GPU_TEST_IMAGE:-nvidia/cuda:12.6.3-base-ubuntu24.04}" \
            -R "$(id -u):$(id -g)" /data >/dev/null 2>&1 || true
    done
fi
echo "run_local.sh: results are in $repo/test-results/gpu"
exit "$status"
