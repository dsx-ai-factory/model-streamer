#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The daily run: the same three models with every loader, from local disk and from S3, in one report.
#
#     benchmarks/suite.sh MODELS_DIR [s3://BUCKET[/PREFIX]] [bench.py options]
#
#     Qwen3-32B       to one GPU
#     DeepSeek-R1     to every GPU
#     Kimi-K3-REAP    to every GPU, and to host memory
#
# MODELS_DIR is where the models are. One that is missing there is downloaded, from the bucket when it has
# the model and from Hugging Face otherwise. A model that the directory has no room for is left out with a
# note, and the others still run.
#
# With a bucket, the cases are also run from S3, for the models the bucket holds, each under its name
# (s3://BUCKET/Qwen3-32B/...). Nothing is uploaded: a model that is not there is left out of the S3 half
# with a note. The endpoint and credentials come from the environment (AWS_ENDPOINT_URL,
# AWS_ACCESS_KEY_ID, ...). Without a bucket only the local disk is used.
#
# Anything after that is passed on to every bench.py run:
#
#     benchmarks/suite.sh /mnt/models s3://bench --repeats 1 --loaders safetensors,streamer
#
# Optional environment variables:
#
#     NPROC        ranks for the multi-GPU case, one GPU each       (default: every GPU)
#     OUT_DIR      a directory named after the date is made in it   (default: bench_suite)
#     SCRATCH_DIR  where an S3 model is downloaded to for the loaders that read only local files
#                                                                   (default: MODELS_DIR/bench_scratch)
#     PYTHON       the Python that has the loaders installed        (default: python3)
#     S3_OPTIONS        bench.py options for the S3 runs in one process, for example the number of S3
#                       clients that suits the bucket: --env RUNAI_STREAMER_OBJ_CONCURRENCY=4
#     S3_MULTI_OPTIONS  the same for the S3 runs to every GPU, where the setting is per rank
set -euo pipefail

# One line per bench.py run: a name for it, the model's Hugging Face repository and revision, how many
# *.safetensors files it has and their size in GiB, and then the options bench.py is run with. NPROC stands
# for the number of ranks. The revision is pinned so that every run loads the same bytes.
QWEN="Qwen/Qwen3-32B                     9216db5781bf21249d130ec9da846c4624c16137   17   62"
DEEPSEEK="deepseek-ai/DeepSeek-R1        56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad  163  642"
KIMI="runrunway/Kimi-K3-REAP-448experts  d2aff7364ad7c5e3cdcb7458d959aee2a2b079c9   96  780"
DIST=safetensors_dist,instanttensor_dist,streamer_dist
VLLM=vllm_safetensors,vllm_instanttensor,vllm_streamer
VLLM_DIST=vllm_dist_safetensors,vllm_dist_instanttensor,vllm_dist_streamer
# What vLLM needs to start these models; none of it changes how the weights are loaded.
KIMI_VLLM='--vllm-args {"trust_remote_code":true,"limit_mm_per_prompt":{"image":0,"video":0},"max_model_len":4096}'
CASES=(
    "gpu        $QWEN      --tests safetensors_cuda,instanttensor_cuda,streamer_cuda,$VLLM"
    "multi_gpu  $DEEPSEEK  --nproc NPROC --tests $DIST,$VLLM_DIST"
    "multi_gpu  $KIMI      --nproc NPROC --tests $DIST,$VLLM_DIST $KIMI_VLLM"
    "cpu        $KIMI      --tests safetensors_cpu,streamer_cpu"
)

usage() {
    sed -n '5,25p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}
[ $# -ge 1 ] && [ "${1#-}" = "$1" ] || usage
MODELS_DIR=$1
shift
S3_ROOT=
bucket=${1:-}
if [ "$bucket" != "${bucket#s3://}" ]; then
    S3_ROOT=${bucket%/}
    shift
    command -v s5cmd > /dev/null || { echo "a bucket needs s5cmd on PATH (pip install s5cmd)" >&2; exit 1; }
fi

HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python3}
SCRATCH_DIR=${SCRATCH_DIR:-$MODELS_DIR/bench_scratch}
NPROC=${NPROC:-$(nvidia-smi -L 2> /dev/null | wc -l)}
OUT=${OUT_DIR:-bench_suite}/$(date +%F)
mkdir -p "$OUT" "$MODELS_DIR"
OUT=$(cd "$OUT" && pwd)

s5() {  # s5cmd against the endpoint the environment names, as the streamer and bench.py use it
    local endpoint=${AWS_ENDPOINT_URL_S3:-${AWS_ENDPOINT_URL:-${S3_ENDPOINT_URL:-}}}
    s5cmd ${endpoint:+--endpoint-url "$endpoint"} "$@"
}

here() {  # whether MODELS_DIR has all of a model's files
    [ "$(find "$MODELS_DIR/$1" -maxdepth 1 -name '*.safetensors' 2> /dev/null | wc -l)" -eq "$2" ]
}

there() {  # whether the bucket has all of a model's files
    [ -n "$S3_ROOT" ] && [ "$(s5 ls "$S3_ROOT/$1/*.safetensors" 2> /dev/null | wc -l)" -eq "$2" ]
}

# What this machine will run. A model that is not in MODELS_DIR has to be downloaded, and one that there is
# no room for is left out; so is a model's S3 half when there is no room for the copy the loaders that read
# only local files download during the run.
has() { case " $1 " in *" $2 "*) return 0 ;; esac; return 1; }
free=$(($(df --output=avail -B1 "$MODELS_DIR" | tail -1) / 1073741824))
cases=() planned= noroom= absent=
for entry in "${CASES[@]}"; do
    read -r case repo revision files gib options <<< "$entry"
    name=${repo##*/}
    if [ "$options" != "${options#*--nproc}" ] && [ "$NPROC" -lt 2 ]; then
        echo "leaving $name to every GPU out: this machine has $NPROC"
        continue
    fi
    if has "$noroom" "$name"; then
        continue
    elif ! here "$name" "$files" && ! has "$planned" "$name"; then
        if [ "$gib" -gt "$free" ]; then
            echo "leaving $name out: it is not in $MODELS_DIR, and its $gib GiB do not fit the $free GiB free there"
            noroom+=" $name"
            continue
        fi
        free=$((free - gib))
        planned+=" $name"
    fi
    cases+=("$entry")
done
# The copy made during an S3 run goes to SCRATCH_DIR. On the models' filesystem it shares what is left
# after the downloads planned above; on another one it has that filesystem's free space.
room=$free
if [ -n "$S3_ROOT" ]; then
    mkdir -p "$SCRATCH_DIR"
    if [ "$(df --output=target "$SCRATCH_DIR" | tail -1)" != "$(df --output=target "$MODELS_DIR" | tail -1)" ]; then
        room=$(($(df --output=avail -B1 "$SCRATCH_DIR" | tail -1) / 1073741824))
    fi
fi
for entry in ${cases[@]+"${cases[@]}"}; do
    read -r case repo revision files gib options <<< "$entry"
    name=${repo##*/}
    [ -n "$S3_ROOT" ] && ! has "$absent" "$name" || continue
    if ! there "$name" "$files"; then
        echo "leaving $name out of the S3 half: $S3_ROOT/$name does not hold its $files files"
        absent+=" $name"
    elif [ "$gib" -gt "$room" ]; then
        echo "leaving $name out of the S3 half: downloading it during the run needs $gib GiB and $SCRATCH_DIR has $room GiB free"
        absent+=" $name"
    fi
done
if [ ${#cases[@]} -eq 0 ]; then
    echo "nothing to run on this machine" >&2
    exit 1
fi

fetch() {  # make MODELS_DIR hold a model: from the bucket when it is there, from Hugging Face otherwise
    local repo=$1 revision=$2 files=$3 name=${1##*/}
    local dir=$MODELS_DIR/$name
    here "$name" "$files" && return
    # into a directory of its own first, so that an interrupted download is never taken for the model
    rm -rf "$dir.partial"
    mkdir -p "$dir.partial"
    if there "$name" "$files"; then
        echo "downloading $name from $S3_ROOT to $dir"
        s5 cp "$S3_ROOT/$name/*.safetensors" "$dir.partial/" > /dev/null
        s5 cp "$S3_ROOT/$name/*.json" "$dir.partial/" > /dev/null 2>&1 || true  # config.json, for the vLLM tests
    else
        echo "downloading $repo from Hugging Face to $dir"
        HF_HUB_OFFLINE=0 hf download "$repo" --revision "$revision" --include '*.safetensors' '*.json' \
            --local-dir "$dir.partial" > /dev/null
        rm -rf "$dir.partial/.cache"
    fi
    mkdir -p "$dir"
    mv "$dir.partial"/* "$dir/"
    rm -rf "$dir.partial"
}

failed=() names=()
bench() {  # one bench.py run. Leaving 3 means the model does not fit there on this machine: not a failure.
    local name=$1 status=0
    shift
    echo
    echo "=== $name"
    rm -f "$OUT/$name.json"  # a case that gives no results must not report an older run's
    "$PYTHON" "$HERE/bench.py" "$@" --out "$OUT/$name.json" --log-dir "$OUT/logs_$name" || status=$?
    names+=("$name")
    if [ "$status" = 3 ]; then
        echo "left $name out: the model does not fit there on this machine"
    elif [ "$status" != 0 ]; then
        failed+=("$name")
    fi
}

for entry in ${cases[@]+"${cases[@]}"}; do
    read -r case repo revision files gib options <<< "$entry"
    fetch "$repo" "$revision" "$files"
done
for source in disk ${S3_ROOT:+s3}; do  # every case from local disk, then every case from S3
    for entry in ${cases[@]+"${cases[@]}"}; do
        read -r case repo revision files gib options <<< "$entry"
        name=${repo##*/}
        options=${options//NPROC/$NPROC}
        # shellcheck disable=SC2086  # the case's options are split on purpose
        if [ "$source" = disk ]; then
            bench "${name}_$case" "$MODELS_DIR/$name" $options "$@"
        elif ! has "$absent" "$name"; then
            options=$(echo "$options" | sed -E 's/,?vllm_[a-z_]+//g; s/ --vllm-args [^ ]+//')  # vLLM loads local models
            tuned=${S3_OPTIONS:-}
            [ "$case" = multi_gpu ] && tuned=${S3_MULTI_OPTIONS:-}
            bench "${name}_${case}_s3" "$S3_ROOT/$name" $options $tuned --ref-dir "$MODELS_DIR/$name" \
                --scratch-dir "$SCRATCH_DIR" "$@"
        fi
    done
done

results=()
for name in ${names[@]+"${names[@]}"}; do
    if [ -f "$OUT/$name.json" ]; then
        results+=("$OUT/$name.json")
    fi
done
if [ ${#results[@]} -gt 0 ]; then
    echo
    PYTHONPATH=$HERE${PYTHONPATH:+:$PYTHONPATH} "$PYTHON" -m benchmark.markdown "${results[@]}" \
        --out "$OUT/report.md"
fi
if [ ${#failed[@]} -gt 0 ]; then
    echo "failed: ${failed[*]}" >&2
    exit 1
fi
