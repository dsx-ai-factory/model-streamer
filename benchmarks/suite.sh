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
# the model and from Hugging Face otherwise. Before anything is downloaded the script checks that the
# directory has room for what is missing, and stops with what it needs when it has not.
#
# With a bucket, every case is also run from S3. A model that is missing there is uploaded from MODELS_DIR,
# and the bucket is made when it does not exist. The endpoint and credentials come from the environment
# (AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID, ...). Without a bucket only the local disk is used.
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
set -euo pipefail

# case, Hugging Face repository, revision, *.safetensors files, their size in GiB.
# The revision is pinned so that every run loads the same bytes.
CASES=(
    "gpu        Qwen/Qwen3-32B                     9216db5781bf21249d130ec9da846c4624c16137   17   62"
    "multi_gpu  deepseek-ai/DeepSeek-R1            56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad  163  642"
    "multi_gpu  runrunway/Kimi-K3-REAP-448experts  d2aff7364ad7c5e3cdcb7458d959aee2a2b079c9   96  780"
    "cpu        runrunway/Kimi-K3-REAP-448experts  d2aff7364ad7c5e3cdcb7458d959aee2a2b079c9   96  780"
)

usage() {
    sed -n '5,20p' "$0" | sed 's/^# \{0,1\}//'
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

seaweedfs() {  # whether the endpoint is a SeaweedFS S3 gateway, which says so in its Server header
    local endpoint=${AWS_ENDPOINT_URL_S3:-${AWS_ENDPOINT_URL:-${S3_ENDPOINT_URL:-}}}
    [ -n "$endpoint" ] && curl -s -m 10 -D - -o /dev/null "$endpoint" 2> /dev/null | grep -qi '^server: *seaweedfs'
}

here() {  # whether MODELS_DIR has all of a model's files
    [ "$(find "$MODELS_DIR/$1" -maxdepth 1 -name '*.safetensors' 2> /dev/null | wc -l)" -eq "$2" ]
}

there() {  # whether the bucket has all of a model's files
    [ -n "$S3_ROOT" ] && [ "$(s5 ls "$S3_ROOT/$1/*.safetensors" 2> /dev/null | wc -l)" -eq "$2" ]
}

# What this machine will run, and whether MODELS_DIR has room for what has to be fetched first.
cases=() need=0 scratch=0 missing=
for entry in "${CASES[@]}"; do
    read -r case repo revision files gib <<< "$entry"
    if [ "$case" = multi_gpu ] && [ "$NPROC" -lt 2 ]; then
        echo "leaving ${repo##*/} out: it is loaded to several GPUs and this machine has $NPROC"
        continue
    fi
    cases+=("$entry")
    if ! here "${repo##*/}" "$files" && [ "${missing#* "${repo##*/}" }" = "$missing" ]; then
        need=$((need + gib))
        missing+=" ${repo##*/} (${gib} GiB)"
    fi
    if [ -n "$S3_ROOT" ] && [ "$gib" -gt "$scratch" ]; then
        scratch=$gib  # the loaders that read only local files download a model from S3 before loading it
    fi
done
free=$(($(df --output=avail -B1 "$MODELS_DIR" | tail -1) / 1073741824))
if [ $((need + scratch)) -gt "$free" ]; then
    echo "not enough room in $MODELS_DIR: $free GiB free, $((need + scratch)) GiB needed" >&2
    [ -z "$missing" ] || echo "    to download:$missing" >&2
    [ "$scratch" = 0 ] || echo "    for downloads from S3 during the run: $scratch GiB" >&2
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
    else
        echo "downloading $repo from Hugging Face to $dir"
        HF_HUB_OFFLINE=0 hf download "$repo" --revision "$revision" --include '*.safetensors' \
            --local-dir "$dir.partial" > /dev/null
        rm -rf "$dir.partial/.cache"
    fi
    mkdir -p "$dir"
    mv "$dir.partial"/*.safetensors "$dir/"
    rm -rf "$dir.partial"
}

upload() {  # make the bucket hold a model, from MODELS_DIR; the bucket is made when it does not exist
    local name=$1 files=$2 bucket
    there "$name" "$files" && return
    bucket=$(echo "$S3_ROOT" | cut -d/ -f1-3)
    if ! s5 ls "$bucket" > /dev/null 2>&1; then
        echo "making $bucket"
        s5 mb "$bucket" > /dev/null
    fi
    echo "uploading $name to $S3_ROOT/$name"
    if seaweedfs; then
        # SeaweedFS compresses the parts of a multipart upload when they shrink, and then has to
        # decompress them on every read. A file sent in one request, with a content type it takes for
        # already compressed, is stored as it is.
        s5 --numworkers 16 cp -p 20000 --content-type application/zstd "$MODELS_DIR/$name/*.safetensors" \
            "$S3_ROOT/$name/" > /dev/null
    else
        s5 cp "$MODELS_DIR/$name/*.safetensors" "$S3_ROOT/$name/" > /dev/null
    fi
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

how() {  # bench.py's options for a case
    case $1 in
        gpu) echo "--device cuda" ;;
        cpu) echo "--device cpu" ;;
        multi_gpu) echo "--nproc $NPROC" ;;
    esac
}

for entry in ${cases[@]+"${cases[@]}"}; do
    read -r case repo revision files gib <<< "$entry"
    fetch "$repo" "$revision" "$files"
    [ -z "$S3_ROOT" ] || upload "${repo##*/}" "$files"
done
for source in disk ${S3_ROOT:+s3}; do  # every case from local disk, then every case from S3
    for entry in ${cases[@]+"${cases[@]}"}; do
        read -r case repo revision files gib <<< "$entry"
        name=${repo##*/}
        # shellcheck disable=SC2046  # the case's options are split on purpose
        if [ "$source" = disk ]; then
            bench "${name}_$case" "$MODELS_DIR/$name" $(how "$case") "$@"
        else
            bench "${name}_${case}_s3" "$S3_ROOT/$name" $(how "$case") --ref-dir "$MODELS_DIR/$name" \
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
    PYTHONPATH=$HERE${PYTHONPATH:+:$PYTHONPATH} "$PYTHON" -m streamer_bench.markdown "${results[@]}" \
        --out "$OUT/report.md"
fi
if [ ${#failed[@]} -gt 0 ]; then
    echo "failed: ${failed[*]}" >&2
    exit 1
fi
