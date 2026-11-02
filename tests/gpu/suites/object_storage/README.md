# The object storage suite

Features of the streamer that need an object store, tested end to end against a SeaweedFS service that the environment
starts. It is also the example of a suite that is **not** about the content of a model: each test is one feature, seen from
the outside, with a small file that the test writes. `tests/gpu/README.md`, "Adding a test", says how to write another one
by copying this.

## What it tests

| Test | Feature |
|---|---|
| `test_the_service_keeps_what_is_put_in_it` | the environment works: an object that is put in the bucket is read back, and listed |
| `test_stream_a_safetensors_file_from_a_bucket_to_the_gpu` | `SafetensorsStreamer.stream_file("s3://...")`: the names, dtypes, shapes and the bytes (CRC32) of what arrives on the GPU are what was put there |
| `test_a_file_of_a_single_tensor_streams_too` | the end of an object that is not a multiple of a page |
| `test_list_safetensors_finds_the_safetensors_files_under_a_prefix_and_nothing_else` | `list_safetensors` on a prefix: the `.safetensors` files, and not the other objects |
| `test_pull_files_downloads_the_files_of_a_prefix` | `pull_files`: the files of a prefix arrive in a directory, byte for byte |
| `test_streaming_an_object_that_is_not_there_is_an_error` | a missing object raises, it is not an empty model and not a hang |
| `test_a_file_that_is_not_safetensors_is_an_error` | an object that is not a safetensors file raises |

Each test works under a prefix of its own in the bucket (`ObjectStore.new_prefix`) and removes it afterwards, so tests do
not see each other's objects. The tests are an ordinary `unittest` module of one process: no `torchrun`, no ranks.
`suite.sh` runs it with `run_unittest`, which writes the results of every test in the format of every suite.

## What it needs, and how it gets it

- **The service**: the `seaweedfs` service of `tests/gpu/compose.yaml`, behind the profile `object_storage` (the name of
  the suite, which is how `run.sh` knows to start it): SeaweedFS in its single-process mode, one bucket (`test-bucket`) and
  one pair of test keys (`harness/seaweedfs-s3.json`), with a healthcheck that the container waits for. The `tests`
  container reaches it through `AWS_ENDPOINT_URL=http://seaweedfs:9000` and the keys, which is how the streamer and the
  harness find it. `run.sh` starts it when the suite is selected, and removes it with the run. `suite_prepare` waits
  until the bucket answers, and says how to get the service if the environment has none.
- **The S3 plugin of the streamer**: object storage is a package of its own (`runai-model-streamer-s3`), built apart from
  the streamer (`COMPONENT=s3` of the root `Makefile`). A wheel of it in `/wheel` next to the streamer's is installed;
  without one, `GPU_S3_SPEC` (`--s3-spec`) or else the newest on PyPI, with a warning that it may not be the version
  of the streamer under test.
- **boto3**, the client that puts the objects: installed by `suite_prepare` (it is not a dependency of the streamer).
- **A GPU**: the tests stream to `cuda:0`, and fail when there is none, like every GPU test. `GPU_DEVICE=cpu` makes them stream
  to the CPU, for running the test module by hand (see below) on a machine without a GPU: the environment itself
  (`run.sh`, `setup_and_run.sh`) refuses to run without one.

## Running it

```bash
# a released streamer and plugin, from PyPI, on one GPU
tests/gpu/run.sh --suites object_storage --streamer-spec 'runai-model-streamer==0.17.0' \
    --s3-spec 'runai-model-streamer-s3==0.17.0' --gpus 1

# wheels that you built: the streamer and the S3 plugin
tests/gpu/run.sh --suites object_storage --wheel dist/runai_model_streamer-*.whl \
    --wheel dist/runai_model_streamer_s3-*.whl

# together with the streaming suite
tests/gpu/run.sh --suites streaming,object_storage --wheel ... --wheel ...
```

By hand (to work on the tests without a GPU, with `GPU_DEVICE=cpu`): start the service, run the test module in a container on
the compose network that has the streamer, its S3 plugin and boto3 installed, and remove the service. Every compose command
needs `GPU_WHEEL_HOST_DIR` and `GPU_REPO` set (the compose file reads them), and the profile of the service:

```bash
export GPU_WHEEL_HOST_DIR=/tmp GPU_REPO=$PWD
docker compose -p os-try -f tests/gpu/compose.yaml --profile object_storage up -d --wait seaweedfs

docker run --rm --network os-try_default -v "$PWD:/workspace" -w /workspace \
  -e PYTHONPATH=tests/gpu/harness -e GPU_DEVICE=cpu -e AWS_ENDPOINT_URL=http://seaweedfs:9000 \
  -e AWS_ACCESS_KEY_ID=testkey -e AWS_SECRET_ACCESS_KEY=testsecret -e RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 \
  IMAGE-WITH-THE-STREAMER-AND-BOTO3 \
  python3 -m gpu_harness.run_unittest --suite object_storage --run basic tests/gpu/suites/object_storage/test_object_storage.py

docker compose -p os-try -f tests/gpu/compose.yaml --profile object_storage down --volumes
```

## In CI

The nightly runs the streaming suite only. A manual run of the workflow takes `suites=object_storage` (or
`streaming,object_storage`): the build job then also builds the S3 plugin wheel (`COMPONENT=s3`) and uploads it next to
the streamer's, the gpu job downloads both into `/wheel`, and `run.sh` starts the `seaweedfs` service with the suite. The
suite runs in the newest-torch leg only: the torch version is not what it tests.
