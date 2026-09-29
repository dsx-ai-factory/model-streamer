# Model Streamer

> [!IMPORTANT]
> **Repository move and rename notice:** On September 13, 2026, Run:ai Model Streamer was
> renamed to Model Streamer and the repository moved from the run-ai GitHub organization to
> [`dsx-ai-factory/model-streamer`](https://github.com/dsx-ai-factory/model-streamer).
> The `runai-model-streamer` PyPI packages and Python import paths are unchanged.
> Existing repository URLs and standard Git operations continue to work through
> GitHub redirects. If you maintain automation or integrations that reference
> `run-ai/runai-model-streamer`, such as GitHub Actions, webhooks, or pinned
> repository URLs, update them to `dsx-ai-factory/model-streamer`.


## Overview
Model Streamer is a Python SDK designed to facilitate the streaming of tensors from tensors files to GPU memory with concurrency and streaming. It provides an API for loading SafeTensors files and building AI models, allowing loading models seamlessly.

For documentation click [here](docs/README.md)

For benchmarks click [here](docs/src/benchmarks.md)

## Usage
Install the package
```
pip install runai-model-streamer
```

And stream tensors
```
from runai_model_streamer import SafetensorsStreamer

file_path = "model.safetensors"

with SafetensorsStreamer() as streamer:
    streamer.stream_file(file_path)
    for name, tensor in streamer.get_tensors():
        gpu_tensor = tensor.to('CUDA:0')
```

## Licensing and Contributions

NVIDIA-authored code is released under the [Apache License 2.0](LICENSE).
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and the
required Developer Certificate of Origin sign-off.

[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) contains third-party
attributions and license texts. Third-party code retains its upstream terms;
in particular, the glibc-derived function in
[portability.c](cpp/cc/portability/portability.c) retains its LGPL notice.

The Apache 2.0 license links in
[py/runai_model_streamer](py/runai_model_streamer/LICENSE),
[py/runai_model_streamer_s3](py/runai_model_streamer_s3/LICENSE),
[py/runai_model_streamer_gcs](py/runai_model_streamer_gcs/LICENSE), and
[py/runai_model_streamer_azure](py/runai_model_streamer_azure/LICENSE)
point to the root license so each independently distributed package includes
the same license text. Each package also includes the third-party notices.

Maintainers should follow the [release compliance checklist](docs/release-compliance.md)
when updating dependencies or preparing a distribution.

## Development

Our repository is built using devcontainer ([Further reading](https://containers.dev/))

The following commands should run inside the dev container

> [!NOTE]
> You can use devcontainer-cli tool ([Further reading](https://github.com/devcontainers/cli)) by installing it, and run every command with the following prefix `devcontainer exec --workspace-folder . [COMMAND]`

**Build**
```
make build
```

> [!NOTE]
> We build `libstreamers3.so` and statically link it to libssl, libcrypto, and libcurl. if you would like to use your system libraries by dynamic link to them, run `USE_SYSTEM_LIBS=1 make build`

> [!NOTE]
> On successful build, the `.whl` file would be at `py/runai_model_streamer/dist/<PACKAGE_FILE>` and `py/runai_model_streamer_s3/dist/<PACKAGE_FILE>`


**Run tests**
```
make test
```

**Install locally**
```
pip3 install py/runai_model_streamer py/runai_model_streamer_s3
```

> [!IMPORTANT]
> In order to the CPP to run, you need to install libcurl4 and libssl1.1_1
