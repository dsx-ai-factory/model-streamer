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

## Development

Our repository is built using devcontainer ([Further reading](https://containers.dev/))

The devcontainer uses Python 3.11 on Debian Bookworm, with GCC toolchains for
both x86_64 and ARM64. Python and pip come from the official Python image;
native SDKs are built separately in the Dockerfile. Python 3.11 also supports
the existing NumPy 1.24.4 test dependency.

Wheels built in this environment use `manylinux_2_36_x86_64` and
`manylinux_2_36_aarch64` tags and require glibc 2.36 or newer. This raises the
Linux compatibility baseline: these wheels cannot be installed on older
distributions such as Ubuntu 20.04 or 22.04. Building for older distributions
requires an older build environment and validation of its native dependencies;
changing the wheel tag alone does not provide compatibility.

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
> Builds using `USE_SYSTEM_LIBS=1` require the matching system curl and OpenSSL
> libraries at runtime (`libcurl4` and `libssl3` on Debian Bookworm).
