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

The devcontainer uses digest-pinned PyPA `manylinux_2_28` images (AlmaLinux 8,
GCC 14, Python 3.12.15). Python 3.12–3.14 are tested in CI; published packages
require Python 3.12 or newer. Linux wheels use `manylinux_2_28` tags (glibc 2.28 or newer), preserving
compatibility with Ubuntu 20.04 and 22.04
when a supported Python interpreter is installed.

Each container builds its native architecture: x86_64 on an x86_64 host and
aarch64 on an ARM64 host. CI builds and tests both on native runners. `make build`
builds all four packages for that architecture. To build the other architecture,
use a matching host or a Docker container with the corresponding `--platform`;
cross-compiling inside a single container is no longer supported.

Build and test dependencies, including transitive Python and npm packages, are
locked. Public package dependencies retain compatible ranges. See
[dependency pins and updates](.devcontainer/README.md) for the update procedure.

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
> Default builds statically link the cloud SDK dependencies. Builds using
> `USE_SYSTEM_LIBS=1` require matching system curl, OpenSSL, zlib, and libxml2
> libraries; their ABI requirements depend on the chosen build environment.
