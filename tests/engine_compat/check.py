#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed streamer inside an inference engine's environment.

Run inside an official engine image (vLLM or SGLang) after the streamer wheel
under test was installed there (tests/engine_compat/run.sh does both). It
checks what a plain `pip install` cannot:

1. importing the package and streaming a tiny safetensors file, which dlopens
   the native libstreamer.so - the step that catches a glibc/ABI mismatch
   between the wheel's build toolchain and the engine image;
2. importing the engine itself afterwards, which catches the streamer's
   install having replaced a dependency with a version the engine cannot use.

The safetensors file is generated here, so the check needs no network and no
GPU. Exits non-zero on the first failure, with the failing step named.
"""

import sys
import tempfile

ENGINE_MODULES = {
    "vllm": "vllm",
    "sglang": "sglang",
}


def fail(step: str, error: BaseException) -> None:
    print(f"ENGINE-COMPAT FAIL [{step}]: {type(error).__name__}: {error}")
    if isinstance(error, OSError) and "GLIBC" in str(error):
        print(
            "ENGINE-COMPAT HINT: libstreamer.so needs glibc symbols newer than the engine "
            "image ships - the wheel's build toolchain is too new for this environment."
        )
    sys.exit(1)


def stream_roundtrip() -> None:
    import torch
    from safetensors.torch import save_file

    # Importing the package does not load the native library; streaming does.
    from runai_model_streamer import SafetensorsStreamer

    tensors = {
        "a": torch.arange(64, dtype=torch.float32).reshape(8, 8),
        "b": torch.ones(16, dtype=torch.int64),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/model.safetensors"
        save_file(tensors, path)
        streamed = {}
        with SafetensorsStreamer() as streamer:
            streamer.stream_file(path)
            for name, tensor in streamer.get_tensors():
                streamed[name] = tensor.clone()
    if set(streamed) != set(tensors):
        raise AssertionError(f"streamed tensors {sorted(streamed)} != written {sorted(tensors)}")
    for name, expected in tensors.items():
        if not torch.equal(streamed[name], expected):
            raise AssertionError(f"tensor {name!r} does not match after streaming")
    print("ENGINE-COMPAT OK [stream]: streamed 2 tensors through libstreamer.so")


def import_engine(engine: str) -> None:
    import importlib

    module = importlib.import_module(ENGINE_MODULES[engine])
    version = getattr(module, "__version__", "unknown")
    print(f"ENGINE-COMPAT OK [engine-import]: {engine} {version} imports after the install")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ENGINE_MODULES:
        print(f"usage: {sys.argv[0]} {{{'|'.join(ENGINE_MODULES)}}}")
        sys.exit(2)
    engine = sys.argv[1]

    try:
        stream_roundtrip()
    except BaseException as error:  # noqa: B036 - report and exit, whatever broke
        fail("stream", error)
    try:
        import_engine(engine)
    except BaseException as error:  # noqa: B036
        fail("engine-import", error)
    print("ENGINE-COMPAT PASS")


if __name__ == "__main__":
    main()
