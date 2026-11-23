#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed streamer inside an inference engine's environment.

Run inside an official engine image (vLLM or SGLang) after the streamer wheel
under test was installed there (tests/engine_compatibility/run.sh does both). It
checks what a plain `pip install` cannot:

1. streaming a tiny generated safetensors file with the call shape the engines
   use, which dlopens the native libstreamer.so - the step that catches a
   glibc/ABI mismatch between the wheel's build toolchain and the engine image;
2. importing the engine itself afterwards, which catches the streamer's
   install having replaced a dependency with a version the engine cannot use;
3. for vLLM, streaming the same file through vLLM's own
   runai_safetensors_weights_iterator - the engine's real integration, on CPU.

The safetensors file is generated here, so the check needs no network and no
GPU. Exits non-zero on the first failure, with the failing step named.
"""

import sys
import tempfile

ENGINE_MODULES = {
    "vllm": "vllm",
    "sglang": "sglang",
}

TENSOR_NAMES = ["a", "b"]


def fail(step: str, error: BaseException) -> None:
    print(f"ENGINE-COMPATIBILITY FAIL [{step}]: {type(error).__name__}: {error}")
    if isinstance(error, OSError) and "GLIBC" in str(error):
        print(
            "ENGINE-COMPATIBILITY HINT: libstreamer.so needs glibc symbols newer than the engine "
            "image ships - the wheel's build toolchain is too new for this environment."
        )
    sys.exit(1)


def write_model(directory: str):
    import torch
    from safetensors.torch import save_file

    tensors = {
        "a": torch.arange(64, dtype=torch.float32).reshape(8, 8),
        "b": torch.ones(16, dtype=torch.int64),
    }
    path = f"{directory}/model.safetensors"
    save_file(tensors, path)
    return path, tensors


def stream_roundtrip(path: str, tensors) -> None:
    import torch

    # Importing the package does not load the native library; streaming does.
    from runai_model_streamer import SafetensorsStreamer

    streamed = {}
    # The call shape vLLM's and SGLang's loaders use.
    with SafetensorsStreamer() as streamer:
        streamer.stream_files([path], device="cpu", is_distributed=False)
        for name, tensor in streamer.get_tensors():
            streamed[name] = tensor.clone()
    if not streamer.files_to_tensors_metadata:
        raise AssertionError("files_to_tensors_metadata is empty after streaming")
    if set(streamed) != set(tensors):
        raise AssertionError(f"streamed tensors {sorted(streamed)} != written {sorted(tensors)}")
    for name, expected in tensors.items():
        if not torch.equal(streamed[name], expected):
            raise AssertionError(f"tensor {name!r} does not match after streaming")
    print("ENGINE-COMPATIBILITY OK [stream]: streamed 2 tensors through libstreamer.so")


def import_engine(engine: str) -> None:
    import importlib

    module = importlib.import_module(ENGINE_MODULES[engine])
    version = getattr(module, "__version__", "unknown")
    print(f"ENGINE-COMPATIBILITY OK [engine-import]: {engine} {version} imports after the install")


def engine_loader(engine: str, path: str) -> None:
    if engine != "vllm":
        print(
            f"ENGINE-COMPATIBILITY SKIP [engine-loader]: {engine} has no public weights iterator; "
            "the stream step mirrors its loader's call shape"
        )
        return
    from vllm.model_executor.model_loader.weight_utils import runai_safetensors_weights_iterator

    names = sorted(name for name, _ in runai_safetensors_weights_iterator([path], False))
    if names != TENSOR_NAMES:
        raise AssertionError(f"vLLM's iterator yielded {names}, expected {TENSOR_NAMES}")
    print("ENGINE-COMPATIBILITY OK [engine-loader]: vLLM's weights iterator streamed the file")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ENGINE_MODULES:
        print(f"usage: {sys.argv[0]} {{{'|'.join(ENGINE_MODULES)}}}")
        sys.exit(2)
    engine = sys.argv[1]

    with tempfile.TemporaryDirectory() as tmp:
        try:
            path, tensors = write_model(tmp)
            stream_roundtrip(path, tensors)
        except BaseException as error:  # noqa: B036 - report and exit, whatever broke
            fail("stream", error)
        try:
            import_engine(engine)
        except BaseException as error:  # noqa: B036
            fail("engine-import", error)
        try:
            engine_loader(engine, path)
        except BaseException as error:  # noqa: B036
            fail("engine-loader", error)
    print("ENGINE-COMPATIBILITY PASS")


if __name__ == "__main__":
    main()
