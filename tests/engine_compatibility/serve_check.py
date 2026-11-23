#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve a small real model through the engine with load_format=runai_streamer, and ask it for a
completion.

Run inside an official engine image (vLLM or SGLang) on a GPU of compute capability 8.0 or newer,
after the streamer wheel under test was installed there (tests/engine_compatibility/run.sh --serve
does both). This is the end-to-end check: the engine's own serve command loads real weights through
the installed streamer, the server comes up, and one completion comes back. The server log must
show the streamer's marker - an engine that quietly fell back to its default loader fails, because
then nothing was tested.

The model is Qwen/Qwen2.5-0.5B at the revision the GPU harness pins
(tests/gpu/suites/streaming/real_model.py), snapshot-downloaded here because serving needs the
config and tokenizer files too, not only the weights.
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

# The repository and revision of tests/gpu/suites/streaming/real_model.py, duplicated: this
# directory follows the GPU suites' no-cross-imports rule. Apache-2.0, not gated, ~1 GiB.
MODEL_REPO = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"

PORT = 8000
BASE_URL = f"http://127.0.0.1:{PORT}"
LOG_PATH = "/tmp/engine-compatibility-server.log"
DOWNLOAD_ATTEMPTS = 3
HEALTH_DEADLINE_SECONDS = 600
STREAMER_MARKER = "[RunAI Streamer]"

SERVER_COMMANDS = {
    "vllm": lambda model_dir: [
        "vllm", "serve", model_dir,
        "--load-format", "runai_streamer",
        "--max-model-len", "1024",
        "--gpu-memory-utilization", "0.85",
        "--port", str(PORT),
    ],
    "sglang": lambda model_dir: [
        "python3", "-m", "sglang.launch_server",
        "--model-path", model_dir,
        "--load-format", "runai_streamer",
        "--mem-fraction-static", "0.8",
        "--host", "127.0.0.1",
        "--port", str(PORT),
    ],
}


def say(line: str) -> None:
    print(line, flush=True)


def fail(step: str, message: str) -> None:
    say(f"ENGINE-COMPATIBILITY FAIL [{step}]: {message}")
    sys.exit(1)


def download_model(target: str) -> str:
    from huggingface_hub import snapshot_download  # the engine images ship it

    last_error = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            path = snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_dir=target)
            say(f"ENGINE-COMPATIBILITY OK [model]: {MODEL_REPO}@{MODEL_REVISION[:7]} in {path}")
            return path
        except Exception as error:  # noqa: BLE001 - retried, then reported
            last_error = error
            say(f"model download attempt {attempt} of {DOWNLOAD_ATTEMPTS} failed: {error}")
            time.sleep(5 * attempt)
    fail("model", f"could not download {MODEL_REPO}: {last_error}")


def http_json(url: str, payload=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body.strip() else {}


def tail_log(lines: int = 150) -> None:
    try:
        with open(LOG_PATH, errors="replace") as handle:
            content = handle.readlines()
    except OSError:
        return
    say(f"--- last {min(lines, len(content))} lines of the server log ---")
    for line in content[-lines:]:
        say(line.rstrip("\n"))
    say("--- end of the server log ---")


def glibc_hint() -> None:
    try:
        with open(LOG_PATH, errors="replace") as handle:
            if "GLIBC" in handle.read():
                say(
                    "ENGINE-COMPATIBILITY HINT: libstreamer.so needs glibc symbols newer than the "
                    "engine image ships - the wheel's build toolchain is too new for this environment."
                )
    except OSError:
        pass


def wait_for_health(process: subprocess.Popen) -> None:
    deadline = time.monotonic() + HEALTH_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail_log()
            glibc_hint()
            fail("serve", f"the server exited with status {process.returncode} before it was healthy")
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as response:
                if response.status == 200:
                    say("ENGINE-COMPATIBILITY OK [health]: the server is up")
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(5)
    tail_log()
    fail("serve", f"the server was not healthy within {HEALTH_DEADLINE_SECONDS}s")


def completion() -> None:
    models = http_json(f"{BASE_URL}/v1/models")
    model_id = models["data"][0]["id"]
    answer = http_json(
        f"{BASE_URL}/v1/completions",
        {"model": model_id, "prompt": "The capital of France is", "max_tokens": 8},
        timeout=120,
    )
    text = answer["choices"][0]["text"]
    if not text.strip():
        raise AssertionError(f"the completion came back empty: {answer}")
    say(f"ENGINE-COMPATIBILITY OK [completion]: {text.strip()!r}")


def check_streamer_marker() -> None:
    with open(LOG_PATH, errors="replace") as handle:
        log = handle.read()
    if STREAMER_MARKER not in log:
        fail(
            "loader",
            f"the server log has no {STREAMER_MARKER!r} line: the engine did not load the weights "
            "through the streamer (a silent fallback to its default loader?)",
        )
    say("ENGINE-COMPATIBILITY OK [loader]: the weights went through the streamer")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in SERVER_COMMANDS:
        say(f"usage: {sys.argv[0]} {{{'|'.join(SERVER_COMMANDS)}}}")
        sys.exit(2)
    engine = sys.argv[1]

    model_dir = download_model(os.environ.get("MODEL_DIR", "/tmp/engine-compatibility-model"))

    environment = dict(os.environ)
    environment["RUNAI_STREAMER_LOG_TO_STDERR"] = "1"  # the streamer's marker must reach the log
    command = SERVER_COMMANDS[engine](model_dir)
    say(f"starting: {' '.join(command)}")
    with open(LOG_PATH, "wb") as log_file:
        process = subprocess.Popen(
            command, stdout=log_file, stderr=subprocess.STDOUT, env=environment, start_new_session=True
        )
        try:
            wait_for_health(process)
            try:
                completion()
            except Exception as error:  # noqa: BLE001
                tail_log()
                fail("completion", f"{type(error).__name__}: {error}")
            check_streamer_marker()
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
    say("ENGINE-COMPATIBILITY SERVE PASS")


if __name__ == "__main__":
    main()
