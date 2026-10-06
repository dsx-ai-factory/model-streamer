# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from runai_model_streamer.safetensors_streamer.safetensors_streamer import (
    SafetensorsStreamer,
    ObjectStorageModel,
    list_safetensors,
    pull_files,
)
from runai_model_streamer.file_streamer.file_streamer import FileStreamer
from runai_model_streamer.file_streamer.requests_iterator import FileChunks
from runai_model_streamer.distributed_streamer.distributed_streamer import DistributedStreamer

import os
import sys
import logging

__all__ = [
    "SafetensorsStreamer",
    "ObjectStorageModel",
    "DistributedStreamer",
    "FileStreamer",
    "FileChunks",
    "list_safetensors",
    "pull_files",
]

logger_level = os.environ.get("RUNAI_STREAMER_LOG_LEVEL", "INFO").upper()

_logger = logging.getLogger(__name__)
_logger.setLevel(logger_level)

# Honor RUNAI_STREAMER_LOG_TO_STDERR for the Python logs too - the C++ core already routes its own logs to
# stderr on this flag. Without a handler here, the runai_model_streamer logger has none, so under hosts like
# vllm (which configure only their own logger namespace, not root) our INFO records hit logging's WARNING-only
# lastResort and are dropped. Attaching our own stderr handler and disabling propagation makes the Python logs
# appear on stderr and avoids double-logging if the host also configures a root handler. Gated on the flag so
# that, when it is unset, we keep relying on the host's logging config (no behavior change).
if os.environ.get("RUNAI_STREAMER_LOG_TO_STDERR") == "1" and not any(
    getattr(h, "_runai_stderr_handler", False) for h in _logger.handlers
):
    _handler = logging.StreamHandler(sys.stderr)
    _handler._runai_stderr_handler = True   # marker so a re-import does not add a second handler
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _logger.addHandler(_handler)
    _logger.propagate = False