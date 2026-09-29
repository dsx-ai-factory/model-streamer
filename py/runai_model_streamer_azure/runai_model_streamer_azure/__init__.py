# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Azure Blob Storage support for Run:ai Model Streamer

import logging
import os

# Configure Azure SDK logger levels.
# The Azure SDK can be very verbose at INFO level (logging every HTTP request/response),
# so default to WARNING. Users can override by setting environment variables.
_azure_log_level = os.environ.get("AZURE_LOG_LEVEL", "WARNING").upper()
logging.getLogger("azure").setLevel(_azure_log_level)