# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# run_dist_tests.py
import datetime
import faulthandler
import sys
import unittest

import torch.distributed as dist

# The suite finishes in seconds when the ranks stay in step, so a collective still waiting after two
# minutes is a desynced pair, not a slow one. gloo's 30-minute default meant one desync cost the CI
# job 40 minutes of cascading timeouts before anything was reported.
PROCESS_GROUP_TIMEOUT = datetime.timedelta(seconds=120)

# While a stall is in progress, periodically dump every thread's stack to stderr, on every rank. The
# timeout that eventually fires reports only the rank that gave up; these dumps are the only record
# of where the OTHER rank - the one that caused the desync - was sitting.
WATCHDOG_INTERVAL_SECONDS = 90


def run_tests():
    faulthandler.dump_traceback_later(WATCHDOG_INTERVAL_SECONDS, repeat=True, file=sys.stderr)

    # 1. GLOBAL SETUP: Initialize the process group ONCE
    dist.init_process_group("gloo", timeout=PROCESS_GROUP_TIMEOUT)

    # 2. DISCOVER & RUN TESTS: This already sorts alphabetically by default
    loader = unittest.TestLoader()
    suite = loader.discover("runai_model_streamer", pattern="dist_test_*.py")

    # failfast: once one collective desyncs, every later test errors against the same closed pair,
    # so each further "failure" is the same incident again and costs another timeout wait.
    runner = unittest.TextTestRunner(failfast=True)
    result = runner.run(suite)

    faulthandler.cancel_dump_traceback_later()

    # 3. GLOBAL TEARDOWN: Destroy the process group ONCE
    dist.barrier()
    dist.destroy_process_group()
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(run_tests())
