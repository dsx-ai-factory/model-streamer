# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# run_dist_tests.py
import unittest
import torch.distributed as dist

def run_tests():
    # 1. GLOBAL SETUP: Initialize the process group ONCE
    dist.init_process_group("gloo")
    rank = dist.get_rank()

    # 2. DISCOVER & RUN TESTS: This already sorts alphabetically by default
    loader = unittest.TestLoader()
    suite = loader.discover("runai_model_streamer", pattern="dist_test_*.py")
    
    # Run the discovered tests in the default (alphabetical) order
    runner = unittest.TextTestRunner()
    runner.run(suite)

    # 3. GLOBAL TEARDOWN: Destroy the process group ONCE
    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    run_tests()