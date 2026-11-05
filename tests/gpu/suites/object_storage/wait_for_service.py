#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Waits until the object storage service of the environment answers.

    wait_for_service.py

The endpoint, the keys and the bucket come from the environment (AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY, GPU_OBJECT_BUCKET). Exits with status 1 and says how to get a service when there is none.
"""
import sys

from gpu_harness.object_store import ObjectStore, ObjectStoreNotAvailable


def main():
    try:
        store = ObjectStore.from_environment()
        store.wait_until_ready(seconds=90)
        print("object storage: %s, bucket %s" % (store.endpoint, store.bucket))
    except ObjectStoreNotAvailable as error:
        print("ERROR: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
