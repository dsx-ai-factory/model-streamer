# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import tempfile
import shutil
import os
import random
from runai_model_streamer.file_streamer import (FileStreamer, FileChunks)
from runai_model_streamer.file_streamer.requests_iterator import (
    RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME,
)

MIN_NUM_FILES = 1
MAX_NUM_FILES = 20
MIN_CHUNK_NUM = 1
MAX_CHUNK_NUM = 500
# 0, not 16: a zero sized range is legal and is what a zero element tensor produces, so the fuzzer has to
# be able to generate one. It reaches no storage yet still owes exactly one response, which is where
# off-by-one indexing shows up - and at 0 it lands anywhere in a file's list, including first and last.
MIN_CHUNK_SIZE = 0
# Raised from 2048 so the fuzzer moves enough data to fill the engine's window. Measured on one run
# under io_uring_direct, with everything else unchanged:
#
#            bytes read   bounced   deepest queue   avg outstanding
#     2048      1.78 MB    2%             8              3.5
#    16384     40.60 MB    0%            23              8.3
#
# Depth is the point. At 2048 the whole run finished with at most 8 reads in flight, so nothing
# exercised a full window, the re-stage path, or the reap loop under pressure - the parts most likely
# to be wrong.
#
# NOT about O_DIRECT alignment, which was the first guess and the measurement says otherwise. Reads
# are sized by RUNAI_STREAMER_FS_CHUNK_BYTESIZE rather than by range, and the fuzzer's ranges are
# contiguous within a file, so they coalesce into large aligned reads at either setting. Bounces
# happen only at the edges, which is why 2% falls to 0% as the reads get bigger.
MAX_CHUNK_SIZE = 16384


def random_chunks():
    chunk_sizes = []
    content = b""
    for i in range(random.randint(MIN_CHUNK_NUM, MAX_CHUNK_NUM)):
        random_binary_content = os.urandom(
            random.randint(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE)
        )
        chunk_sizes.append(len(random_binary_content))
        content = content + random_binary_content
    return content, chunk_sizes


def random_memory_mode(chunks):
    # -1 unlimited (a single request holds everything), 0 largest-chunk (a request holds one range),
    # anything else is a byte limit. Only the latter two exercise the multi-request packing loop, the
    # per-file range bookkeeping across requests, and buffer reuse.
    #
    # chunks can legitimately be empty: a file whose generated chunk list has a single entry contributes
    # only an initial_offset and no ranges at all, so max()/sum() would have nothing to work with.
    memory_mode = random.choice([-1, 0, 1]) if chunks else -1
    if memory_mode == 1:
        # never below the largest range, or no request could hold it and packing would stall
        memory_limit = random.randint(max(chunks), sum(chunks))
    else:
        memory_limit = memory_mode
    os.environ[RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME] = str(memory_limit)

def random_file_chunks(i, dir):
    file_content, chunk_sizes = random_chunks()
    file_path = os.path.join(dir, f"test_file-{i}.txt")
    with open(file_path, "wb") as file:
        file.write(file_content)
        
    initial_offset = chunk_sizes[0]
    request_sizes = chunk_sizes[1:]

    expected_id_to_results = {}
    for j in range(len(request_sizes)):
        if j == len(request_sizes):
            expected_content = (
                file_content[sum(request_sizes[0:j]) :] + initial_offset
            )
        else:
            expected_content = file_content[
                sum(request_sizes[0:j])
                + initial_offset : sum(request_sizes[0 : j + 1])
                + initial_offset
            ]
        expected_id_to_results[j] = {
            "expected_content": expected_content,
        }
    return expected_id_to_results, FileChunks.contiguous(i, file_path, initial_offset, request_sizes)

class TestFuzzing(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def test_file_streamer(self):
        expected_file_to_id_to_results = {}
        file_to_file_chunks = {}
        files_chunks = []
        
        for i in range(random.randint(MIN_NUM_FILES, MAX_NUM_FILES)):
            expected_id_to_results, file_chunks = random_file_chunks(i, self.temp_dir)
            expected_file_to_id_to_results[file_chunks.id] = expected_id_to_results
            file_to_file_chunks[file_chunks.id] = file_chunks
            files_chunks.append(file_chunks)

        random_memory_mode([size for file_chunks in files_chunks for size in file_chunks.sizes])

        received = set()
        with FileStreamer() as fs:
            fs.stream_files(files_chunks)
            for file, id, dst in fs.get_chunks():
                # file is the unique integer identifier for the file chunks request
                self.assertLess(file, len(files_chunks))
                self.assertGreaterEqual(file, 0)

                file_chunks = file_to_file_chunks[file]
                expected_id_to_results = expected_file_to_id_to_results[file]
                self.assertEqual(
                    dst.numpy().tobytes(),
                    expected_id_to_results[id]["expected_content"],
                )
                received.add((file, id))

        # Checking the content of whatever arrived is not enough: a request that silently ends the
        # stream early (issue #157) delivers only correct chunks and would pass. Assert the exact set.
        expected = {
            (file_id, range_index)
            for file_id, results in expected_file_to_id_to_results.items()
            for range_index in results
        }
        self.assertEqual(received, expected)

    def tearDown(self):
        os.environ.pop(RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME, None)
        shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main()
