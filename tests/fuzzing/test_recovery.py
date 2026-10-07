# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
import unittest

from runai_model_streamer.file_streamer import (FileStreamer, FileChunks)


class TestRecoveryAfterRangeError(unittest.TestCase):
    """A per-range error must not break the streamer for everything after it.

    Runs against the REAL library on purpose. The mock cannot express this: it serves every range
    synchronously from the calling thread, so a range is never genuinely in flight when the error fires
    and the streamer always appears to recover.

    Nothing here is filesystem specific. The drain lives in FileStreamer.get_chunks, above the
    backend, so an object-storage submission fails and recovers through the very same code - the filesystem
    is used only because it needs no emulator. Object storage differs in severity, not behaviour: its reads
    are genuinely in flight when the error fires, so more ranges are still writing into a buffer the next
    stream_files is about to free.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "data.bin")
        self.size = 4096
        with open(self.path, "wb") as f:
            f.write(bytes(i % 256 for i in range(self.size)))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _failing_request(self):
        # One range reads past EOF and fails; the others are valid, so the submission still owes their
        # responses when the fail-fast raise abandons the iteration.
        return [
            FileChunks(id=0, path=self.path, offsets=[0], sizes=[self.size * 100]),
            FileChunks(id=1, path=self.path, offsets=[0, 1024], sizes=[1024, 1024]),
        ]

    def _valid_request(self):
        return [FileChunks(id=0, path=self.path, offsets=[512], sizes=[512])]

    def test_streamer_recovers_after_a_range_error(self):
        with FileStreamer() as streamer:
            streamer.stream_files(self._failing_request())
            with self.assertRaisesRegex(ValueError, "End of file"):
                for _ in streamer.get_chunks():
                    pass

            # Several times, not once: before the fix each retry abandoned its OWN responses in turn, so the
            # backlog sustained itself and the streamer never recovered. A single retry would also fail
            # without the fix, but only repetition shows recovery is durable rather than a one-off catch-up.
            for attempt in range(3):
                streamer.stream_files(self._valid_request())
                delivered = [
                    bytes(tensor.numpy().tobytes()) for _path, _index, tensor in streamer.get_chunks()
                ]
                self.assertEqual(
                    delivered,
                    [bytes(i % 256 for i in range(512, 1024))],
                    f"retry {attempt} returned the wrong bytes",
                )

    def test_streamer_recovers_after_an_abandoned_iteration(self):
        # The other way out of the generator: the caller stops early with no error at all. Closing the
        # generator has to drain just as a raise does - safetensors_pytorch raises inside its own
        # `for ... in get_chunks()` loop, which is this same exit.
        with FileStreamer() as streamer:
            streamer.stream_files(
                [FileChunks(id=0, path=self.path, offsets=[0, 1024, 2048], sizes=[1024, 1024, 1024])]
            )
            for _path, _index, _tensor in streamer.get_chunks():
                break   # two ranges left unconsumed

            streamer.stream_files(self._valid_request())
            delivered = [
                bytes(tensor.numpy().tobytes()) for _path, _index, tensor in streamer.get_chunks()
            ]
            self.assertEqual(delivered, [bytes(i % 256 for i in range(512, 1024))])


if __name__ == "__main__":
    unittest.main()
