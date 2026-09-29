# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import tempfile
import shutil
import os
import mmap
import ctypes
from runai_model_streamer.libstreamer.libstreamer import (
    FILE_ACCESS_ERROR_CODE,
    SUCCESS_ERROR_CODE,
    runai_start,
    runai_set_credentials,
    runai_request,
    runai_response,
)
from runai_model_streamer.s3_utils.s3_utils import (
    S3Credentials,
)


def buffer_address(buffer) -> int:
    # range_dsts are absolute addresses, so the destination buffer's base has to be taken explicitly -
    # the binding no longer accepts a Python buffer object and derives offsets from it
    return ctypes.addressof(ctypes.c_char.from_buffer(buffer))


class TestBindings(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def _drain(self, streamer, submission_id, num_ranges, expected_code=SUCCESS_ERROR_CODE):
        """Consume a submission's responses, asserting each carried expected_code, and return the range
        indices seen.

        Every accepted submission must be drained before its destination buffer goes out of scope:
        runai_request is non-blocking and range_dsts are raw addresses that keep nothing alive, so an
        in-flight submission can be writing into memory the test has already released. Draining also
        checks the submission_done flag is set exactly once, on the last response.

        expected_code is a parameter because a FAILING submission must complete just as strictly as a
        successful one - same count, same flag on the last response. That is the case where dropping a
        response is easiest and hangs the caller hardest.
        """
        seen = []
        for response_number in range(num_ranges):
            result = runai_response(streamer)
            self.assertIsNotNone(result)
            ret, sub_id, _file_index, range_index, submission_done = result
            self.assertEqual(ret, expected_code)
            self.assertEqual(sub_id, submission_id)
            seen.append(range_index)

            # Asserted against the response's POSITION, not merely counted: the flag is the caller's
            # signal that it may release the destination buffers, so setting it early is a
            # use-after-free, and a "set exactly once" count passes for exactly that case. Position, not
            # range index - responses may arrive in any order, and the flag belongs to whichever is
            # delivered last.
            self.assertEqual(
                submission_done,
                response_number == num_ranges - 1,
                f"submission_done must be set only on the last response, but response "
                f"{response_number + 1} of {num_ranges} reported {submission_done}",
            )
        return sorted(seen)

    def test_runai_library(self):
        # Multi-request API: submit returns a submission id, and each response reports its submission id,
        # file/range and whether the submission is complete. Also exercises runai_set_credentials (streamer-
        # scoped, unused for this filesystem read) on the second pass.
        file_path_1 = os.path.join(self.temp_dir, "test_file_1.txt")
        with open(file_path_1, "w") as file:
            file.write("XTest Text1TestText2Test-Text3\n")

        file_path_2 = os.path.join(self.temp_dir, "test_file_2.txt")
        with open(file_path_2, "w") as file:
            file.write("YTest Text44TestText")

        size = 60
        for use_credentials in (False, True):
            self._run_read(file_path_1, file_path_2, size, use_credentials)

    def _run_read(self, file_path_1, file_path_2, size, use_credentials):
        buffer = mmap.mmap(-1, size, mmap.MAP_ANONYMOUS | mmap.MAP_PRIVATE)
        buffer_ptr = id(buffer)

        streamer = runai_start()
        self.assertNotEqual(streamer, 0)

        if use_credentials:
            # credentials are streamer-scoped now: set them once on the streamer (unused for this
            # filesystem read, but exercises the set-credentials path)
            runai_set_credentials(streamer, S3Credentials(
                access_key_id="your_access_key",
                secret_access_key="your_secret_key",
                session_token="your_session_token",
                region_name="us-west-2",
                endpoint="optional_endpoint"))

        # Two files, two ranges each. Every range carries its own source offset and its own destination
        # address, so the layout below is stated outright rather than derived from a base offset plus
        # consecutive sizes: file 1 reads 10 bytes at 1 and 9 at 11, file 2 reads 11 at 1 and 8 at 12.
        base = buffer_address(buffer)
        submission_id = runai_request(
            streamer,
            [file_path_1, file_path_2],
            [2, 2],                                 # num_ranges per file
            [1, 11, 1, 12],                         # range_offsets, flat and grouped by file
            [10, 9, 11, 8],                         # range_sizes
            [base, base + 10, base + 19, base + 30] # range_dsts, absolute addresses
        )

        # collect all four sub-range responses (order across files is not guaranteed)
        expected = {(0, 0), (0, 1), (1, 0), (1, 1)}
        done_count = 0
        for _ in range(len(expected)):
            result = runai_response(streamer)
            self.assertIsNotNone(result)
            ret, sub_id, file_index, range_index, submission_done = result
            self.assertEqual(ret, SUCCESS_ERROR_CODE)
            self.assertEqual(sub_id, submission_id)
            self.assertIn((file_index, range_index), expected)
            expected.discard((file_index, range_index))
            if submission_done:
                done_count += 1
        self.assertEqual(expected, set())
        # submission_done is set exactly once - on the submission's last response
        self.assertEqual(done_count, 1)

        # verify the data landed in the shared buffer, copyless
        self.assertEqual(bytes(buffer[0:10]), b"Test Text1")
        self.assertEqual(bytes(buffer[10:19]), b"TestText2")
        self.assertEqual(bytes(buffer[19:30]), b"Test Text44")
        self.assertEqual(bytes(buffer[30:38]), b"TestText")
        self.assertEqual(id(buffer), buffer_ptr)

    def test_zero_sized_range_is_answered(self):
        # A zero-sized range is a range: it gets its own response, like any other. The caller's indexing
        # counts on exactly one response per range, so swallowing it would shift every later index.
        file_path = os.path.join(self.temp_dir, "zeros.txt")
        with open(file_path, "w") as file:
            file.write("abcde")

        buffer = mmap.mmap(-1, 64, mmap.MAP_ANONYMOUS | mmap.MAP_PRIVATE)
        base = buffer_address(buffer)

        streamer = runai_start()
        submission_id = runai_request(
            streamer, [file_path], [2], [0, 0], [0, 5], [base, base]
        )

        seen = []
        for _ in range(2):
            result = runai_response(streamer)
            self.assertIsNotNone(result)
            ret, sub_id, file_index, range_index, submission_done = result
            self.assertEqual(ret, SUCCESS_ERROR_CODE)
            self.assertEqual(sub_id, submission_id)
            seen.append(range_index)

        self.assertEqual(sorted(seen), [0, 1])
        self.assertEqual(bytes(buffer[0:5]), b"abcde")

    def test_scattered_ranges_and_destinations(self):
        # Ranges that are non-contiguous and unordered in BOTH the file and the destination buffer -
        # the shape the per-range API exists for. Every other test here happens to submit ranges that
        # tile one span, which a backend that seeks once per file and walks its destination forward
        # would serve correctly by accident; this one it cannot.
        file_path = os.path.join(self.temp_dir, "scattered.txt")
        with open(file_path, "w") as file:
            file.write("ABCDEFGHIJKLMNOPQRST")   # 20 bytes, every byte distinct

        buffer = mmap.mmap(-1, 64, mmap.MAP_ANONYMOUS | mmap.MAP_PRIVATE)   # zero filled
        base = buffer_address(buffer)

        # last bytes first, then the first bytes, then the middle; destinations in a third order again
        offsets = [15, 0, 8]
        sizes = [5, 3, 2]
        dsts = [base + 40, base + 10, base + 0]

        streamer = runai_start()
        submission_id = runai_request(streamer, [file_path], [3], offsets, sizes, dsts)

        seen = []
        for _ in range(3):
            result = runai_response(streamer)
            self.assertIsNotNone(result)
            ret, sub_id, file_index, range_index, submission_done = result
            self.assertEqual(ret, SUCCESS_ERROR_CODE)
            self.assertEqual(sub_id, submission_id)
            self.assertEqual(file_index, 0)
            seen.append(range_index)

        # one response per range, indexed as submitted
        self.assertEqual(sorted(seen), [0, 1, 2])

        self.assertEqual(bytes(buffer[40:45]), b"PQRST")
        self.assertEqual(bytes(buffer[10:13]), b"ABC")
        self.assertEqual(bytes(buffer[0:2]), b"IJ")

        # nothing was written outside the requested destinations
        self.assertEqual(bytes(buffer[2:10]), b"\x00" * 8)
        self.assertEqual(bytes(buffer[13:40]), b"\x00" * 27)
        self.assertEqual(bytes(buffer[45:64]), b"\x00" * 19)

    def test_zero_sized_ranges_on_unopenable_file_are_still_answered(self):
        # A zero-sized range reads nothing, but it does NOT skip the file: the streamer opens the file for
        # every batch before it looks at any range size, so a range in an unopenable file fails with it -
        # verified against the real library, which returns FileAccessError here.
        #
        # The point of the test is that the ranges are ANSWERED, not what they answer. The response counter
        # is raised per range regardless of size, so a range whose response is never produced hangs the
        # caller (runai_response blocks indefinitely), and the failing path is where dropping one is
        # easiest.
        missing = os.path.join(self.temp_dir, "does_not_exist.bin")
        self.assertFalse(os.path.exists(missing))

        buffer = mmap.mmap(-1, 64, mmap.MAP_ANONYMOUS | mmap.MAP_PRIVATE)
        base = buffer_address(buffer)

        streamer = runai_start()
        submission_id = runai_request(streamer, [missing], [2], [0, 0], [0, 0], [base, base])

        # Both ranges must be accounted for individually: counting two responses would also accept the
        # same range answered twice while the other was dropped, which is the failure this test exists to
        # catch. _drain also checks the submission-done flag lands on the last response.
        self.assertEqual(
            self._drain(streamer, submission_id, 2, expected_code=FILE_ACCESS_ERROR_CODE), [0, 1]
        )

    def test_unopenable_file_reports_one_error_per_range(self):
        # A file that cannot be opened fails each of its ranges, as the real streamer does. The failure
        # mode being guarded is not a wrong code but a HANG: dropping the responses would leave the
        # caller waiting on a count that can never be reached.
        missing = os.path.join(self.temp_dir, "also_missing.bin")
        self.assertFalse(os.path.exists(missing))

        buffer = mmap.mmap(-1, 64, mmap.MAP_ANONYMOUS | mmap.MAP_PRIVATE)
        base = buffer_address(buffer)

        streamer = runai_start()
        submission_id = runai_request(streamer, [missing], [3], [0, 8, 16], [8, 8, 8], [base, base + 8, base + 16])

        seen = []
        for _ in range(3):
            result = runai_response(streamer)
            self.assertIsNotNone(result)
            ret, sub_id, _file_index, range_index, _done = result
            self.assertEqual(sub_id, submission_id)
            self.assertNotEqual(ret, SUCCESS_ERROR_CODE)
            seen.append(range_index)

        # exactly one response per range, none dropped and none duplicated
        self.assertEqual(sorted(seen), [0, 1, 2])

    def test_mismatched_array_lengths_are_rejected(self):
        # The C side takes the range count as sum(num_ranges) and indexes all three flat arrays up to it,
        # so a mismatch here is an out-of-bounds read in native code, not an exception: ctypes silently
        # zero-pads an array built from too few initializers. These must fail in Python.
        file_path = os.path.join(self.temp_dir, "lengths.txt")
        with open(file_path, "w") as file:
            file.write("abcdefgh")

        buffer = mmap.mmap(-1, 64, mmap.MAP_ANONYMOUS | mmap.MAP_PRIVATE)
        base = buffer_address(buffer)
        streamer = runai_start()

        # Matched on the MESSAGE, not merely on ValueError: several guards raise ValueError here, so a
        # bare assertRaises passes when a different one fires. Removing the guard a case is named for
        # would then leave the case green while testing nothing - deleting the uint32 check, for
        # instance, makes that input trip the sum check instead, which is also a ValueError. The
        # patterns are distinctive fragments, so rewording a message does not break them.

        # num_ranges shorter than paths - the second file would silently lose all of its ranges
        with self.assertRaisesRegex(ValueError, "entries but there are"):
            runai_request(streamer, [file_path, file_path], [1], [0, 4], [4, 4], [base, base + 4])

        # sum(num_ranges) larger than the range arrays - this is the out-of-bounds read
        with self.assertRaisesRegex(ValueError, "but the range arrays hold"):
            runai_request(streamer, [file_path], [3], [0], [4], [base])

        # the parallel arrays disagree with each other
        with self.assertRaisesRegex(ValueError, "must be parallel"):
            runai_request(streamer, [file_path], [2], [0], [4, 4], [base, base + 4])
        with self.assertRaisesRegex(ValueError, "must be parallel"):
            runai_request(streamer, [file_path], [2], [0, 4], [4, 4], [base])

        # The correctly shaped version of the same submission is accepted - and is DRAINED before the test
        # returns. runai_request is non-blocking and the destinations are raw addresses that keep nothing
        # alive, so leaving a submission in flight lets the buffer be released while the library is still
        # writing into it.
        submission_id = runai_request(streamer, [file_path], [2], [0, 4], [4, 4], [base, base + 4])
        self.assertIsNotNone(submission_id)
        self.assertEqual(self._drain(streamer, submission_id, 2), [0, 1])
        self.assertEqual(bytes(buffer[0:8]), b"abcdefgh")

    def test_out_of_range_values_are_rejected(self):
        # ctypes converts out-of-range integers SILENTLY - c_uint32(-1) becomes 4294967295 and
        # c_uint32(2**32) becomes 0 - so a check against the Python values can agree while the C side
        # receives something else. These must be rejected before the conversion.
        file_path = os.path.join(self.temp_dir, "ranges.txt")
        with open(file_path, "w") as file:
            file.write("abcdefgh")

        buffer = mmap.mmap(-1, 64, mmap.MAP_ANONYMOUS | mmap.MAP_PRIVATE)
        base = buffer_address(buffer)
        streamer = runai_start()

        # Matched on the MESSAGE (see test_mismatched_array_lengths_are_rejected): each case must trip
        # the guard it is named for, or it silently stops covering that guard.

        # The case a sum-only check cannot catch: sum([-1, 1]) == 0 matches an empty range array, but
        # ctypes turns -1 into 4294967295 and the C side then indexes ~4 billion ranges.
        with self.assertRaisesRegex(ValueError, "unsigned 32-bit integer"):
            runai_request(streamer, [file_path, file_path], [-1, 1], [], [], [])

        # wraps to 0, silently costing the file all of its ranges
        with self.assertRaisesRegex(ValueError, "unsigned 32-bit integer"):
            runai_request(streamer, [file_path], [2**32], [0], [4], [base])

        # negative size wraps to an enormous read length; negative offset to an enormous seek
        with self.assertRaisesRegex(ValueError, "range size must fit"):
            runai_request(streamer, [file_path], [1], [0], [-1], [base])
        with self.assertRaisesRegex(ValueError, "range offset must fit"):
            runai_request(streamer, [file_path], [1], [-1], [4], [base])
        with self.assertRaisesRegex(ValueError, "range destination must be"):
            runai_request(streamer, [file_path], [1], [0], [4], [-1])

        # The other end of the 64-bit range, which wraps just as silently and in a different way for each
        # field: c_uint64(2**64) becomes 0, so an oversized size is a zero-length read and an oversized
        # offset reads from the start of the file; c_void_p(2**64) becomes NULL.
        with self.assertRaisesRegex(ValueError, "range size must fit"):
            runai_request(streamer, [file_path], [1], [0], [2**64], [base])
        with self.assertRaisesRegex(ValueError, "range offset must fit"):
            runai_request(streamer, [file_path], [1], [2**64], [4], [base])
        with self.assertRaisesRegex(ValueError, "range destination must be"):
            runai_request(streamer, [file_path], [1], [0], [4], [2**64])

        # a valid submission of the same shape still goes through, and is drained before returning
        submission_id = runai_request(streamer, [file_path], [1], [0], [4], [base])
        self.assertIsNotNone(submission_id)
        self.assertEqual(self._drain(streamer, submission_id, 1), [0])
        self.assertEqual(bytes(buffer[0:4]), b"abcd")

    def test_file_without_ranges_produces_no_response(self):
        # A file may carry no ranges at all. It is accepted and simply contributes nothing - the
        # submission completes on the responses of the files that do have ranges.
        empty_path = os.path.join(self.temp_dir, "no_ranges.txt")
        open(empty_path, "w").close()
        data_path = os.path.join(self.temp_dir, "data.txt")
        with open(data_path, "w") as file:
            file.write("payload")

        buffer = mmap.mmap(-1, 64, mmap.MAP_ANONYMOUS | mmap.MAP_PRIVATE)

        streamer = runai_start()
        submission_id = runai_request(
            streamer, [empty_path, data_path], [0, 1], [0], [7], [buffer_address(buffer)]
        )

        result = runai_response(streamer)
        self.assertIsNotNone(result)
        ret, sub_id, file_index, range_index, submission_done = result
        self.assertEqual(ret, SUCCESS_ERROR_CODE)
        self.assertEqual(sub_id, submission_id)
        self.assertEqual(file_index, 1)
        self.assertTrue(submission_done)
        self.assertEqual(bytes(buffer[0:7]), b"payload")

    def test_response_error_is_returned_not_raised(self):
        # A per-sub-range error (here EOF: asking for more bytes than the file holds) must be SURFACED by the
        # binding as data - a tuple with a non-Success response_code - NOT raised and NOT returned as None.
        # And since it is the submission's last (only) response, submission_done must be True, so a
        # multi-submission caller can attribute the failure to its submission and still learn it completed.
        file_path = os.path.join(self.temp_dir, "small.txt")
        with open(file_path, "w") as file:
            file.write("short")   # 5 bytes

        buffer = mmap.mmap(-1, 64, mmap.MAP_ANONYMOUS | mmap.MAP_PRIVATE)

        streamer = runai_start()
        self.assertNotEqual(streamer, 0)

        # one range asking for 20 bytes from a 5-byte file -> the read falls short (EOF error)
        submission_id = runai_request(
            streamer, [file_path], [1], [0], [20], [buffer_address(buffer)]
        )

        result = runai_response(streamer)
        # not raised, and not the teardown FinishedError (which would return None)
        self.assertIsNotNone(result)
        response_code, sub_id, file_index, range_index, submission_done = result
        self.assertNotEqual(response_code, SUCCESS_ERROR_CODE)   # error surfaced as data, not raised
        self.assertEqual(sub_id, submission_id)                  # attributable to its submission
        self.assertTrue(submission_done)                         # error was the submission's last response

    def tearDown(self):
        shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main()
