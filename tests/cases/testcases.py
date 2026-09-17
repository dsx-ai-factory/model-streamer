import unittest
import tempfile
import shutil
import os
import sys
import glob
import random
import string
import subprocess
import resource
import numpy as np
from unittest.mock import patch
from safetensors.torch import safe_open
from tests.safetensors.generator import create_random_safetensors, create_sized_safetensors
from tests.safetensors.comparison import tensor_maps_are_equal
from runai_model_streamer.safetensors_streamer.safetensors_streamer import (
    SafetensorsStreamer,
    list_safetensors,
    pull_files
)
from runai_model_streamer.file_streamer.file_streamer import FileStreamer
from runai_model_streamer.file_streamer.requests_iterator import (
    FileChunks,
    RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME,
)

METADATA_SUFFIX = ['safetensors', 'json', 'config', 'xml', 'pt', 'bin']
FILE_COUNT = 5

# Object storage splits a range into chunks of RUNAI_STREAMER_CHUNK_BYTESIZE (default 8 MiB, minimum
# 5 MiB), so the scattered-ranges file is sized to hold one range that spans several of them.
MIB = 1024 * 1024
SCATTERED_BIG_FILE_BYTESIZE = 12 * MIB
SCATTERED_SMALL_FILE_BYTESIZE = 1 * MIB


def positional_bytes(nbytes, seed):
    """Content in which every 4-byte word encodes its own position (offset/4, plus a per-file seed).

    Position-sensitive on purpose: with a constant or random-but-uniform filler, a range read from the
    wrong offset, two ranges whose destinations were swapped, or a chunk written at the wrong place
    inside its range all still compare equal. Here each of those changes the bytes.

    Returns exactly nbytes, whether or not that is a multiple of 4: a whole number of words is generated
    and the tail trimmed, so the last word may be partial. Truncating `nbytes // 4` instead would return
    a SHORT buffer for any size that is not a multiple of 4, and the resulting file would be shorter than
    the fixture believes - surfacing later as an EOF error from whichever range reaches the end, which
    says nothing about the real cause.
    """
    num_words = -(-nbytes // 4)   # ceil, so the buffer covers nbytes before trimming
    return (np.arange(num_words, dtype=np.uint32) + np.uint32(seed)).tobytes()[:nbytes]

def random_letters(x):
    return ''.join(random.choices(string.ascii_letters, k=x))

def create_random_files(dir):
    file_path = os.path.join(dir, f"{random_letters(5)}.{random.choice(METADATA_SUFFIX)}")
    with open(file_path, "w") as file:
        file.write(random_letters(15))
    return file_path

# Runs in a subprocess so a crash during mid-stream teardown surfaces as a signal exit
# (and a core dump) instead of taking down the test runner. Each process: (1) kills a
# streamer mid-stream at a randomized moment with reads still outstanding, then (2) creates
# a NEW streamer in the SAME process and streams the whole file, checking every tensor
# against the reference -- proving the killed streamer did not invalidate the next one (the
# reused, process-global object-storage client still serves correct data). Exits non-zero on
# any mismatch. The URL, reference file path, and credentials/endpoint come from the
# environment (inherited from the parent), exactly like the in-process tests.
_TEARDOWN_CHILD_PROGRAM = """
import os, random, time
import torch
from safetensors.torch import safe_open
from runai_model_streamer.safetensors_streamer.safetensors_streamer import SafetensorsStreamer

url = os.environ["TEARDOWN_TEST_URL"]
expected_file = os.environ["TEARDOWN_EXPECTED_FILE"]

# 1. Kill a streamer mid-stream at a randomized moment, reads still outstanding.
with SafetensorsStreamer() as run_sf:
    run_sf.stream_file(url, None, "cpu")
    if random.random() < 0.5:
        # Wait until the first tensor is delivered, then leave the rest in flight.
        for _name, _tensor in run_sf.get_tensors():
            break
    else:
        # Exit after a very short delay, often before any response arrives.
        time.sleep(random.uniform(0, 0.01))

# 2. A NEW streamer in the SAME process must still stream the whole file correctly.
got = {}
with SafetensorsStreamer() as run_sf:
    run_sf.stream_file(url, None, "cpu")
    for name, tensor in run_sf.get_tensors():
        # clone: the yielded tensor is a VIEW into a ring buffer that is recycled once the
        # generator advances, so anything compared after the loop must own its data
        got[name] = tensor.clone()

expected = {}
with safe_open(expected_file, framework="pt", device="cpu") as f:
    for name in f.keys():
        expected[name] = f.get_tensor(name)

if set(got) != set(expected):
    raise SystemExit("reused streamer tensor names differ: %s vs %s" % (sorted(got), sorted(expected)))
for name in expected:
    if not torch.equal(got[name], expected[name]):
        raise SystemExit("reused streamer returned wrong data for tensor %s after a mid-stream kill" % name)
"""


def compatibility_test_cases(backend_class, scheme, bucket_name):
    class TestObjectStorageCompatibility(unittest.TestCase):
        def setUp(self):
            self.temp_dir = tempfile.mkdtemp()
            self.server = backend_class()
            self.bucket_name = bucket_name
            self.scheme = scheme
            self.server.wait_for_startup()

        def test_safetensors_streamer(self):
            file_path = create_random_safetensors(self.temp_dir)
            self.server.upload_file(self.bucket_name, "", file_path)

            our = {}
            with SafetensorsStreamer() as run_sf:
                run_sf.stream_file(f"{self.scheme}://{self.bucket_name}/model.safetensors", None, "cpu")
                for name, tensor in run_sf.get_tensors():
                    # clone: the yielded tensor is a VIEW into a ring buffer that is recycled
                    # once the generator advances, so a comparison after the loop must own it
                    our[name] = tensor.clone()

            their = {}
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    their[name] = f.get_tensor(name)

            equal, message = tensor_maps_are_equal(our, their)
            if not equal:
                self.fail(f"Tensor mismatch: {message}")

        def test_safetensors_streamer_with_tensor_names(self):
            file_path = create_random_safetensors(self.temp_dir)
            self.server.upload_file(self.bucket_name, "", file_path)

            with safe_open(file_path, framework="pt", device="cpu") as f:
                all_names = list(f.keys())
            # A genuine subset when the generator produced more than one tensor - MIN_NUM_TENSORS
            # is 1, so this must not assume there is more than one to split.
            requested = set(all_names[: max(1, len(all_names) // 2)])

            our = {}
            with SafetensorsStreamer() as run_sf:
                run_sf.stream_file(
                    f"{self.scheme}://{self.bucket_name}/model.safetensors", None, "cpu",
                    tensor_names=requested,
                )
                for name, tensor in run_sf.get_tensors():
                    our[name] = tensor.clone()

            self.assertEqual(set(our.keys()), requested)

            their = {}
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for name in requested:
                    their[name] = f.get_tensor(name)

            equal, message = tensor_maps_are_equal(our, their)
            if not equal:
                self.fail(f"Tensor mismatch: {message}")

        def _upload_positional_file(self, filename, bytesize, seed):
            """Write a position-encoded file locally, upload it, and return (url, local content)."""
            content = positional_bytes(bytesize, seed)
            path = os.path.join(self.temp_dir, filename)
            with open(path, "wb") as f:
                f.write(content)
            self.server.upload_file(self.bucket_name, "", path)
            return f"{self.scheme}://{self.bucket_name}/{filename}", content

        def _stream_and_check_ranges(self, files, memory_limit):
            """Stream the given scattered ranges and assert every range delivered the exact bytes.

            files: list of (url, local_content, [(offset, size), ...]) - the ranges in the order they
            are submitted, which is deliberately NOT the order they appear in the file.
            """
            requests = [
                FileChunks(
                    id=i,
                    path=url,
                    offsets=[offset for offset, _ in ranges],
                    sizes=[size for _, size in ranges],
                )
                for i, (url, _content, ranges) in enumerate(files)
            ]

            got = {}
            with patch.dict(os.environ, {RUNAI_STREAMER_MEMORY_LIMIT_ENV_VAR_NAME: str(memory_limit)}):
                with FileStreamer() as streamer:
                    streamer.stream_files(requests)
                    for file_id, range_index, tensor in streamer.get_chunks():
                        key = (file_id, range_index)
                        self.assertNotIn(key, got, f"range {key} was delivered twice")
                        # uint8 tensor of shape (1, size) viewing the streamer's buffer; copy the bytes
                        # out now, since the buffer is reused once the next request is submitted
                        got[key] = tensor.numpy().tobytes()

            expected_keys = {
                (i, j)
                for i, (_url, _content, ranges) in enumerate(files)
                for j in range(len(ranges))
            }
            self.assertEqual(
                set(got),
                expected_keys,
                "every range must produce exactly one response, including zero-sized ones",
            )

            for i, (url, content, ranges) in enumerate(files):
                for j, (offset, size) in enumerate(ranges):
                    expected = content[offset:offset + size]
                    self.assertEqual(
                        got[(i, j)],
                        expected,
                        f"wrong bytes for range {j} of {url} (offset={offset}, size={size}, "
                        f"memory_limit={memory_limit})",
                    )

        def test_scattered_ranges(self):
            """Read arbitrary, non-contiguous, unordered ranges into scattered destinations.

            This is the per-range destinations API used as a caller would use it, rather than through
            the safetensors path - which only ever submits one contiguous, ascending span per file and
            so never produces the shapes this exercises. Each awkward shape below targets a specific
            position-based assumption:

              - ranges NOT in file order, and with gaps between them, so one file yields several
                contiguous transfers (hence several batches) rather than a single span;
              - a range larger than the object-storage chunk size, so it is split across several
                backend reads that must land back to back at the right place;
              - zero-sized ranges FIRST, in the middle and LAST in a file's list, each doing no backend
                read at all yet still producing exactly one response, in order;
              - two ranges reading the SAME source bytes into DIFFERENT destinations, which nothing
                can satisfy by aliasing or by reusing a destination;
              - two files in one submission, so a response must be attributed to the right file.

            Run twice: once with an unlimited memory cap (the whole thing is one submission) and once
            with a cap smaller than the total (the ranges are split across several submissions, so the
            buffer is reused and the per-request destination packing is re-done).
            """
            big_url, big_content = self._upload_positional_file(
                f"scattered_big_{random_letters(5)}.bin", SCATTERED_BIG_FILE_BYTESIZE, seed=0
            )
            small_url, small_content = self._upload_positional_file(
                f"scattered_small_{random_letters(5)}.bin", SCATTERED_SMALL_FILE_BYTESIZE, seed=1 << 20
            )

            # Zero-sized ranges appear FIRST, in the middle, and LAST. Position matters on its own: the
            # first range seeds the destination cursor and the coalescing pass, and the last one is the
            # range whose response carries submission-done - so a zero-sized range dropped or mis-indexed
            # at either edge fails differently than one in the middle.
            big_ranges = [
                (3 * MIB, 0),          # zero-sized, FIRST
                (11 * MIB, 1024),      # near EOF: not in file order
                (64, 1024),            # far earlier in the file
                (2 * MIB, 9 * MIB),    # spans several object-storage chunks
                (1 * MIB, 4096),
                (5 * MIB, 0),          # zero-sized, MIDDLE
                (1 * MIB, 4096),       # same source bytes as above, different destination
                (0, 4),                # the very first bytes, requested late
            ]
            small_ranges = [
                (SCATTERED_SMALL_FILE_BYTESIZE - 512, 512),   # the tail
                (0, SCATTERED_SMALL_FILE_BYTESIZE),           # the whole file
                (4096, 8192),
                (SCATTERED_SMALL_FILE_BYTESIZE, 0),           # zero-sized, LAST, and at EOF
            ]
            files = [
                (big_url, big_content, big_ranges),
                (small_url, small_content, small_ranges),
            ]

            # -1 = unlimited: one submission holding every range.
            self._stream_and_check_ranges(files, memory_limit=-1)

            # Just above the largest single range - the smallest cap that can still make progress. It
            # splits even ONE file's ranges across submissions, so the buffer is reused between them and
            # the responses of the later submissions have to be mapped back to the right global range
            # index. A cap large enough to split only at a file boundary would not cover that.
            largest_range = max(size for _offset, size in big_ranges + small_ranges)
            self._stream_and_check_ranges(files, memory_limit=largest_range + 2048)

        def test_teardown_while_streaming(self):
            """
            Kill the streamer (context-manager __exit__ -> runai_end) while object-storage
            reads are still in flight, without draining all tensors, and assert the process
            does not crash (no core dump). This exercises the persistent-responder teardown
            (stop() with outstanding responses) plus the S3 client lifecycle.

            Each teardown runs in a fresh subprocess: an in-process segfault/abort during
            runai_end would kill the test runner itself, so isolation is what makes a crash
            observable -- as a negative return code (killed by a signal, i.e. the condition
            that drops a core). A crash and a clean error are reported distinctly. Core dumps
            are enabled in the child so a crash also drops a core we detect on disk as a
            supplementary signal (skipped when the kernel pipes cores elsewhere).

            An unlimited memory limit dispatches the whole file in a single wave, so all
            chunks are outstanding in the capacity queue / object-storage client at once.
            Several fresh processes randomize the teardown moment to cover both "after a
            response was consumed" and "before any response arrived". After each kill the
            child creates a NEW streamer in the SAME process and streams the whole file,
            checking every tensor's data against the reference -- so a killed streamer must
            not invalidate the next one that reuses the process-global object-storage client.
            Distinct per-tensor payloads make that data check catch wrong offsets and
            cross-tensor swaps. A final fresh-process stream then confirms the backend is
            healthy across process boundaries too.
            """
            file_path = create_sized_safetensors(self.temp_dir)
            self.server.upload_file(self.bucket_name, "", file_path)
            url = f"{self.scheme}://{self.bucket_name}/{os.path.basename(file_path)}"

            run_cwd = tempfile.mkdtemp()
            try:
                env = dict(
                    os.environ,
                    RUNAI_STREAMER_MEMORY_LIMIT="-1",
                    TEARDOWN_TEST_URL=url,
                    TEARDOWN_EXPECTED_FILE=file_path,
                )
                # The child runs with cwd=run_cwd, so any RELATIVE entry in LD_LIBRARY_PATH
                # (e.g. the azure harness's "../cpp/bazel-bin/azure") would stop resolving and
                # the child would load the wrong object-storage plugin. Resolve entries against
                # the parent's cwd so the child loads the same libs the parent test does.
                ld_library_path = os.environ.get("LD_LIBRARY_PATH")
                if ld_library_path:
                    env["LD_LIBRARY_PATH"] = os.pathsep.join(
                        os.path.abspath(p) if p else p
                        for p in ld_library_path.split(os.pathsep)
                    )
                for i in range(5):
                    proc = subprocess.run(
                        [sys.executable, "-c", _TEARDOWN_CHILD_PROGRAM],
                        cwd=run_cwd,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        # Enable core dumps in the child so a crash actually drops a core.
                        preexec_fn=lambda: resource.setrlimit(
                            resource.RLIMIT_CORE,
                            (resource.RLIM_INFINITY, resource.RLIM_INFINITY),
                        ),
                    )
                    # A crash during teardown == killed by a signal == negative return code
                    # (the condition that produces a core). This is the primary assertion.
                    self.assertGreaterEqual(
                        proc.returncode,
                        0,
                        f"streamer CRASHED during mid-stream teardown (iteration {i}): "
                        f"killed by signal {-proc.returncode}.\nstderr:\n{proc.stderr}",
                    )
                    cores = glob.glob(os.path.join(run_cwd, "core*"))
                    self.assertEqual(
                        cores, [], f"core dump(s) produced during mid-stream teardown: {cores}"
                    )
                    # A clean non-zero exit means the stream itself errored (not a crash).
                    self.assertEqual(
                        proc.returncode,
                        0,
                        f"streamer errored during mid-stream teardown (iteration {i}, "
                        f"return code {proc.returncode}).\nstderr:\n{proc.stderr}",
                    )
            finally:
                shutil.rmtree(run_cwd, ignore_errors=True)

            # The library and backend must remain healthy: a fresh full stream succeeds.
            our = {}
            with SafetensorsStreamer() as run_sf:
                run_sf.stream_file(url, None, "cpu")
                for name, tensor in run_sf.get_tensors():
                    # clone: the yielded tensor is a VIEW into a ring buffer that is recycled
                    # once the generator advances, so a comparison after the loop must own it
                    our[name] = tensor.clone()

            their = {}
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    their[name] = f.get_tensor(name)

            equal, message = tensor_maps_are_equal(our, their)
            if not equal:
                self.fail(f"Tensor mismatch after mid-stream teardown: {message}")

        def test_safetensors_truncated_file_body(self):
            """
            Tests the scenario where the header is valid, but the file ends unexpectedly 
            (EOF) while reading the tensor data.
            """
            import struct
            import json

            # 1. Manually craft a corrupted file
            # Define a header expecting 100 bytes of data
            header = {
                "tensor_truncated": {
                    "dtype": "U8",
                    "shape": [100000000],
                    "data_offsets": [0, 100000000]
                }
            }
            header_json = json.dumps(header).encode('utf-8')
            
            filename = f"truncated_{random_letters(5)}.safetensors"
            file_path = os.path.join(self.temp_dir, filename)
            
            with open(file_path, "wb") as f:
                # Write 8-byte header length
                f.write(struct.pack('<Q', len(header_json)))
                # Write Header
                f.write(header_json)
                # Write Data: Only write 10 bytes instead of the expected 100
                f.write(b'\x00' * 10)

            # 2. Upload the corrupted file
            self.server.upload_file(self.bucket_name, "", file_path)

            # 3. Stream and expect a ValueError during iteration
            with SafetensorsStreamer() as run_sf:
                # The stream_file call might succeed (it only reads the header),
                # but the iteration MUST fail when it hits the EOF in the body.
                run_sf.stream_file(f"{self.scheme}://{self.bucket_name}/{filename}", None, "cpu")
                
                with self.assertRaises(ValueError):
                    for name, tensor in run_sf.get_tensors():
                        pass

        def test_list_files(self):
            file_paths = [create_random_files(self.temp_dir) for _ in range(FILE_COUNT)]

            directory = random_letters(10)
            for file_path in file_paths:
                self.server.upload_file(self.bucket_name, directory, file_path)

            safetensors_files = [f"{self.scheme}://{self.bucket_name}/{directory}/{os.path.basename(fp)}" for fp in file_paths if fp.endswith('.safetensors')]
            
            result_files = list_safetensors(f"{self.scheme}://{self.bucket_name}/{directory}/")
            self.assertEqual(sorted(result_files), sorted(safetensors_files))

        def test_list_files_is_not_recursive(self):
            file_paths = [create_random_files(self.temp_dir) for _ in range(FILE_COUNT)]

            directory = random_letters(10).rstrip("/")
            subdirectory = directory + "/" + random_letters(10).rstrip("/")
            for file_path in file_paths:
                self.server.upload_file(self.bucket_name, subdirectory, file_path)

            safetensors_files = [f"{self.scheme}://{self.bucket_name}/{subdirectory}/{os.path.basename(fp)}" for fp in file_paths if fp.endswith('.safetensors')]
            
            result_files = list_safetensors(f"{self.scheme}://{self.bucket_name}/{directory}/")
            self.assertEqual(len(result_files), 0)
            result_files = list_safetensors(f"{self.scheme}://{self.bucket_name}/{subdirectory}")
            self.assertEqual(sorted(result_files), sorted(safetensors_files))

        def test_pull_files(self):
            file_paths = [create_random_files(self.temp_dir) for _ in range(FILE_COUNT)]
            directory = random_letters(10)
            for file_path in file_paths:
                self.server.upload_file(self.bucket_name, directory, file_path)

            pull_dir = tempfile.mkdtemp()

            pull_files(f"{self.scheme}://{self.bucket_name}/{directory}", pull_dir, ignore_pattern=[
                                                    "*.pt", "*.safetensors", "*.bin"
                                                ])

            pulled_files = os.listdir(pull_dir)
            original_files = [os.path.basename(fp) for fp in file_paths if not (fp.endswith("pt") or fp.endswith("safetensors") or fp.endswith("bin"))]

            self.assertEqual(sorted(pulled_files), sorted(original_files))

        def test_pull_files_is_recursive(self):
            """
            This test expects pull_files to be recursive. 
            With the current implementation of list_files, this test SHOULD FAIL.
            """
            # 1. Setup paths: a root directory and a nested subdirectory
            base_directory = f"root_{random_letters(5)}"
            sub_directory = "nested_folder"
            
            # Create two files: one in the root, one in the nested folder
            root_file_path = create_random_files(self.temp_dir)
            root_filename = os.path.basename(root_file_path)
            
            nested_file_path = create_random_files(self.temp_dir)
            nested_filename = os.path.basename(nested_file_path)

            # 2. Upload to S3
            # s3://bucket/root_abc/file1.json
            self.server.upload_file(self.bucket_name, base_directory, root_file_path)
            # s3://bucket/root_abc/nested_folder/file2.json
            self.server.upload_file(self.bucket_name, f"{base_directory}/{sub_directory}", nested_file_path)

            # 3. Prepare local destination
            pull_dst = tempfile.mkdtemp()
            
            try:
                # 4. Pull from the base directory
                pull_files(f"{self.scheme}://{self.bucket_name}/{base_directory}", pull_dst)

                # 5. Check results
                pulled_files_flat = []
                for root, dirs, files in os.walk(pull_dst):
                    for file in files:
                        # Create relative paths like 'file1.json' or 'nested_folder/file2.json'
                        rel_path = os.path.relpath(os.path.join(root, file), pull_dst)
                        pulled_files_flat.append(rel_path)

                # Expectation: Both files should be there if recursive
                expected_nested_path = os.path.join(sub_directory, nested_filename)
                
                self.assertIn(root_filename, pulled_files_flat, "Root file was not pulled")
                
                # THIS IS WHERE IT WILL FAIL:
                self.assertIn(expected_nested_path, pulled_files_flat, 
                              f"Recursive file {expected_nested_path} was not pulled. "
                              "The current implementation of list_files is likely non-recursive.")

            finally:
                shutil.rmtree(pull_dst)

        def tearDown(self):
            shutil.rmtree(self.temp_dir)

    return TestObjectStorageCompatibility


def list_files_test_cases(backend_class, scheme, bucket_name):
    class TestListFilesCompatibility(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.server = backend_class()
            cls.server.wait_for_startup()
            cls.bucket_name = bucket_name
            cls.scheme = scheme

        def setUp(self):
            self.temp_dir = tempfile.mkdtemp()
            self.directory = random_letters(10)

        def tearDown(self):
            shutil.rmtree(self.temp_dir)

        def _upload(self, directory, file_paths):
            for fp in file_paths:
                self.server.upload_file(self.bucket_name, directory, fp)

        def _write(self, filename, content=b"x"):
            path = os.path.join(self.temp_dir, filename)
            with open(path, "wb") as f:
                f.write(content)
            return path

        def _prefix(self, subdir=""):
            base = f"{self.scheme}://{self.bucket_name}/{self.directory}"
            return f"{base}/{subdir}" if subdir else f"{base}/"

        def test_basic_listing(self):
            files = [create_random_files(self.temp_dir) for _ in range(3)]
            self._upload(self.directory, files)

            results = FileStreamer().list_files(self._prefix())
            result_paths = {r[0] for r in results}

            for fp in files:
                expected = f"{self.scheme}://{self.bucket_name}/{self.directory}/{os.path.basename(fp)}"
                self.assertIn(expected, result_paths)

        def test_returns_correct_sizes(self):
            content = b"known content 123"
            fp = self._write("known.bin", content)
            self._upload(self.directory, [fp])

            results = FileStreamer().list_files(self._prefix())
            by_path = {r[0]: r[1] for r in results}
            expected_path = f"{self.scheme}://{self.bucket_name}/{self.directory}/known.bin"
            self.assertEqual(by_path[expected_path], len(content))

        def test_recursive(self):
            root_file = create_random_files(self.temp_dir)
            nested_file = create_random_files(self.temp_dir)
            self._upload(self.directory, [root_file])
            self._upload(f"{self.directory}/subdir", [nested_file])

            nested_path = f"{self.scheme}://{self.bucket_name}/{self.directory}/subdir/{os.path.basename(nested_file)}"

            recursive_paths = {r[0] for r in FileStreamer().list_files(self._prefix(), is_recursive=True)}
            non_recursive_paths = {r[0] for r in FileStreamer().list_files(self._prefix(), is_recursive=False)}

            self.assertIn(nested_path, recursive_paths)
            self.assertNotIn(nested_path, non_recursive_paths)

        def test_allow_pattern(self):
            fp_st = self._write("model.safetensors", b"model data")
            fp_js = self._write("config.json", b"config data")
            self._upload(self.directory, [fp_st, fp_js])

            results = FileStreamer().list_files(self._prefix(), allow_patterns=["*.safetensors"])
            paths = {r[0] for r in results}
            self.assertTrue(all(p.endswith(".safetensors") for p in paths))
            self.assertFalse(any(p.endswith(".json") for p in paths))

        def test_ignore_pattern(self):
            fp_st = self._write("model.safetensors", b"model data")
            fp_js = self._write("config.json", b"config data")
            self._upload(self.directory, [fp_st, fp_js])

            results = FileStreamer().list_files(self._prefix(), ignore_patterns=["*.json"])
            paths = {r[0] for r in results}
            self.assertFalse(any(p.endswith(".json") for p in paths))
            self.assertTrue(any(p.endswith(".safetensors") for p in paths))

        def test_empty_prefix(self):
            results = FileStreamer().list_files(self._prefix())
            self.assertEqual(results, [])

    return TestListFilesCompatibility