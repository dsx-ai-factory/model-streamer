# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import os
import shutil
import sys
import tempfile
import multiprocessing
import time

if sys.platform != "linux":
    raise unittest.SkipTest("ObjectStorageModel uses fcntl and fork — Linux only")

import fcntl

import runai_model_streamer.safetensors_streamer.safetensors_streamer as _ss_module
from runai_model_streamer.safetensors_streamer.safetensors_streamer import ObjectStorageModel

SOURCE_FILES = {
    "model.safetensors": b"fake-weights",
    "config.json": b'{"model_type": "test"}',
    "tokenizer.json": b'{"version": "1.0"}',
}

# Fake object storage URL used wherever ObjectStorageModel requires a valid scheme.
# The patched pull_files ignores it and copies from self.src_dir instead.
FAKE_MODEL_PATH = "s3://fake-bucket/fake-model"


class TestObjectStorageModel(unittest.TestCase):

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _create_source_dir():
        src = tempfile.mkdtemp()
        for name, content in SOURCE_FILES.items():
            with open(os.path.join(src, name), "wb") as f:
                f.write(content)
        return src

    def _fake_pull_files(
        self, model_path, dst, allow_pattern=None, ignore_pattern=None, s3_credentials=None
    ):
        """Simulates a download by copying source files into dst."""
        for fname in os.listdir(self.src_dir):
            shutil.copy2(os.path.join(self.src_dir, fname), os.path.join(dst, fname))

    def _slow_fake_pull_files(
        self, model_path, dst, allow_pattern=None, ignore_pattern=None, s3_credentials=None
    ):
        """Slow variant so concurrent processes actually overlap."""
        time.sleep(0.15)
        for fname in os.listdir(self.src_dir):
            shutil.copy2(os.path.join(self.src_dir, fname), os.path.join(dst, fname))

    @staticmethod
    def _worker(dst_dir, result_queue):
        """Entry point for each worker process (fork-safe static method)."""
        try:
            with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=dst_dir) as obj:
                skipped = obj._skip
                obj.pull_files()
            result_queue.put({"status": "ok", "skipped": skipped})
        except Exception as exc:
            result_queue.put({"status": "error", "error": str(exc)})

    def _run_concurrent(self, num_processes):
        """
        Patch pull_files in the parent then fork workers. With fork, children
        inherit the patched module attribute and therefore call _slow_fake_pull_files.
        _slow_fake_pull_files is an instance method bound to self, so children
        inherit self.src_dir via the forked address space.
        """
        original = _ss_module.pull_files
        _ss_module.pull_files = self._slow_fake_pull_files

        try:
            ctx = multiprocessing.get_context("fork")
            result_queue = ctx.Queue()
            processes = [
                ctx.Process(
                    target=self._worker,
                    args=(self.dst_dir, result_queue),
                )
                for _ in range(num_processes)
            ]
            for p in processes:
                p.start()
            timed_out = []
            for p in processes:
                p.join(timeout=30)
                if p.is_alive():
                    p.terminate()
                    timed_out.append(p)
            for p in timed_out:
                p.join(timeout=5)
            if timed_out:
                self.fail(f"Worker processes {[p.pid for p in timed_out]} timed out and were terminated")

            results = []
            for _ in range(num_processes):
                try:
                    results.append(result_queue.get_nowait())
                except Exception:
                    pass  # process died without reporting
            self.assertEqual(
                len(results), num_processes,
                f"Only {len(results)}/{num_processes} workers reported a result (others may have crashed)",
            )
        finally:
            _ss_module.pull_files = original

        return results

    # -----------------------------------------------------------------------
    # setUp / tearDown
    # -----------------------------------------------------------------------

    # Subclass sets this to True to run all tests with dst pre-existing,
    # exercising the shutil.rmtree cleanup path in __init__.
    dst_pre_exists = False

    def setUp(self):
        self.src_dir = self._create_source_dir()
        parent = tempfile.mkdtemp()
        self.dst_dir = os.path.join(parent, "model")
        if self.dst_pre_exists:
            os.makedirs(self.dst_dir)

    def tearDown(self):
        shutil.rmtree(self.src_dir, ignore_errors=True)
        # Removes dst, the sibling .lock file, and the parent in one call.
        shutil.rmtree(os.path.dirname(self.dst_dir), ignore_errors=True)

    # -----------------------------------------------------------------------
    # Single-process: basic download
    # -----------------------------------------------------------------------

    def test_files_present_after_done(self):
        original = _ss_module.pull_files
        _ss_module.pull_files = self._fake_pull_files
        try:
            with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=self.dst_dir) as obj:
                obj.pull_files()
        finally:
            _ss_module.pull_files = original

        for name in SOURCE_FILES:
            self.assertTrue(
                os.path.exists(os.path.join(self.dst_dir, name)),
                f"{name} missing from dst",
            )

    def test_sentinel_written_after_done(self):
        original = _ss_module.pull_files
        _ss_module.pull_files = self._fake_pull_files
        try:
            with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=self.dst_dir) as obj:
                obj.pull_files()
        finally:
            _ss_module.pull_files = original

        self.assertTrue(
            os.path.exists(os.path.join(self.dst_dir, ObjectStorageModel.SENTINEL_NAME))
        )

    def test_sentinel_not_written_before_exit(self):
        original = _ss_module.pull_files
        _ss_module.pull_files = self._fake_pull_files
        try:
            with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=self.dst_dir) as obj:
                obj.pull_files()
                self.assertFalse(
                    os.path.exists(os.path.join(self.dst_dir, ObjectStorageModel.SENTINEL_NAME))
                )
        finally:
            _ss_module.pull_files = original

    # -----------------------------------------------------------------------
    # Single-process: idempotency
    # -----------------------------------------------------------------------

    def test_second_call_skips_download(self):
        call_count = [0]
        fake = self._fake_pull_files

        def counting(model_path, dst, allow_pattern=None, ignore_pattern=None, s3_credentials=None):
            call_count[0] += 1
            fake(model_path, dst, allow_pattern, ignore_pattern, s3_credentials)

        original = _ss_module.pull_files
        _ss_module.pull_files = counting
        try:
            with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=self.dst_dir) as obj1:
                obj1.pull_files()

            with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=self.dst_dir) as obj2:
                obj2.pull_files()
        finally:
            _ss_module.pull_files = original

        self.assertEqual(call_count[0], 1, "pull_files should be called exactly once")

    def test_second_call_files_still_accessible(self):
        original = _ss_module.pull_files
        _ss_module.pull_files = self._fake_pull_files
        try:
            with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=self.dst_dir) as obj1:
                obj1.pull_files()

            with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=self.dst_dir) as obj2:
                obj2.pull_files()
        finally:
            _ss_module.pull_files = original

        for name in SOURCE_FILES:
            self.assertTrue(os.path.exists(os.path.join(self.dst_dir, name)))

    # -----------------------------------------------------------------------
    # Single-process: two pull_files calls before done (vLLM pattern)
    # -----------------------------------------------------------------------

    def test_two_pull_files_calls_one_done(self):
        """Matches the vLLM pattern: pull model files then tokenizer files."""
        call_count = [0]
        fake = self._fake_pull_files

        def counting(model_path, dst, allow_pattern=None, ignore_pattern=None, s3_credentials=None):
            call_count[0] += 1
            fake(model_path, dst, allow_pattern, ignore_pattern, s3_credentials)

        original = _ss_module.pull_files
        _ss_module.pull_files = counting
        try:
            with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=self.dst_dir) as obj:
                obj.pull_files(allow_pattern=["*.safetensors"])
                obj.pull_files(ignore_pattern=["*.safetensors"])
        finally:
            _ss_module.pull_files = original

        self.assertEqual(call_count[0], 2)
        self.assertTrue(
            os.path.exists(os.path.join(self.dst_dir, ObjectStorageModel.SENTINEL_NAME))
        )

    # -----------------------------------------------------------------------
    # Single-process: stale dst cleanup
    # -----------------------------------------------------------------------

    def test_dst_trailing_slash_lock_outside_dst(self):
        """dst with a trailing slash must still place the lock file beside dst, not inside it.
        If normalization is missing, shutil.rmtree(dst) deletes the lock inode and a racing
        process creates a fresh inode at the same path, breaking mutual exclusion silently."""
        original = _ss_module.pull_files
        _ss_module.pull_files = self._fake_pull_files
        trailing_dst = self.dst_dir + "/"
        try:
            with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=trailing_dst) as obj:
                obj.pull_files()
        finally:
            _ss_module.pull_files = original

        expected_lock = self.dst_dir + ".lock"
        self.assertTrue(
            os.path.exists(expected_lock),
            "lock file must be a sibling of dst, not inside it",
        )
        self.assertFalse(
            os.path.exists(os.path.join(self.dst_dir, ".lock")),
            "lock file must not be inside dst",
        )

    def test_stale_dst_cleaned_before_download(self):
        os.makedirs(self.dst_dir, exist_ok=True)
        stale = os.path.join(self.dst_dir, "stale.bin")
        with open(stale, "wb") as f:
            f.write(b"old data")

        original = _ss_module.pull_files
        _ss_module.pull_files = self._fake_pull_files
        try:
            with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=self.dst_dir) as obj:
                obj.pull_files()
        finally:
            _ss_module.pull_files = original

        self.assertFalse(os.path.exists(stale), "stale file should have been removed")
        for name in SOURCE_FILES:
            self.assertTrue(os.path.exists(os.path.join(self.dst_dir, name)))

    # -----------------------------------------------------------------------
    # Single-process: invalid model_path rejected before any side effects
    # -----------------------------------------------------------------------

    def test_invalid_model_path_raises_before_lock(self):
        if self.dst_pre_exists:
            self.skipTest("dst is pre-created by setUp; cannot test that dst is absent after ValueError")

        with self.assertRaises(ValueError):
            ObjectStorageModel(model_path="/local/path/model", dst=self.dst_dir)

        self.assertFalse(
            os.path.exists(self.dst_dir),
            "dst must not be created for an invalid model_path",
        )
        self.assertFalse(
            os.path.exists(self.dst_dir + ".lock"),
            "lock file must not be created for an invalid model_path",
        )

    # -----------------------------------------------------------------------
    # Single-process: empty download raises and releases lock
    # -----------------------------------------------------------------------

    def test_empty_download_raises_runtime_error(self):
        def empty_pull_files(model_path, dst, allow_pattern=None, ignore_pattern=None, s3_credentials=None):
            pass  # downloads nothing

        original = _ss_module.pull_files
        _ss_module.pull_files = empty_pull_files
        try:
            with self.assertRaises(RuntimeError):
                with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=self.dst_dir) as obj:
                    obj.pull_files()
        finally:
            _ss_module.pull_files = original

        self.assertFalse(
            os.path.exists(os.path.join(self.dst_dir, ObjectStorageModel.SENTINEL_NAME)),
            "sentinel must not be written after an empty download",
        )

    def test_empty_download_releases_lock(self):
        def empty_pull_files(model_path, dst, allow_pattern=None, ignore_pattern=None, s3_credentials=None):
            pass  # downloads nothing

        original = _ss_module.pull_files
        _ss_module.pull_files = empty_pull_files
        try:
            with self.assertRaises(RuntimeError):
                with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=self.dst_dir) as obj:
                    obj.pull_files()
        finally:
            _ss_module.pull_files = original

        # Verify the lock file is free by acquiring it non-blocking
        lock_path = self.dst_dir + ".lock"
        with open(lock_path, "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
            except BlockingIOError:
                self.fail("lock was not released after empty download")

    # -----------------------------------------------------------------------
    # Single-process: __del__ releases lock when context manager is not used
    # -----------------------------------------------------------------------

    def test_del_releases_lock(self):
        obj = ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=self.dst_dir)
        obj.__del__()

        lock_path = self.dst_dir + ".lock"
        with open(lock_path, "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
            except BlockingIOError:
                self.fail("lock was not released by __del__")

    def test_del_idempotent_after_exit(self):
        """Calling __del__ after __exit__ must not raise."""
        original = _ss_module.pull_files
        _ss_module.pull_files = self._fake_pull_files
        try:
            with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=self.dst_dir) as obj:
                obj.pull_files()
        finally:
            _ss_module.pull_files = original

        obj.__del__()  # lock_file is already closed; must be a no-op

    # -----------------------------------------------------------------------
    # Single-process: exception during pull suppresses sentinel
    # -----------------------------------------------------------------------

    def test_sentinel_not_written_when_pull_raises(self):
        def raising_pull_files(model_path, dst, allow_pattern=None, ignore_pattern=None, s3_credentials=None):
            raise RuntimeError("simulated download failure")

        original = _ss_module.pull_files
        _ss_module.pull_files = raising_pull_files
        try:
            with self.assertRaises(RuntimeError):
                with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=self.dst_dir) as obj:
                    obj.pull_files()
        finally:
            _ss_module.pull_files = original

        self.assertFalse(
            os.path.exists(os.path.join(self.dst_dir, ObjectStorageModel.SENTINEL_NAME)),
            "sentinel must not be written after a failed download",
        )

    def test_lock_released_when_pull_raises(self):
        def raising_pull_files(model_path, dst, allow_pattern=None, ignore_pattern=None, s3_credentials=None):
            raise RuntimeError("simulated download failure")

        original = _ss_module.pull_files
        _ss_module.pull_files = raising_pull_files
        try:
            with self.assertRaises(RuntimeError):
                with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=self.dst_dir) as obj:
                    obj.pull_files()
        finally:
            _ss_module.pull_files = original

        lock_path = self.dst_dir + ".lock"
        with open(lock_path, "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
            except BlockingIOError:
                self.fail("lock was not released after pull_files raised")

    def test_retry_after_failed_download(self):
        """A second attempt must re-download when the first failed (no sentinel)."""
        call_count = [0]
        fake = self._fake_pull_files

        def failing_then_succeeding(model_path, dst, allow_pattern=None, ignore_pattern=None, s3_credentials=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("simulated first-attempt failure")
            fake(model_path, dst, allow_pattern, ignore_pattern, s3_credentials)

        original = _ss_module.pull_files
        _ss_module.pull_files = failing_then_succeeding
        try:
            with self.assertRaises(RuntimeError):
                with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=self.dst_dir) as obj:
                    obj.pull_files()

            with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=self.dst_dir) as obj:
                obj.pull_files()
        finally:
            _ss_module.pull_files = original

        self.assertEqual(call_count[0], 2, "pull_files must be called again after a failed attempt")
        self.assertTrue(
            os.path.exists(os.path.join(self.dst_dir, ObjectStorageModel.SENTINEL_NAME))
        )

    # -----------------------------------------------------------------------
    # Single-process: trailing slash normalisation
    # -----------------------------------------------------------------------

    def test_model_path_normalised_to_trailing_slash(self):
        """Both with and without trailing slash must arrive at pull_files with a trailing slash."""
        received_paths = []
        fake = self._fake_pull_files

        def capturing(model_path, dst, allow_pattern=None, ignore_pattern=None, s3_credentials=None):
            received_paths.append(model_path)
            fake(model_path, dst, allow_pattern, ignore_pattern, s3_credentials)

        original = _ss_module.pull_files
        _ss_module.pull_files = capturing
        try:
            with ObjectStorageModel(model_path="s3://fake-bucket/fake-model", dst=self.dst_dir) as obj:
                obj.pull_files()
        finally:
            _ss_module.pull_files = original

        self.assertEqual(len(received_paths), 1)
        self.assertTrue(
            received_paths[0].endswith("/"),
            f"model_path passed to pull_files must end with '/'; got {received_paths[0]!r}",
        )

    # -----------------------------------------------------------------------
    # Single-process: accepted object-storage schemes
    # -----------------------------------------------------------------------

    def test_gs_path_accepted(self):
        original = _ss_module.pull_files
        _ss_module.pull_files = self._fake_pull_files
        try:
            with ObjectStorageModel(model_path="gs://fake-bucket/fake-model", dst=self.dst_dir) as obj:
                obj.pull_files()
        finally:
            _ss_module.pull_files = original

        self.assertTrue(
            os.path.exists(os.path.join(self.dst_dir, ObjectStorageModel.SENTINEL_NAME))
        )

    def test_azure_path_accepted(self):
        original = _ss_module.pull_files
        _ss_module.pull_files = self._fake_pull_files
        try:
            with ObjectStorageModel(model_path="az://fake-container/fake-model", dst=self.dst_dir) as obj:
                obj.pull_files()
        finally:
            _ss_module.pull_files = original

        self.assertTrue(
            os.path.exists(os.path.join(self.dst_dir, ObjectStorageModel.SENTINEL_NAME))
        )

    # -----------------------------------------------------------------------
    # Multi-process tests
    # -----------------------------------------------------------------------

    def test_concurrent_all_processes_succeed(self):
        results = self._run_concurrent(4)
        errors = [r for r in results if r["status"] != "ok"]
        self.assertEqual(errors, [], f"Some processes failed: {errors}")

    def test_concurrent_only_one_process_downloads(self):
        results = self._run_concurrent(4)
        downloaders = [r for r in results if r["status"] == "ok" and not r["skipped"]]
        self.assertEqual(
            len(downloaders),
            1,
            f"Expected exactly 1 downloading process, got {len(downloaders)}",
        )

    def test_concurrent_files_present_after_download(self):
        self._run_concurrent(4)
        for name in SOURCE_FILES:
            self.assertTrue(
                os.path.exists(os.path.join(self.dst_dir, name)),
                f"{name} missing after concurrent download",
            )

    def test_concurrent_sentinel_present_after_download(self):
        self._run_concurrent(4)
        self.assertTrue(
            os.path.exists(os.path.join(self.dst_dir, ObjectStorageModel.SENTINEL_NAME))
        )

    def test_concurrent_lock_upgrade_no_deadlock(self):
        """All processes start simultaneously before any sentinel exists, forcing them
        all to compete for the SH→EX upgrade. The old direct-promotion code would
        deadlock here; the release-then-acquire fix must complete within the timeout."""
        num_processes = 4
        # Sleep long enough that all processes acquire SH and reach the upgrade
        # point before any of them releases.
        original = _ss_module.pull_files
        _ss_module.pull_files = self._slow_fake_pull_files

        try:
            ctx = multiprocessing.get_context("fork")
            result_queue = ctx.Queue()
            # Use a barrier so all processes enter __init__ at the same time,
            # maximising the chance that they all hold SH simultaneously.
            barrier = ctx.Barrier(num_processes)

            def worker_with_barrier(dst_dir, q):
                try:
                    barrier.wait(timeout=10)
                    with ObjectStorageModel(model_path=FAKE_MODEL_PATH, dst=dst_dir) as obj:
                        obj.pull_files()
                    q.put({"status": "ok"})
                except Exception as exc:
                    q.put({"status": "error", "error": str(exc)})

            processes = [
                ctx.Process(target=worker_with_barrier, args=(self.dst_dir, result_queue))
                for _ in range(num_processes)
            ]
            for p in processes:
                p.start()
            timed_out = []
            for p in processes:
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()
                    timed_out.append(p)
            for p in timed_out:
                p.join(timeout=5)
        finally:
            _ss_module.pull_files = original

        self.assertEqual(timed_out, [], f"Processes deadlocked: {[p.pid for p in timed_out]}")
        results = []
        for _ in range(num_processes):
            try:
                results.append(result_queue.get_nowait())
            except Exception:
                pass
        errors = [r for r in results if r["status"] != "ok"]
        self.assertEqual(errors, [], f"Some processes failed: {errors}")


class TestObjectStorageModelDstPreExists(TestObjectStorageModel):
    """Re-runs all tests with dst already existing before ObjectStorageModel is
    constructed, exercising the shutil.rmtree + os.makedirs cleanup path in __init__."""
    dst_pre_exists = True


if __name__ == "__main__":
    unittest.main()
