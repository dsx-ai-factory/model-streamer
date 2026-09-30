import os
import random
import unittest
from collections import Counter
from unittest.mock import patch
from typing import List, Dict, Tuple

# Assuming the classes and functions are in these locations for the test.
# You may need to adjust the import paths based on your project structure.
from runai_model_streamer.distributed_streamer.partition import (
    partition,
    partition_by_chunks,
    partition_by_files,
    partition_by_spans,
    partition_span_for_rank,
    partition_for_rank,
    get_total_number_of_chunks,
    get_total_size_of_partition,
    count_runs,
)
from runai_model_streamer.file_streamer import FileChunks

class TestPartitioning(unittest.TestCase):
    def setUp(self):
        """Set up a standard set of requests for use in multiple tests.

        The ids are deliberately NOT the requests' positions in this list. A source map's first element is
        the original FileChunks.id (see DistributedStreamer.rank_dicts_map), and the only production caller
        happens to assign id == position - so fixtures that do the same cannot tell the two apart, and a
        policy emitting positional indices would pass. These ids make that distinguishable.
        """
        self.requests: List[FileChunks] = [
            FileChunks.contiguous(101, path="file_A.dat", offset=1000, sizes=[100, 50, 200]), # Total: 350
            FileChunks.contiguous(7, path="file_B.dat", offset=0, sizes=[400]),               # Total: 400
            FileChunks.contiguous(55, path="file_A.dat", offset=5000, sizes=[80, 20]),        # Total: 100
            FileChunks.contiguous(23, path="file_C.dat", offset=800, sizes=[300, 150]),       # Total: 450
        ]
        self.total_size = sum(r.total_size() for r in self.requests) # 1300

    def _get_partition_total_size(self, partition: List[Tuple[FileChunks, dict]]) -> int:
        """Helper to calculate the total byte size of a single partition."""
        return sum(fc.total_size() for fc, _ in partition)

    def _verify_all_chunks_present(self, original_requests: List[FileChunks], partitions: List[List[Tuple[FileChunks, dict]]]):
        """
        Verifies that every individual range from the original requests is present in the output
        partitions, preserving path, offset, and size.

        Counted as a multiset, not a set: zero-sized ranges share an offset with the range that
        follows them, so two of them at the same offset are indistinguishable as tuples and a set
        would silently collapse them.
        """
        def census(requests):
            counter = Counter()
            for req in requests:
                for offset, size in zip(req.offsets, req.sizes):
                    counter[(req.path, offset, size)] += 1
            return counter

        original = census(original_requests)
        output = census([fc for partition in partitions for fc, _ in partition])

        self.assertEqual(original, output,
                         "Every original range must be present in the output partitions with correct path, offset, and size")

    def _verify_chunk_maps(self, original_requests: List[FileChunks], partitions: List[List[Tuple[FileChunks, dict]]]):
        """Verifies the correctness of the source maps for partitioned ranges."""
        total_mapped_chunks = 0
        original_chunk_data = {}
        for req in original_requests:
            for chunk_idx, (offset, size) in enumerate(zip(req.offsets, req.sizes)):
                original_chunk_data[(req.id, chunk_idx)] = (req.path, offset, size)

        for partition in partitions:
            for new_fc, source_map in partition:
                total_mapped_chunks += len(source_map)
                for new_chunk_idx, (orig_req_idx, orig_chunk_idx, map_chunk_size) in source_map.items():
                    original_pos_tuple = (orig_req_idx, orig_chunk_idx)
                    self.assertIn(original_pos_tuple, original_chunk_data)

                    orig_path, orig_offset, orig_size = original_chunk_data[original_pos_tuple]

                    self.assertEqual(new_fc.sizes[new_chunk_idx], orig_size, "Range size in new object must match original")
                    self.assertEqual(map_chunk_size, orig_size, "Range size in map must match original")
                    # the offset is carried, not derived from a prefix sum over the preceding sizes
                    self.assertEqual(new_fc.offsets[new_chunk_idx], orig_offset, "Offset must match original")
                    self.assertEqual(new_fc.path, orig_path, "Path must match original")

        self.assertEqual(len(original_chunk_data), total_mapped_chunks, "Total number of mapped ranges must equal total original ranges")

    def test_partition_by_files(self):
        """Tests for the partition_by_files function."""
        # Test case 1: Basic partitioning into 3 parts
        n = 3
        partitions = partition_by_files(self.requests, n)

        self.assertEqual(len(partitions), n, "Should return the correct number of partitions")

        partition_sizes = [self._get_partition_total_size(p) for p in partitions]
        self.assertEqual(sum(partition_sizes), self.total_size, "Total size should be conserved")
        self.assertCountEqual(partition_sizes, [450, 400, 450])

        self._verify_all_chunks_present(self.requests, partitions)
        self._verify_chunk_maps(self.requests, partitions)

        # Test case 2: Partitioning into 1 part
        n = 1
        partitions = partition_by_files(self.requests, n)
        self.assertEqual(len(partitions), n)
        self.assertEqual(self._get_partition_total_size(partitions[0]), self.total_size)

        self._verify_all_chunks_present(self.requests, partitions)
        self._verify_chunk_maps(self.requests, partitions)

        # Test case 3: Empty input list
        partitions = partition_by_files([], n)
        self.assertEqual(len(partitions), n)
        self.assertTrue(all(len(p) == 0 for p in partitions))

        # Test case 4: More partitions than requests
        n = 5
        partitions = partition_by_files(self.requests, n)
        self.assertEqual(len(partitions), n)
        self.assertEqual(sum(self._get_partition_total_size(p) for p in partitions), self.total_size)
        self.assertTrue(any(len(p) == 0 for p in partitions))

        self._verify_all_chunks_present(self.requests, partitions)
        self._verify_chunk_maps(self.requests, partitions)

    def test_source_maps_reference_original_ids_not_positions(self):
        """Both policies must key the source map on the original FileChunks.id.

        The receiving ranks look a tensor up by the id carried in the broadcast metadata
        (DistributedStreamer.rank_dicts_map -> safetensors metadata), so a policy that emitted the
        request's POSITION in the input list would hand back the wrong tensor whenever the caller's ids
        are not 0..n-1. Asserted for both policies together because they are interchangeable behind
        RUNAI_STREAMER_PARTITION_POLICY and must agree on this.
        """
        original_ids = {request.id for request in self.requests}
        positions = set(range(len(self.requests)))
        self.assertTrue(original_ids.isdisjoint(positions),
                        "fixture must use ids that are not positions, or this test proves nothing")

        for policy in (partition_by_files, partition_by_chunks):
            for n in (1, 3, 5):
                mapped_ids = {
                    source_id
                    for partition in policy(self.requests, n)
                    for _new_fc, source_map in partition
                    for source_id, _chunk_idx, _size in source_map.values()
                }
                self.assertEqual(
                    mapped_ids, original_ids,
                    f"{policy.__name__} with n={n} mapped to {sorted(mapped_ids)}; "
                    f"expected the original ids {sorted(original_ids)}",
                )

    def test_partition_by_chunks(self):
        """Tests for the partition_by_chunks function."""
        # Test case 1: Basic partitioning into 3 parts
        n = 3
        partitions = partition_by_chunks(self.requests, n)

        self.assertEqual(len(partitions), n, "Should return the correct number of partitions")

        partition_sizes = [self._get_partition_total_size(p) for p in partitions]
        self.assertEqual(sum(partition_sizes), self.total_size, "Total size should be conserved")
        self.assertCountEqual(partition_sizes, [450, 420, 430])

        self._verify_all_chunks_present(self.requests, partitions)
        self._verify_chunk_maps(self.requests, partitions)

        # Test case 2: Partitioning into 1 part
        n = 1
        partitions = partition_by_chunks(self.requests, n)
        self.assertEqual(len(partitions), n)
        self.assertEqual(self._get_partition_total_size(partitions[0]), self.total_size)

        self._verify_all_chunks_present(self.requests, partitions)
        self._verify_chunk_maps(self.requests, partitions)

    def test_one_file_chunks_per_path_per_rank(self):
        """A rank's ranges for a path are emitted as a single FileChunks, whether or not they are
        contiguous. Previously the reconstruction split them into one FileChunks per contiguous run,
        which on a 66-shard model produced 16308 entries with a mean run length of 1.14."""
        contiguous_requests = [
            FileChunks.contiguous(0, path="A", offset=0, sizes=[10, 20]),
            FileChunks.contiguous(1, path="A", offset=30, sizes=[5]),
        ]
        partitions = partition_by_chunks(contiguous_requests, 1)
        self.assertEqual(len(partitions[0]), 1)

        new_fc, _ = partitions[0][0]
        self.assertListEqual(new_fc.offsets, [0, 10, 30])
        self.assertListEqual(new_fc.sizes, [10, 20, 5])

        self._verify_all_chunks_present(contiguous_requests, partitions)
        self._verify_chunk_maps(contiguous_requests, partitions)

    def test_non_contiguous_ranges_share_one_file_chunks(self):
        """Ranges carry their own offsets, so a gap no longer forces a second FileChunks."""
        gapped_requests = [
            FileChunks.contiguous(0, path="A", offset=0, sizes=[10]),
            FileChunks.contiguous(1, path="A", offset=9000, sizes=[10]),
        ]
        partitions = partition_by_chunks(gapped_requests, 1)
        self.assertEqual(len(partitions[0]), 1)

        new_fc, _ = partitions[0][0]
        self.assertListEqual(new_fc.offsets, [0, 9000])
        self.assertListEqual(new_fc.sizes, [10, 10])

        self._verify_all_chunks_present(gapped_requests, partitions)
        self._verify_chunk_maps(gapped_requests, partitions)

    def test_ranges_are_sorted_by_offset(self):
        """Sorting is what lets the C++ assigner coalesce a rank's ranges back into contiguous
        transfers - it only merges ranges that arrive in ascending file order."""
        requests = [FileChunks(0, path="A", offsets=[500, 0, 250], sizes=[10, 10, 10])]
        partitions = partition_by_chunks(requests, 1)

        new_fc, _ = partitions[0][0]
        self.assertListEqual(new_fc.offsets, [0, 250, 500])

    def test_partition_by_chunks_with_zero_size_chunks(self):
        """Zero-sized ranges are kept, not dropped.

        A zero-element tensor is a real safetensors entry that the reference implementation yields
        with its shape intact, and the C++ layer answers one response per range including this one.
        Dropping them here would make the distributed path yield fewer tensors than the
        single-process path.
        """
        requests_with_zero = [
            FileChunks.contiguous(0, path="Z.dat", offset=0, sizes=[10, 50, 0, 100]),
            FileChunks.contiguous(1, path="Y.dat", offset=10, sizes=[0, 0, 25]),
        ]
        n = 2
        partitions = partition_by_chunks(requests_with_zero, n)

        self.assertEqual(len(partitions), n)
        total_size = sum(self._get_partition_total_size(p) for p in partitions)
        expected_size = (10 + 50 + 100) + 25
        self.assertEqual(total_size, expected_size)

        # every range is accounted for, the zero-sized ones included
        self.assertEqual(get_total_number_of_chunks(partitions), 7)

        self._verify_all_chunks_present(requests_with_zero, partitions)
        self._verify_chunk_maps(requests_with_zero, partitions)

    def test_zero_sized_range_sorts_before_a_range_at_the_same_offset(self):
        """A zero-sized range shares the offset of whatever follows it. Sorted after that range it
        would land at the far end of it and break the contiguous transfer twice; sorted before, both
        the file offsets and the destinations stay adjacent and the transfer survives."""
        requests = [FileChunks.contiguous(0, path="A", offset=0, sizes=[10, 0, 50])]
        partitions = partition_by_chunks(requests, 1)

        new_fc, _ = partitions[0][0]
        self.assertListEqual(new_fc.offsets, [0, 10, 10])
        self.assertListEqual(new_fc.sizes, [10, 0, 50])

    def tearDown(self):
        """No cleanup needed for these tests."""
        pass

if __name__ == "__main__":
    unittest.main()


class TestPartitionBySpans(unittest.TestCase):
    """Each rank gets one contiguous span of the model, cut at tensor boundaries.

    What these check is correctness at the edges. The value of the policy - fewer runs, same balance -
    is measured in plan_partition_spans.md against models far larger than a unit test can build.
    """

    def _files(self, count: int, per_file: int, size: int = 100) -> List[FileChunks]:
        return [
            FileChunks.contiguous(1000 + f, path=f"shard_{f}.st", offset=0, sizes=[size] * per_file)
            for f in range(count)
        ]

    def _loads(self, partitions) -> List[int]:
        return [sum(fc.total_size() for fc, _ in p) for p in partitions]

    def _flat(self, partitions):
        """(path, offset, size) for every range, across every rank."""
        out = []
        for partition in partitions:
            for fc, _ in partition:
                out.extend(zip([fc.path] * len(fc.sizes), fc.offsets, fc.sizes))
        return out

    def test_every_range_is_assigned_exactly_once(self):
        # The property the whole policy rests on. A cut that drops or duplicates a tensor would still
        # look balanced and still look contiguous.
        requests = self._files(5, 7)
        partitions = partition_by_spans(requests, 3)

        expected = Counter()
        for request in requests:
            expected.update(zip([request.path] * len(request.sizes), request.offsets, request.sizes))

        self.assertEqual(Counter(self._flat(partitions)), expected)

    def test_each_rank_gets_one_contiguous_span(self):
        # Contiguity is the point. Within a file, a rank's ranges must be consecutive - no gaps that
        # another rank owns.
        requests = self._files(4, 10)
        partitions = partition_by_spans(requests, 4)

        for partition in partitions:
            for fc, _ in partition:
                cursor = fc.offsets[0]
                for offset, size in zip(fc.offsets, fc.sizes):
                    self.assertEqual(offset, cursor, "a rank's ranges must be adjacent in the file")
                    cursor += size

    def test_fewer_files_than_ranks_leaves_no_rank_idle(self):
        # The case that rules out assigning whole files: 4 shards across 8 ranks. Whole files would
        # leave four ranks with nothing, which is a normal Llama-3-8B deployment at TP=8.
        partitions = partition_by_spans(self._files(4, 100), 8)

        loads = self._loads(partitions)
        self.assertTrue(all(load > 0 for load in loads), f"a rank got nothing: {loads}")

    def test_more_ranks_than_tensors(self):
        # Some ranks genuinely have nothing to read. That must be an empty partition rather than a
        # crash or a missing rank - DistributedStreamer already handles an empty share.
        partitions = partition_by_spans(self._files(1, 3), 8)

        self.assertEqual(len(partitions), 8)
        self.assertEqual(sum(1 for p in partitions if p), 3)
        self.assertEqual(sum(self._loads(partitions)), 300)

    def test_single_rank_gets_everything(self):
        partitions = partition_by_spans(self._files(3, 5), 1)

        self.assertEqual(len(partitions), 1)
        self.assertEqual(len(self._flat(partitions)), 15)

    def test_one_huge_tensor_bounds_the_split(self):
        # No partition can be more even than the largest tensor, because it has to live somewhere
        # whole. The search must start from that bound rather than fight it.
        requests = [FileChunks.contiguous(1, path="a.st", offset=0, sizes=[1000] + [10] * 20)]
        partitions = partition_by_spans(requests, 4)

        loads = self._loads(partitions)
        self.assertEqual(sum(loads), 1200)
        self.assertGreaterEqual(max(loads), 1000, "the big tensor cannot be split")
        self.assertLess(max(loads), 1200, "and the rest must still be spread")

    def test_zero_sized_ranges_are_kept(self):
        # A zero-element tensor is a real safetensors entry. Dropping it would make the distributed
        # path yield fewer tensors than the single-process path.
        requests = [FileChunks(id=9, path="a.st", offsets=[0, 100, 100], sizes=[100, 0, 50])]
        partitions = partition_by_spans(requests, 2)

        self.assertEqual(len(self._flat(partitions)), 3)
        self.assertIn(("a.st", 100, 0), self._flat(partitions))

    def test_empty_input(self):
        self.assertEqual(partition_by_spans([], 4), [[], [], [], []])

    def test_rejects_a_non_positive_rank_count(self):
        with self.assertRaises(ValueError):
            partition_by_spans(self._files(2, 2), 0)

    def test_source_maps_reference_original_ids_not_positions(self):
        # Same trap the other policies have a test for: the map's first element is the original
        # FileChunks.id, and a policy emitting the position instead would send the wrong tensor.
        requests = [
            FileChunks.contiguous(101, path="a.st", offset=0, sizes=[10, 10]),
            FileChunks.contiguous(7, path="b.st", offset=0, sizes=[10, 10]),
        ]
        partitions = partition_by_spans(requests, 2)

        seen_ids = {
            origin[0]
            for partition in partitions
            for _, source_map in partition
            for origin in source_map.values()
        }
        self.assertEqual(seen_ids, {101, 7})

    def test_ranges_within_a_file_are_ordered_by_offset(self):
        # Adjacency in the stream must mean adjacency on disk, or the spans are contiguous in name
        # only. The input here is deliberately out of order.
        requests = [FileChunks(id=1, path="a.st", offsets=[300, 0, 100], sizes=[50, 100, 200])]
        partitions = partition_by_spans(requests, 1)

        fc = partitions[0][0][0]
        self.assertEqual(fc.offsets, [0, 100, 300])

    def test_similar_files_split_on_file_boundaries(self):
        # Not a requirement, but the reason no whole-file phase is needed: when files are alike and
        # there are at least as many as ranks, the best split lands on file boundaries by itself.
        partitions = partition_by_spans(self._files(8, 20), 8)

        for partition in partitions:
            self.assertEqual(len(partition), 1, "each rank should get exactly one whole file")


class TestPartitionRandomised(unittest.TestCase):
    """Random models against every policy: nothing lost, nothing duplicated, nothing mis-mapped.

    Seeded and looped rather than truly random, so a failure is reproducible: the seed is in the
    subTest label, and re-running reproduces exactly that model.

    Run against all three policies because the property is not specific to any of them. A policy that
    drops or duplicates a tensor still looks balanced and still looks contiguous, so nothing else here
    would notice.
    """

    POLICIES = (partition_by_chunks, partition_by_files, partition_by_spans)

    def _random_model(self, rng: random.Random) -> List[FileChunks]:
        """A model with an awkward shape on purpose.

        Includes empty files, zero-sized ranges, wildly uneven sizes, and two requests sharing a path -
        all of which occur in practice and each of which has broken something before.
        """
        requests = []
        next_id = rng.randrange(1000)

        for index in range(rng.randint(1, 12)):
            count = rng.randint(0, 40)

            offsets, sizes = [], []
            cursor = rng.randrange(4096)
            for _ in range(count):
                # A few zero-sized ranges, and a few very large tensors among small ones.
                if rng.random() < 0.15:
                    size = 0
                elif rng.random() < 0.1:
                    size = rng.randint(1 << 18, 1 << 22)
                else:
                    size = rng.randint(1, 4096)
                offsets.append(cursor)
                sizes.append(size)
                cursor += size

            # Ids are not positions, and two requests may share a path.
            next_id += rng.randint(1, 5)
            path = f"shard_{index if rng.random() < 0.8 else 0}.st"
            requests.append(FileChunks(id=next_id, path=path, offsets=offsets, sizes=sizes))

        return requests

    def _expected(self, requests: List[FileChunks]):
        """Every tensor, keyed by the identity the broadcast uses."""
        return {
            (request.id, index): (request.path, request.offsets[index], request.sizes[index])
            for request in requests
            for index in range(len(request.sizes))
        }

    def _produced(self, partitions):
        """Every tensor the partition emitted, keyed the same way, with what it says it is."""
        produced = []
        for partition in partitions:
            for file_chunks, source_map in partition:
                for new_index, (original_id, original_index, size) in source_map.items():
                    produced.append((
                        (original_id, original_index),
                        (file_chunks.path, file_chunks.offsets[new_index], file_chunks.sizes[new_index]),
                        size,
                    ))
        return produced

    def test_every_tensor_appears_exactly_once_and_unchanged(self):
        for seed in range(300):
            rng = random.Random(seed)
            requests = self._random_model(rng)
            ranks = rng.randint(1, 10)
            expected = self._expected(requests)

            for policy in self.POLICIES:
                with self.subTest(seed=seed, policy=policy.__name__, ranks=ranks):
                    partitions = policy(requests, ranks)

                    self.assertEqual(len(partitions), ranks, "one partition per rank, even if empty")

                    produced = self._produced(partitions)
                    keys = Counter(key for key, _, _ in produced)

                    # Nothing duplicated. A tensor read twice is wasted bandwidth at best, and at worst
                    # two ranks broadcast the same tensor and the receivers disagree.
                    duplicated = [key for key, count in keys.items() if count > 1]
                    self.assertEqual(duplicated, [], f"tensors assigned more than once: {duplicated[:5]}")

                    # Nothing lost. A missing tensor hangs the load: the chunk count never reaches zero.
                    missing = set(expected) - set(keys)
                    self.assertEqual(missing, set(), f"tensors assigned to no rank: {sorted(missing)[:5]}")

                    # And each one still describes the tensor it claims to be. A partition can be a
                    # perfect bijection and still hand back the wrong bytes.
                    for key, described, size in produced:
                        self.assertEqual(described, expected[key], f"{key} was re-pointed")
                        self.assertEqual(size, expected[key][2], f"{key} has the wrong size in its map")

    def test_no_bytes_are_gained_or_lost(self):
        """The cheap whole-model check, which catches a class the per-tensor one can miss."""
        for seed in range(300, 400):
            rng = random.Random(seed)
            requests = self._random_model(rng)
            ranks = rng.randint(1, 10)
            total = sum(request.total_size() for request in requests)

            for policy in self.POLICIES:
                with self.subTest(seed=seed, policy=policy.__name__, ranks=ranks):
                    partitions = policy(requests, ranks)
                    partitioned = sum(fc.total_size() for p in partitions for fc, _ in p)
                    self.assertEqual(partitioned, total)


class TestPartitionSpanForRank(unittest.TestCase):
    """Building one rank's span must give exactly what building all of them would have given.

    That equivalence is the whole safety argument for the optimisation, so it is what these test -
    not the timing, which belongs in a benchmark.
    """

    def _random_model(self, rng: random.Random) -> List[FileChunks]:
        return TestPartitionRandomised()._random_model(rng)

    def test_matches_the_all_ranks_form(self):
        for seed in range(200):
            rng = random.Random(seed)
            requests = self._random_model(rng)
            ranks = rng.randint(1, 10)
            everything = partition_by_spans(requests, ranks)

            for rank in range(ranks):
                with self.subTest(seed=seed, ranks=ranks, rank=rank):
                    mine = partition_span_for_rank(requests, ranks, rank)

                    # The ids differ - one rank numbering from 0 cannot match a global numbering - so
                    # compare what the ids are FOR: which file, which ranges, and where they came from.
                    self.assertEqual(
                        [(fc.path, fc.offsets, fc.sizes) for fc, _ in mine.partition],
                        [(fc.path, fc.offsets, fc.sizes) for fc, _ in everything[rank]],
                    )
                    self.assertEqual(
                        [source_map for _, source_map in mine.partition],
                        [source_map for _, source_map in everything[rank]],
                    )

    def test_byte_totals_match_the_partitions_they_replace(self):
        # These stand in for log_partition_info, which used to read every rank's objects.
        for seed in range(200, 300):
            rng = random.Random(seed)
            requests = self._random_model(rng)
            ranks = rng.randint(1, 10)
            everything = partition_by_spans(requests, ranks)
            expected = [sum(fc.total_size() for fc, _ in p) for p in everything]

            with self.subTest(seed=seed, ranks=ranks):
                self.assertEqual(partition_span_for_rank(requests, ranks, 0).sizes_by_rank, expected)

    def test_total_chunks_is_the_whole_model_not_the_rank(self):
        # The broadcast loop counts this down to zero across ALL ranks. A rank reporting only its own
        # share would either stop early or wait for tensors nobody is sending.
        for seed in range(300, 400):
            rng = random.Random(seed)
            requests = self._random_model(rng)
            ranks = rng.randint(1, 10)
            expected = get_total_number_of_chunks(partition_by_spans(requests, ranks))

            for rank in range(ranks):
                with self.subTest(seed=seed, ranks=ranks, rank=rank):
                    mine = partition_span_for_rank(requests, ranks, rank)
                    self.assertEqual(mine.total_chunks, expected)
                    self.assertEqual(mine.total_chunks, sum(len(r.sizes) for r in requests))

    def test_the_spans_together_cover_everything_exactly_once(self):
        # The same property the randomised test checks for the all-ranks form, asked of the per-rank
        # form directly - because that is the one production will call.
        for seed in range(400, 500):
            rng = random.Random(seed)
            requests = self._random_model(rng)
            ranks = rng.randint(1, 10)

            expected = Counter(
                (request.id, index)
                for request in requests
                for index in range(len(request.sizes))
            )
            produced = Counter()
            for rank in range(ranks):
                for _, source_map in partition_span_for_rank(requests, ranks, rank).partition:
                    produced.update((origin[0], origin[1]) for origin in source_map.values())

            with self.subTest(seed=seed, ranks=ranks):
                self.assertEqual(produced, expected)

    def test_rejects_a_rank_outside_the_partition(self):
        requests = [FileChunks.contiguous(1, path="a.st", offset=0, sizes=[10])]
        for bad in (-1, 4, 99):
            with self.subTest(rank=bad):
                with self.assertRaises(ValueError):
                    partition_span_for_rank(requests, 4, bad)

    def test_empty_input_still_reports_every_rank(self):
        # log_partition_info prints one line per rank whatever happened, so the list must be full
        # length rather than empty.
        mine = partition_span_for_rank([], 4, 2)
        self.assertEqual(mine.partition, [])
        self.assertEqual(mine.sizes_by_rank, [0, 0, 0, 0])
        self.assertEqual(mine.total_chunks, 0)


class TestPartitionForRank(unittest.TestCase):
    """The dispatcher must give every policy the same answer the old path gave.

    DistributedStreamer no longer builds every rank's share, so this is what guarantees the two
    policies that cannot skip that work are unaffected. `spans` has its own equivalence tests; these
    cover all three through the interface production calls.
    """

    POLICIES = ("chunks", "files", "spans")

    def _random_model(self, rng: random.Random) -> List[FileChunks]:
        return TestPartitionRandomised()._random_model(rng)

    def test_matches_the_old_build_everything_path(self):
        for policy in self.POLICIES:
            with patch.dict(os.environ, {"RUNAI_STREAMER_PARTITION_POLICY": policy}):
                for seed in range(60):
                    rng = random.Random(seed)
                    requests = self._random_model(rng)
                    ranks = rng.randint(1, 8)

                    # What DistributedStreamer used to do: build all of them, then read three things.
                    everything = partition(requests, ranks)
                    old_sizes = [get_total_size_of_partition(p) for p in everything]
                    old_total = get_total_number_of_chunks(everything)

                    for rank in range(ranks):
                        with self.subTest(policy=policy, seed=seed, ranks=ranks, rank=rank):
                            mine = partition_for_rank(requests, ranks, rank)

                            self.assertEqual(
                                [(fc.path, fc.offsets, fc.sizes) for fc, _ in mine.partition],
                                [(fc.path, fc.offsets, fc.sizes) for fc, _ in everything[rank]],
                            )
                            self.assertEqual(
                                [source_map for _, source_map in mine.partition],
                                [source_map for _, source_map in everything[rank]],
                            )
                            self.assertEqual(mine.sizes_by_rank, old_sizes)
                            self.assertEqual(mine.total_chunks, old_total)

    def test_every_policy_reports_a_size_for_every_rank(self):
        # log_partition_info prints one entry per rank. A short list would silently drop ranks from
        # the log, which is where an uneven partition would first be noticed.
        for policy in self.POLICIES:
            with patch.dict(os.environ, {"RUNAI_STREAMER_PARTITION_POLICY": policy}):
                for ranks in (1, 3, 8):
                    with self.subTest(policy=policy, ranks=ranks):
                        requests = self._random_model(random.Random(ranks))
                        mine = partition_for_rank(requests, ranks, 0)
                        self.assertEqual(len(mine.sizes_by_rank), ranks)
                        self.assertEqual(
                            sum(mine.sizes_by_rank),
                            sum(r.total_size() for r in requests),
                        )

    def test_every_policy_rejects_a_rank_outside_the_partition(self):
        requests = [FileChunks.contiguous(1, path="a.st", offset=0, sizes=[10])]
        for policy in self.POLICIES:
            with patch.dict(os.environ, {"RUNAI_STREAMER_PARTITION_POLICY": policy}):
                for bad in (-1, 4):
                    with self.subTest(policy=policy, rank=bad):
                        with self.assertRaises(ValueError):
                            partition_for_rank(requests, 4, bad)

    def test_an_unknown_policy_is_still_rejected(self):
        # The dispatcher must not quietly fall through to a default. A typo in the variable should
        # fail loudly rather than pick a policy nobody asked for.
        with patch.dict(os.environ, {"RUNAI_STREAMER_PARTITION_POLICY": "spanz"}):
            with self.assertRaises(ValueError):
                partition_for_rank([FileChunks.contiguous(1, "a.st", 0, [10])], 2, 0)


class TestPartitionWithGappedInput(unittest.TestCase):
    """tensor_names filtering produces gapped FileChunks (ranges with holes where excluded
    tensors sat) - no existing fixture here covers that. Confirms partition_for_rank handles it.
    """

    POLICIES = ("chunks", "files", "spans")

    def _gapped_requests(self) -> List[FileChunks]:
        # Offsets 0, 10, 30, 35, 80: each gap stands in for an excluded tensor.
        return [
            FileChunks(id=1, path="model-00001.safetensors", offsets=[0, 10, 30, 35, 80], sizes=[10, 15, 5, 40, 20]),
            FileChunks(id=2, path="model-00002.safetensors", offsets=[5, 200], sizes=[3, 7]),
        ]

    def test_every_kept_range_is_assigned_exactly_once(self):
        requests = self._gapped_requests()
        expected = Counter(
            (request.id, index)
            for request in requests
            for index in range(len(request.sizes))
        )
        for policy in self.POLICIES:
            with patch.dict(os.environ, {"RUNAI_STREAMER_PARTITION_POLICY": policy}):
                for ranks in (1, 2, 3):
                    produced = Counter()
                    for rank in range(ranks):
                        for _, source_map in partition_for_rank(requests, ranks, rank).partition:
                            produced.update((origin[0], origin[1]) for origin in source_map.values())
                    with self.subTest(policy=policy, ranks=ranks):
                        self.assertEqual(produced, expected)

    def test_byte_totals_and_chunk_count_are_preserved(self):
        requests = self._gapped_requests()
        total_bytes = sum(r.total_size() for r in requests)
        total_chunks = sum(len(r.sizes) for r in requests)
        for policy in self.POLICIES:
            with patch.dict(os.environ, {"RUNAI_STREAMER_PARTITION_POLICY": policy}):
                for ranks in (1, 2, 3):
                    mine = partition_for_rank(requests, ranks, 0)
                    with self.subTest(policy=policy, ranks=ranks):
                        # The gaps themselves are never assigned to anyone - only the kept
                        # tensors' own bytes count, which is what this equality confirms.
                        self.assertEqual(sum(mine.sizes_by_rank), total_bytes)
                        self.assertEqual(mine.total_chunks, total_chunks)


class TestRunCount(unittest.TestCase):
    """The number that prices direct I/O: how many sequential reads a rank's share becomes.

    It is not a performance test - it asserts the count is right, so that a log reporting it can be
    believed and so a policy that quietly loses locality shows up as a number rather than as a slower
    load with no explanation.
    """

    def _files(self, count: int, per_file: int) -> List[FileChunks]:
        return [
            FileChunks.contiguous(100 + f, path=f"s{f}.st", offset=0, sizes=[64] * per_file)
            for f in range(count)
        ]

    def test_adjacent_ranges_are_one_run(self):
        partition = [(FileChunks.contiguous(1, "a.st", 0, [10, 20, 30]), {})]
        self.assertEqual(count_runs(partition), 1)

    def test_a_gap_starts_a_new_run(self):
        # 10 at 0, then 20 at 500 - the second does not continue the first.
        partition = [(FileChunks(id=1, path="a.st", offsets=[0, 500], sizes=[10, 20]), {})]
        self.assertEqual(count_runs(partition), 2)

    def test_a_second_file_always_starts_a_new_run(self):
        # A read cannot span two files however the offsets line up, which is what makes the file count
        # the floor.
        partition = [
            (FileChunks.contiguous(1, "a.st", 0, [10]), {}),
            (FileChunks.contiguous(2, "b.st", 10, [10]), {}),
        ]
        self.assertEqual(count_runs(partition), 2)

    def test_zero_sized_ranges_do_not_break_a_run(self):
        # A zero-element tensor sits at the same offset as the range after it and consumes nothing, so
        # it must not be counted as a discontinuity.
        partition = [(FileChunks(id=1, path="a.st", offsets=[0, 10, 10], sizes=[10, 0, 20]), {})]
        self.assertEqual(count_runs(partition), 1)

    def test_an_empty_partition_has_no_runs(self):
        self.assertEqual(count_runs([]), 0)
        self.assertEqual(count_runs([(FileChunks(id=1, path="a.st", offsets=[], sizes=[]), {})]), 0)

    def test_spans_reaches_the_floor_and_chunks_does_not(self):
        # The whole reason the policy exists, as a number: a rank's share under `spans` is one read per
        # file, while under `chunks` it approaches one read per tensor.
        requests = self._files(count=8, per_file=50)
        total_tensors = 8 * 50

        span_runs = sum(count_runs(p) for p in partition_by_spans(requests, 4))
        chunk_runs = sum(count_runs(p) for p in partition_by_chunks(requests, 4))

        self.assertLessEqual(span_runs, 8 + 3, "spans should reach one read per file, plus the cuts")
        self.assertGreater(chunk_runs, span_runs * 10,
                           f"chunks scatters: {chunk_runs} reads against {span_runs}")
        self.assertLessEqual(chunk_runs, total_tensors)

    def test_the_floor_is_one_read_per_file_touched(self):
        # A run cannot span two files, so no policy can beat the number of files a rank holds.
        requests = self._files(count=6, per_file=30)
        for policy in (partition_by_spans, partition_by_chunks, partition_by_files):
            for partition in policy(requests, 3):
                with self.subTest(policy=policy.__name__):
                    self.assertGreaterEqual(count_runs(partition), len(partition))
