import unittest
from unittest.mock import Mock, patch

import torch.distributed as dist

from runai_model_streamer.distributed_streamer.distributed_streamer import (
    _distributedStreamer,
    _distributedStreamerParams,
)


class TestBorrowedProcessGroup(unittest.TestCase):
    def test_rank_discovery_is_scoped_to_process_group(self):
        process_group = Mock(spec=dist.ProcessGroup)
        params = _distributedStreamerParams(process_group)

        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_world_size", return_value=2) as get_world_size,
            patch.object(dist, "get_rank", return_value=8),
            patch.object(dist, "get_global_rank", side_effect=[8, 9]),
            patch.object(dist, "new_group") as new_group,
            patch.object(dist, "all_gather_object") as all_gather_object,
        ):
            result = params.find_local_ranks()

        self.assertEqual(result, (2, 8, [[8, 9]]))
        get_world_size.assert_called_with(group=process_group)
        new_group.assert_not_called()
        all_gather_object.assert_not_called()

    def test_single_rank_group_preserves_global_rank(self):
        process_group = Mock(spec=dist.ProcessGroup)
        params = _distributedStreamerParams(process_group)

        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_world_size", return_value=1),
            patch.object(dist, "get_rank", return_value=7),
            patch.object(dist, "get_global_rank", return_value=7),
        ):
            result = params.find_local_ranks()

        self.assertEqual(result, (1, 7, [[7]]))

    def test_borrowed_process_group_is_not_destroyed(self):
        process_group = Mock(spec=dist.ProcessGroup)
        streamer = _distributedStreamer(Mock(), process_group)
        streamer.distribution_group = process_group

        with (
            patch.object(dist, "barrier") as barrier,
            patch.object(dist, "destroy_process_group") as destroy_process_group,
        ):
            streamer.__exit__(None, None, None)

        barrier.assert_called_once_with(group=process_group)
        destroy_process_group.assert_not_called()
        self.assertIsNone(streamer.distribution_group)

    def test_borrowed_process_group_is_used_without_creating_a_group(self):
        process_group = Mock(spec=dist.ProcessGroup)
        streamer = _distributedStreamer(Mock(), process_group)
        streamer.groups_by_ranks = [[8, 9]]

        with patch.object(dist, "new_group") as new_group:
            result = streamer.create_distribution_group()

        self.assertIs(result, process_group)
        self.assertEqual(streamer.local_group_global_ranks, [8, 9])
        new_group.assert_not_called()


if __name__ == "__main__":
    unittest.main()
