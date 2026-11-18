# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""broadcast() sends the slot it was given, and yields views into THAT slot.

The distribution loop is wired to the staged source, whose slot is a view of the staging buffer. So
sending the staging buffer and sending the slot are the same bytes, and no end-to-end test can tell
them apart. A source that is read into directly has no staging buffer at all, and then the two stop
being the same thing.

These tests give broadcast() a slot that is NOT the staging buffer, which is the only way to see
which of the two it uses."""

import unittest
from unittest.mock import patch

import torch

from runai_model_streamer.distributed_streamer.distribution_source import SlotRange
from runai_model_streamer.distributed_streamer.distributed_streamer import (
    _distributedStreamer,
    write_metadata_rows,
)


GROUP = object()


class Sender(unittest.TestCase):
    """One rank, which is therefore always the broadcasting rank."""

    def setUp(self):
        self.streamer = _distributedStreamer.__new__(_distributedStreamer)
        self.streamer.group_size = 1
        self.streamer.original_group_rank = 0
        self.streamer.local_group_global_ranks = [0]
        self.streamer.distribution_group = GROUP
        self.streamer.rank = 0

        self.sent_tensors = []

    def run_broadcast(self, sent, ranges, chunks_to_read=None, count=None):
        """Drive the sender branch and collect both what went out and what came back.

        The decoy is what the loop used to pass: a buffer the same length as the slot, holding
        different bytes. Nothing here may touch it."""
        self.decoy = torch.zeros(len(sent) if sent is not None else 1, dtype=torch.uint8)

        metadata = torch.zeros(16, 4, dtype=torch.int64)
        write_metadata_rows(metadata, ranges)

        chunks_to_read = [len(ranges)] if chunks_to_read is None else chunks_to_read

        def record(tensor, src=None, group=None):
            self.sent_tensors.append(tensor)

        with patch(
            "runai_model_streamer.distributed_streamer.distributed_streamer.dist.broadcast",
            side_effect=record,
        ):
            return list(
                self.streamer.broadcast(
                    chunks_to_read,
                    metadata,
                    torch.zeros(16, 4, dtype=torch.int64),
                    sent,
                    self.decoy,
                    len(ranges) if count is None else count,
                )
            ), metadata

    def slot(self):
        """Position-sensitive bytes, so a wrong offset cannot read as a right one."""
        return torch.arange(1, 1 + 96, dtype=torch.uint8)

    def test_the_slot_itself_goes_on_the_wire(self):
        sent = self.slot()
        ranges = [SlotRange(4, 0, 0, 40), SlotRange(4, 1, 64, 32)]

        self.run_broadcast(sent, ranges)

        data = self.sent_tensors[1]        # [0] is the metadata
        self.assertIs(data, sent, "the sender passes the source's own tensor through")
        self.assertEqual(data.data_ptr(), sent.data_ptr())
        self.assertNotEqual(data.data_ptr(), self.decoy.data_ptr())

    def test_the_yields_are_views_into_the_slot(self):
        sent = self.slot()
        ranges = [SlotRange(4, 0, 0, 40), SlotRange(4, 1, 64, 32)]

        yielded, _ = self.run_broadcast(sent, ranges)

        self.assertEqual([(f, i) for f, i, _ in yielded], [(4, 0), (4, 1)])
        for buffer, r in zip([b for _, _, b in yielded], ranges):
            self.assertEqual(buffer.data_ptr(), sent.data_ptr() + r.offset)
            self.assertEqual(len(buffer), r.size)
            self.assertEqual(buffer.tolist(), sent[r.offset : r.offset + r.size].tolist())

    def test_a_yield_follows_the_slot_when_it_changes(self):
        """A view, not a copy. The loop releases the slot only after the yields for this reason."""
        sent = self.slot()
        ranges = [SlotRange(4, 0, 0, 40)]

        yielded, _ = self.run_broadcast(sent, ranges)
        buffer = yielded[0][2]
        sent[7] = 200

        self.assertEqual(buffer[7].item(), 200)

    def test_what_is_sent_is_the_size_the_metadata_implies(self):
        """The sender sizes the collective by the slot, the receivers by these rows.

        They are two different numbers in two different places and a mismatch is a hang, not an
        error. The contract makes them equal: the slot is the used prefix, so it ends where the
        last range ends."""
        sent = self.slot()
        ranges = [SlotRange(4, 0, 0, 40), SlotRange(4, 1, 64, 32)]

        _, metadata = self.run_broadcast(sent, ranges)

        last = metadata[len(ranges)]
        self.assertEqual(self.sent_tensors[1].numel(), 96)
        self.assertEqual(last[3].item() + last[2].item(), 96)

    def test_a_rank_with_nothing_to_send_sends_only_metadata(self):
        """sent is None then, and the count guard is what keeps it unread."""
        yielded, _ = self.run_broadcast(None, [], chunks_to_read=[0])

        self.assertEqual(yielded, [])
        self.assertEqual(len(self.sent_tensors), 1, "the metadata, and no data collective")


class Receiver(unittest.TestCase):
    """The other half of the loop, which no other test enters.

    Every test above sets group_size to 1, so this rank is always the broadcasting one and the
    receiving branch never runs. Two ranks are enough to reach it: rank 0 sends, we are rank 1."""

    RECEIVE_CAPACITY = 4096

    def setUp(self):
        self.streamer = _distributedStreamer.__new__(_distributedStreamer)
        self.streamer.group_size = 2
        self.streamer.original_group_rank = 1
        self.streamer.local_group_global_ranks = [0, 1]
        self.streamer.distribution_group = GROUP
        self.streamer.rank = 1

        self.received_buffer = torch.arange(
            1, 1 + self.RECEIVE_CAPACITY, dtype=torch.int64).to(torch.uint8)
        self.received_metadata = torch.zeros(16, 4, dtype=torch.int64)
        self.sent_tensors = []

    def receive(self, offset, size):
        """Run one round where rank 0 sends a batch of one range at (offset, size).

        The metadata arrives through the collective, so the fake writes it into the tensor the
        production code hands over - which is how the real one learns what is coming."""
        peer_rows = [(7, 3, size, offset)]

        def fake(tensor, src=None, group=None):
            self.sent_tensors.append((tensor, src))
            if tensor is self.received_metadata:
                tensor[0, 0] = len(peer_rows)
                for i, row in enumerate(peer_rows):
                    tensor[i + 1] = torch.tensor(list(row))

        chunks_to_read = [1]
        with patch(
            "runai_model_streamer.distributed_streamer.distributed_streamer.dist.broadcast",
            side_effect=fake,
        ):
            return list(
                self.streamer.broadcast(
                    chunks_to_read,
                    torch.zeros(16, 4, dtype=torch.int64),
                    self.received_metadata,
                    None,                       # nothing of our own to send
                    self.received_buffer,
                    0,
                )
            )

    def test_a_peer_sending_more_than_this_rank_can_hold_is_refused(self):
        """Slicing would CLAMP instead, and the collective would then disagree about its size: an
        abort under gloo, a hang under NCCL."""
        with self.assertRaises(RuntimeError) as caught:
            self.receive(offset=0, size=self.RECEIVE_CAPACITY + 1)

        message = str(caught.exception)
        self.assertIn(str(self.RECEIVE_CAPACITY + 1), message, "it names what the peer is sending")
        self.assertIn(str(self.RECEIVE_CAPACITY), message, "and what this rank can hold")

    def test_it_refuses_before_posting_the_data_collective(self):
        """Posting it and then raising is a different failure: the peer would already be sending
        into a collective this rank has abandoned."""
        with self.assertRaises(RuntimeError):
            self.receive(offset=0, size=self.RECEIVE_CAPACITY + 1)

        self.assertEqual(len(self.sent_tensors), 1, "the metadata, and nothing after it")
        self.assertIs(self.sent_tensors[0][0], self.received_metadata)

    def test_a_batch_that_exactly_fills_this_rank_is_accepted(self):
        """The boundary the real fixture sits on: a slot's used prefix is the whole receive size."""
        yielded = self.receive(offset=0, size=self.RECEIVE_CAPACITY)

        self.assertEqual(len(yielded), 1)
        file_id, range_index, buffer = yielded[0]
        self.assertEqual((file_id, range_index), (7, 3))
        self.assertEqual(len(buffer), self.RECEIVE_CAPACITY)
        self.assertEqual(buffer.data_ptr(), self.received_buffer.data_ptr())

    def test_a_shorter_batch_yields_the_prefix_and_not_the_whole_buffer(self):
        yielded = self.receive(offset=512, size=1024)

        _, _, buffer = yielded[0]
        self.assertEqual(len(buffer), 1024)
        self.assertEqual(buffer.data_ptr(), self.received_buffer.data_ptr() + 512)
        self.assertEqual(buffer.tolist(), self.received_buffer[512:1536].tolist())

    def test_what_is_received_into_is_only_the_prefix_the_peer_filled(self):
        """The data collective must be posted with the peer's size, not with the whole buffer."""
        self.receive(offset=0, size=1024)

        data = self.sent_tensors[1][0]
        self.assertEqual(data.numel(), 1024)
        self.assertEqual(data.data_ptr(), self.received_buffer.data_ptr())


if __name__ == "__main__":
    unittest.main()
