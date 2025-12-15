from collections import deque

from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.engine.sequence import Sequence, SequenceStatus
from nanovllm_voxcpm.engine.block_manager import BlockManager

import uuid

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        self._id_to_seq: dict[str, Sequence] = {}

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self._id_to_seq[seq.seq_id] = seq

        self.waiting.append(seq)
    
    def cancel(self, seq_id: str):
        try:
            seq = self._id_to_seq.pop(seq_id)
        except KeyError:
            return

        self.block_manager.deallocate(seq)
        if seq.status == SequenceStatus.RUNNING:
            self.running.remove(seq)
        elif seq.status == SequenceStatus.WAITING:
            self.waiting.remove(seq)
        return

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            tokens_to_compute = len(seq) - seq.num_cached_tokens
            if tokens_to_compute > 0:
                num_seqs += 1
                scheduled_seqs.append(seq)
                num_batched_tokens += tokens_to_compute

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
    
    def finish(self, seq: Sequence):
        seq.status = SequenceStatus.FINISHED
        self.block_manager.deallocate(seq)
        self.running.remove(seq)
        self._id_to_seq.pop(seq.seq_id)

    def stats(self) -> dict:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
        }
