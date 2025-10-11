from copy import copy
from enum import Enum, auto
from itertools import count

from typing import Generic, TypeVar

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


PlayloadType = TypeVar("PlayloadType")
class Sequence(Generic[PlayloadType]):
    counter = count()

    def __init__(self, token_ids: list[int | bytes], block_size: int, custom_payload: PlayloadType = None):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.block_size = block_size

        self.custom_payload = custom_payload
        self.stoped = False

    def __len__(self):
        return self.num_tokens

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i) -> list[int | bytes]:
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int | bytes):
        self.token_ids.append(token_id)
        self.num_tokens += 1
