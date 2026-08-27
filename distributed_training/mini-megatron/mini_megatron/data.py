# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
except ModuleNotFoundError:
    torch = None
    DataLoader = object
    Dataset = object

from mini_megatron.config import DataConfig
from mini_megatron.distributed import DistributedContext


class ByteTokenizer:
    """Tiny byte-level tokenizer skeleton for local tutorials without HF dependencies."""

    pad_id: int = 0
    offset: int = 1

    def encode(self, text: str) -> list[int]:
        """Encode UTF-8 bytes into positive token IDs."""
        return [byte + self.offset for byte in text.encode("utf-8")]


class MockTokenDataset(Dataset):
    """Deterministic random token dataset skeleton for language-model smoke tests."""

    def __init__(self, *, vocab_size: int, seq_length: int, num_samples: int, seed: int) -> None:
        """Store mock dataset settings."""
        self.vocab_size=vocab_size
        self.seq_length=seq_length
        self.num_samples=num_samples
        self.seed=seed
        generator=torch.Generator()
        generator.manual_seed(seed)
        self.tokens=torch.randint(0,vocab_size,(num_samples,seq_length+1),generator=generator)


    def __len__(self) -> int:
        """Return the number of mock samples."""
        return self.num_samples


    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return shifted next-token prediction tensors for one sample."""
        return {
            "input_ids":self.tokens[index,:-1],
            "labels":self.tokens[index,1:]
        }


class JsonlTokenDataset(Dataset):
    """JSONL dataset skeleton accepting either `tokens` arrays or `text` strings."""

    def __init__(self, path: str | Path, *, seq_length: int, vocab_size: int) -> None:
        """Load JSONL records and store tokenization settings."""
        self.path=path
        self.seq_length=seq_length
        self.vocab_size=vocab_size
        self.byte_tokenizer=ByteTokenizer()
        self.records: list[dict[str, Any]] = []
        with open(path,"r") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                # 允许跳过空行
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {self.path}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"JSONL line {line_number} must contain an object"
                    )

                self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)


    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]

        tokens = self._tokens_from_record(record)
        tokens = self._pad_or_trim(tokens)

        token_tensor = torch.tensor(
            tokens,
            dtype=torch.long,
        )

        return {
            "input_ids": token_tensor[:-1],
            "labels": token_tensor[1:],
        }

    def _tokens_from_record(self, record: dict[str, Any]) -> list[int]:
        if "tokens" in record:
            tokens = record["tokens"]

            if not isinstance(tokens, list):
                raise ValueError("record['tokens'] must be a list")

            if not all(isinstance(token, int) for token in tokens):
                raise ValueError("all values in record['tokens'] must be integers")

            return list(tokens)

        if "text" in record:
            text = record["text"]

            if not isinstance(text, str):
                raise ValueError("record['text'] must be a string")

            return self.byte_tokenizer.encode(text)

        raise ValueError("record must contain either 'tokens' or 'text'")


    def _pad_or_trim(self, tokens: list[int]) -> list[int]:
        """Pad or trim tokens to `seq_length + 1`."""
        target_length = self.seq_length + 1

        if len(tokens) >= target_length:
            return tokens[:target_length]

        padding_length = target_length - len(tokens)

        return tokens + [self.byte_tokenizer.pad_id] * padding_length


def collate_lm_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack language-model samples into a microbatch."""
    if not samples:
        raise ValueError("samples must not be empty")

    return {
        key: torch.stack([sample[key] for sample in samples], dim=0)
        for key in samples[0]
    }



def build_dataloader(data_config: DataConfig, ctx: DistributedContext, *, seed: int, micro_batch_size: int) -> DataLoader:
    if data_config.dataset_path is None:
        dataset=MockTokenDataset(vocab_size=data_config.vocab_size,seq_length=data_config.seq_length,num_samples=data_config.mock_num_samples,seed=seed)
    else:
        dataset=JsonlTokenDataset(path=data_config.dataset_path,seq_length=data_config.seq_length,vocab_size=data_config.vocab_size)
    sampler=None
    shuffle=None
    if ctx.is_distributed:
        sampler=DistributedSampler(
            dataset,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=True,
            seed=seed
        )
        shuffle=False
    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        collate_fn=collate_lm_batch,
        drop_last=True
    )
