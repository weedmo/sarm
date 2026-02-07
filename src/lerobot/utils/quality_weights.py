#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Quality-based RA-BC weights for conditional behavior cloning.

Assigns per-sample loss weights based on episode quality labels (bad/good/expert).
API-compatible with InterventionRABCWeights / RABCWeights so it can be passed
directly to update_policy() as rabc_weights_provider.

CombinedWeights multiplies weights from multiple providers (e.g. intervention + quality).
"""

import logging

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

QUALITY_LABELS = ("bad", "good", "expert")
QUALITY_MAP = {"bad": 0, "good": 1, "expert": 2}
QUALITY_MAP_INV = {v: k for k, v in QUALITY_MAP.items()}

DEFAULT_QUALITY_WEIGHTS = {"bad": 0.3, "good": 0.7, "expert": 1.0}


class QualityWeights:
    """
    Compute per-sample RA-BC weights based on episode quality labels.

    Each frame has an `episode_quality` int (0=bad, 1=good, 2=expert).
    Weights are looked up from a quality→weight mapping and normalized
    to sum to batch_size (same convention as RABCWeights).

    Args:
        dataset: LeRobotDataset with an "episode_quality" column.
        quality_weights: Mapping from quality label string to weight float.
        device: Device to return tensors on.
    """

    def __init__(
        self,
        dataset: LeRobotDataset,
        quality_weights: dict[str, float] | None = None,
        device: torch.device | None = None,
    ):
        self.quality_weights = quality_weights or dict(DEFAULT_QUALITY_WEIGHTS)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = 1e-6

        # Build int → float weight lookup
        self._weight_lookup: dict[int, float] = {
            QUALITY_MAP[label]: weight for label, weight in self.quality_weights.items()
        }

        # Build global_index → episode_quality lookup from dataset
        self._quality_lookup: dict[int, int] | None = None
        self._build_lookup(dataset)

    def _build_lookup(self, dataset: LeRobotDataset) -> None:
        """Build global index → quality int lookup from dataset."""
        if dataset.hf_dataset is not None and "episode_quality" in dataset.hf_dataset.column_names:
            self._quality_lookup = {}
            for i in range(len(dataset.hf_dataset)):
                row = dataset.hf_dataset[i]
                idx = int(row["index"])
                quality = int(row["episode_quality"])
                self._quality_lookup[idx] = quality
            counts = {}
            for q in self._quality_lookup.values():
                label = QUALITY_MAP_INV.get(q, f"unknown({q})")
                counts[label] = counts.get(label, 0) + 1
            logging.info(f"Built quality lookup: {len(self._quality_lookup)} frames, distribution: {counts}")
        else:
            logging.warning(
                "Dataset does not have 'episode_quality' column. "
                "Will attempt to read from batch tensors at runtime."
            )

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """
        Compute quality-based weights for a batch.

        Fast path: reads "episode_quality" directly from batch tensor.
        Fallback: looks up from prebuilt global_index → quality mapping.

        Returns:
            Tuple of:
            - Weights tensor (batch_size,) normalized to sum to batch_size
            - Stats dict with raw_mean_weight, quality_distribution
        """
        default_quality = QUALITY_MAP["good"]

        # Fast path: episode_quality in batch
        if "episode_quality" in batch:
            qualities = batch["episode_quality"]
            if isinstance(qualities, torch.Tensor):
                qualities = qualities.squeeze(-1).cpu().numpy()
            elif isinstance(qualities, np.ndarray):
                qualities = qualities.squeeze(-1)
            qualities = qualities.astype(np.int64)
        elif self._quality_lookup is not None:
            indices = batch.get("index")
            if indices is None:
                batch_size = self._get_batch_size(batch)
                ones = torch.ones(batch_size, device=self.device)
                return ones, {"raw_mean_weight": 1.0, "quality_distribution": {}}
            if isinstance(indices, torch.Tensor):
                indices = indices.cpu().numpy().tolist()
            qualities = np.array(
                [self._quality_lookup.get(int(idx), default_quality) for idx in indices],
                dtype=np.int64,
            )
        else:
            batch_size = self._get_batch_size(batch)
            ones = torch.ones(batch_size, device=self.device)
            return ones, {"raw_mean_weight": 1.0, "quality_distribution": {}}

        # Map quality int → weight float
        default_weight = self._weight_lookup.get(default_quality, 1.0)
        raw_weights = np.array(
            [self._weight_lookup.get(int(q), default_weight) for q in qualities],
            dtype=np.float32,
        )

        # Stats
        quality_dist = {}
        for q in qualities:
            label = QUALITY_MAP_INV.get(int(q), f"unknown({q})")
            quality_dist[label] = quality_dist.get(label, 0) + 1
        raw_mean_weight = float(np.mean(raw_weights))
        batch_stats = {
            "raw_mean_weight": raw_mean_weight,
            "quality_distribution": quality_dist,
        }

        weights = torch.tensor(raw_weights, device=self.device, dtype=torch.float32)

        # Normalize to sum to batch_size
        batch_size = len(weights)
        weight_sum = weights.sum() + self.epsilon
        weights = weights * batch_size / weight_sum

        return weights, batch_stats

    def _get_batch_size(self, batch: dict) -> int:
        """Determine batch size from batch."""
        for key in ["action", "index", "episode_quality"]:
            if key in batch:
                val = batch[key]
                if isinstance(val, torch.Tensor | np.ndarray):
                    return val.shape[0]
        return 1

    def get_stats(self) -> dict:
        """Get statistics about the quality weights."""
        if self._quality_lookup is not None:
            total = len(self._quality_lookup)
            counts = {}
            for q in self._quality_lookup.values():
                label = QUALITY_MAP_INV.get(q, f"unknown({q})")
                counts[label] = counts.get(label, 0) + 1
        else:
            total = 0
            counts = {}
        return {
            "total_frames": total,
            "quality_distribution": counts,
            "quality_weights": dict(self.quality_weights),
        }


class CombinedWeights:
    """
    Combine multiple weight providers by multiplying their weights and re-normalizing.

    Each provider must implement compute_batch_weights(batch) → (tensor, stats).
    The final weights are the element-wise product, re-normalized to sum to batch_size.

    Args:
        providers: List of weight provider objects.
    """

    def __init__(self, *providers):
        self.providers = list(providers)
        self.epsilon = 1e-6

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """
        Compute combined weights from all providers.

        Returns:
            Tuple of:
            - Combined weights tensor (batch_size,) normalized to sum to batch_size
            - Stats dict with per-provider stats and combined_raw_mean_weight
        """
        all_weights = []
        all_stats = {}

        for i, provider in enumerate(self.providers):
            w, s = provider.compute_batch_weights(batch)
            all_weights.append(w)
            provider_name = type(provider).__name__
            all_stats[f"provider_{i}_{provider_name}"] = s

        if not all_weights:
            batch_size = self._get_batch_size(batch)
            device = all_weights[0].device if all_weights else torch.device("cpu")
            ones = torch.ones(batch_size, device=device)
            return ones, {"combined_raw_mean_weight": 1.0}

        # Element-wise product of all provider weights
        combined = all_weights[0]
        for w in all_weights[1:]:
            combined = combined * w

        # Re-normalize to sum to batch_size
        batch_size = len(combined)
        weight_sum = combined.sum() + self.epsilon
        combined = combined * batch_size / weight_sum

        all_stats["combined_raw_mean_weight"] = float(combined.mean().item())

        return combined, all_stats

    def _get_batch_size(self, batch: dict) -> int:
        for key in ["action", "index"]:
            if key in batch:
                val = batch[key]
                if isinstance(val, torch.Tensor | np.ndarray):
                    return val.shape[0]
        return 1

    def get_stats(self) -> dict:
        """Get combined statistics from all providers."""
        combined_stats = {}
        for i, provider in enumerate(self.providers):
            if hasattr(provider, "get_stats"):
                provider_name = type(provider).__name__
                combined_stats[f"provider_{i}_{provider_name}"] = provider.get_stats()
        return combined_stats
