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
Intervention-based RA-BC weights for DAgger-style post-training.

Assigns higher loss weights to human-intervention frames vs policy-autonomy frames.
API-compatible with RABCWeights so it can be passed directly to update_policy()
as rabc_weights_provider.
"""

import logging

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class InterventionRABCWeights:
    """
    Compute per-sample RA-BC weights based on intervention flags.

    Human-intervention frames receive `intervention_weight` (default 1.0),
    policy-autonomy frames receive `policy_base_weight` (default 0.3).
    Weights are normalized to sum to batch_size (same convention as RABCWeights).

    Args:
        dataset: LeRobotDataset with an "is_intervention" column.
        intervention_weight: Weight for human-intervention frames.
        policy_base_weight: Weight for policy-autonomy frames.
        device: Device to return tensors on.
    """

    def __init__(
        self,
        dataset: LeRobotDataset,
        intervention_weight: float = 1.0,
        policy_base_weight: float = 0.3,
        device: torch.device | None = None,
    ):
        self.intervention_weight = intervention_weight
        self.policy_base_weight = policy_base_weight
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = 1e-6

        # Build global_index → is_intervention lookup from dataset as fallback
        self._intervention_lookup: dict[int, int] | None = None
        self._build_lookup(dataset)

    def _build_lookup(self, dataset: LeRobotDataset) -> None:
        """Build global index → intervention flag lookup from dataset."""
        if dataset.hf_dataset is not None and "is_intervention" in dataset.hf_dataset.column_names:
            self._intervention_lookup = {}
            for i in range(len(dataset.hf_dataset)):
                row = dataset.hf_dataset[i]
                idx = int(row["index"])
                flag = int(row["is_intervention"])
                self._intervention_lookup[idx] = flag
            logging.info(
                f"Built intervention lookup: {len(self._intervention_lookup)} frames, "
                f"{sum(self._intervention_lookup.values())} intervention frames"
            )
        else:
            logging.warning(
                "Dataset does not have 'is_intervention' column. "
                "Will attempt to read from batch tensors at runtime."
            )

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """
        Compute intervention-based weights for a batch.

        Fast path: reads "is_intervention" directly from batch tensor.
        Fallback: looks up from prebuilt global_index → flag mapping.

        Args:
            batch: Training batch dict with tensors.

        Returns:
            Tuple of:
            - Weights tensor (batch_size,) normalized to sum to batch_size
            - Stats dict with raw_mean_weight, num_zero_weight, num_full_weight
        """
        # Fast path: is_intervention directly in batch
        if "is_intervention" in batch:
            flags = batch["is_intervention"]
            if isinstance(flags, torch.Tensor):
                flags = flags.squeeze(-1).cpu().numpy()
            elif isinstance(flags, np.ndarray):
                flags = flags.squeeze(-1)
            flags = flags.astype(np.float32)
        elif self._intervention_lookup is not None:
            # Fallback: look up by global index
            indices = batch.get("index")
            if indices is None:
                batch_size = self._get_batch_size(batch)
                ones = torch.ones(batch_size, device=self.device)
                return ones, {"raw_mean_weight": 1.0, "num_zero_weight": 0, "num_full_weight": batch_size}
            if isinstance(indices, torch.Tensor):
                indices = indices.cpu().numpy().tolist()
            flags = np.array(
                [float(self._intervention_lookup.get(int(idx), 0)) for idx in indices],
                dtype=np.float32,
            )
        else:
            # No intervention data available, use uniform weights
            batch_size = self._get_batch_size(batch)
            ones = torch.ones(batch_size, device=self.device)
            return ones, {"raw_mean_weight": 1.0, "num_zero_weight": 0, "num_full_weight": batch_size}

        # Compute raw weights: intervention → intervention_weight, policy → policy_base_weight
        raw_weights = np.where(
            flags > 0.5,
            self.intervention_weight,
            self.policy_base_weight,
        )

        # Stats before normalization
        raw_mean_weight = float(np.mean(raw_weights))
        num_full_weight = int(np.sum(flags > 0.5))
        num_zero_weight = 0  # We don't zero out any samples
        batch_stats = {
            "raw_mean_weight": raw_mean_weight,
            "num_zero_weight": num_zero_weight,
            "num_full_weight": num_full_weight,
        }

        weights = torch.tensor(raw_weights, device=self.device, dtype=torch.float32)

        # Normalize to sum to batch_size
        batch_size = len(weights)
        weight_sum = weights.sum() + self.epsilon
        weights = weights * batch_size / weight_sum

        return weights, batch_stats

    def _get_batch_size(self, batch: dict) -> int:
        """Determine batch size from batch."""
        for key in ["action", "index", "is_intervention"]:
            if key in batch:
                val = batch[key]
                if isinstance(val, torch.Tensor | np.ndarray):
                    return val.shape[0]
        return 1

    def get_stats(self) -> dict:
        """Get statistics about the intervention weights."""
        if self._intervention_lookup is not None:
            total = len(self._intervention_lookup)
            intervention = sum(self._intervention_lookup.values())
            policy = total - intervention
        else:
            total = intervention = policy = 0
        return {
            "total_frames": total,
            "intervention_frames": intervention,
            "policy_frames": policy,
            "intervention_ratio": intervention / max(total, 1),
            "intervention_weight": self.intervention_weight,
            "policy_base_weight": self.policy_base_weight,
        }
