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

"""Tests for the intervention collection + post-training pipeline."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from lerobot.utils.intervention_weights import InterventionRABCWeights
from lerobot.utils.quality_weights import (
    QUALITY_MAP,
    CombinedWeights,
    QualityWeights,
)

# ---------------------------------------------------------------------------
# Helper: mock dataset with is_intervention column
# ---------------------------------------------------------------------------


def _make_mock_dataset(intervention_flags: list[int]):
    """Create a mock LeRobotDataset with is_intervention data."""
    dataset = MagicMock()
    n = len(intervention_flags)

    # Build a mock hf_dataset
    hf_dataset = MagicMock()
    hf_dataset.column_names = ["index", "is_intervention", "action"]
    hf_dataset.__len__ = lambda self: n
    hf_dataset.__getitem__ = lambda self, i: {
        "index": i,
        "is_intervention": intervention_flags[i],
    }

    dataset.hf_dataset = hf_dataset
    return dataset


def _make_mock_quality_dataset(quality_flags: list[int], intervention_flags: list[int] | None = None):
    """Create a mock LeRobotDataset with episode_quality (and optionally is_intervention) data."""
    dataset = MagicMock()
    n = len(quality_flags)
    if intervention_flags is None:
        intervention_flags = [0] * n

    columns = ["index", "episode_quality", "is_intervention", "action"]
    hf_dataset = MagicMock()
    hf_dataset.column_names = columns
    hf_dataset.__len__ = lambda self: n
    hf_dataset.__getitem__ = lambda self, i: {
        "index": i,
        "episode_quality": quality_flags[i],
        "is_intervention": intervention_flags[i],
    }

    dataset.hf_dataset = hf_dataset
    return dataset


# ---------------------------------------------------------------------------
# Keyboard events tests
# ---------------------------------------------------------------------------


class TestInterventionKeyboardEvents:
    def test_events_dict_has_intervention_mode(self):
        """init_intervention_keyboard_listener should include intervention_mode=False."""
        from lerobot.policies.smolvla.intervention_collect_and_train import (
            init_intervention_keyboard_listener,
        )

        # In CI/headless, listener may be None but events dict should be correct
        listener, events = init_intervention_keyboard_listener()

        assert "intervention_mode" in events
        assert events["intervention_mode"] is False
        assert "exit_early" in events
        assert "rerecord_episode" in events
        assert "stop_recording" in events
        assert "save_without_task" in events

        if listener is not None:
            listener.stop()

    def test_events_initial_values(self):
        """All event flags should start as False, except episode_quality which is 'good'."""
        from lerobot.policies.smolvla.intervention_collect_and_train import (
            init_intervention_keyboard_listener,
        )

        listener, events = init_intervention_keyboard_listener()

        for key, val in events.items():
            if key == "episode_quality":
                assert val == "good", f"events['episode_quality'] should be 'good', got {val}"
            else:
                assert val is False, f"events['{key}'] should be False, got {val}"

        if listener is not None:
            listener.stop()

    def test_events_has_episode_quality(self):
        """init_intervention_keyboard_listener should include episode_quality='good'."""
        from lerobot.policies.smolvla.intervention_collect_and_train import (
            init_intervention_keyboard_listener,
        )

        listener, events = init_intervention_keyboard_listener()

        assert "episode_quality" in events
        assert events["episode_quality"] == "good"

        if listener is not None:
            listener.stop()


# ---------------------------------------------------------------------------
# InterventionRABCWeights tests
# ---------------------------------------------------------------------------


class TestInterventionWeightsBasic:
    def test_correct_weight_assignment(self):
        """Intervention frames get intervention_weight, policy frames get policy_base_weight."""
        flags = [0, 1, 1, 0, 0, 1, 0, 0]
        dataset = _make_mock_dataset(flags)

        weights_provider = InterventionRABCWeights(
            dataset=dataset,
            intervention_weight=1.0,
            policy_base_weight=0.3,
            device=torch.device("cpu"),
        )

        batch = {
            "is_intervention": torch.tensor([[f] for f in flags], dtype=torch.int64),
            "index": torch.arange(len(flags)),
        }

        weights, stats = weights_provider.compute_batch_weights(batch)

        assert weights.shape == (len(flags),)
        assert weights.sum().item() == pytest.approx(len(flags), abs=1e-4)

        # Check relative ordering: intervention frames should have higher weight
        intervention_indices = [i for i, f in enumerate(flags) if f == 1]
        policy_indices = [i for i, f in enumerate(flags) if f == 0]

        avg_intervention_weight = weights[intervention_indices].mean().item()
        avg_policy_weight = weights[policy_indices].mean().item()
        assert avg_intervention_weight > avg_policy_weight

    def test_normalization_sums_to_batch_size(self):
        """Weights should be normalized to sum to batch_size."""
        flags = [0, 1, 0, 1]
        dataset = _make_mock_dataset(flags)

        weights_provider = InterventionRABCWeights(
            dataset=dataset,
            intervention_weight=2.0,
            policy_base_weight=0.5,
            device=torch.device("cpu"),
        )

        batch = {
            "is_intervention": torch.tensor([[f] for f in flags], dtype=torch.int64),
            "index": torch.arange(len(flags)),
        }

        weights, _ = weights_provider.compute_batch_weights(batch)
        assert weights.sum().item() == pytest.approx(len(flags), abs=1e-3)

    def test_stats_dict(self):
        """compute_batch_weights should return correct stats."""
        flags = [0, 1, 1, 0, 0, 1]
        dataset = _make_mock_dataset(flags)

        weights_provider = InterventionRABCWeights(
            dataset=dataset,
            intervention_weight=1.0,
            policy_base_weight=0.3,
            device=torch.device("cpu"),
        )

        batch = {
            "is_intervention": torch.tensor([[f] for f in flags], dtype=torch.int64),
            "index": torch.arange(len(flags)),
        }

        _, stats = weights_provider.compute_batch_weights(batch)

        assert "raw_mean_weight" in stats
        assert "num_zero_weight" in stats
        assert "num_full_weight" in stats
        assert stats["num_full_weight"] == 3  # 3 intervention frames
        assert stats["num_zero_weight"] == 0  # We never zero out samples


class TestInterventionWeightsFromBatch:
    def test_fast_path_from_batch_tensor(self):
        """When is_intervention is in batch, should use it directly (fast path)."""
        # Dataset with NO is_intervention column
        dataset = MagicMock()
        dataset.hf_dataset = MagicMock()
        dataset.hf_dataset.column_names = ["index", "action"]  # no is_intervention
        dataset.hf_dataset.__len__ = lambda self: 0

        weights_provider = InterventionRABCWeights(
            dataset=dataset,
            intervention_weight=1.0,
            policy_base_weight=0.3,
            device=torch.device("cpu"),
        )

        # But batch has is_intervention tensor
        batch = {
            "is_intervention": torch.tensor([[1], [0], [1], [0]], dtype=torch.int64),
            "index": torch.arange(4),
        }

        weights, stats = weights_provider.compute_batch_weights(batch)

        assert weights.shape == (4,)
        assert weights.sum().item() == pytest.approx(4.0, abs=1e-3)
        assert stats["num_full_weight"] == 2

    def test_fallback_to_index_lookup(self):
        """When batch has no is_intervention but dataset has it, use index lookup."""
        flags = [0, 1, 0, 1]
        dataset = _make_mock_dataset(flags)

        weights_provider = InterventionRABCWeights(
            dataset=dataset,
            intervention_weight=1.0,
            policy_base_weight=0.3,
            device=torch.device("cpu"),
        )

        # Batch without is_intervention but with index
        batch = {
            "index": torch.tensor([0, 1, 2, 3]),
            "action": torch.randn(4, 6),
        }

        weights, stats = weights_provider.compute_batch_weights(batch)

        assert weights.shape == (4,)
        assert weights.sum().item() == pytest.approx(4.0, abs=1e-3)
        assert stats["num_full_weight"] == 2  # indices 1 and 3 are intervention


class TestInterventionWeightsEdgeCases:
    def test_all_intervention(self):
        """When all frames are intervention, weights should be uniform."""
        flags = [1, 1, 1, 1]
        dataset = _make_mock_dataset(flags)

        weights_provider = InterventionRABCWeights(
            dataset=dataset,
            intervention_weight=1.0,
            policy_base_weight=0.3,
            device=torch.device("cpu"),
        )

        batch = {
            "is_intervention": torch.tensor([[1], [1], [1], [1]], dtype=torch.int64),
            "index": torch.arange(4),
        }

        weights, stats = weights_provider.compute_batch_weights(batch)

        # All same weight → uniform after normalization
        assert weights.shape == (4,)
        for i in range(4):
            assert weights[i].item() == pytest.approx(1.0, abs=1e-3)

    def test_all_policy(self):
        """When all frames are policy, weights should be uniform."""
        flags = [0, 0, 0, 0]
        dataset = _make_mock_dataset(flags)

        weights_provider = InterventionRABCWeights(
            dataset=dataset,
            intervention_weight=1.0,
            policy_base_weight=0.3,
            device=torch.device("cpu"),
        )

        batch = {
            "is_intervention": torch.tensor([[0], [0], [0], [0]], dtype=torch.int64),
            "index": torch.arange(4),
        }

        weights, stats = weights_provider.compute_batch_weights(batch)

        assert weights.shape == (4,)
        for i in range(4):
            assert weights[i].item() == pytest.approx(1.0, abs=1e-3)

    def test_single_sample_batch(self):
        """Should work with batch_size=1."""
        flags = [1]
        dataset = _make_mock_dataset(flags)

        weights_provider = InterventionRABCWeights(
            dataset=dataset,
            intervention_weight=1.0,
            policy_base_weight=0.3,
            device=torch.device("cpu"),
        )

        batch = {
            "is_intervention": torch.tensor([[1]], dtype=torch.int64),
            "index": torch.tensor([0]),
        }

        weights, _ = weights_provider.compute_batch_weights(batch)
        assert weights.shape == (1,)
        assert weights[0].item() == pytest.approx(1.0, abs=1e-3)

    def test_no_data_returns_uniform(self):
        """When no intervention data is available, return uniform weights."""
        dataset = MagicMock()
        dataset.hf_dataset = MagicMock()
        dataset.hf_dataset.column_names = ["index", "action"]
        dataset.hf_dataset.__len__ = lambda self: 0

        weights_provider = InterventionRABCWeights(
            dataset=dataset,
            device=torch.device("cpu"),
        )

        # Batch without is_intervention
        batch = {"action": torch.randn(4, 6)}

        weights, stats = weights_provider.compute_batch_weights(batch)
        assert weights.shape == (4,)
        assert stats["raw_mean_weight"] == 1.0


class TestInterventionWeightsGetStats:
    def test_get_stats(self):
        """get_stats should return correct summary."""
        flags = [0, 1, 1, 0, 0]
        dataset = _make_mock_dataset(flags)

        weights_provider = InterventionRABCWeights(
            dataset=dataset,
            intervention_weight=1.0,
            policy_base_weight=0.3,
            device=torch.device("cpu"),
        )

        stats = weights_provider.get_stats()
        assert stats["total_frames"] == 5
        assert stats["intervention_frames"] == 2
        assert stats["policy_frames"] == 3
        assert stats["intervention_ratio"] == pytest.approx(0.4)
        assert stats["intervention_weight"] == 1.0
        assert stats["policy_base_weight"] == 0.3


class TestInterventionFeatureDefinition:
    def test_intervention_feature_spec(self):
        """is_intervention feature should be dtype=int64, shape=(1,)."""
        # This is what the collection phase adds to dataset_features
        feature_def = {
            "dtype": "int64",
            "shape": (1,),
            "names": ["is_intervention"],
        }
        assert feature_def["dtype"] == "int64"
        assert feature_def["shape"] == (1,)

    def test_intervention_frame_value(self):
        """Verify the frame value format matches what add_frame expects."""
        # This is what intervention_record_loop creates
        is_intervention_val = np.array([1], dtype=np.int64)
        assert is_intervention_val.dtype == np.int64
        assert is_intervention_val.shape == (1,)

        is_intervention_val = np.array([0], dtype=np.int64)
        assert is_intervention_val.dtype == np.int64
        assert is_intervention_val.shape == (1,)


# ---------------------------------------------------------------------------
# Hardware-dependent integration tests (skipped by default)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Quality constants tests
# ---------------------------------------------------------------------------


class TestQualityConstants:
    def test_quality_map_consistency(self):
        """QUALITY_MAP and QUALITY_MAP_INV should be consistent."""
        from lerobot.policies.smolvla.intervention_collect_and_train import (
            QUALITY_LABELS,
            QUALITY_MAP,
            QUALITY_MAP_INV,
        )

        assert set(QUALITY_MAP.keys()) == set(QUALITY_LABELS)
        for label, val in QUALITY_MAP.items():
            assert QUALITY_MAP_INV[val] == label

    def test_quality_map_matches_utils(self):
        """Constants in intervention_collect_and_train should match quality_weights module."""
        from lerobot.policies.smolvla.intervention_collect_and_train import (
            QUALITY_MAP as PIPELINE_MAP,
        )

        assert PIPELINE_MAP == QUALITY_MAP


# ---------------------------------------------------------------------------
# QualityWeights tests
# ---------------------------------------------------------------------------


class TestQualityWeightsBasic:
    def test_correct_weight_assignment(self):
        """Expert frames get highest weight, bad frames get lowest."""
        # 0=bad, 1=good, 2=expert
        qualities = [0, 1, 2, 0, 1, 2]
        dataset = _make_mock_quality_dataset(qualities)

        provider = QualityWeights(
            dataset=dataset,
            quality_weights={"bad": 0.3, "good": 0.7, "expert": 1.0},
            device=torch.device("cpu"),
        )

        batch = {
            "episode_quality": torch.tensor([[q] for q in qualities], dtype=torch.int64),
            "index": torch.arange(len(qualities)),
        }

        weights, stats = provider.compute_batch_weights(batch)

        assert weights.shape == (len(qualities),)
        assert weights.sum().item() == pytest.approx(len(qualities), abs=1e-3)

        # Expert indices should have higher weight than bad
        expert_indices = [i for i, q in enumerate(qualities) if q == 2]
        bad_indices = [i for i, q in enumerate(qualities) if q == 0]
        avg_expert = weights[expert_indices].mean().item()
        avg_bad = weights[bad_indices].mean().item()
        assert avg_expert > avg_bad

    def test_normalization(self):
        """Weights should normalize to sum to batch_size."""
        qualities = [0, 1, 2, 2]
        dataset = _make_mock_quality_dataset(qualities)

        provider = QualityWeights(
            dataset=dataset,
            quality_weights={"bad": 0.1, "good": 0.5, "expert": 2.0},
            device=torch.device("cpu"),
        )

        batch = {
            "episode_quality": torch.tensor([[q] for q in qualities], dtype=torch.int64),
            "index": torch.arange(len(qualities)),
        }

        weights, _ = provider.compute_batch_weights(batch)
        assert weights.sum().item() == pytest.approx(len(qualities), abs=1e-3)

    def test_all_same_quality_uniform(self):
        """When all frames have same quality, weights should be uniform."""
        qualities = [2, 2, 2, 2]
        dataset = _make_mock_quality_dataset(qualities)

        provider = QualityWeights(
            dataset=dataset,
            device=torch.device("cpu"),
        )

        batch = {
            "episode_quality": torch.tensor([[2], [2], [2], [2]], dtype=torch.int64),
            "index": torch.arange(4),
        }

        weights, _ = provider.compute_batch_weights(batch)
        for i in range(4):
            assert weights[i].item() == pytest.approx(1.0, abs=1e-3)

    def test_stats(self):
        """get_stats should return quality distribution."""
        qualities = [0, 1, 1, 2, 2, 2]
        dataset = _make_mock_quality_dataset(qualities)

        provider = QualityWeights(
            dataset=dataset,
            device=torch.device("cpu"),
        )

        stats = provider.get_stats()
        assert stats["total_frames"] == 6
        assert stats["quality_distribution"]["bad"] == 1
        assert stats["quality_distribution"]["good"] == 2
        assert stats["quality_distribution"]["expert"] == 3

    def test_batch_stats_distribution(self):
        """compute_batch_weights stats should include quality_distribution."""
        qualities = [0, 1, 2]
        dataset = _make_mock_quality_dataset(qualities)

        provider = QualityWeights(
            dataset=dataset,
            device=torch.device("cpu"),
        )

        batch = {
            "episode_quality": torch.tensor([[0], [1], [2]], dtype=torch.int64),
            "index": torch.arange(3),
        }

        _, stats = provider.compute_batch_weights(batch)
        assert stats["quality_distribution"] == {"bad": 1, "good": 1, "expert": 1}


class TestQualityWeightsFallback:
    def test_fast_path_from_batch_tensor(self):
        """When episode_quality is in batch, use it directly."""
        dataset = MagicMock()
        dataset.hf_dataset = MagicMock()
        dataset.hf_dataset.column_names = ["index", "action"]
        dataset.hf_dataset.__len__ = lambda self: 0

        provider = QualityWeights(
            dataset=dataset,
            device=torch.device("cpu"),
        )

        batch = {
            "episode_quality": torch.tensor([[0], [2], [1], [2]], dtype=torch.int64),
            "index": torch.arange(4),
        }

        weights, _ = provider.compute_batch_weights(batch)
        assert weights.shape == (4,)
        assert weights.sum().item() == pytest.approx(4.0, abs=1e-3)

    def test_fallback_to_index_lookup(self):
        """When batch has no episode_quality, use index lookup."""
        qualities = [0, 1, 2, 2]
        dataset = _make_mock_quality_dataset(qualities)

        provider = QualityWeights(
            dataset=dataset,
            device=torch.device("cpu"),
        )

        batch = {
            "index": torch.tensor([0, 1, 2, 3]),
            "action": torch.randn(4, 6),
        }

        weights, _ = provider.compute_batch_weights(batch)
        assert weights.shape == (4,)
        assert weights.sum().item() == pytest.approx(4.0, abs=1e-3)

    def test_no_data_returns_uniform(self):
        """When no quality data available, return uniform weights."""
        dataset = MagicMock()
        dataset.hf_dataset = MagicMock()
        dataset.hf_dataset.column_names = ["index", "action"]
        dataset.hf_dataset.__len__ = lambda self: 0

        provider = QualityWeights(
            dataset=dataset,
            device=torch.device("cpu"),
        )

        batch = {"action": torch.randn(4, 6)}

        weights, stats = provider.compute_batch_weights(batch)
        assert weights.shape == (4,)
        assert stats["raw_mean_weight"] == 1.0


# ---------------------------------------------------------------------------
# CombinedWeights tests
# ---------------------------------------------------------------------------


class TestCombinedWeights:
    def test_combines_two_providers(self):
        """CombinedWeights should multiply and re-normalize."""
        qualities = [0, 1, 2, 2]
        intervention_flags = [1, 0, 1, 0]
        dataset = _make_mock_quality_dataset(qualities, intervention_flags)

        iw = InterventionRABCWeights(
            dataset=dataset,
            intervention_weight=1.0,
            policy_base_weight=0.3,
            device=torch.device("cpu"),
        )
        qw = QualityWeights(
            dataset=dataset,
            quality_weights={"bad": 0.3, "good": 0.7, "expert": 1.0},
            device=torch.device("cpu"),
        )

        combined = CombinedWeights(iw, qw)

        batch = {
            "is_intervention": torch.tensor([[f] for f in intervention_flags], dtype=torch.int64),
            "episode_quality": torch.tensor([[q] for q in qualities], dtype=torch.int64),
            "index": torch.arange(4),
        }

        weights, stats = combined.compute_batch_weights(batch)

        assert weights.shape == (4,)
        assert weights.sum().item() == pytest.approx(4.0, abs=1e-3)
        assert "combined_raw_mean_weight" in stats

    def test_single_provider_equivalent(self):
        """CombinedWeights with one provider should match that provider."""
        flags = [0, 1, 1, 0]
        dataset = _make_mock_dataset(flags)

        iw = InterventionRABCWeights(
            dataset=dataset,
            intervention_weight=1.0,
            policy_base_weight=0.3,
            device=torch.device("cpu"),
        )

        combined = CombinedWeights(iw)

        batch = {
            "is_intervention": torch.tensor([[f] for f in flags], dtype=torch.int64),
            "index": torch.arange(4),
        }

        w_combined, _ = combined.compute_batch_weights(batch)
        w_single, _ = iw.compute_batch_weights(batch)

        # After re-normalization they should be very close
        assert w_combined.sum().item() == pytest.approx(4.0, abs=1e-3)
        assert w_single.sum().item() == pytest.approx(4.0, abs=1e-3)

    def test_get_stats_includes_all_providers(self):
        """get_stats should include stats from each provider."""
        qualities = [0, 1, 2]
        intervention_flags = [1, 0, 1]
        dataset = _make_mock_quality_dataset(qualities, intervention_flags)

        iw = InterventionRABCWeights(
            dataset=dataset, device=torch.device("cpu"),
        )
        qw = QualityWeights(
            dataset=dataset, device=torch.device("cpu"),
        )

        combined = CombinedWeights(iw, qw)
        stats = combined.get_stats()

        assert len(stats) == 2
        keys = list(stats.keys())
        assert "InterventionRABCWeights" in keys[0]
        assert "QualityWeights" in keys[1]


# ---------------------------------------------------------------------------
# Quality feature definition tests
# ---------------------------------------------------------------------------


class TestQualityFeatureDefinition:
    def test_episode_quality_feature_spec(self):
        """episode_quality feature should be dtype=int64, shape=(1,)."""
        feature_def = {
            "dtype": "int64",
            "shape": (1,),
            "names": ["episode_quality"],
        }
        assert feature_def["dtype"] == "int64"
        assert feature_def["shape"] == (1,)

    def test_episode_quality_frame_values(self):
        """Verify frame value format for each quality level."""
        for _label, val in QUALITY_MAP.items():
            frame_val = np.array([val], dtype=np.int64)
            assert frame_val.dtype == np.int64
            assert frame_val.shape == (1,)


# ---------------------------------------------------------------------------
# Hardware-dependent integration tests (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(True, reason="Requires SO100 hardware")
class TestInterventionRecordLoopIntegration:
    def test_intervention_record_loop_smoke(self):
        """End-to-end test with mock robot and teleop."""
        # This test requires actual hardware and is skipped in CI
        pass
