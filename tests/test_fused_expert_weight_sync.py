# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

"""CPU-only tests for the fused MoE expert weight sync path in WeightSyncWorkerExtension.

The GPU/server behavior (identity sync, mutate/restore, cross-backend logprob parity) is
validated end to end against live `trl vllm-serve` servers; see the monorepo's
docs/vllm-flashinfer-moe-weight-sync.md.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from trl.scripts.vllm_serve import WeightSyncWorkerExtension

from .testing_utils import require_vllm

# Real-tensor fixtures for the packed-expert fallback tests: a DeepSeek-V3.2 checkpoint
# shard (asymmetric expert shapes catch orientation bugs synthetic squares would hide).
DS32_SNAPSHOT = os.environ.get(
    "DS32_SNAPSHOT",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3.2-Speciale/"
        "snapshots/c562883eda916fd699a44bbf1f4987a53b96008b"
    ),
)
requires_ds32_checkpoint = pytest.mark.skipif(
    not Path(DS32_SNAPSHOT, "model-00001-of-000163.safetensors").exists(),
    reason="DeepSeek-V3.2-Speciale checkpoint shard not available (set DS32_SNAPSHOT)",
)


class TestFusedExpertWeightNameMatching:
    @pytest.mark.parametrize(
        "name, expected_module, expected_proj",
        [
            (
                "language_model.model.layers.3.mlp.experts.gate_up_proj",
                "language_model.model.layers.3.mlp.experts",
                "gate_up_proj",
            ),
            ("model.layers.0.mlp.experts.down_proj", "model.layers.0.mlp.experts", "down_proj"),
        ],
    )
    def test_matches_fused_expert_weights(self, name, expected_module, expected_proj):
        match = WeightSyncWorkerExtension._FUSED_EXPERT_WEIGHT_RE.match(name)
        assert match is not None
        assert match.group("module") == expected_module
        assert match.group("proj") == expected_proj

    @pytest.mark.parametrize(
        "name",
        [
            # Dense MLP projections do not live under `.experts.`.
            "model.layers.3.mlp.gate_up_proj",
            # The shared expert is a plain MLP, not part of the fused expert weights.
            "model.layers.3.mlp.shared_expert.gate_up_proj",
            # Per-expert 2D form is not packed.
            "model.layers.3.mlp.experts.0.gate_proj.weight",
            "lm_head.weight",
            "model.layers.3.self_attn.q_proj.weight",
        ],
    )
    def test_rejects_other_weights(self, name):
        assert WeightSyncWorkerExtension._FUSED_EXPERT_WEIGHT_RE.match(name) is None


class TestMaybeUpdateFusedExpertParam:
    def _make_extension(self, model: nn.Module) -> WeightSyncWorkerExtension:
        extension = WeightSyncWorkerExtension()
        extension.model_runner = SimpleNamespace(model=model)
        return extension

    def test_returns_false_for_2d_tensor(self):
        extension = self._make_extension(nn.Module())
        handled = extension._maybe_update_fused_expert_param(
            "model.layers.0.mlp.experts.gate_up_proj", torch.zeros(4, 4)
        )
        assert not handled

    def test_returns_false_for_non_expert_name(self):
        extension = self._make_extension(nn.Module())
        handled = extension._maybe_update_fused_expert_param("lm_head.weight", torch.zeros(2, 4, 4))
        assert not handled

    def test_returns_false_when_module_missing(self):
        # A 3D tensor with a matching name form, but the model has no such submodule: the caller
        # should fall back to the plain load_weights path.
        extension = self._make_extension(nn.Module())
        handled = extension._maybe_update_fused_expert_param(
            "model.layers.0.mlp.experts.gate_up_proj", torch.zeros(2, 4, 4)
        )
        assert not handled

    def test_returns_false_for_module_without_fused_moe_params(self):
        model = nn.Module()
        mlp = nn.Module()
        mlp.experts = nn.Linear(4, 4)
        model.add_module("mlp", mlp)
        extension = self._make_extension(model)
        handled = extension._maybe_update_fused_expert_param("mlp.experts.gate_up_proj", torch.zeros(2, 4, 4))
        assert not handled


@require_vllm
class TestMoeConversionApiGuard:
    def test_guard_passes_on_pinned_vllm(self):
        # Fails loudly (RuntimeError) if the installed vLLM changed the private conversion API
        # this path depends on. Validated on vllm==0.25.0.
        api = WeightSyncWorkerExtension._get_moe_conversion_api()
        assert len(api) == 4


class _RecordingExpertLayer(nn.Module):
    """Stands in for a vLLM RoutedExperts layer: records expert-aware weight_loader calls."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.calls: list[tuple[str, str, int, torch.Tensor]] = []

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
        self.calls.append((weight_name, shard_id, expert_id, loaded_weight.clone()))
        return True if return_success else None


class _NoPackedSupportModel(nn.Module):
    """A model whose load_weights ignores packed-3D names (deepseek_v2.py behavior)."""

    def load_weights(self, weights):
        return set()


class _PackedSupportModel(nn.Module):
    """A model whose load_weights handles packed names natively (qwen3_5.py behavior)."""

    def __init__(self):
        super().__init__()
        self.loaded: list[str] = []

    def load_weights(self, weights):
        names = {name for name, _ in weights}
        self.loaded.extend(names)
        return names


def _make_extension(model: nn.Module) -> WeightSyncWorkerExtension:
    extension = WeightSyncWorkerExtension()
    extension.model_runner = SimpleNamespace(model=model)
    return extension


class TestLoadPackedExpertTensor:
    """Slicing/orientation/routing tests for the per-expert fallback (no vLLM needed)."""

    HIDDEN = 16
    INTERMEDIATE = 4
    EXPERTS = 3

    def _linear_convention_tensors(self):
        """gate_up [E, 2I, H] (gate rows first) + down [E, H, I], the deepseek_v32 trainer layout."""
        torch.manual_seed(0)
        gate = torch.randn(self.EXPERTS, self.INTERMEDIATE, self.HIDDEN)
        up = torch.randn(self.EXPERTS, self.INTERMEDIATE, self.HIDDEN)
        down = torch.randn(self.EXPERTS, self.HIDDEN, self.INTERMEDIATE)
        gate_up = torch.cat([gate, up], dim=1)
        return gate, up, down, gate_up

    def test_native_packed_loader_short_circuits(self):
        model = _PackedSupportModel()
        extension = _make_extension(model)
        layer = _RecordingExpertLayer(self.HIDDEN)
        _, _, down, gate_up = self._linear_convention_tensors()
        loaded = extension._load_packed_expert_tensor(
            "model.layers.3.mlp.experts.gate_up_proj", gate_up, layer, None, "gate_up_proj"
        )
        assert loaded
        assert model.loaded == ["model.layers.3.mlp.experts.gate_up_proj"]
        assert layer.calls == []  # fallback never engaged

    def test_gate_up_linear_convention_slicing(self):
        extension = _make_extension(_NoPackedSupportModel())
        layer = _RecordingExpertLayer(self.HIDDEN)
        gate, up, _, gate_up = self._linear_convention_tensors()
        loaded = extension._load_packed_expert_tensor(
            "model.layers.3.mlp.experts.gate_up_proj", gate_up, layer, None, "gate_up_proj"
        )
        assert loaded
        assert len(layer.calls) == 2 * self.EXPERTS
        for expert_id in range(self.EXPERTS):
            name_g, shard_g, eid_g, tensor_g = layer.calls[2 * expert_id]
            name_u, shard_u, eid_u, tensor_u = layer.calls[2 * expert_id + 1]
            assert name_g == f"model.layers.3.mlp.experts.{expert_id}.gate_proj.weight"
            assert name_u == f"model.layers.3.mlp.experts.{expert_id}.up_proj.weight"
            assert (shard_g, shard_u) == ("w1", "w3")
            assert (eid_g, eid_u) == (expert_id, expert_id)
            assert torch.equal(tensor_g, gate[expert_id])
            assert torch.equal(tensor_u, up[expert_id])

    def test_down_linear_convention_slicing(self):
        extension = _make_extension(_NoPackedSupportModel())
        layer = _RecordingExpertLayer(self.HIDDEN)
        _, _, down, _ = self._linear_convention_tensors()
        loaded = extension._load_packed_expert_tensor(
            "model.layers.3.mlp.experts.down_proj", down, layer, None, "down_proj"
        )
        assert loaded
        assert len(layer.calls) == self.EXPERTS
        for expert_id, (name, shard, eid, tensor) in enumerate(layer.calls):
            assert name == f"model.layers.3.mlp.experts.{expert_id}.down_proj.weight"
            assert shard == "w2"
            assert eid == expert_id
            assert torch.equal(tensor, down[expert_id])

    def test_transposed_convention_slicing(self):
        """qwen3_5-style [E, H, 2I] / [E, I, H] wire layouts are transposed per expert."""
        extension = _make_extension(_NoPackedSupportModel())
        layer = _RecordingExpertLayer(self.HIDDEN)
        gate, up, down, gate_up = self._linear_convention_tensors()
        gate_up_t = gate_up.transpose(1, 2).contiguous()  # [E, H, 2I]
        loaded = extension._load_packed_expert_tensor(
            "model.layers.3.mlp.experts.gate_up_proj", gate_up_t, layer, None, "gate_up_proj"
        )
        assert loaded
        assert torch.equal(layer.calls[0][3], gate[0])
        assert torch.equal(layer.calls[1][3], up[0])

        layer.calls.clear()
        down_t = down.transpose(1, 2).contiguous()  # [E, I, H]
        extension._load_packed_expert_tensor(
            "model.layers.3.mlp.experts.down_proj", down_t, layer, None, "down_proj"
        )
        assert torch.equal(layer.calls[0][3], down[0])

    def test_ambiguous_orientation_raises(self):
        extension = _make_extension(_NoPackedSupportModel())
        layer = _RecordingExpertLayer(hidden_size=8)
        square = torch.randn(2, 8, 8)  # H == 2I: cannot infer orientation
        with pytest.raises(RuntimeError, match="orientation"):
            extension._load_packed_expert_tensor(
                "model.layers.3.mlp.experts.gate_up_proj", square, layer, None, "gate_up_proj"
            )

    def test_no_expert_loaded_returns_false(self):
        """If every expert is remote (EP), success must be False so the caller raises."""

        class _AllRemoteLayer(_RecordingExpertLayer):
            def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
                return False if return_success else None

        extension = _make_extension(_NoPackedSupportModel())
        layer = _AllRemoteLayer(self.HIDDEN)
        _, _, down, _ = self._linear_convention_tensors()
        loaded = extension._load_packed_expert_tensor(
            "model.layers.3.mlp.experts.down_proj", down, layer, None, "down_proj"
        )
        assert not loaded


@requires_ds32_checkpoint
class TestLoadPackedExpertTensorRealWeights:
    """Round-trip with real DeepSeek-V3.2 expert tensors: pack them the way the HF-fork
    trainer holds them in memory, run the fallback, and require bitwise-equal per-expert
    checkpoint tensors on the other side."""

    LAYER = 3
    EXPERTS = (0, 1)

    @pytest.fixture
    def real_tensors(self):
        from safetensors import safe_open

        handle = safe_open(str(Path(DS32_SNAPSHOT, "model-00001-of-000163.safetensors")), framework="pt")
        tensors = {}
        for expert in self.EXPERTS:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.{self.LAYER}.mlp.experts.{expert}.{proj}.weight"
                tensors[(expert, proj)] = handle.get_tensor(key).to(torch.bfloat16)
        return tensors

    def test_bitwise_roundtrip(self, real_tensors):
        hidden = real_tensors[(0, "gate_proj")].shape[1]
        extension = _make_extension(_NoPackedSupportModel())
        layer = _RecordingExpertLayer(hidden)

        # Pack exactly like DeepseekV32Experts holds them: gate_up [E, 2I, H], down [E, H, I]
        gate_up = torch.stack(
            [
                torch.cat([real_tensors[(e, "gate_proj")], real_tensors[(e, "up_proj")]], dim=0)
                for e in self.EXPERTS
            ]
        )
        down = torch.stack([real_tensors[(e, "down_proj")] for e in self.EXPERTS])

        base = f"model.layers.{self.LAYER}.mlp.experts"
        extension._load_packed_expert_tensor(f"{base}.gate_up_proj", gate_up, layer, None, "gate_up_proj")
        extension._load_packed_expert_tensor(f"{base}.down_proj", down, layer, None, "down_proj")

        received = {(eid, name.rsplit(".", 2)[-2]): tensor for name, _, eid, tensor in layer.calls}
        for (position, proj), expected in (
            ((0, "gate_proj"), real_tensors[(self.EXPERTS[0], "gate_proj")]),
            ((1, "up_proj"), real_tensors[(self.EXPERTS[1], "up_proj")]),
            ((1, "down_proj"), real_tensors[(self.EXPERTS[1], "down_proj")]),
        ):
            assert torch.equal(received[(position, proj)], expected), f"mismatch for expert {position} {proj}"
