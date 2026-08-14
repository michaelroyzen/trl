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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from trl.scripts.vllm_serve import WeightSyncWorkerExtension

from .testing_utils import require_vllm


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
