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

"""CPU-only tests for data parallelism of dense models in `trl vllm-serve`.

vLLM (PR #30739, shipped in 0.25.0) refuses its offline/env-var data parallelism for models without experts. The
server therefore runs dense DP ranks as standalone tensor-parallel engines on disjoint `CUDA_VISIBLE_DEVICES` slices
and offsets their weight-sync ranks, while MoE models keep using vLLM's lockstep DP engines. These tests pin down that
split without GPUs or an installed vLLM; the live multi-GPU behavior is covered by the server smoke test in the
monorepo (start `trl vllm-serve --tensor-parallel-size 4 --data-parallel-size 2` on a dense model, check
`/get_world_size/ == 8` and one `update_named_param` round trip).
"""

import os
import sys
import types
from itertools import chain
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import transformers

import trl.import_utils as trl_import_utils
from trl.scripts.vllm_serve import (
    WEIGHT_SYNC_RANK_OFFSET_ENV,
    ScriptArguments,
    WeightSyncWorkerExtension,
    config_declares_experts,
    configure_worker_env,
    is_moe_model,
    llm_worker,
    weight_sync_rank_offset,
)


VLLM_DP_KEYS = ("VLLM_DP_RANK", "VLLM_DP_RANK_LOCAL", "VLLM_DP_SIZE", "VLLM_DP_MASTER_PORT")


class TestConfigDeclaresExperts:
    def test_dense_config(self):
        assert not config_declares_experts(SimpleNamespace(model_type="qwen3_5_text", hidden_size=5120))

    @pytest.mark.parametrize("key", ["num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"])
    def test_expert_count_keys(self, key):
        assert config_declares_experts(SimpleNamespace(**{key: 256}))

    @pytest.mark.parametrize("value", [None, 0, False])
    def test_non_positive_expert_counts_are_dense(self, value):
        # Dense Qwen3.5 configs materialize `num_experts=None`; a bool must not be mistaken for an expert count.
        assert not config_declares_experts(SimpleNamespace(num_experts=value))

    def test_looks_inside_text_config(self):
        # Multimodal wrappers (e.g. Qwen3_5MoeForConditionalGeneration) keep the expert count on the text sub-config.
        text_config = SimpleNamespace(num_experts=128)
        config = SimpleNamespace(text_config=text_config, get_text_config=lambda: text_config)
        assert config_declares_experts(config)

    def test_text_config_without_experts_is_dense(self):
        text_config = SimpleNamespace(num_experts=None)
        config = SimpleNamespace(text_config=text_config, get_text_config=lambda: text_config)
        assert not config_declares_experts(config)


class TestIsMoeModel:
    def test_reads_config_and_forwards_loading_args(self, monkeypatch):
        calls = {}

        def fake_from_pretrained(model, revision=None, trust_remote_code=False):
            calls.update(model=model, revision=revision, trust_remote_code=trust_remote_code)
            return SimpleNamespace(num_experts=256)

        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", staticmethod(fake_from_pretrained))
        assert is_moe_model("org/moe", revision="abc", trust_remote_code=True)
        assert calls == {"model": "org/moe", "revision": "abc", "trust_remote_code": True}

    def test_dense_config(self, monkeypatch):
        monkeypatch.setattr(
            transformers.AutoConfig, "from_pretrained", staticmethod(lambda *a, **k: SimpleNamespace(num_experts=None))
        )
        assert not is_moe_model("org/dense")

    def test_unloadable_config_defaults_to_dense(self, monkeypatch, caplog):
        def fail(*args, **kwargs):
            raise OSError("no config.json")

        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", staticmethod(fail))
        with caplog.at_level("WARNING", logger="trl.scripts.vllm_serve"):
            assert not is_moe_model("org/unknown")
        assert "assuming a dense model" in caplog.text


class TestConfigureWorkerEnv:
    @staticmethod
    def _configure(environ, rank, *, tp, dp, moe, master_port=51000):
        configure_worker_env(
            environ,
            data_parallel_rank=rank,
            data_parallel_size=dp,
            tensor_parallel_size=tp,
            master_port=master_port,
            moe=moe,
        )
        return environ

    @pytest.mark.parametrize(
        "rank, expected_devices, expected_offset",
        [(0, "0,1,2,3", "0"), (1, "4,5,6,7", "4")],
    )
    def test_dense_ranks_get_disjoint_gpu_slices_and_rank_offsets(self, rank, expected_devices, expected_offset):
        env = self._configure({"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}, rank, tp=4, dp=2, moe=False)
        assert env["CUDA_VISIBLE_DEVICES"] == expected_devices
        assert env[WEIGHT_SYNC_RANK_OFFSET_ENV] == expected_offset
        assert not any(key in env for key in VLLM_DP_KEYS)

    def test_dense_slices_are_disjoint_and_cover_all_gpus(self):
        tp, dp = 2, 4
        shards = [
            self._configure({"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}, rank, tp=tp, dp=dp, moe=False)[
                "CUDA_VISIBLE_DEVICES"
            ].split(",")
            for rank in range(dp)
        ]
        assert [len(shard) for shard in shards] == [tp] * dp
        assert sorted(chain.from_iterable(shards)) == [str(i) for i in range(tp * dp)]
        offsets = [
            self._configure({"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}, rank, tp=tp, dp=dp, moe=False)[
                WEIGHT_SYNC_RANK_OFFSET_ENV
            ]
            for rank in range(dp)
        ]
        assert offsets == ["0", "2", "4", "6"]

    def test_dense_defaults_to_first_tp_times_dp_gpus_when_unset(self):
        env = self._configure({}, 1, tp=2, dp=2, moe=False)
        assert env["CUDA_VISIBLE_DEVICES"] == "2,3"

    def test_dense_respects_non_identity_visible_device_lists(self):
        # Arbitrary ordering and UUID-style entries are sliced as opaque strings.
        env = self._configure({"CUDA_VISIBLE_DEVICES": " 7, 5 ,GPU-aaaa,GPU-bbbb"}, 1, tp=2, dp=2, moe=False)
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-aaaa,GPU-bbbb"

    def test_dense_removes_inherited_vllm_dp_env(self):
        env = {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "VLLM_DP_RANK": "1",
            "VLLM_DP_RANK_LOCAL": "1",
            "VLLM_DP_SIZE": "2",
            "VLLM_DP_MASTER_IP": "10.0.0.1",
            "VLLM_DP_MASTER_PORT": "1234",
        }
        self._configure(env, 0, tp=1, dp=2, moe=False)
        assert not any(key.startswith("VLLM_DP_") for key in env)

    def test_dense_single_rank_keeps_all_visible_gpus(self):
        # data_parallel_size=1 (today's TP-only server) is a no-op apart from the zero offset.
        env = self._configure({"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}, 0, tp=8, dp=1, moe=False)
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
        assert env[WEIGHT_SYNC_RANK_OFFSET_ENV] == "0"

    def test_dense_raises_when_too_few_gpus_are_visible(self):
        with pytest.raises(RuntimeError, match="needs 4 GPUs"):
            self._configure({"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5"}, 1, tp=4, dp=2, moe=False)

    def test_moe_path_is_unchanged(self):
        # vLLM's lockstep data parallelism: only the VLLM_DP_* variables, no GPU slicing, no rank offset.
        env = self._configure({"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, 1, tp=2, dp=2, moe=True, master_port=51000)
        assert env == {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "VLLM_DP_RANK": "1",
            "VLLM_DP_RANK_LOCAL": "1",
            "VLLM_DP_SIZE": "2",
            "VLLM_DP_MASTER_PORT": "51000",
        }
        assert WEIGHT_SYNC_RANK_OFFSET_ENV not in env


class TestWeightSyncRankOffset:
    def test_defaults_to_zero(self):
        assert weight_sync_rank_offset({}) == 0

    def test_reads_env(self):
        assert weight_sync_rank_offset({WEIGHT_SYNC_RANK_OFFSET_ENV: "4"}) == 4

    def test_reads_process_env_by_default(self, monkeypatch):
        monkeypatch.setenv(WEIGHT_SYNC_RANK_OFFSET_ENV, "6")
        assert weight_sync_rank_offset() == 6
        monkeypatch.delenv(WEIGHT_SYNC_RANK_OFFSET_ENV)
        assert weight_sync_rank_offset() == 0


def _install_fake_vllm_distributed(monkeypatch, tp_rank: int):
    """Stub the vLLM distributed modules `init_communicator` imports; returns the StatelessProcessGroup stub."""
    stateless_pg = MagicMock(name="StatelessProcessGroup")
    stateless_pg.create.return_value = MagicMock(name="pg")
    parallel_state = types.ModuleType("vllm.distributed.parallel_state")
    parallel_state.get_world_group = lambda: SimpleNamespace(rank=tp_rank)
    utils = types.ModuleType("vllm.distributed.utils")
    utils.StatelessProcessGroup = stateless_pg
    pynccl = types.ModuleType("vllm.distributed.device_communicators.pynccl")
    pynccl.PyNcclCommunicator = MagicMock(name="PyNcclCommunicator")
    device_communicators = types.ModuleType("vllm.distributed.device_communicators")
    distributed = types.ModuleType("vllm.distributed")
    vllm = types.ModuleType("vllm")
    for name, module in {
        "vllm": vllm,
        "vllm.distributed": distributed,
        "vllm.distributed.parallel_state": parallel_state,
        "vllm.distributed.utils": utils,
        "vllm.distributed.device_communicators": device_communicators,
        "vllm.distributed.device_communicators.pynccl": pynccl,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return stateless_pg


class TestInitCommunicatorRankOffset:
    @pytest.mark.parametrize("offset, expected_rank", [(None, 2), ("0", 2), ("4", 6)])
    def test_rank_is_tp_rank_plus_offset(self, monkeypatch, offset, expected_rank):
        stateless_pg = _install_fake_vllm_distributed(monkeypatch, tp_rank=2)
        # Skip the same-device guard (needs a real accelerator) and the XPU branch.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(transformers, "is_torch_xpu_available", lambda: False)
        monkeypatch.setattr(trl_import_utils, "is_vllm_ascend_available", lambda: False)
        if offset is None:
            monkeypatch.delenv(WEIGHT_SYNC_RANK_OFFSET_ENV, raising=False)
        else:
            monkeypatch.setenv(WEIGHT_SYNC_RANK_OFFSET_ENV, offset)

        extension = WeightSyncWorkerExtension()
        extension.device = torch.device("cpu")
        extension.init_communicator("0.0.0.0", 51216, world_size=9, client_device_uuid="GPU-trainer")

        stateless_pg.create.assert_called_once_with(host="0.0.0.0", port=51216, rank=expected_rank, world_size=9)
        assert extension.client_rank == 8
        assert extension.communicator is not None


class _FakeConnection:
    """Feeds `llm_worker` a single shutdown command and records what it sends."""

    def __init__(self):
        self.sent = []

    def send(self, obj):
        self.sent.append(obj)

    def recv(self):
        return {"type": "shutdown"}


@pytest.fixture
def restore_environ():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


class TestLlmWorkerWiring:
    """Runs `llm_worker` against a fake `vllm.LLM` to check the environment is prepared before the engine exists."""

    def _run_worker(self, monkeypatch, *, moe: bool, rank: int) -> tuple[dict, list]:
        constructed = {}

        class FakeLLM:
            def __init__(self, **kwargs):
                constructed["kwargs"] = kwargs
                constructed["env_at_init"] = dict(os.environ)

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

        script_args = ScriptArguments(model="org/model", tensor_parallel_size=2, data_parallel_size=2)
        connection = _FakeConnection()
        llm_worker(script_args, rank, 51000, connection, moe)
        return constructed, connection.sent

    def test_dense_rank_runs_standalone_engine_on_its_gpu_slice(self, monkeypatch, restore_environ):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        for key in VLLM_DP_KEYS:
            os.environ.pop(key, None)

        constructed, sent = self._run_worker(monkeypatch, moe=False, rank=1)

        env = constructed["env_at_init"]
        assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
        assert env[WEIGHT_SYNC_RANK_OFFSET_ENV] == "2"
        assert not any(key in env for key in VLLM_DP_KEYS)
        kwargs = constructed["kwargs"]
        assert kwargs["tensor_parallel_size"] == 2
        assert "data_parallel_size" not in kwargs  # a plain TP engine, never vLLM DP
        assert kwargs["worker_extension_cls"] == "trl.scripts.vllm_serve.WeightSyncWorkerExtension"
        assert sent == [{"status": "ready"}]

    def test_moe_rank_uses_vllm_data_parallel_env(self, monkeypatch, restore_environ):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        os.environ.pop(WEIGHT_SYNC_RANK_OFFSET_ENV, None)

        constructed, sent = self._run_worker(monkeypatch, moe=True, rank=1)

        env = constructed["env_at_init"]
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        assert {key: env[key] for key in VLLM_DP_KEYS} == {
            "VLLM_DP_RANK": "1",
            "VLLM_DP_RANK_LOCAL": "1",
            "VLLM_DP_SIZE": "2",
            "VLLM_DP_MASTER_PORT": "51000",
        }
        assert WEIGHT_SYNC_RANK_OFFSET_ENV not in env
        assert constructed["kwargs"]["tensor_parallel_size"] == 2
        assert sent == [{"status": "ready"}]

    def test_detects_moe_from_config_when_not_told(self, monkeypatch, restore_environ):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        monkeypatch.setattr(
            transformers.AutoConfig, "from_pretrained", staticmethod(lambda *a, **k: SimpleNamespace(num_experts=8))
        )
        constructed = {}

        class FakeLLM:
            def __init__(self, **kwargs):
                constructed["env_at_init"] = dict(os.environ)

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

        script_args = ScriptArguments(model="org/moe", tensor_parallel_size=2, data_parallel_size=2)
        llm_worker(script_args, 0, 51000, _FakeConnection())  # `moe` left to be detected
        assert constructed["env_at_init"]["VLLM_DP_SIZE"] == "2"
