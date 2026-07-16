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

import argparse
import base64
import json
import logging
import math
import os
import re
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection


# We use CUDA with multiprocessing, so we must use the 'spawn' start method. Otherwise, we will get the following
# error: RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use
# the 'spawn' start method
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class WeightSyncWorkerExtension:
    """
    A vLLM worker extension that enables weight synchronization between a client and multiple server workers.

    This worker uses a `StatelessProcessGroup` to establish communication and a `PyNcclCommunicator` or
    `ProcessGroupXCCL` to handle efficient GPU-based communication using NCCL. The primary purpose of this class is to
    receive updated model weights from a client process and distribute them to all worker processes participating in
    model inference.
    """

    # The following attributes are initialized when `init_communicator` method is called.
    communicator = None  # Communicator for weight updates
    client_rank = None  # Source rank for broadcasting updated weights

    # Fused (packed 3D) MoE expert tensors, e.g. `...mlp.experts.gate_up_proj` for Qwen3.5-MoE.
    _FUSED_EXPERT_WEIGHT_RE = re.compile(r"(?P<module>.+\.experts)\.(?P<proj>gate_up_proj|down_proj)$")

    # Packed expert tensors including block-scale companions, used only during layerwise
    # (quantized) weight-update rounds, e.g. `...mlp.experts.gate_up_proj_scale_inv`.
    _PACKED_EXPERT_ANY_RE = re.compile(
        r"(?P<module>.+\.experts)\.(?P<proj>gate_up_proj|down_proj)(?P<scale>_scale_inv)?$"
    )

    # True between begin_weight_update() and end_weight_update() on quantized models:
    # parameters are restored to checkpoint layout and weight loaders are wrapped by
    # vLLM's layerwise reload machinery, so the kernel-format staging path must not run.
    _layerwise_update_active = False

    # Unquantized MoE backends whose kernel-format conversion is validated to be an independent,
    # per-tensor transformation of w13 and w2 (see `_update_fused_expert_param`). AITER (ROCm) is
    # deliberately excluded until validated.
    _SUPPORTED_UNQUANTIZED_MOE_BACKENDS = frozenset(
        {"FLASHINFER_TRTLLM", "FLASHINFER_CUTLASS", "TRITON", "BATCHED_TRITON"}
    )

    # Set to the imported vLLM private-API handles after the first successful guard check.
    _moe_conversion_api = None
    _layerwise_reload_api = None

    def init_communicator(self, host: str, port: int, world_size: int, client_device_uuid: str) -> None:
        """
        Initializes the weight update communicator using a stateless process group.

        This method creates a `StatelessProcessGroup` that allows external training processes to communicate with vLLM
        workers without interfering with the global torch distributed group.

        Args:
            host (`str`):
                Hostname or IP address of the master node.
            port (`int`):
                Port number to be used for communication.
            world_size (`int`):
                Total number of participating processes in the update group.
            client_device_uuid (`str`):
                UUID of the device of client main process. Used to assert that devices are different from vllm workers
                devices.
        """
        import torch
        import torch.distributed.distributed_c10d as c10d
        from transformers import is_torch_xpu_available
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.parallel_state import get_world_group
        from vllm.distributed.utils import StatelessProcessGroup

        from trl.import_utils import is_vllm_ascend_available

        if is_vllm_ascend_available():
            from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator

        if self.communicator is not None:
            raise RuntimeError("Weight update group already initialized. Call close_communicator first.")

        # TODO: will remove after torch xpu 2.9 support uuid in get_device_properties
        if torch.cuda.is_available() or (
            is_torch_xpu_available() and hasattr(torch.xpu.get_device_properties(self.device), "uuid")
        ):
            accelerator_module = torch.xpu if is_torch_xpu_available() else torch.cuda
            if client_device_uuid == str(accelerator_module.get_device_properties(self.device).uuid):
                raise RuntimeError(
                    f"Attempting to use the same CUDA device (UUID: {client_device_uuid}) for multiple distinct "
                    "roles/ranks within the same communicator. This setup is unsupported and will likely lead to program "
                    "hangs or incorrect behavior. Ensure that trainer is using different devices than vLLM server."
                )
        # Get the rank of the current worker in the global world group.
        rank = get_world_group().rank

        if is_torch_xpu_available():
            store = torch.distributed.TCPStore(host_name=host, port=port, world_size=world_size, is_master=(rank == 0))
            prefixed_store = c10d.PrefixStore("client2server", store)
            xccl_options = c10d.ProcessGroupXCCL.Options()
            pg = c10d.ProcessGroupXCCL(
                store=prefixed_store,
                rank=rank,
                size=world_size,
                options=xccl_options,
            )
            self.communicator = pg
        else:
            # Create a stateless process group to manage communication between training processes and vLLM workers.
            # Initialize the NCCL-based communicator for weight synchronization.
            pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
            self.communicator = PyNcclCommunicator(pg, device=self.device)

        # The client process that sends updated weights has the highest rank (world_size - 1).
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: str, shape: Sequence[int]) -> None:
        """
        Receives updated weights from the client process and updates the named parameter in the model.

        Args:
            name (`str`):
                Name of the weight tensor being updated.
            dtype (`str`):
                Data type of the weight tensor as a string (e.g., `"torch.float32"`).
            shape (`Sequence[int]`):
                Shape of the weight tensor.
        """
        import torch
        from transformers import is_torch_xpu_available

        if self.communicator is None:
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

        dtype = getattr(torch, dtype.split(".")[-1])
        # Allocate memory for the incoming weight tensor on the correct device.
        weight = torch.empty(shape, dtype=dtype, device=self.device)

        if is_torch_xpu_available():
            # Use XCCL to broadcast the updated weights from the client (src) to all workers.
            self.communicator.broadcast(weight, root=self.client_rank)
            self.communicator.barrier()
        else:
            # Use NCCL to broadcast the updated weights from the client (src) to all workers.
            self.communicator.broadcast(weight, src=self.client_rank)
            self.communicator.group.barrier()

        # Load the received weights into the model.
        #
        # Layerwise (quantized) rounds: between begin_weight_update() and end_weight_update()
        # the model's parameters are restored to checkpoint layout and every weight loader is
        # wrapped by vLLM's layerwise reload machinery (deferred per-layer re-processing, with
        # results copied back into the original kernel-format storage). Tensors arrive in
        # checkpoint format (e.g. fp8 weight + `*_scale_inv` scales) and are routed through the
        # ordinary `load_weights`; packed 3D expert tensors are sliced into per-expert
        # checkpoint keys first.
        #
        # Non-layerwise rounds (unquantized models): fused (packed 3D) MoE expert tensors need
        # special handling: unquantized MoE backends such as FlashInfer TRT-LLM re-lay-out the
        # expert weight parameters into a kernel block format after the bulk startup load, and an
        # incremental `load_weights` against such a converted parameter crashes the engine core
        # (FlashInfer TRT-LLM) or silently corrupts the weights (FlashInfer CUTLASS). See
        # monorepo docs/vllm-flashinfer-moe-weight-sync.md.
        if self._layerwise_update_active:
            self._load_checkpoint_format_tensor(name, weight)
        elif not self._maybe_update_fused_expert_param(name, weight):
            self.model_runner.model.load_weights(weights=[(name, weight)])

    @classmethod
    def _get_layerwise_reload_api(cls):
        """Import and guard vLLM's layerwise reload machinery (private API).

        Validated against vllm==0.25.0: `record_metadata_for_reloading` is called at model
        construction (model_loader/utils.py), so standard-layout metadata is always available;
        `initialize_layerwise_reload` restores checkpoint-layout params with wrapped loaders;
        `finalize_layerwise_processing` re-runs per-layer quantization processing (e.g.
        Fp8MoEMethod/Fp8LinearMethod process_weights_after_loading, FlashInfer TRT-LLM
        kernel-format shuffles) and copies results into the original tensor storage, keeping
        captured CUDA graphs valid. This is the same call sequence as
        GPUModelRunner.reload_weights (gpu_model_runner.py), driven from the RL sync protocol
        instead of a checkpoint iterator.
        """
        if cls._layerwise_reload_api is not None:
            return cls._layerwise_reload_api

        guard_msg = (
            "WeightSyncWorkerExtension: vLLM's layerwise reload API changed (validated on "
            "vllm==0.25.0). Re-validate the quantized weight-update path against this vLLM "
            "version before syncing (a broken reload corrupts every subsequent rollout). "
        )
        try:
            from vllm.model_executor.model_loader.reload.layerwise import (
                finalize_layerwise_processing,
                initialize_layerwise_reload,
            )
        except ImportError as e:
            raise RuntimeError(guard_msg + f"Import failed: {e}") from e

        import inspect

        init_params = list(inspect.signature(initialize_layerwise_reload).parameters)
        finalize_params = list(inspect.signature(finalize_layerwise_processing).parameters)
        if init_params != ["model"] or finalize_params != ["model", "model_config"]:
            raise RuntimeError(
                guard_msg
                + f"Signatures changed: initialize={init_params}, finalize={finalize_params}."
            )

        cls._layerwise_reload_api = (initialize_layerwise_reload, finalize_layerwise_processing)
        return cls._layerwise_reload_api

    def get_server_quantization(self) -> str | None:
        """Return the server model's quantization method (e.g. "fp8"), or None.

        The sync client uses this to decide whether to send checkpoint-format quantized
        tensors (weight + scale pairs inside a begin/end layerwise round) or plain bf16.
        """
        quantization = getattr(self.model_runner.model_config, "quantization", None)
        return str(quantization) if quantization is not None else None

    def begin_weight_update(self) -> None:
        """Arm vLLM's layerwise reload for a full checkpoint-format weight-update round.

        Only valid for quantized models. After this call the model is NOT runnable until
        end_weight_update(): every layer's parameters are restored to checkpoint layout on
        the meta device and re-materialize as their weights stream in.
        """
        if self._layerwise_update_active:
            raise RuntimeError("begin_weight_update called while a weight update is already active.")
        self._guard_no_full_cudagraphs()
        initialize_layerwise_reload, _ = self._get_layerwise_reload_api()
        initialize_layerwise_reload(self.model_runner.model)
        self._layerwise_update_active = True

    def _guard_no_full_cudagraphs(self) -> None:
        """Refuse layerwise weight updates under FULL CUDA graphs.

        Empirically (vllm==0.25.0, tiny DeepSeek-V3.2 fp8, B300): after a layerwise
        reload round, every parameter/buffer is value- AND address-stable, eager and
        PIECEWISE-graph outputs are bitwise identical to pre-round, but captured
        FULL-decode graphs produce different (deterministic, wrong) outputs - the
        finalize step rebuilds quantized MoE kernels (`Fp8MoEMethod._setup_kernel`)
        and the full graph keeps stale internal state. Fail loudly instead of
        corrupting rollouts; serve with `--cudagraph-mode PIECEWISE` (or
        `--enforce-eager True`) for RL sync against quantized models.
        """
        try:
            compilation_config = self.model_runner.vllm_config.compilation_config
            mode = compilation_config.cudagraph_mode
        except AttributeError as e:
            raise RuntimeError(
                "WeightSyncWorkerExtension: could not inspect the CUDA graph mode "
                f"(vLLM internals changed: {e}). Re-validate layerwise weight updates "
                "against this vLLM version."
            ) from e
        has_full = getattr(mode, "has_full_cudagraphs", None)
        full_active = has_full() if callable(has_full) else "FULL" in str(mode)
        if full_active:
            raise RuntimeError(
                f"Layerwise weight updates are unsafe under CUDA graph mode {mode}: "
                "captured FULL graphs keep stale quantized-kernel state after the "
                "post-update kernel rebuild (validated on vllm==0.25.0). Launch the "
                "server with --cudagraph-mode PIECEWISE or --enforce-eager True."
            )

    def end_weight_update(self) -> None:
        """Finalize a layerwise weight-update round: process remaining layers and restore
        kernel formats (per-layer quantization post-processing + copy-back)."""
        if not self._layerwise_update_active:
            raise RuntimeError("end_weight_update called without begin_weight_update.")
        _, finalize_layerwise_processing = self._get_layerwise_reload_api()
        try:
            finalize_layerwise_processing(self.model_runner.model, self.model_runner.model_config)
        finally:
            self._layerwise_update_active = False

    def _load_checkpoint_format_tensor(self, name: str, weight) -> None:
        """Route one checkpoint-format tensor through load_weights during a layerwise round.

        Packed 3D expert tensors (weights or `_scale_inv` block scales) are sliced into
        per-expert checkpoint keys first, since models like DeepSeek only map per-expert
        names; everything else goes through load_weights directly. Loading nothing is an
        error for weights but tolerated for auxiliary tensors vLLM does not use.
        """
        model = self.model_runner.model
        match = self._PACKED_EXPERT_ANY_RE.match(name)
        if match is not None and weight.dim() == 3:
            pairs = self._packed_expert_checkpoint_pairs(
                name, weight, match.group("proj"), is_scale=match.group("scale") is not None
            )
            loaded = set()
            for ckpt_key, tensor_2d in pairs:
                loaded |= set(model.load_weights(weights=[(ckpt_key, tensor_2d)]) or ())
            if not loaded:
                raise RuntimeError(
                    f"Layerwise weight update for '{name}' loaded nothing via per-expert keys."
                )
            return
        try:
            loaded = model.load_weights(weights=[(name, weight)])
        except KeyError:
            loaded = None  # unknown name: models may raise instead of skipping
        if not loaded:
            logging.getLogger(__name__).warning(
                "Layerwise weight update: '%s' did not map onto the model; skipped.", name
            )

    def _packed_expert_checkpoint_pairs(
        self, name: str, weight, proj: str, is_scale: bool, hidden_size: int | None = None
    ):
        """Slice a packed 3D expert tensor into (checkpoint key, 2D tensor) pairs.

        Handles both the weight tensors (`[E, 2I, H]` gate-first / `[E, H, I]`, linear
        convention) and their block-scale companions (same layout divided by the 128-block
        in both trailing dims). Orientation is inferred against the model's hidden size
        (scaled by the block size for `_scale_inv` tensors); ambiguity raises.
        """
        import math as _math

        if hidden_size is None:
            hidden_size = self.model_runner.model_config.hf_text_config.hidden_size
        hidden = _math.ceil(hidden_size / 128) if is_scale else hidden_size
        base = name.rsplit(".", 1)[0]
        suffix = "weight_scale_inv" if is_scale else "weight"
        _, dim1, dim2 = weight.shape

        pairs = []
        for expert_id in range(weight.shape[0]):
            if proj == "gate_up_proj":
                if dim2 == hidden and dim1 != hidden:  # [E, 2I, H] rows-first
                    intermediate = dim1 // 2
                    gate = weight[expert_id, :intermediate, :]
                    up = weight[expert_id, intermediate:, :]
                elif dim1 == hidden and dim2 != hidden:  # [E, H, 2I] transposed
                    intermediate = dim2 // 2
                    gate = weight[expert_id, :, :intermediate].t().contiguous()
                    up = weight[expert_id, :, intermediate:].t().contiguous()
                else:
                    raise RuntimeError(
                        f"Cannot infer packed gate_up orientation for '{name}': shape "
                        f"{tuple(weight.shape)} vs hidden {hidden} (is_scale={is_scale})."
                    )
                pairs.append((f"{base}.{expert_id}.gate_proj.{suffix}", gate))
                pairs.append((f"{base}.{expert_id}.up_proj.{suffix}", up))
            else:
                if dim1 == hidden and dim2 != hidden:  # [E, H, I]
                    down = weight[expert_id, :, :]
                elif dim2 == hidden and dim1 != hidden:  # [E, I, H] transposed
                    down = weight[expert_id, :, :].t().contiguous()
                else:
                    raise RuntimeError(
                        f"Cannot infer packed down orientation for '{name}': shape "
                        f"{tuple(weight.shape)} vs hidden {hidden} (is_scale={is_scale})."
                    )
                pairs.append((f"{base}.{expert_id}.down_proj.{suffix}", down))
        return pairs

    @classmethod
    def _get_moe_conversion_api(cls):
        """Import and guard the private vLLM MoE kernel-format conversion API.

        `_update_fused_expert_param` relies on vLLM internals that are not a stable API. This
        guard was validated against vllm==0.25.0 and fails loudly (rather than corrupting
        weights silently) if a vLLM upgrade changes the surface.
        """
        if cls._moe_conversion_api is not None:
            return cls._moe_conversion_api

        import inspect

        guard_msg = (
            "WeightSyncWorkerExtension: vLLM's private MoE kernel-format conversion API changed "
            "(validated on vllm==0.25.0). Re-validate the fused-expert weight sync path against "
            "this vLLM version (see docs/vllm-flashinfer-moe-weight-sync.md). "
        )
        try:
            from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import FusedMoEModularMethod
            from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
                convert_to_unquantized_kernel_format,
            )
            from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
                UnquantizedFusedMoEMethod,
            )
            from vllm.model_executor.utils import set_weight_attrs
        except ImportError as e:
            raise RuntimeError(guard_msg + f"Import failed: {e}") from e

        expected_params = ["unquantized_backend", "moe_config", "w13_weight", "w2_weight"]
        actual_params = list(inspect.signature(convert_to_unquantized_kernel_format).parameters)
        if actual_params != expected_params:
            raise RuntimeError(
                guard_msg
                + f"convert_to_unquantized_kernel_format signature changed: expected {expected_params}, "
                + f"got {actual_params}."
            )

        cls._moe_conversion_api = (
            FusedMoEModularMethod,
            UnquantizedFusedMoEMethod,
            convert_to_unquantized_kernel_format,
            set_weight_attrs,
        )
        return cls._moe_conversion_api

    def _maybe_update_fused_expert_param(self, name: str, weight) -> bool:
        """Route a fused (packed 3D) MoE expert tensor through the staging + re-conversion path.

        Returns `True` if the tensor was handled here, `False` if the caller should fall back to
        the plain `model.load_weights` path (dense tensors, quantized MoE methods, unknown name
        forms, or a model without a matching MoE module).
        """
        if weight.dim() != 3:
            return False
        match = self._FUSED_EXPERT_WEIGHT_RE.match(name)
        if match is None:
            return False

        model = self.model_runner.model
        try:
            experts_module = model.get_submodule(match.group("module"))
        except AttributeError:
            return False
        # The `.experts` attribute is a MoERunner in vLLM >= 0.25; the expert weight parameters
        # live on its RoutedExperts child. Older layouts keep them on the module itself.
        layer = getattr(experts_module, "routed_experts", experts_module)
        if not (hasattr(layer, "w13_weight") and hasattr(layer, "w2_weight") and hasattr(layer, "quant_method")):
            return False

        FusedMoEModularMethod, UnquantizedFusedMoEMethod, _, _ = self._get_moe_conversion_api()
        quant_method = layer.quant_method
        if isinstance(quant_method, FusedMoEModularMethod):
            quant_method = quant_method.old_quant_method
        if not isinstance(quant_method, UnquantizedFusedMoEMethod):
            # Quantized MoE methods manage their own weight layouts; keep the legacy behavior.
            return False
        if quant_method.moe_kernel is None:
            # Weights have not been kernel-format converted yet; the plain path is still valid.
            return False

        backend = quant_method.unquantized_backend
        if backend.name not in self._SUPPORTED_UNQUANTIZED_MOE_BACKENDS:
            raise NotImplementedError(
                f"Fused expert weight sync is not supported for unquantized MoE backend "
                f"'{backend.name}' (tensor '{name}'). Supported backends: "
                f"{sorted(self._SUPPORTED_UNQUANTIZED_MOE_BACKENDS)}."
            )

        self._update_fused_expert_param(name, weight, layer, quant_method, match.group("proj"))
        logging.getLogger(__name__).debug(
            "Synced fused expert tensor '%s' via staging + %s kernel-format re-conversion.",
            name,
            backend.name,
        )
        return True

    def _update_fused_expert_param(self, name: str, weight, layer, quant_method, proj: str) -> None:
        """Update one fused expert tensor on a (possibly kernel-format-converted) FusedMoE layer.

        The full unsharded checkpoint-form tensor (`gate_up_proj [E, hidden, 2*intermediate]` or
        `down_proj [E, intermediate, hidden]`) arrives on every worker. Mirroring how the bulk
        startup load works, this:

        1. Registers a standard-layout staging parameter (the shape `create_weights` produces:
           `w13 [E_local, 2I_part, H]` / `w2 [E_local, H, I_part]`) in place of the converted
           parameter and routes the tensor through `model.load_weights`, so vLLM's validated
           fused-expert loader handles the transpose, gate/up split, expert mapping (EP) and TP
           sharding.
        2. Re-runs vLLM's kernel-format conversion (`convert_to_unquantized_kernel_format`) on
           the staged tensor and copies the result into the original parameter storage, keeping
           tensor addresses stable so captured CUDA graphs remain valid. The w13/w2 conversions
           are independent per-tensor permutations for the supported backends, so the counterpart
           parameter is never touched; its slot is fed a discarded dummy tensor.

        Under backends without a kernel-format conversion (e.g. Triton) this degrades to plain
        copies through the same code path.
        """
        import torch

        _, _, convert_to_unquantized_kernel_format, set_weight_attrs = self._get_moe_conversion_api()

        target_name = "w13_weight" if proj == "gate_up_proj" else "w2_weight"
        param = getattr(layer, target_name)

        # Standard-layout shapes, mirroring UnquantizedFusedMoEMethod.create_weights.
        moe_config = layer.moe_config
        num_local_experts = moe_config.num_local_experts
        intermediate = moe_config.intermediate_size_per_partition
        hidden = layer.hidden_size
        w13_up_dim = 2 * intermediate if moe_config.is_act_and_mul else intermediate
        w13_shape = (num_local_experts, w13_up_dim, hidden)
        w2_shape = (num_local_experts, hidden, intermediate)

        staging = torch.nn.Parameter(
            torch.empty(
                w13_shape if target_name == "w13_weight" else w2_shape,
                dtype=layer.params_dtype,
                device=param.device,
            ),
            requires_grad=False,
        )
        set_weight_attrs(staging, {"weight_loader": layer.weight_loader})

        # Temporarily swap the staging parameter in so the regular load path writes the
        # standard-layout local shard into it, then restore the original parameter.
        setattr(layer, target_name, staging)
        try:
            loaded = self._load_packed_expert_tensor(name, weight, layer, staging, proj)
        finally:
            setattr(layer, target_name, param)
        if not loaded:
            raise RuntimeError(
                f"Fused expert weight sync for '{name}' did not load any weights into the "
                "staging tensor; the weight name does not map onto the model as expected."
            )

        # Convert the staged tensor to the backend's kernel format. The counterpart slot gets a
        # dummy tensor whose converted output is discarded (conversions are per-tensor).
        dummy = torch.empty(
            w2_shape if target_name == "w13_weight" else w13_shape,
            dtype=layer.params_dtype,
            device=param.device,
        )
        if target_name == "w13_weight":
            w13_new, _ = convert_to_unquantized_kernel_format(
                quant_method.unquantized_backend, moe_config=moe_config, w13_weight=staging.data, w2_weight=dummy
            )
            converted = w13_new
        else:
            _, w2_new = convert_to_unquantized_kernel_format(
                quant_method.unquantized_backend, moe_config=moe_config, w13_weight=dummy, w2_weight=staging.data
            )
            converted = w2_new

        if converted.shape != param.data.shape or converted.dtype != param.data.dtype:
            raise RuntimeError(
                f"Fused expert weight sync for '{name}': kernel-format conversion produced "
                f"{converted.dtype} {tuple(converted.shape)} but the parameter storage is "
                f"{param.data.dtype} {tuple(param.data.shape)}. Refusing to write (this would "
                "corrupt the model); re-validate against this vLLM version."
            )
        # Copy into the original storage so data_ptr stays stable for captured CUDA graphs.
        param.data.copy_(converted)

    def _load_packed_expert_tensor(self, name: str, weight, layer, staging, proj: str) -> bool:
        """Load a packed 3D expert tensor into the (swapped-in) staging parameter.

        Preferred path: the model's own `load_weights`, for vLLM models with a native
        packed-3D loader (e.g. qwen3_5's `load_fused_expert_weights` — transformers-5
        checkpoints for those models ship packed tensors, so vLLM handles the layout,
        gate/up split, EP expert mapping and TP narrowing).

        Fallback path: models whose vLLM implementation only understands per-expert 2D
        checkpoint names (e.g. DeepSeek-V3.2: `deepseek_v2.py` has no packed-3D handling,
        so `load_weights` returns nothing for `...experts.gate_up_proj`). The packed
        tensor is sliced per expert into checkpoint-convention 2D tensors and driven
        through the layer's expert-aware `weight_loader` (shard ids w1/w3/w2), which is
        exactly what `load_weights` would do per checkpoint key - reusing vLLM's
        validated EP mapping and TP narrowing rather than reimplementing them.

        Orientation is detected from the wire shape (in-memory transformers layout):
        - deepseek_v32 trainers sync `nn.functional.linear` convention:
          `gate_up [E, 2I, H]` (gate rows first), `down [E, H, I]` - matching the
          per-expert checkpoint convention directly after slicing.
        - transposed layouts (`gate_up [E, H, 2I]` / `down [E, I, H]`, the qwen3_5
          disk convention) are transposed per expert before loading.
        A shape that matches neither (or is ambiguous, `H == 2I` / `H == I`) raises.
        """
        try:
            loaded = self.model_runner.model.load_weights(weights=[(name, weight)])
        except KeyError:
            # Models without a packed-3D loader may raise on the unknown name (e.g.
            # deepseek_v2.py's final params_dict[name] lookup) instead of skipping it.
            loaded = None
        if loaded:
            return True

        shard_ids = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
        expert_key_re = re.compile(r"\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
        any_loaded = False
        for ckpt_key, tensor_2d in self._packed_expert_checkpoint_pairs(
            name, weight, proj, is_scale=False, hidden_size=layer.hidden_size
        ):
            key_match = expert_key_re.search(ckpt_key)
            # Under EP, experts not owned by this rank return success=False; that is
            # expected and must not fail the sync (mirrors load_weights' skip).
            success = layer.weight_loader(
                staging,
                tensor_2d,
                ckpt_key,
                shard_id=shard_ids[key_match.group(2)],
                expert_id=int(key_match.group(1)),
                return_success=True,
            )
            any_loaded = any_loaded or bool(success)
        return any_loaded

    def close_communicator(self) -> None:
        """
        Closes the communicator when weight synchronization is no longer needed.

        This method deletes the NCCL communicator to release associated resources.
        """

        if self.communicator is not None:
            del self.communicator
            self.communicator = None  # Ensure attribute is reset to None
            self.client_rank = None  # Ensure attribute is reset to None


@dataclass
class ScriptArguments:
    r"""
    Arguments for the script.

    Args:
        model (`str`):
            Model name or path to load the model from.
        revision (`str`, *optional*):
            Revision to use for the model. If not specified, the default branch will be used.
        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Number of tensor parallel workers to use.
        data_parallel_size (`int`, *optional*, defaults to `1`):
            Number of data parallel workers to use. For dense models, keep this at 1. Setting this above `1` for dense
            models is not supported/useful and will error out (see vLLM PR #30739).
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host address to run the server on.
        port (`int`, *optional*, defaults to `8000`):
            Port to run the server on.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the
            device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus
            improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors
            during initialization.
        dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for vLLM generation. If set to `"auto"`, the data type will be automatically determined
            based on the model configuration. Find the supported values in the vLLM documentation.
        max_model_len (`int`, *optional*):
            If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced
            `vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model
            context size, which might be much larger than the KV cache, leading to inefficiencies.
        enable_prefix_caching (`bool`, *optional*):
            Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the hardware support
            this feature.
        enable_expert_parallel (`bool`, *optional*):
            Whether to enable expert parallelism in vLLM for MoE models. If set to `True`, experts will be distributed
            across tensor parallel workers.
        moe_backend (`str`, *optional*):
            MoE kernel backend override, forwarded to vLLM's `--moe-backend` engine arg (e.g. `"triton"`,
            `"flashinfer_trtllm"`). When unset, vLLM selects a backend automatically. RL weight sync requires a
            backend that keeps expert weights in the standard layout (e.g. `"triton"`): backends that re-lay-out
            expert weights into kernel block formats after loading crash on incremental expert-weight updates.
        enforce_eager (`bool`, *optional*, defaults to `False`):
            Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always execute the
            model in eager mode. If `False` (default behavior), we will use CUDA graph and eager execution in hybrid.
        vllm_model_impl (`str`, *optional*, defaults to `"vllm"`):
            Model implementation to use for vLLM. Must be one of `"transformers"` or `"vllm"`. `"transformers"`: Use
            the `transformers` backend for model implementation. `"vllm"`: Use the `vllm` library for model
            implementation.
        kv_cache_dtype (`str`, *optional*, defaults to `"auto"`):
            Data type to use for KV cache. If set to `"auto"`, the dtype will default to the model data type.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to trust remote code when loading models. Set to `True` to allow executing code from model
            repositories. This is required for some custom models but introduces security risks.
        language_model_only (`bool`, *optional*, defaults to `False`):
            Whether to force vLLM to load the model in language-model-only mode. Disables all multimodal inputs by
            setting every modality limit to 0, so a multimodal checkpoint serves as a text-only model.
        limit_mm_per_prompt (`str`, *optional*):
            Limits on multimodal items per prompt, as comma-separated key=value pairs (e.g., `"image=0,video=0"`).
            Passed through to vLLM's EngineArgs to restrict the number of multimodal inputs.
        log_level (`str`, *optional*, defaults to `"info"`):
            Log level for uvicorn. Possible choices: `"critical"`, `"error"`, `"warning"`, `"info"`, `"debug"`,
            `"trace"`.
        distributed_executor_backend (`str` or `None`, *optional*):
            Distributed executor backend for vLLM. Set to `"ray"` to distribute tensor parallel workers across multiple
            nodes via a Ray cluster. Required when `tensor_parallel_size` exceeds the number of local GPUs. If not set,
            vLLM defaults to the multiproc backend (single-node only).
        speculative_config (`str`, *optional*):
            JSON string for vLLM speculative decoding config, forwarded to `LLM(speculative_config=...)`. When unset,
            speculative decoding is disabled. Example: `'{"method": "qwen3_next_mtp", "num_speculative_tokens": 5}'`.
    """

    model: str = field(
        metadata={"help": "Model name or path to load the model from."},
    )
    revision: str | None = field(
        default=None,
        metadata={"help": "Revision to use for the model. If not specified, the default branch will be used."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    data_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Number of data parallel workers to use. For dense models, keep this at 1. Setting this above "
            "`1` for dense models is not supported/useful and will error out (see vLLM PR #30739)."
        },
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    max_model_len: int | None = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    enable_prefix_caching: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )
    cudagraph_mode: str | None = field(
        default=None,
        metadata={
            "help": "vLLM CUDA graph mode override (e.g. 'PIECEWISE', 'FULL_AND_PIECEWISE', 'NONE'), forwarded via "
            "compilation_config. REQUIRED to be 'PIECEWISE' (or serve with --enforce-eager) when running RL weight "
            "sync against a quantized (fp8) model: layerwise weight updates leave captured FULL graphs with stale "
            "quantized-kernel state (validated on vllm==0.25.0); begin_weight_update refuses FULL-graph servers."
        },
    )
    enable_expert_parallel: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to enable expert parallelism in vLLM for MoE models. If set to `True`, experts will "
            "be distributed across tensor parallel workers."
        },
    )
    moe_backend: str | None = field(
        default=None,
        metadata={
            "help": "MoE kernel backend override, forwarded to vLLM's `--moe-backend` engine arg (e.g. 'triton', "
            "'flashinfer_trtllm'). When unset, vLLM selects a backend automatically. RL weight sync requires a "
            "backend that keeps expert weights in the standard layout (e.g. 'triton')."
        },
    )
    enforce_eager: bool | None = field(
        default=False,
        metadata={
            "help": "Whether to enforce eager execution. If set to `True`, we will disable CUDA graph and always "
            "execute the model in eager mode. If `False` (default behavior), we will use CUDA graph and eager "
            "execution in hybrid."
        },
    )
    kv_cache_dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for KV cache. If set to 'auto', the dtype will default to the model data type."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to trust remote code when loading models. Set to True to allow executing code from model "
            "repositories. This is required for some custom models but introduces security risks."
        },
    )
    language_model_only: bool = field(
        default=False,
        metadata={
            "help": "Whether to force vLLM to load the model in language-model-only mode. Disables all multimodal "
            "inputs by setting every modality limit to 0, so a multimodal checkpoint serves as a text-only model."
        },
    )
    limit_mm_per_prompt: str | None = field(
        default=None,
        metadata={
            "help": "Limits on multimodal items per prompt, as comma-separated key=value pairs "
            "(e.g., 'image=0,video=0'). Passed through to vLLM's EngineArgs."
        },
    )
    log_level: str = field(
        default="info",
        metadata={
            "help": "Log level for uvicorn. Possible choices: 'critical', 'error', 'warning', 'info', 'debug', "
            "'trace'."
        },
    )
    vllm_model_impl: str = field(
        default="vllm",
        metadata={
            "help": "Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: "
            "Use the `transformers` backend for model implementation. `vllm`: Use the `vllm` library for "
            "model implementation."
        },
    )
    distributed_executor_backend: str | None = field(
        default=None,
        metadata={
            "help": "Distributed executor backend for vLLM. When set to 'ray', vLLM uses Ray to distribute tensor "
            "parallel workers across multiple nodes. Required when tensor_parallel_size exceeds the number of local "
            "GPUs. If not set, vLLM defaults to the multiproc backend (single-node only)."
        },
    )
    speculative_config: str | None = field(
        default=None,
        metadata={
            "help": "JSON string for vLLM speculative decoding config. "
            'Example: \'{"method": "qwen3_next_mtp", "num_speculative_tokens": 5}\''
        },
    )


def llm_worker(
    script_args: ScriptArguments, data_parallel_rank: int, master_port: int, connection: Connection
) -> None:
    from vllm import LLM

    # Set required environment variables for DP to work with vLLM
    os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank)
    os.environ["VLLM_DP_SIZE"] = str(script_args.data_parallel_size)
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    # Optional engine args: only forward when explicitly set, so vLLM's config validation never
    # sees `None` where it expects a bool/dict and engine defaults are preserved otherwise.
    optional_engine_kwargs = {}
    if script_args.cudagraph_mode is not None:
        optional_engine_kwargs["compilation_config"] = {"cudagraph_mode": script_args.cudagraph_mode}
    if script_args.enable_expert_parallel is not None:
        optional_engine_kwargs["enable_expert_parallel"] = script_args.enable_expert_parallel
    if script_args.moe_backend is not None:
        optional_engine_kwargs["moe_backend"] = script_args.moe_backend
    if script_args.language_model_only:
        optional_engine_kwargs["language_model_only"] = True
    if script_args.limit_mm_per_prompt:
        limit_mm_per_prompt = {}
        for item in script_args.limit_mm_per_prompt.split(","):
            key, value = item.split("=")
            limit_mm_per_prompt[key.strip()] = int(value.strip())
        optional_engine_kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt

    llm = LLM(
        model=script_args.model,
        revision=script_args.revision,
        tensor_parallel_size=script_args.tensor_parallel_size,
        gpu_memory_utilization=script_args.gpu_memory_utilization,
        enforce_eager=script_args.enforce_eager,
        dtype=script_args.dtype,
        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
        # This is particularly useful here because we generate completions from the same prompts.
        enable_prefix_caching=script_args.enable_prefix_caching,
        kv_cache_dtype=script_args.kv_cache_dtype,
        max_model_len=script_args.max_model_len,
        worker_extension_cls="trl.scripts.vllm_serve.WeightSyncWorkerExtension",
        trust_remote_code=script_args.trust_remote_code,
        model_impl=script_args.vllm_model_impl,
        distributed_executor_backend=script_args.distributed_executor_backend,
        # Important so temperature scaling/logit tweaking affects the TIS log probs
        logprobs_mode="processed_logprobs",
        speculative_config=json.loads(script_args.speculative_config) if script_args.speculative_config else None,
        **optional_engine_kwargs,
    )

    # Send ready signal to parent process
    connection.send({"status": "ready"})

    while True:
        # Wait for commands from the parent process
        try:
            command = connection.recv()
        except KeyboardInterrupt:
            llm.collective_rpc(method="close_communicator")
            break

        # Handle commands
        if command["type"] in ["call", "fire_and_forget"]:
            method_name = command["method"]
            args, kwargs = command.get("args", ()), command.get("kwargs", {})
            method = getattr(llm, method_name)
            result = method(*args, **kwargs)
            if command["type"] == "call":
                connection.send(result)
        elif command["type"] == "shutdown":
            break


def chunk_list(lst: list, n: int) -> list[list]:
    """
    Split list `lst` into `n` evenly distributed sublists.

    Example:
    ```python
    >>> chunk_list([1, 2, 3, 4, 5, 6], 2)
    [[1, 2, 3], [4, 5, 6]]

    >>> chunk_list([1, 2, 3, 4, 5, 6], 4)
    [[1, 2], [3, 4], [5], [6]]

    >>> chunk_list([1, 2, 3, 4, 5, 6], 8)
    [[1], [2], [3], [4], [5], [6], [], []]
    ```
    """
    k, r = divmod(len(lst), n)
    return [lst[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)] for i in range(n)]


def main(script_args: ScriptArguments):
    import asyncio

    from transformers import is_vision_available

    from trl.generation.vllm_generation import extract_logprobs
    from trl.import_utils import (
        is_fastapi_available,
        is_pydantic_available,
        is_uvicorn_available,
        is_vllm_available,
    )

    if not is_fastapi_available():
        raise ImportError(
            "FastAPI is required to run the vLLM serve script. Please install it using `pip install fastapi`."
        )

    if not is_pydantic_available():
        raise ImportError(
            "Pydantic is required to run the vLLM serve script. Please install it using `pip install pydantic`."
        )

    if not is_uvicorn_available():
        raise ImportError(
            "Uvicorn is required to run the vLLM serve script. Please install it using `pip install uvicorn`."
        )

    if not is_vllm_available():
        raise ImportError("vLLM is required to run the vLLM serve script. Please install it using `pip install vllm`.")

    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.utils.network_utils import get_open_port

    if is_vision_available():
        from PIL import Image

    logger = logging.getLogger(__name__)

    # Spawn dp workers, and setup pipes for communication
    master_port = get_open_port()
    connections = []
    processes = []
    for data_parallel_rank in range(script_args.data_parallel_size):
        parent_connection, child_connection = Pipe()
        process = Process(target=llm_worker, args=(script_args, data_parallel_rank, master_port, child_connection))
        process.start()
        connections.append(parent_connection)
        processes.append(process)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Wait for all workers to send "ready"
        ready_connections = set()
        while len(ready_connections) < script_args.data_parallel_size:
            for connection in connections:
                msg = connection.recv()
                if isinstance(msg, dict) and msg.get("status") == "ready":
                    ready_connections.add(connection)

        # Start the logprob request batcher background task
        batcher_task = asyncio.create_task(_logprob_batcher())

        yield

        batcher_task.cancel()

        # Wait for processes to terminate
        for process in processes:
            process.join(timeout=10)  # Wait for 10 seconds for the process to terminate
            if process.is_alive():
                logger.warning(f"Process {process} is still alive after 10 seconds, attempting to terminate...")
                process.terminate()
                process.join()  # ensure process termination after calling terminate()

    app = FastAPI(lifespan=lifespan)

    # Define the endpoints for the model server
    @app.get("/health/")
    async def health():
        """
        Health check endpoint to verify that the server is running.
        """
        return {"status": "ok"}

    @app.get("/get_world_size/")
    async def get_world_size():
        """
        Retrieves the world size of the LLM engine, which is `tensor_parallel_size * data_parallel_size`.

        Returns:
            `dict`:
                A dictionary containing the world size.

        Example response:
        ```json
        {"world_size": 8}
        ```
        """
        return {"world_size": script_args.tensor_parallel_size * script_args.data_parallel_size}

    class GenerateRequest(BaseModel):
        prompts: list[str] | list[list[int]]
        images: list[list[str] | None] | None = None
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        logprobs: int | None = 0
        structured_outputs_regex: str | None = None
        generation_kwargs: dict = field(default_factory=dict)

    class GenerateResponse(BaseModel):
        prompt_ids: list[list[int]]
        completion_ids: list[list[int]]
        logprobs: list[list[list[float | None]]] | None
        logprob_token_ids: list[list[list[int]]] | None

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """
        Generates completions for the provided prompts.

        Args:
            request (`GenerateRequest`):
                - `prompts` (list of `str` or list of list of `int`): A list of prompts. It accepts either text strings
                  or pre-tokenized token ID lists. When text strings are provided, `images` can optionally be included.
                - `images` (list of list of `str` or `None`, *optional*): A list of image lists. Each element is a list
                  of base64-encoded images for the corresponding prompt, or `None` if no images for that prompt.
                - `n` (`int`, *optional*, defaults to `1`): Number of completions to generate for each prompt.
                - `repetition_penalty` (`float`, *optional*, defaults to `1.0`): Repetition penalty to apply during
                  generation.
                - `temperature` (`float`, *optional*, defaults to `1.0`): Temperature for sampling. Higher values lead
                  to more random outputs.
                - `top_p` (`float`, *optional*, defaults to `1.0`): Top-p (nucleus) sampling parameter. It controls the
                  diversity of the generated text.
                - `top_k` (`int`, *optional*, defaults to `-1`): Top-k sampling parameter. If set to `-1`, it disables
                  top-k sampling.
                - `min_p` (`float`, *optional*, defaults to `0.0`): Minimum probability threshold for sampling.
                - `max_tokens` (`int`, *optional*, defaults to `16`): Maximum number of tokens to generate for each
                  completion.
                - `logprobs` (`int`, *optional*, defaults to `0`): Number of top logprobs to return per token. When 0,
                  only the sampled token's logprob is returned. When N>0, returns up to N+1 logprobs sorted by
                  descending probability, because vLLM always includes the sampled token's logprob (which may fall
                  outside the top-N).
                - `structured_outputs_regex` (`str`, *optional*): A regex pattern for structured outputs. If provided,
                  the model will only generate tokens that match this regex pattern.
                - `generation_kwargs` (`dict`, *optional*): Additional generation parameters to pass to the vLLM
                  `SamplingParams`. This can include parameters like `seed`, `frequency_penalty`, etc. If it contains
                  keys that conflict with the other parameters, they will override them.

        Returns:
            `GenerateResponse`:
                - `prompt_ids` (list of list of `int`): A list of lists of token IDs for each input prompt.
                - `completion_ids` (list of list of `int`): A list of lists of token IDs for each generated completion.
                - `logprobs` (list of list of list of `float`): Per-token logprobs of shape (num_sequences, seq_len,
                  num_logprobs), sorted by descending probability.
                - `logprob_token_ids` (list of list of list of `int`): Token IDs corresponding to each logprob, same
                  shape as `logprobs`.

        Example request (text prompts):
        ```json
        {"prompts": ["Hello world", "What is AI?"]}
        ```

        Example request (token IDs):
        ```json
        {"prompts": [[101, 102], [201, 202]]}
        ```

        Example response:
        ```json
        {
          "prompt_ids": [[101, 102], [201, 202]],
          "completion_ids": [[103, 104, 105], [203, 204, 205]],
          "logprobs": [[[-0.1], [-0.2], [-0.3]], [[-0.4], [-0.5], [-0.6]]],
          "logprob_token_ids": [[[103], [104], [105]], [[203], [204], [205]]]
        }
        ```
        """
        # Build vLLM-compatible prompt inputs
        is_token_ids = request.prompts and isinstance(request.prompts[0], list)
        request.images = request.images or [None] * len(request.prompts)

        prompts = []
        for prompt, image_list in zip(request.prompts, request.images, strict=True):
            row = {"prompt_token_ids": prompt} if is_token_ids else {"prompt": prompt}
            if image_list is not None:
                row["multi_modal_data"] = {"image": [Image.open(BytesIO(base64.b64decode(img))) for img in image_list]}
            prompts.append(row)

        generation_kwargs = {
            "n": request.n,
            "repetition_penalty": request.repetition_penalty,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "max_tokens": request.max_tokens,
            "logprobs": request.logprobs,
        }
        generation_kwargs.update(request.generation_kwargs)

        # Structured outputs, if enabled
        if request.structured_outputs_regex is not None:
            if generation_kwargs.get("structured_outputs") is not None:
                logger.warning(
                    "Both `structured_outputs_regex` and `generation_kwargs['structured_outputs']` are set; "
                    "`structured_outputs_regex` takes precedence."
                )
            generation_kwargs["structured_outputs"] = StructuredOutputsParams(regex=request.structured_outputs_regex)
        elif isinstance(structured_outputs_kwargs := generation_kwargs.get("structured_outputs"), dict):
            generation_kwargs["structured_outputs"] = StructuredOutputsParams(**structured_outputs_kwargs)
        sampling_params = SamplingParams(**generation_kwargs)

        # Evenly distribute prompts across DP ranks
        chunked_prompts = chunk_list(prompts, script_args.data_parallel_size)

        # Send the prompts to each worker
        for connection, prompts in zip(connections, chunked_prompts, strict=True):
            # When the number of prompts is less than data_parallel_size, some workers will receive empty prompts.
            # However, vLLM requires that we always send at least one prompt. So we send a placeholder prompt to comply
            # with vLLM's requirement, and we later ignore the result.
            if not prompts:
                prompts = ["<placeholder>"]
            kwargs = {"prompts": prompts, "sampling_params": sampling_params}
            connection.send({"type": "call", "method": "generate", "kwargs": kwargs})

        # Receive results
        all_outputs = [connection.recv() for connection in connections]

        # Handle empty prompts (see above)
        all_outputs = [output for output, prompts in zip(all_outputs, chunked_prompts, strict=True) if prompts]

        # Flatten and combine all results
        all_outputs = list(chain.from_iterable(all_outputs))  # from list of list to single list
        prompt_ids = [output.prompt_token_ids for output in all_outputs]
        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
        logprobs, logprob_token_ids = extract_logprobs(all_outputs)

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "logprob_token_ids": logprob_token_ids,
        }

    class SequenceLogprobsRequest(BaseModel):
        sequences: list[list[int]]
        prompt_lengths: list[int]
        top_logprobs: int = 100
        temperature: float = 1.0
        response_format: str = "json"  # "json" (legacy) or "binary" (base64 numpy arrays)

    class SequenceLogprobsResponse(BaseModel):
        logprobs: list[list[list[float | None]]] | None = None
        logprob_token_ids: list[list[list[int]]] | None = None
        # Binary format fields (base64-encoded numpy arrays)
        logprobs_b64: str | None = None
        token_ids_b64: str | None = None
        actual_logprobs_b64: str | None = None
        actual_token_ids_b64: str | None = None
        shape: list[int] | None = None  # [batch_size, max_completion_len, top_logprobs]
        completion_lengths: list[int] | None = None  # actual completion length per sample

    def _run_prompt_logprobs(prompts, sampling_params):
        """Send prompts to DP workers and collect outputs."""
        chunked_prompts = chunk_list(prompts, script_args.data_parallel_size)
        for connection, chunk in zip(connections, chunked_prompts, strict=True):
            if not chunk:
                chunk = [{"prompt_token_ids": [0]}]
            kwargs = {"prompts": chunk, "sampling_params": sampling_params}
            connection.send({"type": "call", "method": "generate", "kwargs": kwargs})
        all_outputs = [connection.recv() for connection in connections]
        all_outputs = [output for output, chunk in zip(all_outputs, chunked_prompts, strict=True) if chunk]
        return list(chain.from_iterable(all_outputs))

    # ── Request batching for get_sequence_logprobs ──
    # Collects concurrent requests into batches and dispatches them together so that
    # all DP workers stay busy. Without this, async endpoint handlers block the event
    # loop during pipe I/O, serializing requests and leaving DP workers idle.
    _logprob_queue: asyncio.Queue = asyncio.Queue()

    # Maximum time (seconds) to wait for more requests before dispatching a batch.
    _BATCH_WAIT_S = 0.005  # 5ms - short enough to not add much latency when lightly loaded
    # Maximum number of HTTP requests to collect per batcher cycle
    _MAX_BATCH_REQUESTS = max(script_args.data_parallel_size * 4, 16)
    # Maximum total tokens per batch. prompt_logprobs materializes full-vocab logits
    # during the forward pass, so each worker can safely handle ~1 max-length sequence.
    # Budget = max_model_len * dp_size gives ~1 sequence per worker at max length.
    _max_model_len = script_args.max_model_len or 8192
    _MAX_BATCH_TOKENS = _max_model_len * script_args.data_parallel_size

    async def _logprob_batcher():
        """Background task that continuously drains the queue, batches requests, and dispatches."""
        loop = asyncio.get_running_loop()

        while True:
            batch = []
            try:
                # Wait for the first request
                batch_tokens = 0
                item = await _logprob_queue.get()
                batch.append(item)
                # Count tokens in this item's sequences
                for prompt in item[0]:
                    batch_tokens += len(prompt.get("prompt_token_ids", []))

                # Collect more requests up to batch limit, timeout, or token budget
                deadline = loop.time() + _BATCH_WAIT_S
                while len(batch) < _MAX_BATCH_REQUESTS and batch_tokens < _MAX_BATCH_TOKENS:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(_logprob_queue.get(), timeout=remaining)
                        # Check if adding this item would exceed the token budget
                        item_tokens = sum(len(p.get("prompt_token_ids", [])) for p in item[0])
                        if batch_tokens + item_tokens > _MAX_BATCH_TOKENS and len(batch) > 0:
                            # Put it back and dispatch current batch
                            await _logprob_queue.put(item)
                            break
                        batch.append(item)
                        batch_tokens += item_tokens
                    except asyncio.TimeoutError:
                        break

                # batch is a list of (prompts, prompt_lengths, top_logprobs, temperature, response_format, future)
                # All items in a batch must share the same (top_logprobs, temperature) pair.
                # Group by those execution parameters to handle mixed requests.
                groups = {}
                for prompts, prompt_lengths, top_logprobs, temperature, response_format, future in batch:
                    key = (top_logprobs, temperature)
                    if key not in groups:
                        groups[key] = []
                    groups[key].append((prompts, prompt_lengths, response_format, future))

                for (top_logprobs, temperature), items in groups.items():
                    # Merge all sequences into a single batch
                    all_prompts = []
                    all_prompt_lengths = []
                    offsets = []  # (start_idx, count) per original request
                    for prompts, prompt_lengths, _response_format, _future in items:
                        start = len(all_prompts)
                        all_prompts.extend(prompts)
                        all_prompt_lengths.extend(prompt_lengths)
                        offsets.append((start, len(prompts)))

                    sampling_params = SamplingParams(
                        max_tokens=1,
                        temperature=temperature,
                        prompt_logprobs=top_logprobs,
                    )

                    # Dispatch to workers in a thread to avoid blocking the event loop
                    try:
                        all_outputs = await loop.run_in_executor(
                            None, _run_prompt_logprobs, all_prompts, sampling_params
                        )

                        # Split results back to individual requests
                        for (start, count), (_, prompt_lengths, response_format, future) in zip(
                            offsets, items, strict=True
                        ):
                            outputs_slice = all_outputs[start : start + count]
                            if not future.done():
                                future.set_result((outputs_slice, prompt_lengths, top_logprobs, response_format))
                    except Exception as e:
                        # Signal error to all waiting requests in this execution-parameter group
                        for *_, future in items:
                            if not future.done():
                                future.set_exception(e)
            except Exception as e:
                # Prevent killing the batcher task — signal error to all unfulfilled futures
                for *_, future in batch:
                    if not future.done():
                        future.set_exception(e)

    def _format_logprob_response(all_outputs, prompt_lengths, top_k, response_format):
        """Format vLLM outputs into the response dict (runs in any thread)."""
        import numpy as np

        batch_size = len(all_outputs)
        use_binary = response_format == "binary"

        if use_binary:
            from starlette.responses import Response

            comp_lengths = []
            for output, prompt_length in zip(all_outputs, prompt_lengths, strict=True):
                prompt_lps = output.prompt_logprobs
                if prompt_lps is None:
                    raise ValueError("prompt_logprobs is None.")
                comp_lengths.append(len(prompt_lps) - prompt_length)

            max_comp_len = max(comp_lengths) if comp_lengths else 0

            # logprobs_arr / token_ids_arr: teacher's sorted top-k logprobs + token ids (for forward KL).
            # actual_logprobs_arr / actual_token_ids_arr: actual token's teacher logprob (for reverse KL).
            logprobs_arr = np.full((batch_size, max_comp_len, top_k), float("-inf"), dtype=np.float32)
            token_ids_arr = np.zeros((batch_size, max_comp_len, top_k), dtype=np.int32)
            actual_logprobs_arr = np.full((batch_size, max_comp_len, 1), float("-inf"), dtype=np.float32)
            actual_token_ids_arr = np.zeros((batch_size, max_comp_len, 1), dtype=np.int32)

            for i, (output, prompt_length) in enumerate(zip(all_outputs, prompt_lengths, strict=True)):
                prompt_lps = output.prompt_logprobs
                seq_tokens = output.prompt_token_ids
                if comp_lengths[i] == 0:
                    continue

                for pos in range(prompt_length, len(prompt_lps)):
                    lp = prompt_lps[pos]
                    if lp is None:
                        continue
                    t = pos - prompt_length
                    actual_token = seq_tokens[pos]

                    # Actual token's logprob (for reverse KL)
                    if actual_token in lp:
                        val = lp[actual_token].logprob
                        if not math.isnan(val):
                            actual_logprobs_arr[i, t, 0] = val
                        actual_token_ids_arr[i, t, 0] = actual_token

                    # Teacher's top-k logprobs (for forward KL)
                    if top_k == 1:
                        # Fast path: find rank-1 directly instead of sorting
                        for token_id, logprob_obj in lp.items():
                            if logprob_obj.rank == 1:
                                val = logprob_obj.logprob
                                if not math.isnan(val):
                                    logprobs_arr[i, t, 0] = val
                                token_ids_arr[i, t, 0] = token_id
                                break
                    else:
                        sorted_items = sorted(lp.items(), key=lambda x: x[1].rank)
                        for k_idx, (token_id, logprob_obj) in enumerate(sorted_items[:top_k]):
                            val = logprob_obj.logprob
                            if not math.isnan(val):
                                logprobs_arr[i, t, k_idx] = val
                            token_ids_arr[i, t, k_idx] = token_id

            payload = {
                "logprobs_b64": base64.b64encode(logprobs_arr.tobytes()).decode("ascii"),
                "token_ids_b64": base64.b64encode(token_ids_arr.tobytes()).decode("ascii"),
                "actual_logprobs_b64": base64.b64encode(actual_logprobs_arr.tobytes()).decode("ascii"),
                "actual_token_ids_b64": base64.b64encode(actual_token_ids_arr.tobytes()).decode("ascii"),
                "shape": [batch_size, max_comp_len, top_k],
                "completion_lengths": comp_lengths,
            }

            try:
                import orjson

                return Response(content=orjson.dumps(payload), media_type="application/json")
            except ImportError:
                return payload
        else:
            all_logprobs = []
            all_token_ids = []
            for output, prompt_length in zip(all_outputs, prompt_lengths, strict=True):
                prompt_lps = output.prompt_logprobs
                if prompt_lps is None:
                    raise ValueError("prompt_logprobs is None.")
                seq_logprobs = []
                seq_token_ids = []
                for pos in range(prompt_length, len(prompt_lps)):
                    lp = prompt_lps[pos]
                    if lp is None:
                        seq_logprobs.append([])
                        seq_token_ids.append([])
                        continue
                    sorted_items = sorted(lp.items(), key=lambda x: x[1].rank)
                    seq_token_ids.append([token_id for token_id, _ in sorted_items])
                    seq_logprobs.append(
                        [None if math.isnan(item.logprob) else item.logprob for _, item in sorted_items]
                    )
                all_logprobs.append(seq_logprobs)
                all_token_ids.append(seq_token_ids)
            return {"logprobs": all_logprobs, "logprob_token_ids": all_token_ids}

    @app.post("/get_sequence_logprobs/", response_model=SequenceLogprobsResponse)
    async def get_sequence_logprobs(request: SequenceLogprobsRequest):
        """
        Computes teacher logprobs for existing token sequences without generating new tokens.

        Concurrent requests are automatically batched and dispatched together to maximize GPU utilization across DP
        workers. This avoids the event-loop-blocking problem where synchronous pipe I/O serializes requests despite
        having multiple DP workers.

        Args:
            request (`SequenceLogprobsRequest`):
                - `sequences` (list of list of `int`): Full token sequences (prompt + completion) per sample.
                - `prompt_lengths` (list of `int`): Number of prompt tokens per sequence; completion logprobs start
                  after each prompt.
                - `top_logprobs` (`int`, *optional*, defaults to `100`): Number of top teacher logprobs to return per
                  completion position (sorted by vLLM rank).
                - `temperature` (`float`, *optional*, defaults to `1.0`): Sampling temperature passed to vLLM for
                  logprob computation.
                - `response_format` (`str`, *optional*, defaults to `"json"`): Either `"json"` (nested lists,
                  backward-compatible) or `"binary"` (base64-encoded numpy arrays for fast serialization).

        Returns:
            `SequenceLogprobsResponse` or Starlette `Response`:
                When `response_format` is `"json"`, a JSON object with:
                - `logprobs` (list of list of list of `float` or `None`): Top-k teacher logprobs per completion token.
                - `logprob_token_ids` (list of list of list of `int`): Token IDs aligned with `logprobs`.
                When `response_format` is `"binary"`, a JSON response (Starlette `Response` if `orjson` is installed)
                whose body is a JSON object with base64-encoded float32/int32 arrays: `logprobs_b64`, `token_ids_b64`,
                `actual_logprobs_b64`, `actual_token_ids_b64`, plus `shape` (`list[int]`, `[batch_size,
                max_completion_len, top_k]`) and `completion_lengths` (`list[int]`).
        """
        if len(request.sequences) != len(request.prompt_lengths):
            raise ValueError("sequences and prompt_lengths must have the same length.")

        for i, (seq, pl) in enumerate(zip(request.sequences, request.prompt_lengths, strict=True)):
            if pl < 0 or pl > len(seq):
                raise ValueError(
                    f"Sequence {i} has prompt_length={pl} which is out of range [0, {len(seq)}]. "
                    f"prompt_length must be between 0 and the sequence length inclusive."
                )

        # Validate sequence lengths against max_model_len to prevent worker OOM crashes
        if _max_model_len:
            for i, seq in enumerate(request.sequences):
                if len(seq) > _max_model_len:
                    raise ValueError(
                        f"Sequence {i} has length {len(seq)} which exceeds max_model_len={_max_model_len}. "
                        f"Truncate sequences or increase --max-model-len."
                    )

        prompts = [{"prompt_token_ids": seq} for seq in request.sequences]

        # Submit to the batching queue and await result
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await _logprob_queue.put(
            (
                prompts,
                list(request.prompt_lengths),
                request.top_logprobs,
                request.temperature,
                request.response_format,
                future,
            )
        )

        # Wait for the batcher to process our request
        all_outputs, prompt_lengths, top_k, response_format = await future

        return await loop.run_in_executor(
            None, _format_logprob_response, all_outputs, prompt_lengths, top_k, response_format
        )

    class ChatRequest(BaseModel):
        messages: list[list[dict]]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        logprobs: int | None = 0
        structured_outputs_regex: str | None = None
        generation_kwargs: dict = field(default_factory=dict)
        chat_template_kwargs: dict = field(default_factory=dict)
        tools: list | None = None

    class ChatResponse(BaseModel):
        prompt_ids: list[list[int]]
        completion_ids: list[list[int]]
        logprobs: list[list[list[float | None]]] | None
        logprob_token_ids: list[list[list[int]]] | None

    @app.post("/chat/", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        Generates completions for the provided chat messages.

        Args:
            request (`ChatRequest`):
                - `messages` (list of `dict`): A list of messages (dicts with "role" and "content" keys) for the model
                  to generate completions.
                - `n` (`int`, *optional*, defaults to `1`): Number of completions to generate for each prompt.
                - `repetition_penalty` (`float`, *optional*, defaults to `1.0`): Repetition penalty to apply during
                  generation.
                - `temperature` (`float`, *optional*, defaults to `1.0`): Temperature for sampling. Higher values lead
                  to more random outputs.
                - `top_p` (`float`, *optional*, defaults to `1.0`): Top-p (nucleus) sampling parameter. It controls the
                  diversity of the generated text.
                - `top_k` (`int`, *optional*, defaults to `-1`): Top-k sampling parameter. If set to `-1`, it disables
                  top-k sampling.
                - `min_p` (`float`, *optional*, defaults to `0.0`): Minimum probability threshold for sampling.
                - `max_tokens` (`int`, *optional*, defaults to `16`): Maximum number of tokens to generate for each
                  completion.
                - `logprobs` (`int`, *optional*, defaults to `0`): Number of top logprobs to return per token. When 0,
                  only the sampled token's logprob is returned. When N>0, returns up to N+1 logprobs sorted by
                  descending probability, because vLLM always includes the sampled token's logprob (which may fall
                  outside the top-N).
                - `structured_outputs_regex` (`str`, *optional*): A regex pattern for structured outputs. If provided,
                  the model will only generate tokens that match this regex pattern.
                - `generation_kwargs` (`dict`, *optional*): Additional generation parameters to pass to the vLLM
                  `SamplingParams`. This can include parameters like `seed`, `frequency_penalty`, etc. If it contains
                  keys that conflict with the other parameters, they will override them.
                - `chat_template_kwargs` (`dict`, *optional*): Additional keyword arguments to pass to the chat
                  template.

        Returns:
            `ChatResponse`:
                - `prompt_ids` (list of list of `int`): A list of lists of token IDs for each input prompt.
                - `completion_ids` (list of list of `int`): A list of lists of token IDs for each generated completion.
                - `logprobs` (list of list of list of `float`): Per-token logprobs of shape (num_sequences, seq_len,
                  num_logprobs), sorted by descending probability.
                - `logprob_token_ids` (list of list of list of `int`): Token IDs corresponding to each logprob, same
                  shape as `logprobs`.

        Example request:
        ```bash
        curl -X POST 'http://0.0.0.0:8000/chat/' \
          -H 'Content-Type: application/json' \
          -d '{"messages": [[{ "role": "user", "content": "Hello!" }]]}'
        ```

        Example response:
        ```json
        {
            "prompt_ids": [[151644, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198]],
            "completion_ids": [[151667, 198, 32313, 11, 279]],
            "logprobs": [[[-0.0003], [-3.58e-07], [-0.0902], [-6.39e-05], [-0.0387]]],
            "logprob_token_ids": [[[151667], [198], [32313], [11], [279]]]
        }
        ```
        """
        # Convert PIL images to base64 strings
        for message_list in request.messages:
            for message in message_list:
                if isinstance(message["content"], list):
                    for part in message["content"]:
                        if part["type"] == "image_pil":
                            part["image_pil"] = Image.open(BytesIO(base64.b64decode(part["image_pil"])))

        generation_kwargs = {
            "n": request.n,
            "repetition_penalty": request.repetition_penalty,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "max_tokens": request.max_tokens,
            "logprobs": request.logprobs,
        }
        generation_kwargs.update(request.generation_kwargs)

        # Structured outputs, if enabled
        if request.structured_outputs_regex is not None:
            if generation_kwargs.get("structured_outputs") is not None:
                logger.warning(
                    "Both `structured_outputs_regex` and `generation_kwargs['structured_outputs']` are set; "
                    "`structured_outputs_regex` takes precedence."
                )
            generation_kwargs["structured_outputs"] = StructuredOutputsParams(regex=request.structured_outputs_regex)
        elif isinstance(structured_outputs_kwargs := generation_kwargs.get("structured_outputs"), dict):
            generation_kwargs["structured_outputs"] = StructuredOutputsParams(**structured_outputs_kwargs)
        sampling_params = SamplingParams(**generation_kwargs)

        # Evenly distribute prompts across DP ranks
        chunked_messages = chunk_list(request.messages, script_args.data_parallel_size)

        # Send the messages to each worker
        for connection, messages in zip(connections, chunked_messages, strict=True):
            # When the number of messages is less than data_parallel_size, some workers will receive empty messages.
            # However, vLLM requires that we always send at least one prompt. So we send a placeholder prompt to comply
            # with vLLM's requirement, and we later ignore the result.
            if not messages:
                messages = [[{"role": "user", "content": "<placeholder>"}]]
            kwargs = {
                "messages": messages,
                "sampling_params": sampling_params,
                "chat_template_kwargs": request.chat_template_kwargs,
                "tools": request.tools,
            }
            connection.send({"type": "call", "method": "chat", "kwargs": kwargs})

        # Receive results
        all_outputs = [connection.recv() for connection in connections]

        # Handle empty prompts (see above)
        all_outputs = [output for output, prompts in zip(all_outputs, chunked_messages, strict=True) if prompts]

        # Flatten and combine all results
        all_outputs = list(chain.from_iterable(all_outputs))  # from list of list to single list
        prompt_ids = [output.prompt_token_ids for output in all_outputs]
        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
        logprobs, logprob_token_ids = extract_logprobs(all_outputs)

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "logprob_token_ids": logprob_token_ids,
        }

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int
        client_device_uuid: str

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest):
        """
        Initializes the communicator for synchronizing model weights between a client and multiple server workers.

        Args:
            request (`InitCommunicatorRequest`):
                - `host` (`str`): Hostname or IP address of the master node.
                - `port` (`int`): Port number to be used for communication.
                - `world_size` (`int`): Total number of participating processes in the group.
                - `client_device_uuid` (`str`): UUID of the device of client main process. Used to assert that devices
                  are different from vLLM workers devices.
        """
        world_size = script_args.tensor_parallel_size * script_args.data_parallel_size + 1

        # The function init_communicator is called this way: init_communicator(host, port, world_size)
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc(method="init_communicator", args=(host, port, world_size))
        kwargs = {
            "method": "init_communicator",
            "args": (request.host, request.port, world_size, request.client_device_uuid),
        }
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})

        return {"message": "Request received, initializing communicator"}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest):
        """
        Updates the model weights with the provided tensor.

        Once this endpoint is called, the client process should broadcast the updated weights to all server workers.

        Args:
            request (`UpdateWeightsRequest`):
                - `name` (`str`): Name of the weight tensor being updated.
                - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                - `shape` (list of `int`): Shape of the weight

        """
        # The function update_named_param is called this way: update_named_param("name", "torch.float32", (10, 10))
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc("update_named_param", args=("name", "torch.float32", (10, 10)))
        kwargs = {"method": "update_named_param", "args": (request.name, request.dtype, tuple(request.shape))}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})

        return {"message": "Request received, updating named parameter"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        """
        Resets the prefix cache for the model.
        """
        for connection in connections:
            connection.send({"type": "call", "method": "reset_prefix_cache"})
        # Wait for and collect all results
        all_outputs = [connection.recv() for connection in connections]
        success = all(output for output in all_outputs)
        return {"message": "Request received, resetting prefix cache status: " + str(success)}

    @app.get("/get_server_quantization/")
    async def get_server_quantization():
        """
        Returns the served model's quantization method (e.g. "fp8"), or null for an
        unquantized model. The sync client uses this to decide between the plain bf16
        per-tensor sync and the checkpoint-format layerwise round (weight + scale pairs
        inside begin/end_weight_update).
        """
        kwargs = {"method": "get_server_quantization"}
        connections[0].send({"type": "call", "method": "collective_rpc", "kwargs": kwargs})
        output = connections[0].recv()
        quantization = output[0] if isinstance(output, list) and output else None
        return {"quantization": quantization}

    @app.post("/begin_weight_update/")
    async def begin_weight_update():
        """
        Arms vLLM's layerwise reload on every worker for a full checkpoint-format weight
        update (quantized models). The model is unrunnable until /end_weight_update/.
        """
        kwargs = {"method": "begin_weight_update"}
        for connection in connections:
            connection.send({"type": "call", "method": "collective_rpc", "kwargs": kwargs})
        for connection in connections:
            connection.recv()
        return {"message": "Layerwise weight update armed"}

    @app.post("/end_weight_update/")
    async def end_weight_update():
        """
        Finalizes a layerwise weight update on every worker: re-runs per-layer quantization
        processing and restores kernel formats.
        """
        kwargs = {"method": "end_weight_update"}
        for connection in connections:
            connection.send({"type": "call", "method": "collective_rpc", "kwargs": kwargs})
        for connection in connections:
            connection.recv()
        return {"message": "Layerwise weight update finalized"}

    @app.post("/close_communicator/")
    async def close_communicator():
        """
        Closes the weight update group and cleans up associated resources.
        """
        kwargs = {"method": "close_communicator"}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
        return {"message": "Request received, closing communicator"}

    # Start the server
    uvicorn.run(app, host=script_args.host, port=script_args.port, log_level=script_args.log_level)


def make_parser(subparsers: argparse._SubParsersAction | None = None, prog: str | None = None):
    from trl import TrlParser

    if subparsers is not None:
        parser = subparsers.add_parser("vllm-serve", help="Run the vLLM serve script", dataclass_types=ScriptArguments)
    else:
        parser = TrlParser(ScriptArguments, prog=prog)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args)
