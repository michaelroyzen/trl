# Fork notes: DeepSeek-V3.2 GRPO weight sync (branch `deepseek-32-grpo-v1.8.0`)

Fork-specific engineering notes for this branch (offshoot of
`flashinfer-weight-sync-v1.8.0`, which is v1.8.0 + the FlashInfer TRT-LLM bf16 MoE
sync fix). Written 2026-07-16 after single-node bring-up on 8x NVIDIA B300 (sm103),
vllm==0.25.0, torch==2.11.0+cu130. The companion operational doc (measurements,
runbook, checkpoint tooling) lives in the monorepo: `docs/deepseek-32-grpo.md`.

## What this branch adds

Three commits, each independently validated:

1. `cbf3bc8d` - **Defer bitsandbytes import in `vllm_generation.py`.**
2. `e99deac1` - **Per-expert fallback for packed 3D expert sync** (models whose vLLM
   implementation has no fused packed-3D loader, e.g. DeepSeek).
3. `f506e163` - **fp8 requant-on-sync via vLLM layerwise reload** (bf16 trainer ->
   natively-fp8-served model), plus the `--cudagraph-mode` serve arg and the
   FULL-cudagraph guard.

## What we tried, what succeeded, what failed

### Serving DeepSeek-V3.2 at all (failed twice before succeeding)

- **FAILED: every forked DP worker segfaulted at startup** ("Cannot re-initialize CUDA
  in forked subprocess", then a segfault in `cuGetProcAddress` during worker imports).
  Bisection with a fork-after-import repro showed `trl.generation.vllm_generation`'s
  module-scope `import bitsandbytes` initializes CUDA on bitsandbytes 0.49.x, and
  `trl vllm-serve`'s `main()` imports that module in the parent BEFORE spawning its
  per-DP-rank `llm_worker` `multiprocessing.Process` (fork start method). This broke
  ALL serving on the environment, not just DeepSeek (a lockfile regeneration had bumped
  bitsandbytes). **FIX: import bnb lazily at its only use site** (colocate-mode
  quantization detection). If you see this segfault again, check for new module-scope
  imports that initialize CUDA.
- **Footgun rediscovered:** `pkill -f 'vllm[-]serve'` self-matches when the *invoking*
  command line contains the literal text `trl.scripts.vllm_serve` elsewhere (the
  bracket trick protects the pattern occurrence only). Kill by PID.

### Packed 3D expert sync for DeepSeek (bf16 path)

- **FAILED: the existing staged sync loaded nothing** for
  `model.layers.N.mlp.experts.gate_up_proj`: unlike `qwen3_5.py`
  (`load_fused_expert_weights`), vLLM's `deepseek_v2.py` only maps per-expert 2D
  checkpoint names - and its final `params_dict[name]` lookup **raises KeyError** on
  unknown names rather than skipping (now caught).
- **SUCCEEDED: per-expert slicing fallback** (`_load_packed_expert_tensor`):
  `load_weights` first (qwen behavior unchanged); if nothing loads, slice the packed
  wire tensor into checkpoint-convention 2D tensors driven through the layer's
  expert-aware `weight_loader` (shard ids w1/w3/w2), reusing vLLM's EP mapping and TP
  narrowing. Orientation is inferred from the wire shape - `[E, 2I, H]` gate-first
  (deepseek_v32 trainer layout) vs transposed `[E, H, 2I]` (qwen disk layout) - and
  **ambiguous shapes raise**. This matters: a test model with `2I == H` failed here by
  design; keep `moe_intermediate_size * 2 != hidden_size` in test fixtures.
- Validated live (tiny arch-faithful server, FlashInfer TRT-LLM bf16 MoE backend,
  default CUDA graphs): identity sync bitwise-preserves greedy ids + logprobs;
  zero/restore of `lm_head` and packed `gate_up_proj` shift and recover exactly.
  CPU tests: `tests/test_fused_expert_weight_sync.py` (19, incl. a bitwise round-trip
  on real checkpoint expert tensors when `DS32_SNAPSHOT` is available).

### fp8 requant-on-sync (bf16 trainer -> fp8 server)

- **Design that worked:** drive vLLM's own layerwise-reload lifecycle from the existing
  per-tensor sync protocol. `record_metadata_for_reloading` already runs at model
  construction in vllm 0.25, so the server only needs `initialize_layerwise_reload`
  (new `begin_weight_update`) -> per-tensor `load_weights` of checkpoint-format
  tensors (fp8 weight + `weight_scale_inv` pairs; packed experts sliced per expert,
  scales included) -> `finalize_layerwise_processing` (new `end_weight_update`), which
  re-runs per-layer quantization processing (`Fp8MoEMethod._setup_kernel`, FlashInfer
  kernel-format shuffles, MLA `W_UK/W_UV` re-derivation) with storage-stable copies.
  The client (`vllm_generation.py`) auto-detects an fp8 server
  (`get_server_quantization`) and re-quantizes manifest-listed tensors with vLLM's
  `per_block_cast_to_fp8(use_ue8m0=True)` - power-of-two scales make the identity
  round value-exact, and fp8 wire tensors halve sync bytes. NCCL handles
  float8_e4m3 natively (`ncclFloat8e4m3`), no dtype workaround needed.

- **FAILED (and root-caused): FULL CUDA graphs after a layerwise round.** The fp8
  identity gate failed on a default-graph server. Exhaustive in-process diffing showed
  the round is *perfect* at the tensor level - 0/78 params, 0/14 buffers, 0/6 derived
  MLA tensors changed value, 0 data_ptrs moved - and eager + PIECEWISE outputs are
  bitwise identical to pre-round. But **captured FULL-decode graphs produce
  deterministic wrong outputs** afterwards: the finalize step rebuilds the quantized
  MoE kernel and the captured graph retains stale internal kernel state. Mode bisect:
  `PIECEWISE` OK, `FULL_DECODE_ONLY` broken. **FIX: `begin_weight_update` refuses
  FULL-graph servers loudly; serve RL fp8 with the new `--cudagraph-mode PIECEWISE`
  (or `--enforce-eager True`).** The bf16 staging path is unaffected (it copies into
  original storage without kernel rebuilds) and keeps default graphs. Upstream issue
  worth filing.

- **FAILED (by design, do not do this): partial layerwise rounds.** During debugging we
  bisected by syncing tensor *families* individually - every partial round corrupted
  the model, sometimes unrecoverably until a full round. `initialize_layerwise_reload`
  restores ALL layers to meta; layers receiving only some of their tensors get
  "delayed processing" intended for padding slack, not missing tensors. **A layerwise
  round must send every trainer parameter** (the production `sync_weights` does).
  Single-tensor updates outside a round still work through the legacy paths.

- **Quirk (accepted): requantized (weight, scale) pairs are value-exact but not always
  byte-identical** to the original checkpoint - where a block's original scale was not
  tight, `per_block_cast_to_fp8` picks a smaller power-of-two scale (measured: 2/184
  tensors on the tiny model; dequantized values bit-identical, and power-of-two
  rescaling commutes with fp8 GEMM rounding, so logits are unchanged). Identity gates
  therefore assert greedy-id + logprob equality, not parameter-byte equality.

- Validated live (tiny fp8 server, PIECEWISE and eager): full checkpoint-format
  identity round preserves greedy + logprobs bitwise; scaled-lm_head mutate round
  shifts logprobs and an identity round restores baseline exactly; a 2-optimizer-step
  `GRPOTrainer` run against the server trains with per-step fp8 layerwise sync
  (begin/end rounds visible server-side). At 671B, per-tensor sync extrapolates to
  ~2 min/step (fp8) / ~3.5 min/step (bf16); packed session transfers
  (`trl/experimental/async_grpo/weight_transfer.py` client + these begin/end hooks)
  are the natural next step.

## Operational requirements

- fp8 sync needs `fp8_sync_manifest.json` next to the trainer's bf16 checkpoint
  (written by the monorepo's `dequantize_fp8_checkpoint.py`): the original
  checkpoint's fp8 weight keys, i.e. exactly what to re-quantize.
- fp8 rollout servers MUST run `--cudagraph-mode PIECEWISE` (enforced at
  `begin_weight_update`).
- MTP / speculative decoding stays off for RL sync (drafter weights are never synced).
- As before: any server-side sync exception wedges the weight-update group; restart
  the server before retrying.
