# Mechanism map — record #92 stack (0.665 min, upstream 4ea6b93)

> How the current record actually works, read from the code. Compiled 2026-11-09.
> Read `train_gpt.py` for the run outline; `track_1_short/` for model/optimizer/data/schedules; `track_1_short/perf/` = kernels/systems (skimmed, one line each).
> Headline config read from code: 11 layers (one removed, two attention-less), head_dim 128 (64 on narrow layers), 6 heads, model_dim 768, MLP 2816, |V|=50304, 1194 steps, ≤49,152 tokens/rank/step.

## File map

**Run skeleton:** `train_gpt.py` (setup → warmup + graph capture → timed loop → validation). **Package:** `config.py` (Hyperparameters + 5 TRAINING_STAGES), `schedule.py` (stage lookup, batch-tied LR with linear cooldown to floor, ANVIL rail beta), `training.py` (param table: anvil vs adam, sharded vs replicated, lr/wd multipliers; comms order), `data.py` (BOS-aligned varlen batches, async prefetch), `ngram_table.py` (84.6M-row hashed table: hashing, sharding, row cache, row-sparse Adam), `sampled_softmax.py` (per-step candidate set), `sharded_rows.py` (row ownership + pull/gradient routes), `canonical_mask.py` (impossible-token mask, built on a forked child during training, used only at validation), `tail_average.py` (4 tail accumulators shipped before final val), `distributed.py`, `run_log.py`.

**Model:** `model/gpt.py` (embeddings: token + n-gram + value; 11-block forward with MUDD machinery; FP8 caches; losses), `model/attention.py` (YaRN + partial rebuild, varlen FA3, XSA, attention gate, dynamic width composition), `model/layers.py` (RMS norm, transposed lm_head), `model/prefix_prediction.py` (prefix-token aux-loss table).

**Optimizer:** `optim/anvil.py` — ANVIL (twin-rail momentum + whitening cascade + lane equalizer + mantissa/cautious decay) and Adam, with explicitly scheduled reduce/gather phases.

**perf/ (one line each):** `bank_scalars.py` (all ANVIL scalars in one pinned buffer, one H2D/step), `deferred_gathers.py` (leaves all-gathers in flight, waits right before consumer), `pinned_batches.py` (pinned host staging), `replicated_adam.py` (flat replicated Adam: one all_reduce + one fused Triton launch), `residual_fusion.py` (scale/scale_add/rms_norm fusions), `row_prefetch.py`, `sampled_softmax_overlap.py` (host thread + side stream for candidate set), `value_embed_pull.py` (row-sparse VE gradient/update), `cuda_graphs/step_graphs.py` (fwd+bwd graph per StepGraphKey), `cuda_graphs/optimizer_graphs.py` (one graph per ANVIL bank), `cuda_graphs/fp8_refresh_graphs.py`, `cuda_graphs/capture_plan.py`, `cuda_graphs/baked_addresses.py` (address census), `cuda_graphs/capture_support.py`; kernels: `cross_entropy.py` (fp8 lm_head GEMM + fused softcapped CE, NVRTC), `sampled_cross_entropy.py`, `polar_express.py` (ANVIL cascade matmuls), `qkv_rope.py` (packed FP8 QKV + fused QK-RMSNorm + RoPE), `fp8_attention_quant.py`, `mlp.py` (fused relu², all-GEMM fp8), `ngram_embed.py`, `ngram_adam.py` (owner-side row merge + per-row Adam replaying missed β₂), `value_embed.py`, `row_scatter.py`, `replicated_adam.py`, `transpose.py`, `lerp.py`.

## Techniques → mechanism → code

Win types: **SE** sample-efficiency, **WC** wallclock, **MEM** memory.

1. **Muon → ANVIL lineage.** No vanilla Muon left; ANVIL is the successor. Momentum whitened by six quintic iterations on the Gram matrix (norm pre-divided by 1.05·‖X‖_F); symmetry-exploiting Triton kernels. `optim/anvil.py:36-144`, `perf/kernels/polar_express.py`. SE+WC.
2. **ANVIL** = four deviations: (a) twin-rail momentum (fast EMA β 0.85→0.93, slow β=0.98, blended 0.4385·fast + 0.5615·slow after `RAIL_ENGAGE_STEP=514`, then Nesterov); (b) re-derived orthogonalization (above); (c) per-lane energy equalizer (NorMuon-style low-rank variance estimate, EMA β₂=0.9, reciprocal-sqrt per lane, matrix rescaled to pre-equalization norm); (d) cautious decay gated on the *slow* rail, in a mantissa-tracked fp32 view of bf16 weights (sub-ulp accumulation). `anvil.py:48,59-76,747-801`; rails chosen `training.py:174-183`. Mostly SE.
3. **Cautious Weight Decay + LR schedule.** Adam gates decay on its own update: `mask=(update*p)>0`, `eff_wd = lr²·wd·wd_mul` (schedule-coupled, quadratic fall). lm_head/embed `wd_mul=150`. LR decays linearly to absolute floor 0.30 over the last 80% of main stages. `anvil.py:687-731`, `schedule.py:9,67-75`, `training.py:57-120`. SE.
4. **Value embeddings.** One `[4·50304, 768]` bf16 param, planes per `VALUE_EMBED_LAYERS=(1,2,8,10)`, gathered rows added to V before FA3, gated by learned `ve_gate_bank` (layer 10 uses MUDD lanes). Value-residual lineage (arXiv 2410.17897). `model/gpt.py:71-75,224-227,601-610,676-694,716`; `model/attention.py:202-215`. SE; row-sparse pull keeps it cheap.
5. **Hashed n-gram table.** `[84,602,880, 768]` = 6.5e10 elements, bigram+trigram channels, 10.6M rows/rank shard. XOR hashes, reserved sentinel rows, ±1 sign rows from an 8192-row pool (#299 sign trick lets n-grams share rows without averaging directions). Updates only at events (odd steps → %4==3 from step 336): rows pulled once per event into a cache, gradients to a zeros sink, per-event per-slot sum, routed back, owner-side Adam with **β₁=0 and one fp32 second moment per row**, missed β₂ decays replayed lazily. `ngram_table.py:53-140,242-350`; consumption `gpt.py:166,347-348,494-496,592-599,720`. SE+WC+MEM.
6. **Sampled softmax.** Early steps normalize over shared candidate set P=10,240→14,336→24,576→full (from step 1107): all batch targets + stride-permutation negatives (NEGATIVE_STRIDE=20011, staggered per rank — uniform coverage without RNG). lm_head GEMM + CE + both grad GEMMs at width P; full softmax only at validation. Prefix targets absent from C are dropped (-1 = kernel no-op). `sampled_softmax.py:45-52,79-139,242-250`; `model/gpt.py:786-793`. WC+MEM. (Biased normalizer — see open question 16.)
7. **MTP + late untie.** MTP = weighted CE over t..t+k (stage 0: [1,0.5,0.25]→[1,0.5,0]). Embed/lm_head tied at init; while tied only lm_head is communicated (embed.gradᵀ folded into lm_head.grad). **Untie at step 1175/1194 (~98%, the extension stage — not 2/3 as commonly cited).** `gpt.py:182-185,794-800`; `anvil.py:499-527,606-614,659-666`; `training.py:202-203`. SE + WC (halved embedding comms for 98% of run).
8. **Long/short sliding-window FA3.** Windows per stage `(1,3)→(3,7)→(5,11)→(6,13)` blocks of 128; `LONG_WINDOW_LAYERS=(3,10)` get long window + **partial key offset** (non-rotating key dims shifted one position — "previous block" info without extra attention). Patched FA3 pinned revision. YaRN re-derives rotary freqs on window growth, partial table rebuild on the clock, `complete_yarn_tables()` before validation. Final validation bumps ws_long 13→20. `gpt.py:38,52-53,561-562`; `attention.py:42-117,174,218-220`; `training.py:137-172`; `config.py:72`. WC+SE.
9. **Attention micro-architecture.** Sparse attention gate (per-head multiplicative on FA3 output, layers 3/10). Learnable XSA (removes output component aligned with own value vector, layers 1/3). Paired head attention (layers 0,2,5: interleaved rotary table doubles effective seq, halves window). Dynamically composable MHA: parameter-free attention modules, weights composed at forward from shared banks; per-layer runtime width decision; unused rows never get gradient. Depth cut: layer 7 removed whole (its cache still feeds every later attention layer), layers 4/6 no attention, 9 MLP-only. **Widths are 64 and 128 — there is no 96-dim QK in this code** (that's record #90's description; the merged stack uses 64). `gpt.py:30-49,505-530`; `attention.py:108-115,135-155,204-230`. SE+WC+MEM.
10. **Skip architecture.** `x0 = norm(smear(embed))` re-injected at `X0_INJECT_LAYERS=(0,1,2,4,5,7)` with per-token MUDD gates (not every block — 6 and 10 use MUDD coefficients instead). block3→block6 skip at layer 6 (σ(skip_lambda)·gate, init −1.5). **MUDD** (arXiv 2502.12170 trimmed): two GELU MLPs produce per-token coefficients — pre-gate (23 lanes), post-gate at layer 4 (13 lanes), last-layer MUDD (V sources + residual recombination), post-loop MUDD mixing 10 sources with per-token + per-channel-group deltas (12 groups, zero-init). **Residual stream scaling is geometric growth**: `resid_lambdas=1.1^0.5` per sublayer, folded into MLP down-projection scale. `gpt.py:350-354,362,368-488,643-651,676-686,712-720,732-747,758-772`. SE+WC.
11. **Softcap + asymmetric rescale + FP8.** Softcap `z = 23·sigmoid((logit+5)/7.5)` — the +5 shift centers the useful range; spelled identically in CE kernel, eval path, and backward chain factor; fp16 σ cache because bf16 rounds 1−σ to 0; LSE uses fixed max A. FP8: lm_head (e4m3 logits, e5m2 grad, two `_scaled_mm` backwards, statically clip-free scales); MLP all four GEMMs (static x_scale 2⁻⁴ justified by √768 bound, delayed relu² scales, dual layouts, post-lambda folded); QKV one packed cache per width group (static x_scale 8/448). `gpt.py:111-125,173-180,784-816,857-947`; `perf/kernels/cross_entropy.py:14-28`, `mlp.py`, `qkv_rope.py`. WC+MEM.
12. **Smear + prefix-token aux loss.** Smear: `x[t] += λ·σ(W·x[t−1][:12])·x[t−1]` — rank-12 gated 1-token lookback, zero-init no-op start; smear_gate lr_mul=0.01 (near-frozen). Prefix prediction: aux CE whose target is the longest proper prefix of the token's byte string that's also a token (static vocabulary only); table built on the clock, bucketed + all_reduce(MAX), baked into the step graph; weight 0.25→0. `gpt.py:161,317-320,342-343,361,612-614,795`; `model/prefix_prediction.py:12-45`; `config.py:77-85`; `schedule.py:62-65`; kernel `perf/kernels/cross_entropy.py:187-207`. SE.
13. **Schedules + tail averaging.** Batch 8→16→24→20(taper)→8 (extension) × 2048 × 8, lr_mul ∝ (batch ratio)^0.6/0.5; train_max_seq_len 896→2048→3072, but attention segments capped at VIRTUAL_SEQ_CAP=2560 (loader inserts boundaries; token stream untouched — keeps FA3 max_seqlen and graph keys small). LR linear to floor 0.30 from step 234 — the whole tail-averaging window sits at floor LR. Four accumulators ship before final val: tail-EMA (lm_head+embed, last 298, lerp 0.65), tail-avg (vo+mlp, last 250 @ every 4th), VE-avg, bank-blend (B=0.55 with Richardson correction 0.2007 removing lag); shipped weights' Frobenius norms restored ("decontraction"). `config.py:54-89`; `schedule.py:9,67-75`; `data.py:148-153`; `tail_average.py:30-49,126-167`. SE+WC.
14. **CUDA graphs.** Three families, captured in untimed warmup and sealed (a miss is fatal): step graphs (fwd+bwd pair per StepGraphKey = shapes/windows/candidates), optimizer graphs (one per ANVIL bank, ~50 launches → 1), FP8 refresh graphs (attn; MLP ± lm_head, odd steps). Gradients land in pool buffers, p.grad assigned manually. Baked-address census after in-place reset. `train_gpt.py:205-282`; `perf/cuda_graphs/*`. WC (~3 s + removes host bottleneck).
15. **Data path.** Batches aligned to token 50256 (BOS-as-EoS); every sample = concatenated whole documents cut at max_seq_len; last doc trimmed one token so targets stay in-document; cu_seqlens padded to fixed row count per batch size (one graph shape per batch size). Shards preadv-read with 4 threads; next shard BOS-scanned and preloaded in background threads; loader runs ahead (sampled softmax by 1, row prefetch by cycle+slack); H2D through pinned slots. Loader is NOT under a graph; batch copies into graph static buffers are. `data.py:23-46,74-80,106-129,155-279`. WC.

## One training step (train_gpt.py:75-113 + anvil.py:555-682)

```
1  batch = batches.peek(s)                    eager (loader ran ahead)
2  ngram_slots = row_prefetch.forward_slots(s) eager
3  sampled_softmax.upload(s)                   eager (side copy stream)
4  deferred_gathers.flush()                    eager: wait qk/vo -> [G] attn fp8 refresh;
                                              land VE replica; wait mlp -> [G] mlp refresh (+lm_head on Adam steps)
5  sampled_loss = sampled_softmax.gather(s)    eager (index_select + transpose of lm_head rows)
6  step_graphs.forward(s, batch, slots, cfg)   [G] copy to static buffers, replay
7  row_prefetch.after_forward(s)               eager host (count exchange -> id all_to_all)
8  ngram_grad = step_graphs.backward()         [G] replay; grads to pools; p.grad assigned
9  ngram_table.accumulate_grad(slots, held)    eager ring bookkeeping
10 sampled_softmax.mark_readers_done()         eager
11 row_prefetch.post_prep(s)                   eager
12 sampled_softmax.prefetch(s+1)               host thread (CPU only)
13 training_manager.step_optimizers(s, ...)    ALL EAGER, NOT CAPTURED:
   - one pinned H2D of lr/wd/rails/betas
   - replicated Adam: one flat all_reduce + fused Triton launch (every 2nd step)
   - scatter order: reduce_scatter qk/vo/mlp (bf16 AVG, async); VE slot -> sparse
     ngram+ve gradient exchange; lm_head (with embed.gradᵀ transpose_add); embed
   - work order: wait -> [G] ANVIL bank graph or compiled Adam -> async all_gather
     -> [VE slot: ngram row-merge+Adam, ve shard merge+Adam, serve next cycle]
   - wait lm_head gather -> transpose_copy into embed (while tied); sparse finish
14 deferred_gathers.hand_over(...)             eager; wait moves to next step's (4)
15 batches.take(s)
16 tail_averages.tick(s)                       eager
```

**Under graphs:** model fwd, model bwd, 3 ANVIL bank updates, fp8 refresh halves. **Not under graphs:** all collectives, Adam, sampled-softmax gather, row pulls, YaRN rebuild, tail ticks, data copies.

## Validation path

- val_tokens=10,485,760, val_batch=2,097,152 → 5 steps; `val_loss_every=0` (only final val runs, on the clock).
- Before val (clock running): flush deferred gathers, ws_long 13→20, canonical-mask collect, **tail_averages.ship()**, complete YaRN tables, VE replica gather.
- N-gram rows: one batched count exchange + one id all_to_all for all 5 val batches; cache holds one batch, re-landed per batch (untimed).
- Loss: bf16 path, slabs of 32768 rows, softcap → canonical mask (−60) → per-token CE; sampled_loss ignored at eval — always full softmax. Clock stops before `model.eval()`? No — **the clock stops before model.eval() and restarts after; val batch reads, n-gram pulls and ship() are INSIDE the timed region.** gc.collect() runs with the clock stopped.
- Checkpoint (optional) skips the n-gram table (16 GB/rank, rank-0 only).

## Open questions (to resolve by experiment or by asking the community)

1. **No 96-dim QK** — widths are 64/128 only. Is "96" in #90's description a different config or a typo?
2. **Untie at ~98%, not 2/3** — why push the split to the 19-step extension? (Comms saving vs cold-moment problem?)
3. **LR_FLOOR=0.30 vs code comment "README says 0.20"** (schedule.py:8-9) — which value produced #92?
4. **RAIL_ENGAGE_STEP=514 is absolute**, not fractional — interacts with `SCHEDULE_GROWTH_STEPS` so any step-count sweep moves two things at once. Why 514?
5. **Magic constants without derivation**: RAIL_FAST_WEIGHT=0.4385, RICHARDSON_W=0.2007, TAIL_AVG_RATE=4/53 ("not derived from the window"), BANK_BLEND=0.55, NARROW/WIDE_ATTN_SCALE=0.13/0.085, VALUE_EMBED_LR_MUL=70, wd_mul=150, MUDD_GATE_SCALE=0.1. Which were swept, over what budget?
6. Two sibling MUDD gate scales: `_mudd_scale=0.1` plain float vs `_mudd_gate_scale` nn.Parameter lr_mul=0.1 — why different?
7. `X0_INJECT_LAYERS=(0,1,2,4,5,7)` and `BIGRAM_INJECT_LAYERS=(0,1,4,5,9)` exclude 3,6,8,10 — deliberate or accidental? Layer 8 injects neither.
8. **Layer 7 is "dead"** (no attn, no MLP; MLP bank slot 7 allocated and frozen only to keep the bank even) — is cache[7] worth a whole layer slot? Why keep residual scaling there?
9. Bank-slot↔layer invariant ("slot i is layer i, except 7; 8→also 11") enforced only by comments + reshape attribute.
10. n-gram hash edge cases: `bigram[1:]` computed in place from `bigram[:-1]` so position 1's "previous" is the reserved sentinel, not x[0]; moduli are half−1. Tuned-into-correctness off-by-ones?
11. **FP8_ATTN_X_SCALE=8/448 static** — no amax check or fallback. Observed max? What happens on a spike?
12. smear_gate lr_mul=0.01 — is smear meant to stay near zero-init?
13. Warmup budget coupling: `WARMUP_STEP_BUDGET=40` + `assert split_step in warmup_steps` — a schedule edit can silently invalidate the untie path; compile_steps ±2 window is hand-tuned to graph-key granularity.
14. **Final val cost is timed but barely instrumented** (~0.2–0.5 s?): ships 5 weights, completes 3 YaRN tables, builds the mask — is the canonical mask actually worth ~0 XP (#91 regression)?
15. EVAL_CE_SLAB_ROWS=32768 ⇒ 6.6 GB fp32 logits per slab next to 16 GB table shard — memory headroom stated, not measured.
16. **Sampled-softmax residual bias**: biased normalizer, full resolution only from step 1107/1194. Was the bias at the switch ever measured? Would P=40960 through the taper be free (never tried in code)?

## Candidate modification sites (small diff, measurable)

| # | Site | Change | Headroom |
|---|---|---|---|
| 1 | `sampled_softmax.py:45-49` | Give taper stage (1107–1174) P=40960 instead of full softmax | Full softmax costs ~2.4× the P=24576 GEMM+CE+2 grad GEMMs for 68 steps; zero kernel work, one dict row |
| 2 | `schedule.py:78-90` + `anvil.py:48` | Make rail-engage fractional; sweep rail betas | Four constants with no derivation; touches only the pinned scalar buffer — graph capture unaffected |
| 3 | `tail_average.py:30-47` | Widen TAIL_EMA_WINDOW to the full floor-LR span (step 234+); sweep TAIL_AVG_RATE / BANK_BLEND | LR pinned at floor for ~960 steps; only last 250–298 averaged. Cheapest SE lever in the repo |
| 4 | `gpt.py:46-49,107` | Move one layer between width groups; try MLP_HIDDEN_DIM 2688/2944 | Widths are runtime decisions over shared banks; groups inherited from #360, never re-swept |
| 5 | `attention.py:52-53,93` | Sweep NARROW/WIDE_ATTN_SCALE and the YaRN 0.2·ln growth coefficient | Attention temperature is high-variance; constants are Python floats re-checked per replay |
