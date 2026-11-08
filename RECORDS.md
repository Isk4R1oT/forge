# Record archaeology — what 92 speedrun records teach about pretraining

> Study notes on the full record history of the NanoGPT speedrun (May 2024 → Aug 2026), made while reproducing record #92.
> Sources: upstream README record table, record logs, PRs #360/#367/#375/#379/#391/#393/#396/#397/#399, track-1 short package.
> Compiled 2026-11-09. Δ% computed from consecutive record times.

## TL;DR — the shape of the whole game

- **92.9% of the total gain (45.0 → 3.57 min) came in the first 17 records (7 months).** Since record #18, the field has fought 20 months for 3.1 minutes. Logarithmic decay of record *interval*, not of effort.
- **ML vs systems flipped.** Records 1–21 are nearly all ML (sample-efficiency). Records 22–92 are ~60% systems (comms, kernels, precision). Rough count: ~44 ML, ~44 systems/both.
- **Era boundaries are category-saturation events**, not dates: every era ends when the cheapest remaining instance of its dominant class drops under ~1%.
- **The published table lags reality.** Record #92 (39.9 s) is the last merge, but open PRs already claim **30.7 s** (#397). The true frontier is ~23% below the table.
- **Composability is the currency of the modern era:** the last three records were largely consolidations of open PRs (#391 = #379+#375; #393 = +#371+MoE; #397 = #393+#396).

## Eras

| Era | Records | Dates | Start→End | Δ min | Share of 44.3 | Dominant | Ended because |
|---|---|---|---|---|---|---|---|
| E1 Bootstrap | 1–2 | 06/24 | 45.0→31.4 | 13.6 | 30.7% | SCHED/ARCH | llm.c baseline was undialed — one LR+RoPE pass recovered 30%; class exhausted |
| E2 Muon | 3–4 | 10/24 | 31.4→22.3 | 9.1 | 20.5% | OPT | Muon's win came from *swapping* the optimizer (20.7%), follow-up tuning only 10% |
| E3 Modern arch + systems | 5–8 | 10–11/24 | 22.3→10.8 | 11.5 | 25.9% | ARCH+SYS | arch ideas were gated by serial optimizer cost and torch version, not by lack of ideas |
| E4 Residuals/embeddings/attention | 9–17 | 11–12/24 | 10.8→3.57 | 7.2 | 16.3% | ARCH+ATTN | 3 years of published arch tricks compressed into 9 records; each new idea ≤6% |
| E5 The 2025 grind | 18–40 | 01–10/25 | 3.57→2.358 | 1.21 | 2.7% | SYS+MEM | 23 records, 10 months, 1.2 min; a 4-month dead zone (#21→#22) where nothing was found |
| E6 Optimizer theory | 41–57 | 10/25–01/26 | 2.358→1.878 | 0.48 | 1.1% | OPT | NorMuon/CWD/PolarExpress/mixed-precision: 17 records for 0.48 min, each <1.3% |
| E7 Arch reshuffle + n-grams | 58–77 | 01–03/26 | 1.878→1.435 | 0.44 | 1.0% | SYS(13/20)+ARCH | two model ideas (paired heads, bigram hash) carried the era; rest was 0.3–1% kernel work |
| E8 FP8/sampled-softmax/graphs | 78–92 | 03–08/26 | 1.435→0.665 | 0.77 | 1.7% | MEM+SYS+ML | precision stack finished; #92 exhausted the *composition* play |

## Top-10 biggest jumps (absolute Δs)

| # | Rec | Δ | Mechanism (why so big) |
|---|---|---|---|
| 1 | 2 | −816 s | Baseline repair: untuned LR + learned absolute positions → RoPE. Raised tokens/sec and tokens-per-nat at once. |
| 2 | 5 | −426 s | Four changes as one portfolio: pad embeds, ReLU², zero-init projections (identity-start ⇒ big LR safe), QK-norm (bounds logits). The *combination* paid; QK-norm enabled the doubled LR that #11 then used. |
| 3 | 3 | −390 s | Muon: orthogonalized momentum ⇒ uniform progress in all singular directions. Needs #5's stability to run at its LRs. |
| 4 | 4 | −156 s | NS coefficient tuning + momentum schedule + WD — pure tuning of Muon's family. |
| 5 | 9 | −156 s | Value-residual + embed skip to every layer + softcap: direct token-identity path to every block ⇒ deeper effective gradient flow. Ancestor of MUDD (#81). |
| 6 | 12 | −130 s | 1024 dense causal → 64K long/short sliding-window: O(window) instead of O(ctx) attention cost at unchanged quality. |
| 7 | 6 | −126 s | Sharded Newton–Schulz: removed Muon's ~200 ms/step serial tail. Systems win only meaningful because Muon was now the optimizer. |
| 8 | 8 | −72 s | Untie embed/lm_head (later refined into retie-at-2/3 in #51/#53 — actually at 98% in the current code, see NOTES.md). |
| 9 | 7 | −66 s | PyTorch 2.5.0 — better codegen, same math. Template for #25/#66. |
| 10 | 92 | −27.7 s vs #91 | Six mechanisms at once (see below). A portfolio, not an idea — the only post-2024 record that behaves like an early-era record. |

Modern-era top jumps by %: #92 40.9%, #90 8.1%, #62 5.3%, #85 3.7%, #67 3.5%, #81 3.1%, #58 3.1%. Absolute sizes are 100× smaller than the early era.

## Record #92 (39.91 s, @DevenPzak, PR #360) — decomposition

Best-documented record in the repo: 18 cold-cache runs, 9 same-node runs of record #89, 26 single-mechanism removal legs.
**Headline: 39.914 ± 0.120 s vs #89 at 73.889 ± 0.137 s on the same node = −33.98 s (−46.0%, 1.85×), p=9.4e-10.**

| Component | wall Δ | val Δ (mn) | net @164 ms/mn | share |
|---|---|---|---|---|
| Sampled softmax CE (shared candidate set, full-vocab val) | +8.80 s | −2.4 | **+8.40 s** | 25% |
| Hashed n-gram table (84.6M rows, sparse) | +1.19 s | +35.8 | **+7.05 s** | 21% |
| ANVIL (update rule + anneal/shipping + optimizer graph) | mixed | +25.6 | **+5.08 s** | 15% |
| Full-stack FP8 (MLP fwd+bwd, lm_head cache) | +5.02 s | −0.4 | **+4.96 s** | 15% |
| Mixed-width attention (layer drop, d_qk 64/128, half-V) | +4.95 s | −6.9 | **+3.82 s** | 11% |
| CUDA-graph capture (step + optimizer) | +2.99 s | 0.0 | **+2.99 s** | 9% |
| Sparse grad comms + loader | +1.12 s | +0.5 | **+1.20 s** | 3.5% |
| MUDD residual mix (10-coef) | −0.51 s | +6.1 | **+0.48 s** | 1.4% |
| **Total** | +24.45 s | +58.2 | **+33.98 s** | 100% |

Careful reads:

- **The biggest single item is not the one that looks biggest.** Raw wall-clock removal deltas sum to only 24.45 s; the other 9.5 s comes from *val-loss effects* priced at 164 ms/millinat (the measured marginal cost of buying loss with steps, stated range 120–250).
- **Sampled softmax (+8.40 s) is the largest wall-clock win**: over the first ~93% of the run the CE competes against P=10,240→14,336→24,576 candidates instead of 50,304, full softmax only at validation. Pure wallclock, and it slightly *improves* measured val (−2.4 mn) because the candidate set is target-enriched.
- **The n-gram table (+7.05 s) is the largest algorithmic win**: 84.6M rows × 768 = 6.5e10 parameters, but cost scales with tokens touched, never with table size. The engineering is the story: sharding + row-compacted Adam (β₁=0, one fp32 second moment per row) + exact replay made it possible to grow the table 224×.
- **ANVIL is mostly shipping, not the optimizer**: twin-rail + re-derived spectral maps worth +0.33 s; the tail-shipping (TailEMA, bank tail-blend, VE ship, decontraction) worth +3.82 s.
- **Author's caveat**: the val↔wall conversion rate is fitted so the table sums to the certified −33.98 s; ablations not strictly additive. Treat shares as ±2 pp.
- **Regression note**: #91's canonical masking (#350) is *not* in the #92 trainer — the 27.7 s vs #91 includes losing that trick.

## Where a solo can still win (and what is exhausted)

**Exhausted:** Muon-family optimizer variants (E6, <1.3% each), FP8 (downcast end-to-end already), kernel micro-fusions (≤1% each), layer-drop/schedule reshaping, track-3 step-count frontier (stuck ±10% around ~3000 steps for six months — do not compete there solo).

**Under-exploited classes:**

1. **Validation-path trickery** — explicitly legal (rule 1 freezes token streams only; the metric is "a valid probability model"). #33/#35/#91 all bought steps for free; #376/#378/#381/#399 sit open. Each model change re-opens this class.
2. **Off-clock precomputation** (retrieval / exact n-gram indices) — largest raw numbers in the pool (#367 claims 6.7 s, #380 9.65 s, #366 −6.16 s), but all hinge on a *ruling* about audit-trail compute. **Do not bet without a maintainer statement.**
3. **Exact systems levers** — #396 found 0.66 s in *four* pure no-math levers on an already-optimized 31 s stack (unintended FA3 reshape copies, late reduce-scatters, stacked grad banks). "Audit every record for non-overlapped copies/zero-fills/syncs" is repeatable, deterministic, loss-budget-free — the lowest-risk way for a solo to get on the board.
4. **Hash-table sparsity** — #375's own data: 64.6% of bigram slots and 8% of trigram slots are never touched in a full run. Pruning/reallocating rows is memory-free loss. Unexplored.
5. **Approximate training gradients** — #371 (sampled LM-head gradient, truncated attention backward) is the only open PR in this class; the least crowded non-rules-risky corner.

**The actual bottleneck is merge lag, not ideas.** #397 (30.675 s, n=12, paired seeds, timing-only) has been open since 2026-10-31 and is 23% faster than the last merged record. Rule 4 explicitly encourages consolidating open PRs — the last three records did exactly that.

## Candidate pool for my bet (ranked)

| Priority | PR(s) | Why |
|---|---|---|
| 1 | **#397 + #396 + #390** | Current true frontier, timing-only (no loss budget), four mechanisms broken out per-lever, rule-4-compliant consolidation — "strongly encouraged" by the rules |
| 2 | **#399 vote mechanics alone on stock #92** | ~−2.5% for zero training cost; validate without the rules-ambiguous retrieval index → unambiguously legal |
| 3 | **#391 on #92** (−5.3 s, n=18: CPLM pointer LM + n-gram token normalization) | Lowest-variance ML bet: two orthogonal, independently attested ideas (#379 −4.57 s, #375 −0.42 s) |
| 4 | **#371 on #397** | Sampled LM-head gradient — the only unclaimed approximation class; composes with the 30 s stack |
| Hedge | #375 as a *map*, not a gain | Its real value: documents remaining headroom in the hash table (64.6% of bigram slots never hit) |
