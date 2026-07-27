# FA-v3 backward on gfx950 — status summary

**Date:** 2026-06-26   **Scope:** deterministic FA-v3 **backward** for Meta training, on **MI350** (gfx950, the power-reduced part Meta deploys).

---

## 1. Meta requested attn config + H100 baseline

- **Common to all configs:** **head_dim=64**, GQA group 8, bf16, varlen; full causal **and** sliding-window (SWA); deterministic backward.
- **Shape families** (S_q up to 16K, S_kv up to 16K; benchmarked at B=4):

| family | H_q | H_kv | S_q | S_kv |
|---|--:|--:|--:|--:|
| newshape | 128 | 16 | 2k–16k | 16k |
| 5.5B-tier | 48 | 6 | 4096 | 4k / 8k / 12k / 16k |
| 2.3B-tier | 64 | 8 | 1024 | 1k / 16k |

- **Metric:** **both forward and backward** throughput ≥ **1.4× H100 Tri Dao FA-v3**.

**H100 FA-v3 baseline** (measured, square B=1 H_q=128 H_kv=16 causal, D=64; FLOP conv `4·B·S²·H_q·D/2`, bwd ×2.5):

| S | H100 fwd (TF/s) | 1.4× fwd target | H100 bwd (TF/s) | 1.4× bwd target |
|--:|--:|--:|--:|--:|
| 2048 | 298 | 417 | 226 | 316 |
| 4096 | 445 | 623 | 377 | 528 |
| 8192 | 486 | 680 | 466 | 652 |
| **16384** | **522** (8.42 ms) | **731** | **508** (21.6 ms) | **711** |

At the headline S=16384, D=64: need **fwd ≥ 731 TF/s** and **bwd ≥ 711 TF/s** to hit 1.4× on both passes. (On H100, forward MFU slightly exceeds backward.)

---

## 2. Current status vs the 1.4× target

### 2.1 Forward (production = CK)

MI350 CK forward is ahead of H100 FA3 on all 20 configs (1.1–1.8×) but meets 1.4× on only 3/20 (the smallest shapes); large-S sits ~1.1–1.15×:

| shape group | forward ratio (MI350 CK / H100 FA3) | meets 1.4×? |
|---|---|:--:|
| Set A newshape, SWA (S 2k→16k) | 1.30× → 1.11× | no |
| Set A newshape, full (S 2k→16k) | 1.14× – 1.16× | no |
| Set B 5.5B-tier (H=48/6) | 1.18× – 1.45× | 1 of 4 (only 4096,4096) |
| Set B 2.3B-tier (H=64/8) | 1.20× – 1.76× | 2 of 4 (the 1024,1024 shapes) |

(A research HipKittens forward hd64 kernel *does* clear 1.4× — but it is **non-varlen, no-SWA, not production-ready**, so it is not counted here; those numbers and what they imply for headroom are in §3.3.)

### 2.2 Backward — deterministic (Meta's requirement)

MI350 = **best of two CK branches per config**: exp1 (grid-prune / non-persistent) and **Yi Ding's persistent kernel**. Persistent wins the **large/square full-causal** shapes (e.g. #15 4096×4096 full **4.08→2.95 ms**, #8 16384 **121→105 ms**); exp1 wins SWA and rectangular. (Forward is a *shared* kernel — persistent doesn't change it.) D=64, B=4 varlen:

| case | MI350 (best-of-both) vs H100 FA-v3 (det) | meets 1.4×? |
|---|--:|:--:|
| **SWA** deterministic | **1.51× → 1.88×** (9 of 10; only the tiny 1024² is 1.07×) | **YES** — comfortably |
| **Full causal** deterministic | **0.83× → 1.16×** (worst-case improved from 0.65× exp1-only) | **NO** |

**SWA already meets 1.4×; the unmet backward case is full-causal deterministic** — persistent narrows but does not close it. Tally across the full 20 Meta configs (§2.3): **9/20 meet 1.4× — all SWA; 11/20 miss — all 10 full-causal + 1 tiny SWA.**

**Net:** in production (CK), the 1.4× gap is on **both** passes — forward misses at large-S (17/20), backward misses on all full-causal (11/20). The backward is the larger/harder gap. Encouragingly, a research HK forward hd64 already clears 1.4× (§3.3), so the *pipeline* headroom for hd64 is proven — the remaining work is productionizing it (varlen + SWA) and closing the deterministic backward.

### 2.3 Full 20-config comparison (deterministic backward — both CK branches vs best-H100-det)

<details>
<summary><b>Click to expand — all 20 Meta configs (Set A newshape #1–8, Set B 5.5B-tier #9–20)</b></summary>

All D=64, B=4 varlen, bf16. Times in ms. **exp1** = grid-prune/non-persistent; **persist** = Yi Ding's persistent kernel (bold = the faster branch). bwd ratio = best-H100-det / best-MI350-branch (>1 = MI350 faster; target 1.4×). best-H100-det = faster of FA2/FA3. fwd ratio = MI350 fwd / H100 FA3 fwd (forward is a shared kernel across branches).

#### Set A — newshape (H_q=128, H_kv=16, S_kv=16384, W=2048)
| # | mask | S_q | exp1 bwd | persist bwd | H100-det bwd | bwd ratio | fwd ratio |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | SWA | 2048 | **5.012** | 5.340 | 7.88 (FA2) | **1.57x** | 1.30x |
| 2 | SWA | 4096 | **8.948** | 9.217 | 14.04 (FA2) | **1.57x** | 1.27x |
| 3 | SWA | 8192 | **16.317** | 17.217 | 26.50 (FA2) | **1.62x** | 1.11x |
| 4 | SWA | 16384 | **29.263** | 31.482 | 48.39 (FA2) | **1.65x** | 1.12x |
| 5 | full | 2048 | **23.460** | 24.987 | 24.78 (FA3) | 1.06x | 1.14x |
| 6 | full | 4096 | **43.743** | 46.043 | 41.37 (FA3) | 0.95x | 1.16x |
| 7 | full | 8192 | 81.829 | **77.926** | 66.79 (FA3) | 0.86x | 1.15x |
| 8 | full | 16384 | 120.992 | **105.136** | 86.79 (FA3) | 0.83x | 1.14x |

#### Set B — 5.5B-tier (W=2047); config = (S_q, S_kv, H, H_kv)
| # | config | mask | exp1 bwd | persist bwd | H100-det bwd | bwd ratio | fwd ratio |
|--:|---|---|--:|--:|--:|--:|--:|
| 9  | 4096,4096,48,6   | SWA  | 2.758 | **2.682** | 4.63 (FA2)  | **1.72x** | 1.45x |
| 10 | 4096,16384,48,6  | SWA  | **3.434** | 4.510 | 6.44 (FA2)  | **1.88x** | 1.36x |
| 11 | 4096,8192,48,6   | SWA  | **3.248** | 3.969 | 6.11 (FA2)  | **1.88x** | 1.34x |
| 12 | 4096,12288,48,6  | SWA  | **3.357** | 3.861 | 6.27 (FA2)  | **1.87x** | 1.22x |
| 13 | 1024,1024,64,8   | SWA  | **0.457** | 0.491 | 0.49 (FA3)  | 1.07x | 1.45x |
| 14 | 1024,16384,64,8  | SWA  | **1.582** | 1.751 | 2.39 (FA2)  | **1.51x** | 1.29x |
| 15 | 4096,4096,48,6   | full | 4.077 | **2.947** | 2.65 (FA3)  | 0.90x | 1.38x |
| 16 | 4096,16384,48,6  | full | **16.288** | 17.570 | 15.14 (FA3) | 0.93x | 1.18x |
| 17 | 4096,8192,48,6   | full | 8.133 | **7.874** | 7.17 (FA3)  | 0.91x | 1.29x |
| 18 | 4096,12288,48,6  | full | **12.055** | 12.730 | 11.64 (FA3) | 0.97x | 1.22x |
| 19 | 1024,1024,64,8   | full | **0.456** | 0.488 | 0.52 (FA3)  | 1.13x | 1.76x |
| 20 | 1024,16384,64,8  | full | **6.473** | 6.586 | 7.51 (FA3)  | 1.16x | 1.20x |

#### Tally vs 1.4x (best-of-both branches)
- **Backward det — meets 1.4x: 9/20** (configs 1-4, 9-12, 14, **all SWA**); **misses: 11/20** (all 10 full-causal 0.83-1.16x + tiny SWA #13). Persistent lifts the full-causal *worst case* from 0.65x (exp1) to 0.83x, but none of the full-causal configs reach 1.4x.
- **Forward — meets 1.4x: 3/20** (configs 9, 13, 19, the smallest shapes); **misses: 17/20** (ahead of H100 at 1.1-1.8x, but below 1.4x on all large-S shapes). Forward is branch-independent.

Caveat: MI350 is the power-reduced part; MI355X runs ~10-20% faster.

</details>

---

## 3. Additional backend benchmarks + path to 1.4× at HD=64

## 3.1 Forward

**Results** (additional backends; production CK forward is in §2.1). A research **HK hd64 forward** on the optimized 8-wave pipeline (MI350, B=1, H_q=128, H_kv=16, D=64, causal, **square**; validated cos=0.999995 vs aiter) clears 1.4× at every S:

| S | HK fwd | aiter fwd | H100 FA3 fwd | **HK / H100** | aiter / H100 |
|--:|--:|--:|--:|--:|--:|
| 2048 | 544.9 | 414.8 | 298 | **1.83×** | 1.39× |
| 4096 | 675.9 | 530.1 | 445 | **1.52×** | 1.19× |
| 8192 | 708.9 | 566.1 | 486 | **1.46×** | 1.16× |
| 16384 | 733.5 | 576.3 | 522 | **1.41×** | 1.10× |

- **HK forward = optimized 8-wave pipeline** → 1.41–1.83× (also ~1.27–1.31× over aiter fwd).
- **aiter has NO hd64 forward asm kernels** — its `asm_fmha_v3_fwd_configs.hpp` ships only hd128 and hd192_hd128, so aiter hd64 forward **actually runs CK** (the ck_tile fmha forward), a different/weaker pipeline → 1.10–1.39×. (aiter fwd *is* the optimized 8-wave asm only at hd128.)
- **Not production-ready:** the HK forward is non-varlen, no SWA, square-only — an existence proof, not counted in §2.

**Plan (forward).** Two routes, not mutually exclusive:
1. **Productionize the HK 8-wave hd64 forward** — add **varlen + SWA** (and rectangular S_q≠S_kv). Lower-risk than the backward: the pipeline already exceeds 1.4×, so the work is *coverage*, not new performance.
2. **File a ticket to AITER to add optimized hd64 forward asm kernels** — today aiter has no hd64 fwd asm and silently falls back to CK (~1.1–1.4×); an optimized hd64 fwd asm (like their hd128) would lift aiter's own forward and give a production path without HK.

## 3.2 Backward (the hard gap)

**Results — backend landscape (full-causal):**
| backend | det? | D | vs H100 | note |
|---|:--:|:--:|--:|---|
| HipKittens bf16 | no | 128 | **1.38×** @16K (up to ~2× non-causal/B16) | HK re-optimized aiter's hd128 pipeline; **non-det, D=128 only** |
| HipKittens split+reduce | **yes** | 128 | ~free vs non-det (plain-store, no atomic tax) | deterministic; **D=128 only**, D=64 unported |
| CK (exp1 / persistent) | **yes** | 64 | **0.83–1.16×** full-causal (§2.2) | production deterministic; below target |
| AITER asm | no | 128 | **~1.2×** | optimized gfx950 asm |
| AITER asm | no | 64 | **~parity (1.0×)** | **unoptimized MI300 recompile** (per AITER team) |
| AITER triton (dao_ai + autotune) | yes* | 64 | 0.71–0.76× | far below |

**Why the current deterministic backends miss:**
- **HD=128 reaches 1.38× only *non-deterministically* (HK).** Determinism costs a **~2× fp32-atomic tax** on gfx950 unless done as disjoint-store + fixed-order reduce: gfx950 has **no packed fp32 atomic**, so fp32 atomic-accumulate issues 2× the ops into one shared L2 atomic unit (~1.3 TB/s RMW ≈ 17% of HBM) → drain spills past compute → ~2×. Our **split+reduce avoids this** (plain stores), so deterministic HK is ~free vs non-det — but it's **D=128 only**.
- **No optimized HD=64 backward exists** in any backend — CK is production but below target, aiter asm is an unoptimized recompile, triton is far off.

**Why 1.4× is still possible (headroom):**
1. **The 8-wave pipeline is MEASURED at 1.4× for hd64 — on the forward (§3.1).** A measured hd64 existence proof, not an hd128 extrapolation; the backward is the same pipeline family applied to the bwd math.
2. **gfx950 also does ≥1.4× at hd128** — HK re-optimized aiter's hd128 asm to 1.38× (up to ~2×); deterministic via split+reduce is ~free.
3. **aiter's hd64 is under-served in *both* directions** (no asm forward → CK; unoptimized recompile backward) — a codegen gap, not hardware.
4. **That lift has never been applied to hd64 *backward*** — doing it (HK 8-wave schedule/regalloc + split-store/fixed-order-reduce) is the headroom from CK's 0.83–1.16× → 1.4×.
5. **Caveat:** the backward has lower arithmetic intensity / more memory traffic than the forward, so a 1.4× *backward* is a bigger stretch than the forward we already hit — not automatic.

**Plan (backward):**
1. **Optimized HD=64 deterministic bwd (primary)** — port HK's re-optimized hd128 pipeline to D=64 with **split-store + fixed-order reduce**. No CK/AITER dependency; either hits 1.4× or yields a hardware-grounded reason it can't. (~1–2 mo.)
2. **AITER team** optimizes HD=64 asm + adds a deterministic mode (removes the recompile gap). — the ticket.
3. **CK team** ports the HK hd128 pipeline into CK's deterministic HD=64.
4. **Ask Meta if bf16 dq_acc accumulation is acceptable** — sidesteps the fp32-atomic tax entirely; quick to try.

---

## 4. Action items

- **File tickets to AITER** — request optimized gfx950 **hd64 asm kernels** in both directions: (a) an hd64 **forward** asm (aiter has none today → silently falls back to CK, §3.1), and (b) an optimized hd64 **backward** asm + a **deterministic** mode (current hd64 bwd is an unoptimized MI300 recompile, §3.2). Attach the shape set, FLOP convention, our benchmarks, and the disassembly-level pipeline analysis.
- **TE team owns the HK hd64 port** — (a) productionize the HK 8-wave hd64 **forward** (varlen + SWA + rectangular; pipeline already >1.4×, §3.1), and (b) build the **deterministic hd64 backward** on the HK 8-wave pipeline + split-store/fixed-order-reduce (§3.2 plan 1). This is the path to Meta's 1.4× with no external (AITER/CK) dependency.
