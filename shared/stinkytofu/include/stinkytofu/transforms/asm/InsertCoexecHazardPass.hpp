/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;
class StinkyAsmModule;

/// Insert V_NOPs for MI400/gfx1250 multicycle co-execution data hazards.
///
/// On gfx1250 WMMA/SWMMAC (XDL) and TRANS instructions co-execute with other
/// VALU on separate sub-pipelines. Hardware cannot detect data hazards between
/// co-executing instructions, so software must space a producer and a dependent
/// consumer by a number of VALU-pipe slots (independent VALU or V_NOP). This
/// pass performs a forward walk with a bounded backward scan and inserts the
/// missing V_NOPs before each hazard consumer.
///
/// Hazards handled (only when a VGPR range actually overlaps):
///   - WMMA -> WMMA, prev D feeds A/B (or SWMMAC index): 1 + coexec slots
///   - WMMA -> co-executable VALU (RAW/WAR/WAW on D/A/B):    coexec slots
///   - WMMA -> WMMA, prev D feeds C (accumulation):          0 (HW forwards)
///   - TRANS -> TRANS / TRANS -> XDL WMMA:                   1
///   - TRANS -> core/side VALU:                              0 (HW handles)
///
/// The coexec slot count is read directly from the producer's HwInstDesc
/// coIssueWindow bitmask: popcount(coIssueWindow) == coexec slots. WMMA->VALU
/// needs that many slots; WMMA->WMMA needs one more.
///
/// DISABLE_XDL_ARB_STALL awareness: the pass tracks the SCHED_MODE arb-stall
/// bit during the walk. When co-execution is disabled (arb-stall set — the
/// normal gfx1250 case, emitted by TensileLite) the reduced counts apply:
/// WMMA->WMMA = 1, WMMA->VALU = 0.
///
/// The pass takes full ownership of hazard V_NOPs: it first strips every bare
/// V_NOP (TensileLite's mode-unaware miVALUInstrDataHazard fillers are the only
/// bare-V_NOP source on this path), guarantees co-execution is OFF by ensuring
/// the DISABLE_XDL_ARB_STALL setreg exists (emitting one at the entry anchor if
/// absent), then re-emits the correct mode-aware counts. This avoids the
/// over-emission that results from layering on top of TensileLite's fixed 4/8.
/// Runs whole-kernel, AFTER InsertWaitAluPass (s_wait_alu are not VALU and must
/// already be placed) and BEFORE InsertDelayAluPass.
///
/// The module overload reaches callee Functions; the no-argument overload
/// (stinkytofu-opt single-pass mode, unit tests) processes only the given
/// Function.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertCoexecHazardPass(StinkyAsmModule& module);
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertCoexecHazardPass();

}  // namespace stinkytofu
