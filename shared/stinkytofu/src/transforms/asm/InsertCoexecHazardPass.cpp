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

#include "stinkytofu/transforms/asm/InsertCoexecHazardPass.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

#define DEBUG_TYPE "InsertCoexecHazardPass"

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HwReg.hpp"
#include "stinkytofu/hardware/HwRegHelpers.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace {
using namespace stinkytofu;

// Per-arch co-execution hazard configuration.
//
// The WMMA V_NOP counts are NOT tabulated here: they are derived at runtime from
// each producer's HwInstDesc coIssueWindow bitmask (popcount == coexec slots).
// Only the arch-level rules that cannot be read off a single instruction live in
// this struct.
struct CoexecHazardConfig {
    // TRANS -> TRANS and TRANS -> XDL WMMA spacing (independent op / V_NOP).
    int transToNonCoreSide = 0;
    // Whether HW detects TRANS -> core/side VALU (then no V_NOP is needed).
    bool hwHandlesTransToCoreSide = false;
    // Whether the arch has the SCHED_MODE DISABLE_XDL_ARB_STALL bit. When set at
    // runtime co-execution is OFF and the reduced counts apply.
    bool hasArbStallBit = false;
};

// MI450-B0 / gfx1250.
constexpr CoexecHazardConfig kGfx1250Config = {
    /*transToNonCoreSide=*/1,
    /*hwHandlesTransToCoreSide=*/true,
    /*hasArbStallBit=*/true,
};

// Largest slot budget worth scanning: max WMMA->WMMA count on gfx1250 is
// popcount(0xFF00)+1 = 9. 18 leaves generous headroom and bounds the scan.
constexpr int kMaxSlotBudget = 18;

enum class ProducerKind { WMMA, TRANS, DGEMM };

// A DGEMM/SGEMM producer: a multicycle f32/f64 GEMM op that does NOT use the XDL
// pipe. On gfx1250 this is the f32-input WMMA (v_wmma_f32_16x16x4_f32, SGEMM) or
// an f64 double-precision MACC (v_fma_f64 &c., DGEMM). These have no co-exec
// window, but feeding one's result into a following XDL WMMA still needs a single
// spacer (SPG 4.6.12.1 "1 V_NOP in between two WMMA instructions").
inline bool isDGEMMProducer(const StinkyInstruction& inst) {
    return (isMatrixInstruction(inst) && !isXDLWMMA(inst)) || isDPMACC(inst);
}

// What the current consumer is looking for during a backward scan.
struct ConsumerCtx {
    ProducerKind kind;
    bool consumerIsWmma;  // only meaningful for kind == WMMA
    const StinkyInstruction* consumer;
    bool coexecOff;
};

inline int popcount16(uint16_t v) {
    return __builtin_popcount(static_cast<unsigned>(v));
}

// Does this instruction fill a VALU co-execution [I] slot? Only VALU-pipe ops
// (incl. transcendental, matrix, and bare v_nop) count; S_NOP/S_WAIT_ALU/
// S_DELAY_ALU/SALU/VMEM do not.
inline bool isSlotFiller(const StinkyInstruction& inst) {
    return isVectorALU(inst) || isTranscendental(inst) || isMatrixInstruction(inst) ||
           inst.getUnifiedOpcode() == GFX::v_nop;
}

// A co-executable VALU consumer: a VALU (incl. transcendental) that is not a
// matrix instruction. Matches LLVM's isCoexecutableVALUInst intent.
inline bool isCoexecutableVALU(const StinkyInstruction& inst) {
    return (isVectorALU(inst) || isTranscendental(inst)) && !isMatrixInstruction(inst);
}

// WMMA producer D feeds a WMMA consumer's A/B (or SWMMAC index). D->C
// (accumulation) is intentionally NOT a hazard — HW forwards it.
bool wmmaToWmmaOverlap(const StinkyInstruction& prod, const StinkyInstruction& cons) {
    if (prod.getDestRegs().empty()) return false;
    const StinkyRegister& d = prod.getDestRegs()[0];
    const auto& srcs = cons.getSrcRegs();
    if (srcs.size() > 0 && d.isOverlap(srcs[0])) return true;                   // A
    if (srcs.size() > 1 && d.isOverlap(srcs[1])) return true;                   // B
    if (isSWMMA(cons) && srcs.size() > 2 && d.isOverlap(srcs[2])) return true;  // index
    return false;
}

// WMMA producer D vs a co-executable VALU consumer: RAW (D->src), WAW (D->dst),
// WAR (producer A/B, or SWMMAC index, -> consumer dst).
bool wmmaToValuOverlap(const StinkyInstruction& prod, const StinkyInstruction& cons) {
    if (prod.getDestRegs().empty()) return false;
    const StinkyRegister& d = prod.getDestRegs()[0];
    for (const StinkyRegister& s : cons.getSrcRegs())
        if (d.isOverlap(s)) return true;  // RAW
    for (const StinkyRegister& cd : cons.getDestRegs())
        if (d.isOverlap(cd)) return true;  // WAW
    // WAR: a later VALU overwrites a register the WMMA still reads. Producer
    // inputs are A (src0), B (src1), and for SWMMAC the index (src2).
    const auto& psrc = prod.getSrcRegs();
    const size_t nWar = isSWMMA(prod) ? 3 : 2;
    for (size_t i = 0; i < psrc.size() && i < nWar; ++i)
        for (const StinkyRegister& cd : cons.getDestRegs())
            if (psrc[i].isOverlap(cd)) return true;  // WAR
    return false;
}

// TRANS producer vs consumer: RAW/WAW on producer dst, WAR on producer src.
bool transOverlap(const StinkyInstruction& prod, const StinkyInstruction& cons) {
    for (const StinkyRegister& d : prod.getDestRegs()) {
        for (const StinkyRegister& s : cons.getSrcRegs())
            if (d.isOverlap(s)) return true;  // RAW
        for (const StinkyRegister& cd : cons.getDestRegs())
            if (d.isOverlap(cd)) return true;  // WAW
    }
    for (const StinkyRegister& ps : prod.getSrcRegs())
        for (const StinkyRegister& cd : cons.getDestRegs())
            if (ps.isOverlap(cd)) return true;  // WAR
    return false;
}

class InsertCoexecHazardPass : public StinkyInstPass {
   public:
    static char ID;
    explicit InsertCoexecHazardPass(StinkyAsmModule* module) : module_(module) {}

    const char* getName() const override {
        return "InsertCoexecHazardPass";
    }

    PassID getPassID() const override {
        return &InsertCoexecHazardPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        auto arch = passCtx.getGemmTileConfig().arch;
        archId_ = getGfxArchID(arch[0], arch[1], arch[2]);
        config_ = kGfx1250Config;

        PASS_DEBUG(std::cerr << "[InsertCoexecHazard] run arch=gfx" << arch[0] << arch[1] << arch[2]
                             << "\n");

        // Whole-kernel: process the entry function, then every callee. The pass
        // is invoked on the entry function; callees are reached via the module.
        // If a future driver invokes it directly on a callable, just process it.
        if (func.getIsCallable()) {
            if (!func.empty()) processFunction(func);
            return preserveCFGAnalyses();
        }

        if (!func.empty()) processFunction(func);

        if (module_) {
            for (Function* fn : module_->getFunctions())
                if (fn && fn->getIsCallable() && !fn->empty()) processFunction(*fn);
        }

        return preserveCFGAnalyses();
    }

   private:
    // Detect TensileLite's `s_setreg hwreg(SCHED_MODE, offset=4, size=1), 1`
    // which disables the XDL arb stall (co-execution OFF from here on).
    bool isArbStallSetreg(const StinkyInstruction& inst) const {
        if (!config_.hasArbStallBit) return false;
        return HwReg::isSetregTo(inst, HwReg::schedModeId(archId_),
                                 HwReg::schedModeDisableXdlArbStall(archId_));
    }

    // V_NOPs a consumer needs behind a matched producer.
    int required(ProducerKind kind, int slots, bool consumerIsWmma, bool off) const {
        if (kind == ProducerKind::TRANS) return config_.transToNonCoreSide;
        // DGEMM/SGEMM -> WMMA: a single spacer, independent of co-exec mode
        // (these ops have no [I] co-exec window).
        if (kind == ProducerKind::DGEMM) return 1;
        // WMMA producer.
        if (off) return consumerIsWmma ? 1 : 0;
        return consumerIsWmma ? slots + 1 : slots;
    }

    // Does `prod` match what `ctx` is scanning for?
    bool matches(const StinkyInstruction& prod, const ConsumerCtx& ctx) const {
        if (ctx.kind == ProducerKind::WMMA) {
            if (!isXDLWMMA(prod)) return false;
            return ctx.consumerIsWmma ? wmmaToWmmaOverlap(prod, *ctx.consumer)
                                      : wmmaToValuOverlap(prod, *ctx.consumer);
        }
        if (ctx.kind == ProducerKind::DGEMM) {
            if (!isDGEMMProducer(prod)) return false;
            // Same RAW/WAW/WAR shape as the TRANS overlap (producer dst vs
            // consumer src/dst, producer src vs consumer dst).
            return transOverlap(prod, *ctx.consumer);
        }
        if (!isTranscendental(prod)) return false;
        return transOverlap(prod, *ctx.consumer);
    }

    // Real instructions of `bb` in program order (skips pseudo-insts). Scans use
    // indices into this vector so they never rely on decrementing end(), which in
    // this intrusive list is a null sentinel whose operator-- is a no-op.
    static std::vector<StinkyInstruction*> realInsts(BasicBlock& bb) {
        std::vector<StinkyInstruction*> out;
        for (auto& node : bb) {
            auto* inst = dyn_cast<StinkyInstruction>(&node);
            if (inst && !isPseudoInst(inst)) out.push_back(inst);
        }
        return out;
    }

    // Backward scan of `bb` starting strictly before `startBefore` (or from the
    // end of the block when `startBefore` is null, used when entering a
    // predecessor). Follows predecessors while the slot budget is not exhausted.
    // Returns the maximum (required - existing) over every matching producer
    // reachable within budget; a non-positive result means no insertion is
    // needed. `accExisting` is the slot-filler count already accumulated between
    // the consumer and the scan start.
    //
    // Min-over-paths: the V_NOP count a consumer needs is the MAX shortfall over
    // ALL predecessor paths, i.e. the shortfall computed against the path with the
    // FEWEST fillers — if any runtime path reaches the producer with few fillers,
    // that path must be fully covered. So a block may be re-entered from a
    // predecessor as long as we arrive with STRICTLY FEWER accumulated fillers than
    // any previous entry (a smaller `existing` can only widen the shortfall). The
    // `minExisting` memo records the least `accExisting` each block was entered
    // with from a predecessor; a re-entry that cannot beat it is pruned. Because
    // `existing` only grows along a path, a loop backedge always re-enters with a
    // larger count and is pruned — so the walk terminates, bounded by
    // O(#BB * kMaxSlotBudget). (A single shared boolean visited-set would instead
    // bind each reconvergence block to whichever path reached it first, dropping
    // the other arm and risking an order-dependent under-insertion.)
    //
    // The scan start is located by pointer identity in a freshly collected inst
    // vector, so v_nops inserted before earlier consumers in this same block are
    // correctly counted as existing slot-fillers for later consumers.
    int scanBack(BasicBlock& bb, const StinkyInstruction* startBefore, int accExisting,
                 const ConsumerCtx& ctx, std::unordered_map<const BasicBlock*, int>& minExisting) {
        // Predecessor entries (startBefore == null) are memoized on the fewest
        // fillers seen; prune when this arrival cannot widen the shortfall. The
        // initial consumer-block call (startBefore != null) is never memoized, so a
        // self-loop backedge can still re-enter this block once from its end.
        if (!startBefore) {
            auto it = minExisting.find(&bb);
            if (it != minExisting.end() && it->second <= accExisting) return INT_MIN;
            minExisting[&bb] = accExisting;
        }

        int best = INT_MIN;
        int existing = accExisting;

        const std::vector<StinkyInstruction*> insts = realInsts(bb);
        int start = static_cast<int>(insts.size());
        if (startBefore) {
            for (int i = 0; i < static_cast<int>(insts.size()); ++i)
                if (insts[i] == startBefore) {
                    start = i;
                    break;
                }
        }

        for (int i = start - 1; i >= 0; --i) {
            StinkyInstruction& inst = *insts[i];

            // A call is a hard boundary: do not scan across it (cross-call drains
            // are handled conservatively elsewhere).
            if (isCall(inst)) return best;

            if (matches(inst, ctx)) {
                const int slots = ctx.kind == ProducerKind::WMMA
                                      ? popcount16(inst.getHwInstDesc()->coIssueWindow)
                                      : 0;
                const int need = required(ctx.kind, slots, ctx.consumerIsWmma, ctx.coexecOff);
                best = std::max(best, need - existing);
            }

            if (isSlotFiller(inst)) ++existing;
            if (existing > kMaxSlotBudget) return best;
        }

        // Reached the top of the BB with budget to spare: continue into every
        // predecessor, taking the max shortfall (min fillers) across them.
        for (BasicBlock* pred : bb.getPredecessors())
            best = std::max(best,
                            scanBack(*pred, /*startBefore=*/nullptr, existing, ctx, minExisting));
        return best;
    }

    // Compute, for every BB, whether co-execution is OFF on entry: OFF iff the
    // DISABLE_XDL_ARB_STALL setreg has executed on EVERY path from the function
    // entry to that block. The setreg is monotonic (set once, never re-enabled).
    //
    // This is a forward MUST-analysis (entryOff = AND over predecessors' exitOff).
    // Two seeding/safety points:
    //   1. exitOff is seeded at TOP (true/OFF) for every block and descends to the
    //      greatest fixed point. Seeding at bottom (false) yields the LEAST fixed
    //      point, which pins any block reachable only via a loop backedge to ON even
    //      when the setreg has provably executed — an over-insertion. TOP-seeding
    //      fixes the loop case (verified: post-loop epilogue resolves to OFF).
    //   2. A block with no predecessors is forced ON (en = !preds.empty()). This is
    //      the deliberately conservative choice, NOT dominance: an s_setpc indirect
    //      branch (runtime address) or a CFG-builder gap can make a genuinely
    //      reachable block look pred-less. Treating it as ON — and letting it poison
    //      its successors through the pred edges — guarantees we never under-insert
    //      on a path the static CFG failed to model. A dominator-tree formulation
    //      would instead trust the static CFG and could under-insert there.
    //
    // Direction guarantee: a block treated as ON when it is really OFF only
    // over-inserts V_NOPs (safe); the reverse would under-insert (corruption), so ON
    // is always the conservative default. Callees are processed as independent
    // functions (each entry defaults ON) until a module-level pass can propagate
    // caller mode across call edges.
    void computeCoexecOff(Function& func, std::unordered_map<const BasicBlock*, bool>& entryOff) {
        std::unordered_map<const BasicBlock*, bool> hasSetreg, exitOff;
        for (BasicBlock& bb : func) {
            bool found = false;
            for (auto& node : bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&node);
                if (inst && !isPseudoInst(inst) && isArbStallSetreg(*inst)) {
                    found = true;
                    break;
                }
            }
            hasSetreg[&bb] = found;
            entryOff[&bb] = false;
            exitOff[&bb] = true;  // TOP seed for the greatest-fixed-point must-analysis
        }

        bool changed = true;
        int iters = 0;
        while (changed) {
            changed = false;
            ++iters;
            for (BasicBlock& bb : func) {
                const auto& preds = bb.getPredecessors();
                bool en = !preds.empty();  // no preds => ON (conservative; see note 2)
                for (BasicBlock* p : preds) en = en && exitOff[p];
                if (en != entryOff[&bb]) {
                    entryOff[&bb] = en;
                    changed = true;
                }
                const bool ex = en || hasSetreg[&bb];
                if (ex != exitOff[&bb]) {
                    exitOff[&bb] = ex;
                    changed = true;
                }
            }
        }

        PASS_DEBUG({
            std::cerr << "[InsertCoexecHazard] === computeCoexecOff CFG dump (fixed-point in "
                      << iters << " iters) ===\n";
            for (BasicBlock& bb : func) {
                std::cerr << "[InsertCoexecHazard]   bb \"" << bb.getLabel() << "\""
                          << " setreg=" << (hasSetreg[&bb] ? 1 : 0)
                          << " entryOFF=" << (entryOff[&bb] ? 1 : 0)
                          << " exitOFF=" << (exitOff[&bb] ? 1 : 0) << " preds=[";
                bool first = true;
                for (BasicBlock* p : bb.getPredecessors()) {
                    std::cerr << (first ? "" : ", ") << "\"" << p->getLabel()
                              << "\"(exitOFF=" << (exitOff[p] ? 1 : 0) << ")";
                    first = false;
                }
                std::cerr << "]\n";
            }
            std::cerr << "[InsertCoexecHazard] === end CFG dump ===\n";
        });
    }

    // Strip every bare v_nop in the function. On the gfx1250 TensileLite path the
    // only bare-v_nop source is the mode-unaware miVALUInstrDataHazard filler at
    // the loop->epilogue boundary (activation code uses s_nop, and no earlier
    // StinkyTofu pass emits v_nops), so all bare v_nops are hazard fillers this
    // pass owns. Removing them lets the re-emit below size hazards from scratch
    // for the actual co-exec mode instead of inheriting TensileLite's always-ON
    // 4/8 counts. Returns the number removed.
    size_t stripVNops(Function& func) {
        size_t removed = 0;
        for (BasicBlock& bb : func) {
            for (auto it = bb.begin(); it != bb.end();) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                IRBase* node = it.getNodePtr();
                ++it;  // advance before a possible erase
                if (inst && inst->getUnifiedOpcode() == GFX::v_nop) {
                    bb.removeIR(node);
                    ++removed;
                }
            }
        }
        return removed;
    }

    // Ensure the DISABLE_XDL_ARB_STALL setreg (hwreg(26,4,1)=1) exists so co-exec
    // is OFF and the reduced counts apply. TensileLite already emits it on the
    // normal path (detected, not duplicated); this only fires as a fallback when
    // it is absent, anchoring at the kernel entry label like InsertWaitAluPass.
    void ensureArbStallSetreg(Function& func) {
        if (!config_.hasArbStallBit) return;
        for (BasicBlock& bb : func)
            for (auto& node : bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&node);
                if (inst && !isPseudoInst(inst) && isArbStallSetreg(*inst)) return;  // present
            }

        BasicBlock* entry = func.getEntryBlock();
        if (!entry) return;
        BasicBlock* anchorBB = entry;
        for (BasicBlock& bb : func)
            if (bb.getLabel() == "label_ASM_Start") {
                anchorBB = &bb;
                break;
            }
        auto anchorIt = anchorBB->begin();
        while (anchorIt != anchorBB->end()) {
            auto* inst = dyn_cast<StinkyInstruction>(anchorIt.getNodePtr());
            if (inst && isPseudoInst(inst)) {
                ++anchorIt;
                continue;
            }
            break;
        }
        IRBase* anchor = (anchorIt == anchorBB->end()) ? nullptr : anchorIt.getNodePtr();
        AsmIRBuilder builder(*anchorBB, archId_);
        StinkyInstruction* setreg =
            builder.create(getMCIDByUOp(GFX::s_setreg_IMM32_b32, archId_), anchor);
        const HwReg::SubField arb = HwReg::schedModeDisableXdlArbStall(archId_);
        setreg->addDestReg(
            StinkyRegister::Hwreg(HwReg::schedModeId(archId_), arb.offset, arb.size));
        setreg->addSrcReg(StinkyRegister(1));
        PASS_DEBUG(std::cerr << "[InsertCoexecHazard]   inserted DISABLE_XDL_ARB_STALL setreg at \""
                             << anchorBB->getLabel() << "\"\n");
    }

    void processFunction(Function& func) {
        // Take ownership of hazard v_nops: strip TensileLite's mode-unaware
        // fillers, guarantee co-exec is OFF, then re-emit correct counts.
        const size_t stripped = stripVNops(func);
        ensureArbStallSetreg(func);
        PASS_DEBUG(std::cerr << "[InsertCoexecHazard] stripped " << stripped
                             << " pre-existing v_nop(s)\n");

        std::unordered_map<const BasicBlock*, bool> entryOff;
        computeCoexecOff(func, entryOff);

        for (BasicBlock& bb : func) {
            bool off = entryOff[&bb];
            for (auto it = bb.begin(); it != bb.end();) {
                auto* inst = dyn_cast<StinkyInstruction>(it.getNodePtr());
                if (!inst || isPseudoInst(inst)) {
                    ++it;
                    continue;
                }

                if (isArbStallSetreg(*inst)) {
                    off = true;
                    ++it;
                    continue;
                }

                int toInsert = 0;
                if (isXDLWMMA(*inst)) {
                    toInsert = std::max(toInsert, hazardFor(bb, *inst, ProducerKind::WMMA,
                                                            /*consumerIsWmma=*/true, off));
                    toInsert = std::max(toInsert, hazardFor(bb, *inst, ProducerKind::TRANS,
                                                            /*consumerIsWmma=*/false, off));
                    // DGEMM/SGEMM (f32/f64 GEMM) -> this XDL WMMA: 1 spacer.
                    toInsert = std::max(toInsert, hazardFor(bb, *inst, ProducerKind::DGEMM,
                                                            /*consumerIsWmma=*/true, off));
                } else if (isCoexecutableVALU(*inst)) {
                    toInsert = std::max(toInsert, hazardFor(bb, *inst, ProducerKind::WMMA,
                                                            /*consumerIsWmma=*/false, off));
                    // TRANS -> core/side is HW-handled; only a TRANS consumer needs
                    // the TRANS -> TRANS spacing.
                    if (isTranscendental(*inst))
                        toInsert = std::max(toInsert, hazardFor(bb, *inst, ProducerKind::TRANS,
                                                                /*consumerIsWmma=*/false, off));
                }

                if (toInsert > 0) {
                    insertVNops(bb, it.getNodePtr(), toInsert);
                    PASS_DEBUG(std::cerr << "[InsertCoexecHazard]   inserted " << toInsert
                                         << " v_nop before " << inst->getHwInstDesc()->mnemonic
                                         << " in bb \"" << bb.getLabel() << "\"" << " (coexec "
                                         << (off ? "OFF" : "ON") << ")\n");
                }
                ++it;
            }
        }
    }

    int hazardFor(BasicBlock& bb, const StinkyInstruction& consumer, ProducerKind kind,
                  bool consumerIsWmma, bool off) {
        ConsumerCtx ctx{kind, consumerIsWmma, &consumer, off};
        // The initial call passes startBefore=&consumer and is not memoized, so on
        // a self-loop the consumer's block can still be re-entered once from its
        // end via the backedge to reach a producer at the loop tail. The
        // minExisting memo bounds all predecessor re-entries (each block is
        // re-scanned only on a strictly smaller filler count).
        std::unordered_map<const BasicBlock*, int> minExisting;
        const int r = scanBack(bb, /*startBefore=*/&consumer, /*accExisting=*/0, ctx, minExisting);
        return r > 0 ? r : 0;
    }

    void insertVNops(BasicBlock& bb, IRBase* insertBefore, int n) {
        AsmIRBuilder builder(bb, archId_);
        for (int i = 0; i < n; ++i) builder.create(getMCIDByUOp(GFX::v_nop, archId_), insertBefore);
    }

    StinkyAsmModule* module_ = nullptr;
    GfxArchID archId_ = GfxArchID{};
    CoexecHazardConfig config_;
};

char InsertCoexecHazardPass::ID = 0;
}  // namespace

namespace stinkytofu {
std::unique_ptr<Pass> createInsertCoexecHazardPass(StinkyAsmModule& module) {
    return std::make_unique<InsertCoexecHazardPass>(&module);
}
std::unique_ptr<Pass> createInsertCoexecHazardPass() {
    return std::make_unique<InsertCoexecHazardPass>(nullptr);
}
}  // namespace stinkytofu
