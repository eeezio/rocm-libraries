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

#include "stinkytofu/transforms/asm/WaitAwareScheduleRepairPass.hpp"

#include <cassert>
#include <unordered_set>

#include "dag/RegionDAG.hpp"
#include "dag/WaitAnchoredReadyQueue.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"

#define DEBUG_TYPE "WaitAwareScheduleRepairPass"

namespace {
using namespace stinkytofu;
using namespace stinkytofu::dag;
using namespace stinkytofu::waitcnt;

bool isAnyWaitCnt(const StinkyInstruction& inst) {
    return isWaitCnt(inst) || inst.is(InstFlag::IF_WaitTensorCnt);
}

WaitCountSpec decodeWaitSpec(const StinkyInstruction& wait) {
    WaitCountSpec spec;
    if (const auto* data = wait.getModifier<SWaitCntData>()) {
        if (data->dlcnt >= 0) spec.dsCount = data->dlcnt;
        if (data->vlcnt >= 0) spec.bufferCount = data->vlcnt;
        if (data->kmcnt >= 0) spec.kmCount = data->kmcnt;
    }
    if (const auto* tdata = wait.getModifier<SWaitTensorCntData>()) {
        if (tdata->tlcnt >= 0) spec.tensorCount = tdata->tlcnt;
    }
    return spec;
}

WaitCountSpec mergeWaitSpecs(const WaitCountSpec& a, const WaitCountSpec& b) {
    WaitCountSpec out = a;
    if (b.dsCount != WaitCountSpec::kUnused) out.dsCount = b.dsCount;
    if (b.bufferCount != WaitCountSpec::kUnused) out.bufferCount = b.bufferCount;
    if (b.kmCount != WaitCountSpec::kUnused) out.kmCount = b.kmCount;
    if (b.tensorCount != WaitCountSpec::kUnused) out.tensorCount = b.tensorCount;
    return out;
}

WaitAnchorMap discoverWaitAnchors(const std::vector<StinkyInstruction*>& seq) {
    WaitAnchorMap anchors;
    for (size_t i = 0; i < seq.size(); ++i) {
        if (!isAnyWaitCnt(*seq[i])) continue;

        size_t waitStart = i;
        size_t waitEnd = waitStart + 1;
        while (waitEnd < seq.size() && isAnyWaitCnt(*seq[waitEnd])) ++waitEnd;

        if (waitEnd >= seq.size() || !isMatrixInstruction(*seq[waitEnd])) {
            i = waitEnd;
            continue;
        }

        WaitAnchorInfo info;
        info.anchor = seq[waitEnd];
        WaitCountSpec combined;
        for (size_t w = waitStart; w < waitEnd; ++w) {
            info.waits.push_back(seq[w]);
            combined = mergeWaitSpecs(combined, decodeWaitSpec(*seq[w]));
        }
        info.spec = combined;
        anchors[info.anchor] = std::move(info);
        i = waitEnd;
    }
    return anchors;
}

std::unordered_set<StinkyInstruction*> collectAttachedWaits(const WaitAnchorMap& anchors) {
    std::unordered_set<StinkyInstruction*> attached;
    for (const auto& [anchor, info] : anchors) {
        (void)anchor;
        for (StinkyInstruction* wait : info.waits) attached.insert(wait);
    }
    return attached;
}

bool isHardBoundary(const StinkyInstruction& inst,
                    const std::unordered_set<StinkyInstruction*>& attachedWaits) {
    if (isLabel(inst)) return true;
    if (isAnyWaitCnt(inst) && attachedWaits.count(const_cast<StinkyInstruction*>(&inst)) == 0)
        return true;
    if (hasSideEffect(inst)) return true;
    if (isExecMaskGroup(inst)) return true;
    return false;
}

std::vector<StinkyInstruction*> repairSegment(const std::vector<StinkyInstruction*>& instructions,
                                              const WaitAnchorMap& anchors,
                                              const PassContext& passCtx) {
    if (instructions.empty()) return {};

    RegionDAG dag = buildRegisterDependencyDAG(instructions);
    addCounterOrderEdges(dag, instructions, anchors);

    WaitAnchoredReadyQueue queue(passCtx, anchors, dag);
    std::vector<StinkyInstruction*> scheduled = scheduleWithWaitAnchoredReadyQueue(dag, queue);
    assert(scheduled.size() == instructions.size() &&
           "Repair schedule must include every segment instruction exactly once");
    return scheduled;
}

void emitInstWithWaits(std::vector<IRBase*>& output, StinkyInstruction* inst,
                       const WaitAnchorMap& anchors) {
    auto it = anchors.find(inst);
    if (it != anchors.end()) {
        for (StinkyInstruction* wait : it->second.waits) output.push_back(wait);
    }
    output.push_back(inst);
}

void repairBlock(BasicBlock& bb, const PassContext& passCtx) {
    std::vector<StinkyInstruction*> seq;
    seq.reserve(bb.size());
    for (IRBase& ir : bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) continue;
        seq.push_back(cast<StinkyInstruction>(&ir));
    }
    if (seq.empty()) return;

    const WaitAnchorMap anchors = discoverWaitAnchors(seq);
    const std::unordered_set<StinkyInstruction*> attachedWaits = collectAttachedWaits(anchors);

    std::vector<IRBase*> output;
    output.reserve(bb.size());

    std::vector<StinkyInstruction*> segment;
    segment.reserve(seq.size());

    auto flushSegment = [&]() {
        if (segment.empty()) return;
        const std::vector<StinkyInstruction*> repaired = repairSegment(segment, anchors, passCtx);
        for (StinkyInstruction* inst : repaired) emitInstWithWaits(output, inst, anchors);
        segment.clear();
    };

    for (IRBase& ir : bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) {
            flushSegment();
            output.push_back(&ir);
            continue;
        }

        auto* inst = cast<StinkyInstruction>(&ir);
        if (attachedWaits.count(inst) != 0) continue;

        if (isHardBoundary(*inst, attachedWaits)) {
            flushSegment();
            output.push_back(inst);
            continue;
        }

        segment.push_back(inst);
    }
    flushSegment();

    assert(output.size() == bb.size() && "Repair must preserve instruction count");

    for (IRBase* ir : output) {
        bb.removeIR(ir);
        bb.appendIR(ir);
    }
}

class WaitAwareScheduleRepairPass : public StinkyInstPass {
   public:
    static char ID;

    const char* getName() const override {
        return "WaitAwareScheduleRepairPass";
    }

    PassID getPassID() const override {
        return &WaitAwareScheduleRepairPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        (void)AM;
        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;
            repairBlock(bb, passCtx);
        }
        return PreservedAnalyses::none();
    }
};

char WaitAwareScheduleRepairPass::ID = 0;

}  // namespace

namespace stinkytofu {

std::unique_ptr<Pass> createWaitAwareScheduleRepairPass() {
    return std::make_unique<WaitAwareScheduleRepairPass>();
}

}  // namespace stinkytofu
