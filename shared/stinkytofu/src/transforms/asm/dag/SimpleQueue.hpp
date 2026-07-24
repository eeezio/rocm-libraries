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

#include <cassert>
#include <set>
#include <vector>

#include "ReadyQueue.hpp"
#include "RegionDAG.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitDataflow.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitPlan.hpp"

namespace stinkytofu {
namespace dag {

struct WaitAnchorInfo {
    StinkyInstruction* anchor = nullptr;
    std::vector<StinkyInstruction*> waits;
    waitcnt::WaitCountSpec spec;
};

using WaitAnchorMap = std::unordered_map<StinkyInstruction*, WaitAnchorInfo>;

inline bool counterApplies(waitcnt::CounterKind kind, const waitcnt::WaitCountSpec& spec) {
    using waitcnt::WaitCountSpec;
    switch (kind) {
        case waitcnt::CK_DS:
            return spec.dsCount != WaitCountSpec::kUnused;
        case waitcnt::CK_Buffer:
            return spec.bufferCount != WaitCountSpec::kUnused;
        case waitcnt::CK_KM:
            return spec.kmCount != WaitCountSpec::kUnused;
        case waitcnt::CK_Tensor:
            return spec.tensorCount != WaitCountSpec::kUnused;
        default:
            return false;
    }
}

/// Preserve the meaning of final wait immediates by ordering each counter's
/// producers and wait anchors in their original sequence.
inline void addCounterOrderEdges(RegionDAG& dag,
                                 const std::vector<StinkyInstruction*>& instructions,
                                 const WaitAnchorMap& anchors) {
    using waitcnt::CK_Count;
    using waitcnt::CounterKind;

    for (int ck = 0; ck < CK_Count; ++ck) {
        const auto kind = static_cast<CounterKind>(ck);
        std::vector<unsigned> events;
        events.reserve(instructions.size());

        for (unsigned i = 0; i < instructions.size(); ++i) {
            StinkyInstruction* inst = instructions[i];
            if (waitcnt::classifyMemOp(*inst) == kind) {
                events.push_back(i);
                continue;
            }

            auto anchor = anchors.find(inst);
            if (anchor != anchors.end() && counterApplies(kind, anchor->second.spec))
                events.push_back(i);
        }

        for (size_t i = 1; i < events.size(); ++i) {
            addEdgeById(&dag.nodes[events[i - 1]], &dag.nodes[events[i]], dag.graph);
        }
    }
}

struct CompareDAGNodeByOriginalOrder {
    bool operator()(const DAGNode* a, const DAGNode* b) const {
        return a->id < b->id;
    }
};

class SimpleQueue : public ReadyQueue {
   public:
    explicit SimpleQueue(const PassContext& passCtx) : ReadyQueue(passCtx) {}

    void push(DAGNode* node) override {
        if (isMatrixInstruction(*node->inst))
            wmmaQueue_.insert(node);
        else
            otherQueue_.insert(node);
    }

    DAGNode* pickOne() override {
        assert(!wmmaQueue_.empty() || !otherQueue_.empty());
        DAGNode* node = peekBaseline();
        wmmaQueue_.erase(node);
        otherQueue_.erase(node);
        return node;
    }

    bool empty() const override {
        return wmmaQueue_.empty() && otherQueue_.empty();
    }

   private:
    std::set<DAGNode*, CompareDAGNodeByOriginalOrder> wmmaQueue_;
    std::set<DAGNode*, CompareDAGNodeByOriginalOrder> otherQueue_;

    DAGNode* peekBaseline() {
        if (wmmaQueue_.empty()) return *otherQueue_.begin();
        if (otherQueue_.empty()) return *wmmaQueue_.begin();
        DAGNode* w = *wmmaQueue_.begin();
        DAGNode* o = *otherQueue_.begin();
        return w->id < o->id ? w : o;
    }
};

inline std::vector<StinkyInstruction*> scheduleWithSimpleQueue(RegionDAG& dag, SimpleQueue& queue) {
    std::vector<StinkyInstruction*> scheduled;
    scheduled.reserve(dag.nodes.size());

    for (DAGNode& node : dag.nodes) {
        if (node.inDegree == 0) queue.push(&node);
    }

    while (!queue.empty()) {
        DAGNode* node = queue.pickOne();
        scheduled.push_back(node->inst);
        for (unsigned succId : dag.graph[node->id]) {
            DAGNode& succ = dag.nodes[succId];
            if (--succ.inDegree == 0) queue.push(&succ);
        }
    }

    return scheduled;
}

}  // namespace dag
}  // namespace stinkytofu
