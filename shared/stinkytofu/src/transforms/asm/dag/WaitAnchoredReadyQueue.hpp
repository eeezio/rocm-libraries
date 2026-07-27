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
#include <iostream>
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

using OrderedReadyNodeSet = std::set<DAGNode*, CompareDAGNodeByOriginalOrder>;

/// Selection policy for shortening windows immediately before wait-anchored WMMAs.
///
/// All tuning state and decisions live here so ReadyQueue mechanics remain
/// independent from the repair heuristic.
class WaitAnchoredPickPolicy {
   public:
    WaitAnchoredPickPolicy(const WaitAnchorMap& waitAnchors, RegionDAG& regionDAG)
        : waitAnchors_(waitAnchors), regionDAG_(regionDAG) {}

    /// Return a policy-selected node, or nullptr to request stable baseline order.
    DAGNode* select(const OrderedReadyNodeSet& wmmaQueue,
                    const OrderedReadyNodeSet& otherQueue) const {
        if (!window_.active()) return nullptr;

        // Fill one fewer non-WMMA slot than the original window. Work carried
        // from the preceding window counts against this budget.
        if (window_.otherPicks < window_.otherPickBudget) {
            if (DAGNode* node = findReadyOtherBeforeAnchor(otherQueue)) return node;
        }

        // Producers named by the final wait are mandatory even when they exceed
        // the shortening budget.
        if (DAGNode* node = findReadyWaitAffectingNode(otherQueue)) return node;

        auto readyAnchor = wmmaQueue.find(window_.anchor);
        if (readyAnchor != wmmaQueue.end()) return *readyAnchor;

        // Keep selecting only dependency-path work needed to unlock the anchor.
        return findReadyAnchorPredecessor(otherQueue);
    }

    /// Update policy state after the queue commits a selected node.
    void onPicked(DAGNode& node, bool isWmma) {
        if (isWmma)
            onWmmaPicked(node);
        else
            onOtherPicked();
    }

   private:
    /// State for the currently active interval ending at a wait-anchored WMMA.
    struct WindowState {
        DAGNode* anchor = nullptr;
        const WaitAnchorInfo* anchorInfo = nullptr;
        unsigned startId = 0;
        unsigned originalOtherCount = 0;
        unsigned otherPickBudget = 0;
        unsigned otherPicks = 0;

        bool active() const {
            return anchor != nullptr;
        }

        void reset() {
            *this = {};
        }
    };

    // Move this many otherwise-eligible slots past each wait anchor.
    static constexpr unsigned kSlotsToMovePastAnchor = 1;

    const WaitAnchorMap& waitAnchors_;
    RegionDAG& regionDAG_;
    WindowState window_;

    /// Account for a non-WMMA pick within the active window.
    void onOtherPicked() {
        if (!window_.active()) return;

        ++window_.otherPicks;
        if (window_.otherPicks > window_.originalOtherCount) {
            std::cerr << "[WaitAnchoredReadyQueue pickOne] unable to shorten wait-anchored window: "
                      << "all original instructions are required by the anchor\n";
        }
    }

    /// Close the current window and arm the next wait anchor when present.
    void onWmmaPicked(DAGNode& node) {
        if (window_.active()) {
            assert(&node == window_.anchor &&
                   "Only the active wait anchor may close a wait-anchored window");
            window_.reset();
        }

        std::cerr << "[WaitAnchoredReadyQueue pickOne] erased WMMA dagId=" << node.id << '\n';

        DAGNode* nextWmma = findNextWmmaInOriginalOrder(node.id);
        if (nextWmma == nullptr) return;

        auto waitAnchor = waitAnchors_.find(nextWmma->inst);
        const bool isWaitAnchor = waitAnchor != waitAnchors_.end();
        std::cerr << "[WaitAnchoredReadyQueue pickOne] next original WMMA dagId=" << nextWmma->id
                  << " waitAnchor=" << (isWaitAnchor ? "true" : "false") << '\n';
        if (isWaitAnchor) armWindow(node, *nextWmma, waitAnchor->second);
    }

    /// Initialize shortening state for a wait-anchored WMMA interval.
    void armWindow(const DAGNode& currentWmma, DAGNode& waitAnchor,
                   const WaitAnchorInfo& anchorInfo) {
        window_.anchor = &waitAnchor;
        window_.anchorInfo = &anchorInfo;
        window_.startId = currentWmma.id;
        window_.originalOtherCount = waitAnchor.id - currentWmma.id - 1;
        window_.otherPickBudget = window_.originalOtherCount > kSlotsToMovePastAnchor
                                      ? window_.originalOtherCount - kSlotsToMovePastAnchor
                                      : 0;
        window_.otherPicks = 0;
        std::cerr << "[WaitAnchoredReadyQueue pickOne] armed wait anchor dagId=" << waitAnchor.id
                  << " originalOtherCount=" << window_.originalOtherCount
                  << " otherPickBudget=" << window_.otherPickBudget << '\n';
    }

    /// Test whether a node originally lies inside the active interval.
    bool isInsideActiveWindow(const DAGNode& node) const {
        return window_.active() && node.id > window_.startId && node.id < window_.anchor->id;
    }

    /// Test whether a node produces a counter named by the active wait.
    bool affectsActiveWait(const DAGNode& node) const {
        if (!isInsideActiveWindow(node)) return false;
        assert(window_.anchorInfo != nullptr);
        const waitcnt::CounterKind kind = waitcnt::classifyMemOp(*node.inst);
        return kind != waitcnt::CK_Count && counterApplies(kind, window_.anchorInfo->spec);
    }

    /// Test whether a DAG path connects a node to the active anchor.
    bool reachesActiveWaitAnchor(const DAGNode& start) const {
        if (!window_.active()) return false;

        std::vector<unsigned> worklist{start.id};
        std::vector<bool> visited(regionDAG_.nodes.size(), false);
        visited[start.id] = true;

        while (!worklist.empty()) {
            const unsigned id = worklist.back();
            worklist.pop_back();
            for (unsigned succId : regionDAG_.graph[id]) {
                if (succId == window_.anchor->id) return true;
                if (!visited[succId] && succId < window_.anchor->id) {
                    visited[succId] = true;
                    worklist.push_back(succId);
                }
            }
        }
        return false;
    }

    /// Find the earliest ready producer required by the active wait.
    DAGNode* findReadyWaitAffectingNode(const OrderedReadyNodeSet& otherQueue) const {
        for (DAGNode* node : otherQueue) {
            if (affectsActiveWait(*node)) return node;
        }
        return nullptr;
    }

    /// Find ready dependency work needed to make the anchor ready.
    DAGNode* findReadyAnchorPredecessor(const OrderedReadyNodeSet& otherQueue) const {
        for (DAGNode* node : otherQueue) {
            if (isInsideActiveWindow(*node) && reachesActiveWaitAnchor(*node)) return node;
        }
        return nullptr;
    }

    /// Find stable ready work before the anchor, including carried work.
    DAGNode* findReadyOtherBeforeAnchor(const OrderedReadyNodeSet& otherQueue) const {
        if (!window_.active()) return nullptr;
        for (DAGNode* node : otherQueue) {
            // This includes carried-over work from the previous WMMA window.
            if (node->id < window_.anchor->id) return node;
        }
        return nullptr;
    }

    /// Find the next WMMA in original DAG order.
    DAGNode* findNextWmmaInOriginalOrder(unsigned currentId) const {
        for (unsigned id = currentId + 1; id < regionDAG_.nodes.size(); ++id) {
            DAGNode& candidate = regionDAG_.nodes[id];
            if (isMatrixInstruction(*candidate.inst)) return &candidate;
        }
        return nullptr;
    }
};

class WaitAnchoredReadyQueue : public ReadyQueue {
   public:
    /// Create a stable queue using final wait anchors and the region DAG.
    WaitAnchoredReadyQueue(const PassContext& passCtx, const WaitAnchorMap& waitAnchors,
                           RegionDAG& regionDAG)
        : ReadyQueue(passCtx), policy_(waitAnchors, regionDAG) {}

    /// Add a ready node to the matrix or non-matrix queue.
    void push(DAGNode* node) override {
        if (isMatrixInstruction(*node->inst))
            wmmaQueue_.insert(node);
        else
            otherQueue_.insert(node);
    }

    /// Select and remove the next node, delegating tuning to the pick policy.
    DAGNode* pickOne() override {
        assert(!empty());

        DAGNode* node = policy_.select(wmmaQueue_, otherQueue_);
        if (node == nullptr) node = peekBaseline();

        const bool pickedWmma = removeSelectedNode(node);
        policy_.onPicked(*node, pickedWmma);
        return node;
    }

    bool empty() const override {
        return wmmaQueue_.empty() && otherQueue_.empty();
    }

   private:
    OrderedReadyNodeSet wmmaQueue_;
    OrderedReadyNodeSet otherQueue_;
    WaitAnchoredPickPolicy policy_;

    /// Remove a node from its queue and report whether it is a WMMA.
    bool removeSelectedNode(DAGNode* node) {
        if (isMatrixInstruction(*node->inst)) {
            const size_t erased = wmmaQueue_.erase(node);
            assert(erased == 1 && "Selected WMMA must be in the WMMA ready queue");
            return true;
        }

        const size_t erased = otherQueue_.erase(node);
        assert(erased == 1 && "Selected non-WMMA must be in the other ready queue");
        return false;
    }

    /// Pick the smallest original ID across both ready queues.
    DAGNode* peekBaseline() const {
        if (wmmaQueue_.empty()) return *otherQueue_.begin();
        if (otherQueue_.empty()) return *wmmaQueue_.begin();
        DAGNode* w = *wmmaQueue_.begin();
        DAGNode* o = *otherQueue_.begin();
        return w->id < o->id ? w : o;
    }
};

inline std::vector<StinkyInstruction*> scheduleWithWaitAnchoredReadyQueue(
    RegionDAG& dag, WaitAnchoredReadyQueue& queue) {
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
