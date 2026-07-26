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

class WaitAnchoredReadyQueue : public ReadyQueue {
   public:
    /// Create a stable queue using final wait anchors and the region DAG.
    WaitAnchoredReadyQueue(const PassContext& passCtx, const WaitAnchorMap& waitAnchors,
                           RegionDAG& regionDAG)
        : ReadyQueue(passCtx), waitAnchors_(waitAnchors), regionDAG_(regionDAG) {}

    /// Add a ready node to the matrix or non-matrix queue.
    void push(DAGNode* node) override {
        if (isMatrixInstruction(*node->inst))
            wmmaQueue_.insert(node);
        else
            otherQueue_.insert(node);
    }

    /// Select and remove the next node while applying wait-window shortening.
    DAGNode* pickOne() override {
        assert(!wmmaQueue_.empty() || !otherQueue_.empty());

        DAGNode* node = selectNode();
        const bool pickedWmma = removeSelectedNode(node);

        if (pickedWmma)
            onWmmaPicked(*node);
        else
            onOtherPicked();

        return node;
    }

    /// Return true when neither ready queue contains a node.
    bool empty() const override {
        return wmmaQueue_.empty() && otherQueue_.empty();
    }

   private:
    const WaitAnchorMap& waitAnchors_;
    RegionDAG& regionDAG_;
    std::set<DAGNode*, CompareDAGNodeByOriginalOrder> wmmaQueue_;
    std::set<DAGNode*, CompareDAGNodeByOriginalOrder> otherQueue_;
    DAGNode* activeWaitAnchor_ = nullptr;
    unsigned activeWindowStartId_ = 0;
    unsigned originalOtherCount_ = 0;
    unsigned otherPickBudget_ = 0;
    unsigned otherPicksBeforeAnchor_ = 0;

    /// Use the active-window policy when available, otherwise stable order.
    DAGNode* selectNode() const {
        if (DAGNode* windowPick = selectForWaitAnchoredWindow()) return windowPick;
        return peekBaseline();
    }

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

    /// Account for a non-WMMA pick within the active window.
    void onOtherPicked() {
        if (activeWaitAnchor_ == nullptr) return;

        ++otherPicksBeforeAnchor_;
        if (otherPicksBeforeAnchor_ > originalOtherCount_) {
            std::cerr << "[WaitAnchoredReadyQueue pickOne] unable to shorten wait-anchored window: "
                      << "all original instructions are required by the anchor\n";
        }
    }

    /// Close the current window and arm the next wait anchor when present.
    void onWmmaPicked(DAGNode& node) {
        if (activeWaitAnchor_ != nullptr) {
            assert(&node == activeWaitAnchor_ &&
                   "Only the active wait anchor may close a wait-anchored window");
            clearActiveWindow();
        }

        std::cerr << "[WaitAnchoredReadyQueue pickOne] erased WMMA dagId=" << node.id << '\n';

        DAGNode* nextWmma = findNextWmmaInOriginalOrder(node.id);
        if (nextWmma == nullptr) return;

        const bool isWaitAnchor = waitAnchors_.contains(nextWmma->inst);
        std::cerr << "[WaitAnchoredReadyQueue pickOne] next original WMMA dagId=" << nextWmma->id
                  << " waitAnchor=" << (isWaitAnchor ? "true" : "false") << '\n';
        if (isWaitAnchor) armWaitAnchoredWindow(node, *nextWmma);
    }

    /// Initialize shortening state for a wait-anchored WMMA interval.
    void armWaitAnchoredWindow(const DAGNode& currentWmma, DAGNode& waitAnchor) {
        activeWaitAnchor_ = &waitAnchor;
        activeWindowStartId_ = currentWmma.id;
        originalOtherCount_ = waitAnchor.id - currentWmma.id - 1;
        otherPickBudget_ = originalOtherCount_ > 0 ? originalOtherCount_ - 1 : 0;
        otherPicksBeforeAnchor_ = 0;
        std::cerr << "[WaitAnchoredReadyQueue pickOne] armed wait anchor dagId=" << waitAnchor.id
                  << " originalOtherCount=" << originalOtherCount_
                  << " otherPickBudget=" << otherPickBudget_ << '\n';
    }

    /// Reset the active wait-window state.
    void clearActiveWindow() {
        activeWaitAnchor_ = nullptr;
        activeWindowStartId_ = 0;
        originalOtherCount_ = 0;
        otherPickBudget_ = 0;
        otherPicksBeforeAnchor_ = 0;
    }

    /// Test whether a node originally lies inside the active interval.
    bool isInsideActiveWindow(const DAGNode& node) const {
        return activeWaitAnchor_ != nullptr && node.id > activeWindowStartId_ &&
               node.id < activeWaitAnchor_->id;
    }

    /// Test whether a node produces a counter named by the active wait.
    bool affectsActiveWait(const DAGNode& node) const {
        if (!isInsideActiveWindow(node)) return false;
        const auto anchor = waitAnchors_.find(activeWaitAnchor_->inst);
        assert(anchor != waitAnchors_.end());
        const waitcnt::CounterKind kind = waitcnt::classifyMemOp(*node.inst);
        return kind != waitcnt::CK_Count && counterApplies(kind, anchor->second.spec);
    }

    /// Test whether a DAG path connects a node to the active anchor.
    bool reachesActiveWaitAnchor(const DAGNode& start) const {
        if (activeWaitAnchor_ == nullptr) return false;

        std::vector<unsigned> worklist{start.id};
        std::vector<bool> visited(regionDAG_.nodes.size(), false);
        visited[start.id] = true;

        while (!worklist.empty()) {
            const unsigned id = worklist.back();
            worklist.pop_back();
            for (unsigned succId : regionDAG_.graph[id]) {
                if (succId == activeWaitAnchor_->id) return true;
                if (!visited[succId] && succId < activeWaitAnchor_->id) {
                    visited[succId] = true;
                    worklist.push_back(succId);
                }
            }
        }
        return false;
    }

    /// Find the earliest ready producer required by the active wait.
    DAGNode* findReadyWaitAffectingNode() const {
        for (DAGNode* node : otherQueue_) {
            if (affectsActiveWait(*node)) return node;
        }
        return nullptr;
    }

    /// Find ready dependency work needed to make the anchor ready.
    DAGNode* findReadyAnchorPredecessor() const {
        for (DAGNode* node : otherQueue_) {
            if (isInsideActiveWindow(*node) && reachesActiveWaitAnchor(*node)) return node;
        }
        return nullptr;
    }

    /// Find stable ready work before the anchor, including carried work.
    DAGNode* findReadyOtherBeforeAnchor() const {
        if (activeWaitAnchor_ == nullptr) return nullptr;
        for (DAGNode* node : otherQueue_) {
            // This includes carried-over work from the previous WMMA window.
            if (node->id < activeWaitAnchor_->id) return node;
        }
        return nullptr;
    }

    /// Apply the active window budget and mandatory-counter policy.
    DAGNode* selectForWaitAnchoredWindow() const {
        if (activeWaitAnchor_ == nullptr) return nullptr;

        // Fill exactly one fewer non-WMMA slot than existed in the original
        // window. Carried-over work counts against this budget.
        if (otherPicksBeforeAnchor_ < otherPickBudget_) {
            if (DAGNode* other = findReadyOtherBeforeAnchor()) return other;
        }

        // Counter producers named by the wait are mandatory even if they exceed
        // the shortening budget.
        if (DAGNode* waitAffecting = findReadyWaitAffectingNode()) return waitAffecting;

        auto readyAnchor = wmmaQueue_.find(activeWaitAnchor_);
        if (readyAnchor != wmmaQueue_.end()) return *readyAnchor;

        // Keep selecting only dependency-path work needed to make the anchor ready.
        return findReadyAnchorPredecessor();
    }

    /// Find the next WMMA in original DAG order.
    DAGNode* findNextWmmaInOriginalOrder(unsigned currentId) const {
        for (unsigned id = currentId + 1; id < regionDAG_.nodes.size(); ++id) {
            DAGNode& candidate = regionDAG_.nodes[id];
            if (isMatrixInstruction(*candidate.inst)) return &candidate;
        }
        return nullptr;
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
