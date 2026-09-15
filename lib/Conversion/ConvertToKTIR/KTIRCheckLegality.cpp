//===- KTIRCheckLegality.cpp - KTIR legality check ---------------*- C++ -*-===//
//
// Copyright 2026 The KTIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the KTIR legality verification pass (issue #35).
//
// Inter-tile rules
// ================
//
// The rules are numbered R1-R14 in docs/inter-tile-communication.md §5, which
// is their single point of definition; the numbering here refers to it. Which
// rules apply is a property of the *delivery op*, so the same rule can mean
// different things -- or nothing -- from one op to the next. This is the
// applicability matrix of §5, restricted to the ops that exist today:
//
//   Rule  Owner     consume  reduce   | statement
//   ----  --------  -------  -------  | ------------------------------------
//   R1    produce   y        y        | groups are pairwise disjoint
//   R2    produce   y        y        | the future has exactly one use
//   R3    delivery  y        y        | declared deps are real producers
//   R4    delivery  y        y        | every producer is claimed by someone
//   R8    delivery  y        --       | one source per consumer tile
//   R13   delivery  --       y        | consumer set is a subset of producers
//   R14   delivery  --       y        | reduce mode gate: C == P or |C| == 1
//
// Rules absent from the table above (R5-R7, R9-R12) belong to ops not yet
// built -- the assembling and splitting placements -- and are unimplemented.
// R13/R14 are `--` for consume deliberately, not by omission: a broadcast
// consumer contributes nothing, so it need not be a producer (§10.1, resolved
// for consume and scatter; still open for the assembling ops).
//
// Where each check lives
// ----------------------
//
// R1 and R2 are owned by `inter_tile_produce`, so they are checked once on the
// produce op and hold for whichever delivery op consumes the future -- there is
// nothing per-op to do. R1 needs only that op's own attribute, so it lives in
// InterTileProduceOp::verify (KTDPOps.cpp); R2 needs the result's use-list,
// which a verifier may not inspect, so it is here.
//
// Everything else is owned by a delivery op and needs both sides of the
// def-use edge: `consumer_tiles_per_group` and the dependency set are on the
// delivery op, `producer_tiles_per_group` is on the produce op. That is why
// these checks are in this pass rather than in an op verifier, which can only
// see one op. Checks needing just one op's attributes stay in KTDPOps.cpp.
//
// Static enumeration and its limits
// ---------------------------------
//
// The tile sets are parameterized affine sets -- `(i)[g]` over tile ids, and
// `(p)[c]` or `(p)[c, g]` for dependencies -- so every check first grounds
// them at concrete group values via the helpers in KTDPInterTileHelpers.h.
// Grounding fails when a bound is not statically known, and then the honest
// answer is to check nothing rather than to guess: an unbounded group range
// returns, an unbounded tile range continues to the next group.
//
// Skipping is only safe for a check that rejects on evidence it *has*. Most
// are: they name a tile they read and found wrong, and an unread tile cannot
// make that tile right. R4 is the exception -- it rejects a producer for being
// *absent* from the union of what every consumer claims, so a partial read
// makes it accuse the innocent. That is why the per-consumer read returns
// FailureOr<optional<...>> -- failure for a violation, nullopt for "could not
// read" -- and R4 is a separate function taking the finished union, called only
// when every consumer contributed to it. See declaredProducersOf and
// checkEveryProducerIsClaimed.
//
//===----------------------------------------------------------------------===//

#include "ktir/Conversion/ConvertToKTIR/ConvertToKTIR.h"
#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ktir/Dialect/KTDP/KTDPInterTileHelpers.h"

namespace ktir {
#define GEN_PASS_DEF_KTIRCHECKLEGALITYPASS
#include "ktir/Conversion/Passes.h.inc"
} // namespace ktir

void ktir::populateKTIRLegalTarget(mlir::ConversionTarget &target) {
  target.addLegalDialect<mlir::ktdp::KtdpDialect>();
  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addLegalDialect<mlir::func::FuncDialect>();
  target.addLegalDialect<mlir::linalg::LinalgDialect>();
  target.addLegalDialect<mlir::math::MathDialect>();
  target.addLegalDialect<mlir::memref::MemRefDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addLegalDialect<mlir::tensor::TensorDialect>();
}

namespace ktir {
namespace {
using namespace mlir;
using namespace mlir::ktdp;

/// Checks R3 for one consumer: every producer tile it declares a dependency on
/// must be a producer. Returns the tiles it declared, which the caller unions
/// across the group for R4.
///
/// Returns std::nullopt when the dependency set is not statically enumerable --
/// grounding it at this consumer left the producer-tile range unbounded. That
/// is not a violation: there is no declared tile to find fault with. It does
/// mean the consumer contributes nothing to the group's union, which is why the
/// two cases are distinguishable here and why R4 checks for it.
///
/// Fails, rather than returning std::nullopt, when a tile the consumer *did*
/// declare is not a producer, and emits the diagnostic on \p op before
/// returning -- a dependency on a tile that never produces would wait forever.
[[nodiscard]] mlir::FailureOr<std::optional<llvm::DenseSet<int64_t>>>
declaredProducersOf(mlir::Operation* op, IntegerSet depSet, int64_t consumer,
                    int64_t group,
                    const llvm::DenseSet<int64_t>& producers) {
  auto declared = depTilesOf(depSet, consumer, group);
  if (mlir::failed(declared)) return std::optional<llvm::DenseSet<int64_t>>();

  for (int64_t p : *declared) {
    if (!producers.count(p)) {
      op->emitError("producer_dependency_per_consumer for consumer ")
          << consumer << " group " << group << " references producer tile " << p
          << " which is not in producer_tiles_per_group";
      return mlir::failure();
    }
  }
  return std::optional<llvm::DenseSet<int64_t>>(std::move(*declared));
}

// Checks R4 for one group: every producer must be claimed by some consumer, so
// no tile yields a value nobody reads -- a deadlock risk in a push-based
// lowering. Emits on `op` and returns false if not.
//
// `claimed` must be the union over *every* consumer in the group. This check
// faults a producer for being absent from it, which an incomplete union would
// make an accusation against a tile some unread consumer claims; the caller
// only calls this when the union is complete.
[[nodiscard]] bool checkEveryProducerIsClaimed(mlir::Operation* op,
                                               int64_t group,
                                               const llvm::DenseSet<int64_t>& producers,
                                               const llvm::DenseSet<int64_t>& claimed) {
  for (int64_t p : producers) {
    if (!claimed.count(p)) {
      op->emitError("producer_dependency_per_consumer for group ")
          << group << " does not cover producer tile " << p
          << " (no consumer has it as a dependency)";
      return false;
    }
  }
  return true;
}

// R3 and R4 on `producer_dependency_per_consumer`, per group. Shared by every
// delivery op that accepts the attribute -- today `reduce` and `consume`, later
// the assembling ops -- because both rules read the same three sets and say the
// same thing for all of them. `op` carries the diagnostics so the wording is
// identical whichever op called in.
//
// The two rules have different shapes, which is why they are separate functions
// above rather than one pass over the group:
//
//   R3 is per consumer. Each consumer's declared tiles are checked against the
//      producer set on their own; no other consumer is involved.
//   R4 is per group. It reads the union of every consumer's claims, so it can
//      only run once that union is known to be complete.
//
// Deliberately *not* checked here, because they are per-op rather than shared:
// R8 (consume: one source per consumer tile), R13/R14 (reduce: consumer subset
// and mode gate), and R5-R7 (the assembling ops' disjointness and uniformity,
// unimplemented). Callers own those.
[[nodiscard]] bool checkDepSetSubsetAndCoverage(mlir::Operation* op,
                                                IntegerSet depSet,
                                                IntegerSet consumerSet,
                                                IntegerSet producerSet,
                                                llvm::ArrayRef<int64_t> groupVals) {
  // Two symbols (c, g) is the general spelling; one symbol (c) is legal
  // when the producer/consumer pairing is group-independent.
  if (depSet.getNumSymbols() != 1 && depSet.getNumSymbols() != 2) {
    op->emitError("`producer_dependency_per_consumer` must have "
                  "one symbol (c) or two symbols (c, g)");
    return false;
  }

  for (int64_t g : groupVals) {
    auto consumersOpt = tilesOf(consumerSet, g);
    auto producersOpt = tilesOf(producerSet, g);
    if (mlir::failed(consumersOpt) || mlir::failed(producersOpt)) continue;

    // R3 on each consumer, unioning what they declare for R4.
    llvm::DenseSet<int64_t> claimed;
    bool unionIsComplete = true;
    for (int64_t c : *consumersOpt) {
      auto declared = declaredProducersOf(op, depSet, c, g, *producersOpt);
      if (mlir::failed(declared)) return false; // R3 violated, already emitted
      if (!declared->has_value()) {
        unionIsComplete = false; // nothing readable for this consumer
        continue;
      }
      claimed.insert((*declared)->begin(), (*declared)->end());
    }

    // R4 on the group, once every consumer has contributed to `claimed`.
    if (unionIsComplete &&
        !checkEveryProducerIsClaimed(op, g, *producersOpt, claimed))
      return false;
  }
  return true;
}

struct KTIRCheckLegalityPass
    : impl::KTIRCheckLegalityPassBase<KTIRCheckLegalityPass> {

  void runOnOperation() override {
    bool failed = false;

    // --- Static legality via ConversionTarget + applyPartialConversion ---
    ConversionTarget target(getContext());
    populateKTIRLegalTarget(target);

    mlir::ConversionConfig config;
    mlir::DenseSet<mlir::Operation *> unlegalizedOps;
    config.unlegalizedOps = &unlegalizedOps;

    mlir::FrozenRewritePatternSet emptyPatterns;
    if (mlir::failed(mlir::applyPartialConversion(
            getOperation(), target, emptyPatterns, config)))
      failed = true;

    // --- Cross-op inter-tile invariants via IR walk ---

    // Single-use invariant on inter_tile_produce (§2.3): the future result
    // must have exactly one use — the single delivery op that consumes it.
    getOperation()->walk([&](InterTileProduceOp produceOp) {
      if (!produceOp.getFuture().hasOneUse()) {
        produceOp.emitError("future result must have exactly one use");
        failed = true;
      }
    });

    // inter_tile_reduce: R13 (consumer set is a subset of the producer set),
    // R14 (mode gate), then R3/R4 on the dependency set. R13/R14 are what make
    // reduce stricter than consume -- a tile that consumes a reduced value is
    // required to have contributed to it (§10.1 for reduce is answered "yes"
    // here, and that answer is this op's alone).
    getOperation()->walk([&](InterTileReduceOp reduceOp) {
      auto produceOp =
          reduceOp.getFuture().getDefiningOp<InterTileProduceOp>();
      if (!produceOp) return; // dynamic future — cannot compare statically

      IntegerSet groupsSet   = reduceOp.getGroups();
      IntegerSet consumerSet = reduceOp.getConsumerTilesPerGroup().getValue();
      IntegerSet producerSet = produceOp.getProducerTilesPerGroup().getValue();

      auto groupVals = groupValues(groupsSet);
      if (mlir::failed(groupVals)) return; // unbounded group range — defer

      for (int64_t g : *groupVals) {
        auto cOpt = tilesOf(consumerSet, g);
        auto pOpt = tilesOf(producerSet, g);
        if (mlir::failed(cOpt) || mlir::failed(pOpt)) continue;
        const auto &c = *cOpt;
        const auto &p = *pOpt;

        // C⊆P check: every consumer tile must be a producer tile.
        for (int64_t tile : c) {
          if (!p.count(tile)) {
            reduceOp.emitError("consumer_tiles_per_group for group ")
                << g << " is not a subset of producer_tiles_per_group "
                << "(a consumer tile that did not produce is unsupported; "
                << "see open question Q1)";
            failed = true;
            return;
          }
        }

        // Mode gate: only all-reduce (C == P) or reduce-to-one (|C| == 1).
        if (c == p) continue;
        if (c.size() == 1) continue;
        reduceOp.emitError("consumer_tiles_per_group for group ")
            << g << " is a strict subset of producer_tiles_per_group with "
            << "more than one tile (reduce-to-subset is unsupported; only "
            << "all-reduce and reduce-to-one are supported)";
        failed = true;
        return;
      }

      // producer_dependency_per_consumer checks (R3, R4), when present.
      auto depAttr = reduceOp.getProducerDependencyPerConsumerAttr();
      if (!depAttr) return;

      if (!checkDepSetSubsetAndCoverage(reduceOp, depAttr.getValue(),
                                        consumerSet, producerSet, *groupVals))
        failed = true;
    });

    // inter_tile_consume: R8 (single-source delivery), then R3/R4 on the
    // dependency set. No R13/R14 -- a broadcast consumer contributes nothing,
    // so it need not be a producer, and there is no mode to gate.
    //
    // R8 here is not "one producer per group": `replicate` delivers one
    // contribution into a result sized for one, with no combiner, so what the
    // op needs is that each *consumer tile* have a single source. One producer
    // per group satisfies that with no attribute (broadcast); several producers
    // require the attribute to name each consumer's sender (routing).
    getOperation()->walk([&](InterTileConsumeOp consumeOp) {
      auto produceOp =
          consumeOp.getFuture().getDefiningOp<InterTileProduceOp>();
      if (!produceOp) return; // dynamic future — cannot compare statically

      IntegerSet groupsSet   = consumeOp.getGroups();
      IntegerSet consumerSet = consumeOp.getConsumerTilesPerGroup().getValue();
      IntegerSet producerSet = produceOp.getProducerTilesPerGroup().getValue();
      auto depAttr = consumeOp.getProducerDependencyPerConsumerAttr();

      auto groupVals = groupValues(groupsSet);
      if (mlir::failed(groupVals)) return; // unbounded group range — defer

      for (int64_t g : *groupVals) {
        auto pOpt = tilesOf(producerSet, g);
        if (mlir::failed(pOpt)) continue; // unbounded tile range — defer
        if (pOpt->size() <= 1) continue;  // one producer: R8 holds trivially

        // Several producers, so the attribute is what names each consumer's
        // sender. Without it, a consumer would receive from all of them, which
        // has no defined value for a copy delivery.
        if (!depAttr) {
          consumeOp.emitError("producer_tiles_per_group for group ")
              << g << " has " << pOpt->size()
              << " producers, so producer_dependency_per_consumer is required "
                 "to name each consumer tile's single source";
          failed = true;
          return;
        }

        auto cOpt = tilesOf(consumerSet, g);
        if (mlir::failed(cOpt)) continue;
        for (int64_t c : *cOpt) {
          auto dOpt = depTilesOf(depAttr.getValue(), c, g);
          // Not statically enumerable: skip this consumer rather than guess.
          // Unlike the coverage check below, this test only rejects on a
          // dependency set we could read, so skipping one is conservative.
          if (mlir::failed(dOpt)) continue;
          if (dOpt->size() != 1) {
            consumeOp.emitError("consumer tile ")
                << c << " in group " << g << " depends on " << dOpt->size()
                << " producers; inter_tile_consume delivers one contribution "
                   "per consumer tile, so exactly one is required";
            failed = true;
            return;
          }
        }
      }

      // R3/R4 on the dependency set, shared with reduce.
      if (!depAttr) return;
      if (!checkDepSetSubsetAndCoverage(consumeOp, depAttr.getValue(),
                                        consumerSet, producerSet, *groupVals))
        failed = true;
    });

    if (failed) signalPassFailure();
  }
};

} // namespace
} // namespace ktir
