//===- KtdpInterTileHelpers.h - Inter-tile enumeration helpers ---*- C++ -*-===//
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
// Pure-enumeration helpers shared between the KtdpOps verifier and the
// KTIRCheckLegality pass.  We enumerate concrete integer points rather than
// using Presburger set operations; see the comment in KtdpOps.cpp for the
// trade-offs vs. the pure-Presburger Option A approach.
//
//===----------------------------------------------------------------------===//

#ifndef KTIR_DIALECT_KTDP_KTDPINTERTILEHELPERS_H
#define KTIR_DIALECT_KTDP_KTDPINTERTILEHELPERS_H

#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::ktdp {

// Enumerate the group values in `groupsSet` (a 1D set over `g` with no
// symbols). Returns failure() when the range is not statically bounded.
mlir::FailureOr<llvm::SmallVector<int64_t>>
groupValues(mlir::IntegerSet groupsSet);

// Return the set of tile ids selected by `tileSet` (a set `(i)[g]`) for a
// concrete group value `g`. Returns failure() when the tile-id upper bound
// is not statically known.
mlir::FailureOr<llvm::DenseSet<int64_t>>
tilesOf(mlir::IntegerSet tileSet, int64_t gVal);

// Return the set of producer tiles in `depSet` for a concrete consumer `cVal`
// and group `gVal`. Symbols are ordered [c, g], and `depSet` may carry either
// both -- `(p)[c, g]` -- or only `c`, as `(p)[c]`, which is the legal spelling
// when the pairing is group-independent. `gVal` is ignored in the latter case.
// Returns failure() when the symbol count is not 1 or 2, or when the
// producer-tile upper bound is not statically known.
mlir::FailureOr<llvm::DenseSet<int64_t>>
depTilesOf(mlir::IntegerSet depSet, int64_t cVal, int64_t gVal);

} // namespace mlir::ktdp

#endif // KTIR_DIALECT_KTDP_KTDPINTERTILEHELPERS_H
