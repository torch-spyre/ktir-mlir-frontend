//===- KtdpInterTileHelpers.cpp - Inter-tile enumeration helpers --*- C++ -*-===//
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
// Implements the pure-enumeration helpers declared in KtdpInterTileHelpers.h.
//
// We ground each parameterized affine set at concrete integer points rather
// than reasoning about it symbolically. This was a deliberate switch away from
// a hybrid that used IntegerPolyhedron / isSubsetOf / computeVolume / isEqual
// (see "Replace Presburger verifier with pure enumeration"; the symbolic design
// is PR #25). Enumerating keeps every caller in terms of DenseSet<int64_t>,
// which makes the checks in KTIRCheckLegality.cpp read as the set statements
// the RFC writes -- subset, coverage, cardinality -- and makes their
// diagnostics able to name the offending tile.
//
// The cost is that a set must be statically bounded to be enumerated at all.
// Every helper here returns failure() rather than guessing when a bound is not
// a constant, and callers treat that as "check nothing" -- see the soundness
// note in KTIRCheckLegality.cpp's header, since silence is only safe for checks
// that reject on evidence they have.
//
// Point-by-point emptiness testing is what bounds the cost: it is linear in the
// tile-id range per group, which is a handful of tiles per group in practice.
// A symbolic formulation would avoid that, at the price of diagnostics that
// cannot point at a tile.
//
//===----------------------------------------------------------------------===//

#include "ktir/Dialect/KTDP/KTDPInterTileHelpers.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/IR/IntegerSet.h"

mlir::FailureOr<llvm::SmallVector<int64_t>>
mlir::ktdp::groupValues(mlir::IntegerSet groupsSet) {
  FlatLinearValueConstraints cst(groupsSet);
  std::optional<int64_t> lo =
      cst.getConstantBound64(presburger::BoundType::LB, /*pos=*/0);
  std::optional<int64_t> hi =
      cst.getConstantBound64(presburger::BoundType::UB, /*pos=*/0);
  if (!lo || !hi) return failure();
  llvm::SmallVector<int64_t> vals;
  for (int64_t g = *lo; g <= *hi; ++g) {
    FlatLinearValueConstraints gCst(groupsSet);
    gCst.setAndEliminate(gCst.getVarKindOffset(presburger::VarKind::SetDim),
                         {g});
    if (!gCst.isIntegerEmpty()) vals.push_back(g);
  }
  return vals;
}

mlir::FailureOr<llvm::DenseSet<int64_t>>
mlir::ktdp::tilesOf(mlir::IntegerSet tileSet, int64_t gVal) {
  FlatLinearValueConstraints cst(tileSet);
  cst.setAndEliminate(cst.getVarKindOffset(presburger::VarKind::Symbol),
                      {gVal});
  std::optional<int64_t> hi =
      cst.getConstantBound64(presburger::BoundType::UB, /*pos=*/0);
  if (!hi) return failure();
  llvm::DenseSet<int64_t> out;
  for (int64_t i = 0; i <= *hi; ++i) {
    FlatLinearValueConstraints pt(tileSet);
    pt.setAndEliminate(pt.getVarKindOffset(presburger::VarKind::Symbol),
                       {gVal});
    pt.setAndEliminate(pt.getVarKindOffset(presburger::VarKind::SetDim), {i});
    if (!pt.isIntegerEmpty()) out.insert(i);
  }
  return out;
}

// Bind the symbols of a dependency set `(p)[c]` or `(p)[c, g]` to concrete
// values. The one-symbol spelling is legal whenever the pairing is
// group-independent, so the symbol count -- not a fixed count of two -- decides
// how many values to substitute. Binding a second symbol on a one-symbol set
// would eliminate the set *dimension* `p` instead.
static void bindDepSymbols(mlir::FlatLinearValueConstraints &cst,
                           unsigned numSymbols, int64_t cVal, int64_t gVal) {
  unsigned symBase = cst.getVarKindOffset(mlir::presburger::VarKind::Symbol);
  cst.setAndEliminate(symBase, {cVal}); // fix c (first symbol)
  if (numSymbols > 1)
    cst.setAndEliminate(symBase, {gVal}); // fix g (now first remaining symbol)
}

mlir::FailureOr<llvm::DenseSet<int64_t>>
mlir::ktdp::depTilesOf(mlir::IntegerSet depSet, int64_t cVal, int64_t gVal) {
  unsigned numSymbols = depSet.getNumSymbols();
  if (numSymbols < 1 || numSymbols > 2) return failure();

  FlatLinearValueConstraints cst(depSet);
  bindDepSymbols(cst, numSymbols, cVal, gVal);
  std::optional<int64_t> hi =
      cst.getConstantBound64(presburger::BoundType::UB, /*pos=*/0);
  if (!hi) return failure();
  llvm::DenseSet<int64_t> out;
  for (int64_t p = 0; p <= *hi; ++p) {
    FlatLinearValueConstraints pt(depSet);
    bindDepSymbols(pt, numSymbols, cVal, gVal);
    pt.setAndEliminate(pt.getVarKindOffset(presburger::VarKind::SetDim), {p});
    if (!pt.isIntegerEmpty()) out.insert(p);
  }
  return out;
}
