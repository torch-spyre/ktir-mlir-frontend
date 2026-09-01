//===-- SpyreOpOps.cpp -------------------------------------------*- c++-*-===//
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

// clang-format off
#include "ktir/Dialect/SpyreOp/SpyreOp.h"
// clang-format on

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;
using namespace mlir::spyreop;

//===----------------------------------------------------------------------===//
// SpyreOpDialect
//===----------------------------------------------------------------------===//

void SpyreOpDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "ktir/Dialect/SpyreOp/SpyreOp.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Fused pairs
//===----------------------------------------------------------------------===//

namespace {

/// Gets the type holding a pair of \p scalar, null if there is none for it.
Type fusedPairOf(Type scalar) {
  MLIRContext* context = scalar.getContext();
  if (llvm::isa<Float16Type>(scalar)) return FP16FusedType::get(context);
  if (llvm::isa<DF16Type>(scalar)) return DF16FusedType::get(context);
  if (llvm::isa<Float32Type>(scalar)) return FP32FusedType::get(context);
  return nullptr;
}

/// Gets the type \p fused holds a pair of, null if it is not a fused type.
Type scalarBehind(Type fused) {
  MLIRContext* context = fused.getContext();
  if (llvm::isa<FP16FusedType>(fused)) return Float16Type::get(context);
  if (llvm::isa<DF16FusedType>(fused)) return DF16Type::get(context);
  if (llvm::isa<FP32FusedType>(fused)) return Float32Type::get(context);
  return nullptr;
}

}  // namespace

LogicalResult Exx2Fused::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferred) {
  const Type fused = fusedPairOf(operands.front().getType());
  if (!fused) return failure();
  inferred.push_back(fused);
  return success();
}

LogicalResult LayerNormScaleFused::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferred) {
  const Type scalar = scalarBehind(operands.front().getType());
  if (!scalar) return failure();
  inferred.push_back(scalar);
  return success();
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "ktir/Dialect/SpyreOp/SpyreOp.cpp.inc"
