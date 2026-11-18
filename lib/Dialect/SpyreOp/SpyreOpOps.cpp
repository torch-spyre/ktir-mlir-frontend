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

#include <llvm/ADT/STLExtras.h>
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
// Numeric cast
//===----------------------------------------------------------------------===//

LogicalResult NumericCast::verify() {
  auto input = llvm::cast<RankedTensorType>(getInput().getType());
  auto result = llvm::cast<RankedTensorType>(getResult().getType());
  bool widen = getKind() == "dl16tofp32";
  if (!widen && getKind() != "fp32todl16")
    return emitOpError("kind must be dl16tofp32 or fp32todl16");
  auto native = widen ? input : result;
  auto wide = widen ? result : input;
  const int64_t rank = native.getRank();
  const int64_t packet_axis = rank > 2 ? rank - 3 : 0;
  if (rank < 2 || wide.getRank() != rank || !native.hasStaticShape() ||
      !wide.hasStaticShape() || !llvm::isa<DF16Type>(native.getElementType()) ||
      !wide.getElementType().isF32() || native.getEncoding() ||
      wide.getEncoding() || native.getShape().back() != 64 ||
      wide.getShape().back() != 32 ||
      !llvm::all_of(
          native.getShape(), [](int64_t size) { return size > 0; }) ||
      !llvm::all_of(
          llvm::seq<int64_t>(0, rank - 1),
          [&](int64_t axis) {
            return axis == packet_axis
                       ? wide.getDimSize(axis) % 2 == 0 &&
                             wide.getDimSize(axis) / 2 ==
                                 native.getDimSize(axis)
                       : wide.getDimSize(axis) == native.getDimSize(axis);
          }))
    return emitOpError(
        "requires tensor<Nx64x!spyreop.df16> -> tensor<(2*N)x32xf32> or "
        "tensor<...xNxRx64x!spyreop.df16> -> tensor<...x(2*N)xRx32xf32> "
        "without encodings, with positive static extents");
  return success();
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

LogicalResult Exx2ZeromeanFused::inferReturnTypes(
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
