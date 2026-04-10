//===- Dialects.cpp - CAPI for dialects -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Ktdp-c/Dialects.h"

#include "Ktdp/KtdpDialect.hpp"
#include "Ktdp/KtdpTypes.hpp"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Ktdp, ktdp,
                                      mlir::ktdp::KtdpDialect)

MlirType mlirKtdpAccessTileTypeGet(intptr_t rank, int64_t *shape, MlirType elementType) {
  return wrap(mlir::ktdp::AccessTileType::get(llvm::ArrayRef(shape, static_cast<size_t>(rank)), unwrap(elementType)));
}

bool mlirKtdpTypeIsAAccessTileType(MlirType t) {
  return llvm::isa<mlir::ktdp::AccessTileType>(unwrap(t));
}

MlirTypeID mlirKtdpAccessTileTypeGetTypeID() {
  return wrap(mlir::ktdp::AccessTileType::getTypeID());
}

MlirType mlirKtdpRuntimeArgTypeGet(MlirContext ctx, MlirType underlyingType,
                                    int64_t granularity, int64_t upperbound) {
  std::optional<int64_t> gran = granularity >= 0 ? std::optional<int64_t>(granularity) : std::nullopt;
  std::optional<int64_t> ub = upperbound >= 0 ? std::optional<int64_t>(upperbound) : std::nullopt;
  return wrap(mlir::ktdp::RuntimeArgType::get(unwrap(ctx), unwrap(underlyingType), gran, ub));
}

bool mlirKtdpTypeIsARuntimeArgType(MlirType t) {
  return llvm::isa<mlir::ktdp::RuntimeArgType>(unwrap(t));
}

MlirTypeID mlirKtdpRuntimeArgTypeGetTypeID() {
  return wrap(mlir::ktdp::RuntimeArgType::getTypeID());
}

MlirType mlirKtdpRuntimeArgTypeGetUnderlyingType(MlirType t) {
  return wrap(llvm::cast<mlir::ktdp::RuntimeArgType>(unwrap(t)).getUnderlyingType());
}

int64_t mlirKtdpRuntimeArgTypeGetGranularity(MlirType t) {
  auto v = llvm::cast<mlir::ktdp::RuntimeArgType>(unwrap(t)).getGranularity();
  return v.has_value() ? v.value() : -1;
}

int64_t mlirKtdpRuntimeArgTypeGetUpperbound(MlirType t) {
  auto v = llvm::cast<mlir::ktdp::RuntimeArgType>(unwrap(t)).getUpperbound();
  return v.has_value() ? v.value() : -1;
}
