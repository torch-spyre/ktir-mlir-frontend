//===- Dialects.h - CAPI for dialects -----------------------------*- C -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KTDP_C_DIALECTS_H
#define KTDP_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Ktdp, ktdp);

MLIR_CAPI_EXPORTED MlirType mlirKtdpAccessTileTypeGet(intptr_t rank,
                                                      int64_t *shape,
                                                      MlirType elementType);

MLIR_CAPI_EXPORTED bool mlirKtdpTypeIsAAccessTileType(MlirType t);

MLIR_CAPI_EXPORTED MlirTypeID mlirKtdpAccessTileTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirKtdpRuntimeArgTypeGet(MlirContext ctx,
                                                       MlirType underlyingType,
                                                       int64_t granularity,
                                                       int64_t upperbound);

MLIR_CAPI_EXPORTED bool mlirKtdpTypeIsARuntimeArgType(MlirType t);

MLIR_CAPI_EXPORTED MlirTypeID mlirKtdpRuntimeArgTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirKtdpRuntimeArgTypeGetUnderlyingType(MlirType t);

/// Returns the granularity, or -1 if not set.
MLIR_CAPI_EXPORTED int64_t mlirKtdpRuntimeArgTypeGetGranularity(MlirType t);

/// Returns the upperbound, or -1 if not set.
MLIR_CAPI_EXPORTED int64_t mlirKtdpRuntimeArgTypeGetUpperbound(MlirType t);

#ifdef __cplusplus
}
#endif

#endif // KTDP_C_DIALECTS_H
