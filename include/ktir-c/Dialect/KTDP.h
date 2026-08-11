//===- KTDP.h - CAPI for KTDP dialect -----------------------------*- C -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KTIR_C_DIALECT_KTDP_H_
#define KTIR_C_DIALECT_KTDP_H_

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(KTDP, ktdp);

MLIR_CAPI_EXPORTED MlirType mlirKTDPAccessTileTypeGet(intptr_t rank,
                                                      int64_t *shape,
                                                      MlirType elementType);

MLIR_CAPI_EXPORTED bool mlirTypeIsAKTDPAccessTileType(MlirType t);

MLIR_CAPI_EXPORTED MlirTypeID mlirKTDPAccessTileTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirKTDPRuntimeArgTypeGet(MlirContext ctx,
                                                       MlirType underlyingType,
                                                       int64_t granularity,
                                                       int64_t upperbound);

MLIR_CAPI_EXPORTED bool mlirTypeIsAKTDPRuntimeArgType(MlirType t);

MLIR_CAPI_EXPORTED MlirTypeID mlirKTDPRuntimeArgTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirKTDPRuntimeArgTypeGetUnderlyingType(MlirType t);

/// Returns the granularity, or -1 if not set.
MLIR_CAPI_EXPORTED int64_t mlirKTDPRuntimeArgTypeGetGranularity(MlirType t);

/// Returns the upperbound, or -1 if not set.
MLIR_CAPI_EXPORTED int64_t mlirKTDPRuntimeArgTypeGetUpperbound(MlirType t);

#ifdef __cplusplus
}
#endif

#endif // KTIR_C_DIALECT_KTDP_H_
