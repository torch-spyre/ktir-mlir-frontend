//===- KtdpAttrs.cpp - KTDP dialect attr implementations ------------------===//
//
//===----------------------------------------------------------------------===//

#include "Ktdp/KtdpAttrs.hpp"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::ktdp;

//===----------------------------------------------------------------------===//
// SpyreMemorySpaceAttr::verify
//===----------------------------------------------------------------------===//

LogicalResult SpyreMemorySpaceAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    SpyreMemorySpaceKind value, int32_t core) {
  // Core affinity is only meaningful for core-local memory spaces.
  if (core == -1)
    return success();

  if (core < 0)
    return emitError() << "core affinity must be non-negative, but got: "
                       << core;

  if (value != SpyreMemorySpaceKind::LX &&
      value != SpyreMemorySpaceKind::L0) {
    return emitError()
           << "core affinity is only valid for LX or L0 memory spaces, "
              "but got memory space '"
           << stringifySpyreMemorySpaceKind(value) << "' with core = " << core;
  }
  return success();
}
