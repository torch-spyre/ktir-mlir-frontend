//===- KtdpDialect.cpp - KTDP dialect definition --------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Ktdp/KtdpDialect.hpp"
#include "Ktdp/KtdpAttrs.hpp"
#include "Ktdp/KtdpOps.hpp"
#include "Ktdp/KtdpTypes.hpp"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectImplementation.h"

// Generated dialect definitions
#include "Ktdp/KtdpDialect.cpp.inc"

// Generated enum definitions
#include "Ktdp/KtdpEnums.cpp.inc"

// Generated attr interface definitions
#include "Ktdp/KtdpAttrInterfaces.cpp.inc"

// Generated attribute definitions
#define GET_ATTRDEF_CLASSES
#include "Ktdp/KtdpAttrs.cpp.inc"

// Generated type definitions
#define GET_TYPEDEF_CLASSES
#include "Ktdp/KtdpTypes.cpp.inc"

using namespace mlir;
using namespace mlir::ktdp;

void KtdpDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ktdp/KtdpOps.cpp.inc"
  >();
  addTypes<
    AccessTileType,
    RuntimeArgType
  >();
  addAttributes<
    SpyreMemorySpaceAttr
  >();
}
