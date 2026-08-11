//===- KTDPDialect.cpp - KTDP dialect definition --------------------------===//
//
//===----------------------------------------------------------------------===//

#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectImplementation.h"

// Generated dialect definitions
#include "ktir/Dialect/KTDP/KTDPDialect.cpp.inc"

// Generated enum definitions
#include "ktir/Dialect/KTDP/KTDPEnums.cpp.inc"

// Generated attr interface definitions
#include "ktir/Dialect/KTDP/KTDPAttrInterfaces.cpp.inc"

// Generated attribute definitions
#define GET_ATTRDEF_CLASSES
#include "ktir/Dialect/KTDP/KTDPAttrs.cpp.inc"

// Generated type definitions
#define GET_TYPEDEF_CLASSES
#include "ktir/Dialect/KTDP/KTDPTypes.cpp.inc"

using namespace mlir;
using namespace mlir::ktdp;

void KTDPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ktir/Dialect/KTDP/KTDP.cpp.inc"
  >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "ktir/Dialect/KTDP/KTDPTypes.cpp.inc"
  >();
  addAttributes<
    SpyreMemorySpaceAttr
  >();
}
