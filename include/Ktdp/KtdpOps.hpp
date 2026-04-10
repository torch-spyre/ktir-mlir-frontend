//===- KtdpOps.hpp - KTDP dialect ops public header -------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
#ifndef KTDP_KTDPOPS_HPP_
#define KTDP_KTDPOPS_HPP_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"

#include "Ktdp/KtdpDialect.hpp"
#include "Ktdp/KtdpAttrs.hpp"
#include "Ktdp/KtdpTypes.hpp"

// Ops
#define GET_OP_CLASSES
#include "Ktdp/KtdpOps.hpp.inc"

#endif // KTDP_KTDPOPS_HPP_
