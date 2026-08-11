//===- RegisterEverything.cpp - KTIR registration entry points --*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
// This file implements the main registration entry points for KTIR.
//
// IMPORTANT:
// When you update this file, make sure that any new dialects, passes or
// extensions that you add below are also declared as link dependencies in the
// accompanying `CMakeLists.txt` file!
//
//===----------------------------------------------------------------------===//

#include "ktir/RegisterEverything.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/DialectRegistry.h>

#include "ktir/Dialect/KTDP/KTDPDialect.h"

//===----------------------------------------------------------------------===//
// Exported Only
//===----------------------------------------------------------------------===//

void ktir::registerPasses() {
  // Register the passes defined in KTIR.
  //
  // KTIR does not define any passes yet; its invariants are currently enforced
  // by op verifiers rather than by a standalone pass.
}

void ktir::registerDialects(mlir::DialectRegistry &registry) {
  // Register the dialects defined in KTIR.
  registry.insert<mlir::ktdp::KTDPDialect>();
}

void ktir::registerExtensions(mlir::DialectRegistry &registry) {
  // Register the extensions provided by KTIR.
  //
  // KTIR does not provide any extensions yet.
  (void)registry;
}

//===----------------------------------------------------------------------===//
// Imported and Exported
//===----------------------------------------------------------------------===//

void ktir::registerAllPasses() {
  // Register the passes required from MLIR.

  // Register our own passes.
  registerPasses();
}

void ktir::registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off

  // Register the dialects required from MLIR. These are the upstream dialects
  // that may appear in KTDP IR.
  registry.insert<mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect,
                  mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();

  // clang-format on

  // Register our own dialects.
  registerDialects(registry);
}

void ktir::registerAllExtensions(mlir::DialectRegistry &registry) {
  // Register the extensions required from MLIR.

  // Register our own extensions.
  registerExtensions(registry);
}
