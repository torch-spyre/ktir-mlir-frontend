//===- ktir-opt.cpp - KTIR MLIR optimizer driver ---------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "Ktdp/KtdpDialect.hpp"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registry.insert<mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect,
                  mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect,
                  mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect,
                  mlir::ktdp::KtdpDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "KTIR optimizer driver\n", registry));
}
