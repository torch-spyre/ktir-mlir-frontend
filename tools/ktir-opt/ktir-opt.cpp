//===- ktir-opt.cpp - KTIR MLIR optimizer driver ---------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Ktdp/KtdpDialect.hpp"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::ktdp::KtdpDialect>();
  mlir::registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "KTIR optimizer driver\n", registry));
}
