//===- ktir-opt.cpp - KTIR MLIR optimizer driver ----------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "ktir/RegisterEverything.h"

using namespace mlir;

auto main(int argc, char **argv) -> int {
  registerAllPasses();
  ktir::registerAllPasses();

  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  ktir::registerAllDialects(registry);
  ktir::registerAllExtensions(registry);

  return asMainReturnCode(
      MlirOptMain(argc, argv, "KTIR modular optimizer driver\n", registry));
}
