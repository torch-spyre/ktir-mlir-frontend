//===- ktir-lsp-server.cpp - KTIR LSP server --------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/Tools/mlir-lsp-server/MlirLspServerMain.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "ktir/RegisterEverything.h"

using namespace mlir;

auto main(int argc, char **argv) -> int {
  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  ktir::registerAllDialects(registry);
  ktir::registerAllExtensions(registry);

  return asMainReturnCode(MlirLspServerMain(argc, argv, registry));
}
