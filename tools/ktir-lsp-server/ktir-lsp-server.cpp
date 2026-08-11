//===- ktir-lsp-server.cpp - KTIR LSP server --------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/Tools/mlir-lsp-server/MlirLspServerMain.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "ktir/Dialect/Spyre/SpyreDialect.h"

using namespace mlir;

auto main(int argc, char** argv) -> int {
  DialectRegistry registry;
  registry.insert<ktdp::KtdpDialect, spyre::SpyreDialect>();
  registerAllDialects(registry);
  registerAllExtensions(registry);

  return asMainReturnCode(MlirLspServerMain(argc, argv, registry));
}
