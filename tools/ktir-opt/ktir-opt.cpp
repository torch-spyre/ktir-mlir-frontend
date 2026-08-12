//===- ktir-opt.cpp - KTIR MLIR optimizer driver ----------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "ktir/Conversion/Passes.h"
#include "ktir/Dialect/KTDP/KTDPDialect.h"
#include "ktir/Dialect/SpyreOp/SpyreOpDialect.h"

using namespace mlir;

auto main(int argc, char** argv) -> int {
  registerAllPasses();
  ktir::registerKTIRConversionPasses();

  DialectRegistry registry;
  registry.insert<ktdp::KtdpDialect, spyreop::SpyreOpDialect>();
  registerAllDialects(registry);
  registerAllExtensions(registry);

  return asMainReturnCode(
      MlirOptMain(argc, argv, "KTIR modular optimizer driver\n", registry));
}
