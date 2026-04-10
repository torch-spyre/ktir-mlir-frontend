//===- KtdpDialect.td - KTDP dialect definition ------------*- tablegen -*-===//
//
// Dialect declaration for the ktdp (Kernel Tile Data-Parallel) dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KTDP_DIALECT_TD
#define KTDP_DIALECT_TD

include "mlir/IR/BuiltinDialect.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def Ktdp_Dialect : Dialect {
  let name = "ktdp";
  let summary = "Kernel Tile Data-Parallel (KTDP) dialect for KTIR";
  let description = [{
    The `ktdp` dialect models all memory-view and access-related abstractions
    for tile-based, data-parallel kernels targeting multi-core accelerator
    architectures.  It provides operations for:

    - Constructing logical memory views over existing allocations.
    - Composing distributed memory views that span multiple physical memories.
    - Building access tiles that encode structured collections of logical
      tensor coordinates (direct and indirect / gather-scatter styles).
    - Performing explicit load and store operations driven by access tiles.

    Compute and control-flow operations are delegated to standard MLIR
    dialects (Arith, Math, LinAlg, SCF, Tensor, MemRef, Func).
  }];

  let cppNamespace = "::mlir::ktdp";

  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect",
    "mlir::scf::SCFDialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect"
  ];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // KTDP_DIALECT_TD
