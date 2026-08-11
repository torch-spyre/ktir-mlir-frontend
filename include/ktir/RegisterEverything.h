//===- RegisterEverything.h - KTIR registration entry points ----*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
//
// This file declares the main registration entry points for KTIR.
//
// Downstream users should prefer these over registering dialects by hand, so
// that KTIR remains the single source of truth for the upstream MLIR dialects
// its IR depends on.
//
//===----------------------------------------------------------------------===//

#ifndef KTIR_REGISTEREVERYTHING_H
#define KTIR_REGISTEREVERYTHING_H

namespace mlir {

class DialectRegistry;

} // namespace mlir

namespace ktir {

//===----------------------------------------------------------------------===//
// Exported Only
//===----------------------------------------------------------------------===//

/// Registers all passes defined by KTIR.
void registerPasses();
/// Registers all dialects defined by KTIR.
void registerDialects(mlir::DialectRegistry &registry);
/// Registers all extensions provided by KTIR.
void registerExtensions(mlir::DialectRegistry &registry);

//===----------------------------------------------------------------------===//
// Imported and Exported
//===----------------------------------------------------------------------===//

/// Registers all passes defined and used by KTIR.
void registerAllPasses();
/// Registers all dialects defined and used by KTIR.
void registerAllDialects(mlir::DialectRegistry &registry);
/// Registers all extensions provided and required by KTIR.
void registerAllExtensions(mlir::DialectRegistry &registry);

} // namespace ktir

#endif // KTIR_REGISTEREVERYTHING_H
