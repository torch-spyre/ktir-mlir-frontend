//===- KTDPAttrInterfaces.td - KTDP attr iface definitions -*- tablegen -*-===//
//
// Attribute interface definitions for the ktdp dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KTIR_DIALECT_KTDP_KTDPATTRINTERFACES_TD
#define KTIR_DIALECT_KTDP_KTDPATTRINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// KTDPMemorySpaceAttr -- abstract base (AttrInterface)
//===----------------------------------------------------------------------===//

def KTDP_MemorySpaceAttr : AttrInterface<"KTDPMemorySpaceAttr"> {
  let cppNamespace = "::mlir::ktdp";
  let description = [{
    Abstract base interface for all device-specific memory-space attributes
    within the `ktdp` dialect.  Concrete implementations (e.g.
    `SpyreMemorySpaceAttr`) implement this interface.
  }];

  // TODO: Add query methods (e.g. getMemorySpaceKind()) once a second
  // concrete implementation is introduced and the common API is clear.
  let methods = [];
}

#endif // KTIR_DIALECT_KTDP_KTDPATTRINTERFACES_TD
