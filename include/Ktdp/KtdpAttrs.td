//===- KtdpAttrs.td - KTDP dialect attribute definitions ---*- tablegen -*-===//
//
// Attribute definitions for the ktdp dialect.
//
// Hierarchy:
//   KtdpMemorySpaceAttr  (abstract base / AttrInterface)
//     +-- SpyreMemorySpaceAttr  (concrete, mnemonic: spyre_memory_space)
//
//===----------------------------------------------------------------------===//

#ifndef KTDP_ATTRS_TD
#define KTDP_ATTRS_TD

include "Ktdp/KtdpDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Memory-space enum
//===----------------------------------------------------------------------===//

def Ktdp_SpyreMemorySpaceKind : I32EnumAttr<
    "SpyreMemorySpaceKind",
    "Spyre memory-space enum",
    [
      I32EnumAttrCase<"unspecified", 0, "unspecified">,
      I32EnumAttrCase<"LX",  1, "LX">,
      I32EnumAttrCase<"HBM", 2, "HBM">,
      I32EnumAttrCase<"L0",  3, "L0">
    ]> {
  let cppNamespace = "::mlir::ktdp";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// KtdpMemorySpaceAttr -- abstract base (AttrInterface)
//===----------------------------------------------------------------------===//

def Ktdp_MemorySpaceAttr : AttrInterface<"KtdpMemorySpaceAttr"> {
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

//===----------------------------------------------------------------------===//
// SpyreMemorySpaceAttr -- concrete Spyre target memory-space attribute
//
// Assembly syntax:
//   #ktdp.spyre_memory_space<HBM>
//   #ktdp.spyre_memory_space<LX, core = 7>
//   #ktdp.spyre_memory_space<L0>
//   #ktdp.spyre_memory_space<unspecified>
//
// Parameters:
//   value  -- SpyreMemorySpaceKind enum (unspecified | LX | HBM | L0)
//   core   -- optional core affinity (default -1 = unspecified)
//===----------------------------------------------------------------------===//

def Ktdp_SpyreMemorySpaceAttr
    : AttrDef<Ktdp_Dialect, "SpyreMemorySpace",
              [DeclareAttrInterfaceMethods<Ktdp_MemorySpaceAttr>]> {
  let mnemonic = "spyre_memory_space";
  let summary = "Spyre target memory-space descriptor";
  let description = [{
    Identifies a physical memory space on the Spyre target.

    Valid values:
    - `unspecified` -- memory space not yet determined.
    - `LX`  -- shared on-chip scratchpad (cross-core accessible).
    - `HBM` -- high-bandwidth off-chip memory.
    - `L0`  -- per-core on-chip scratchpad.

    The optional `core` parameter identifies which compute core's local
    memory is referenced (relevant for `LX` and `L0`).

    Examples:
    ```mlir
    memory_space = #ktdp.spyre_memory_space<HBM>
    memory_space = #ktdp.spyre_memory_space<LX, core = 7>
    ```
  }];

  let parameters = (ins
    "::mlir::ktdp::SpyreMemorySpaceKind":$value,
    DefaultValuedParameter<"int32_t", "-1">:$core
  );

  let assemblyFormat = "`<` $value (`,` `core` `=` $core^)? `>`";

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Return true if a core affinity is specified.
    bool hasCoreSpecified() const { return getCore() != -1; }
  }];
}

//===----------------------------------------------------------------------===//
// Runtime argument extraction kind
//===----------------------------------------------------------------------===//

def KTDP_RuntimeArgExtractKind_Value : I32EnumAttrCase<"value", 0>;
def KTDP_RuntimeArgExtractKind_Granularity : I32EnumAttrCase<"granularity", 1>;
def KTDP_RuntimeArgExtractKind_Upperbound : I32EnumAttrCase<"upperbound", 2>;

def KTDP_RuntimeArgExtractKind : I32EnumAttr<"RuntimeArgExtractKind",
    "Kind of extraction from runtime_arg type",
    [
       KTDP_RuntimeArgExtractKind_Value,
       KTDP_RuntimeArgExtractKind_Granularity,
       KTDP_RuntimeArgExtractKind_Upperbound
    ]> {
  let cppNamespace = "::mlir::ktdp";
}

#endif // KTDP_ATTRS_TD
