//===- KTDPAttrs.td - KTDP dialect attribute definitions ---*- tablegen -*-===//
//
// Attribute definitions for the ktdp dialect.
//
// Hierarchy:
//   KTDPMemorySpaceAttr  (abstract base / AttrInterface)
//     +-- SpyreMemorySpaceAttr  (concrete, mnemonic: spyre_memory_space)
//
//===----------------------------------------------------------------------===//

#ifndef KTDP_ATTRS_TD
#define KTDP_ATTRS_TD

include "ktir/Dialect/KTDP/KTDPDialect.td"
include "ktir/Dialect/KTDP/KTDPAttrInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Memory-space enum
//===----------------------------------------------------------------------===//

def KTDP_SpyreMemorySpaceKind : I32EnumAttr<
    "SpyreMemorySpaceKind",
    "Spyre memory-space enum",
    [
      I32EnumAttrCase<"unspecified", 0, "unspecified">,
      I32EnumAttrCase<"LX",  1, "LX">,
      I32EnumAttrCase<"HBM", 2, "HBM">
    ]> {
  let cppNamespace = "::mlir::ktdp";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// SpyreMemorySpaceAttr -- concrete Spyre target memory-space attribute
//
// Assembly syntax:
//   #ktdp.spyre_memory_space<HBM>
//   #ktdp.spyre_memory_space<LX, core = 7>
//   #ktdp.spyre_memory_space<unspecified>
//
// Parameters:
//   value  -- SpyreMemorySpaceKind enum (unspecified | LX | HBM)
//   core   -- optional core affinity (default -1 = unspecified)
//===----------------------------------------------------------------------===//

def KTDP_SpyreMemorySpaceAttr
    : AttrDef<KTDP_Dialect, "SpyreMemorySpace",
              [DeclareAttrInterfaceMethods<KTDP_MemorySpaceAttr>]> {
  let mnemonic = "spyre_memory_space";
  let summary = "Spyre target memory-space descriptor";
  let description = [{
    Identifies a physical memory space on the Spyre target.

    Valid values:
    - `unspecified` -- memory space not yet determined.
    - `LX`  -- shared on-chip scratchpad (cross-core accessible).
    - `HBM` -- high-bandwidth off-chip memory.

    The optional `core` parameter identifies which compute core's local
    memory is referenced (relevant for `LX`).

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
