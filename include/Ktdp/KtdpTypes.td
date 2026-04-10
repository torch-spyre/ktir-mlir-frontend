//===- KtdpTypes.td - KTDP dialect type definitions --------*- tablegen -*-===//
//
// Type definitions for the ktdp dialect.
//
// Types:
//   AccessTileType  -- mnemonic: access_tile
//   RuntimeArgType  -- mnemonic: runtime_arg
//
//===----------------------------------------------------------------------===//

#ifndef KTDP_TYPES_TD
#define KTDP_TYPES_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

include "Ktdp/KtdpDialect.td"

//===----------------------------------------------------------------------===//
// KTDP Type Definitions
//===----------------------------------------------------------------------===//

// Base class for KTDP dialect types
class KTDP_Type<string name, string typeMnemonic, list<Trait> traits = [],
                string baseCppClass = "::mlir::Type">
    : TypeDef<Ktdp_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// AccessTileType
//===----------------------------------------------------------------------===//

def Ktdp_AccessTileType : KTDP_Type<"AccessTile", "access_tile", [
    ShapedTypeInterface, ValueSemantics
  ], "::mlir::TensorType"> {
  let summary = "Multi-dimensional tile with a fixed number of dimensions and index element type";
  let hasCustomAssemblyFormat = 1;
  let description = [{
    Syntax:

    ```
    access_tile-type ::= `access_tile` `<` dimension-list `index` `>`
    dimension-list ::= (dimension `x`)*
    dimension ::= `?` | decimal-literal
    ```

    The access_tile type is similar to the builtin ranked tensor type but is specific
    to the KTDP dialect. It represents aggregate N-dimensional data values with
    a fixed rank with a list of dimensions. Each dimension may be a static
    non-negative decimal constant or be dynamically determined (indicated by `?`).

    Unlike the builtin ranked tensor type, the access_tile type only supports the
    `index` type as its element type. This constraint is enforced during both
    parsing and verification.

    Example:

    ```mlir
    // Known rank but unknown dimensions.
    access_tile<? x ? x index>

    // Partially known dimensions.
    access_tile<? x 64 x index>

    // Full static shape.
    access_tile<1 x 64 x index>

    // AccessTile with rank zero. Represents a scalar.
    access_tile<index>

    // Zero-element dimensions are allowed.
    access_tile<0 x 42 x index>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );

  let genStorageClass = 1;

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType
    ), [{
      return $_get(elementType.getContext(), shape, elementType);
    }]>
  ];

  let extraClassDeclaration = [{
    using TensorType::clone;
    using ShapedType::Trait<AccessTileType>::getElementTypeBitWidth;
    using ShapedType::Trait<AccessTileType>::getRank;
    using ShapedType::Trait<AccessTileType>::getNumElements;
    using ShapedType::Trait<AccessTileType>::isDynamicDim;
    using ShapedType::Trait<AccessTileType>::hasStaticShape;
    using ShapedType::Trait<AccessTileType>::getNumDynamicDims;
    using ShapedType::Trait<AccessTileType>::getDimSize;
    using ShapedType::Trait<AccessTileType>::getDynamicDimIndex;

    /// This is a builder type that keeps local references to arguments.
    /// Arguments that are passed into the builder must outlive the builder.
    class Builder;

    /// Return true since AccessTileType is always ranked.
    bool hasRank() const { return true; }

    /// Return a clone of this type with the given new shape and element type.
    ::mlir::ShapedType cloneWith(
        ::std::optional<::llvm::ArrayRef<int64_t>> shape,
        ::mlir::Type elementType) const {
      return AccessTileType::get(shape.value_or(getShape()),
                                 elementType);
    }

    /// Return a clone of this type with the given new element type and the same
    /// shape as this type.
    AccessTileType clone(::mlir::Type elementType) {
      return ::llvm::cast<AccessTileType>(cloneWith(getShape(), elementType));
    }
  }];

  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// RuntimeArgType
//===----------------------------------------------------------------------===//

def Ktdp_RuntimeArgType : KTDP_Type<"RuntimeArg", "runtime_arg"> {
  let summary = "Runtime argument type with optional granularity and upper bound";
  let hasCustomAssemblyFormat = 1;
  let description = [{
    Syntax:

    ```
    runtime_arg-type ::= `runtime_arg` `<` type (`,` `granularity` `=` integer-literal)?
                         (`,` `upperbound` `=` integer-literal)? `>`
    ```

    The runtime_arg type represents a runtime argument with an underlying type
    and optional granularity and upper bound constraints. The underlying type
    must be a builtin type (e.g., index, integer types, float types).

    - `type`: The underlying type of the runtime argument (mandatory)
    - `granularity`: Optional positive integer specifying the granularity constraint
    - `upperbound`: Optional positive integer specifying the upper bound constraint

    Examples:

    ```mlir
    // Runtime arg with just the underlying type
    !ktdp.runtime_arg<index>

    // Runtime arg with granularity
    !ktdp.runtime_arg<index, granularity=3>

    // Runtime arg with upper bound
    !ktdp.runtime_arg<index, upperbound=300>

    // Runtime arg with both granularity and upper bound
    !ktdp.runtime_arg<index, granularity=3, upperbound=300>

    // Runtime arg with i32 type
    !ktdp.runtime_arg<i32, granularity=4, upperbound=1024>
    ```
  }];

  let parameters = (ins
    "Type":$underlyingType,
    OptionalParameter<"std::optional<int64_t>">:$granularity,
    OptionalParameter<"std::optional<int64_t>">:$upperbound
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "Type":$underlyingType
    ), [{
      return $_get(underlyingType.getContext(), underlyingType,
                   std::nullopt, std::nullopt);
    }]>,
    TypeBuilderWithInferredContext<(ins
      "Type":$underlyingType,
      "std::optional<int64_t>":$granularity,
      "std::optional<int64_t>":$upperbound
    ), [{
      return $_get(underlyingType.getContext(), underlyingType,
                   granularity, upperbound);
    }]>
  ];

  let extraClassDeclaration = [{
    /// This is a builder type that keeps local references to arguments.
    /// Arguments that are passed into the builder must outlive the builder.
    class Builder;

    /// Return a clone of this type with all parameters specified.
    RuntimeArgType clone(::mlir::Type underlyingType,
                         std::optional<int64_t> granularity,
                         std::optional<int64_t> upperbound) {
      return RuntimeArgType::get(underlyingType, granularity, upperbound);
    }
  }];

  let genVerifyDecl = 1;
}

#endif // KTDP_TYPES_TD
