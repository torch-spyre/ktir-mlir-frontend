//===- KtdpOps.td - KTDP dialect operation definitions ------*- tablegen -*-===//
//
// Copyright 2026 The KTIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines KTDP operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef KTDP_OPS_TD
#define KTDP_OPS_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE
include "mlir/IR/EnumAttr.td"

include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/BuiltinAttributes.td"

// Import the KTDP dialect definition
include "ktir/Dialect/KTDP/KTDPDialect.td"

// Import the KTDP type definitions
include "ktir/Dialect/KTDP/KTDPTypes.td"

// Import the KTDP attribute definitions
include "ktir/Dialect/KTDP/KTDPAttrs.td"

// Base class for KTDP dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Ktdp_Op<string mnemonic, list<Trait> traits = []> :
    Op<Ktdp_Dialect, mnemonic, traits> {

}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

def KTDP_RegionTerminatorOp : Ktdp_Op<"region_terminator", [Terminator]> {
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Inter-tile communication
//
// Implements the design in docs/inter-tile-communication.md. Production is
// `ktdp.inter_tile_produce`; the delivery ops implemented so far are
// `ktdp.inter_tile_consume` (copy: broadcast and routing) and
// `ktdp.inter_tile_reduce` (reduction). Both support per-tile synchronization
// via `producer_dependency_per_consumer`. Reduce-scatter, gather, all-to-all
// and scatter are future work; the `!ktdp.tile_future` type and these
// terminators are shared scaffolding for them.
//===----------------------------------------------------------------------===//

def KTDP_YieldPartialOp : Ktdp_Op<"yield_partial",
    [Terminator, ReturnLike, Pure,
     HasParent<"::mlir::ktdp::InterTileProduceOp">]> {
  let summary = "Terminator naming a tile's partial contributions";
  let description = [{
    Terminates the producer region of `ktdp.inter_tile_produce`, yielding one
    value per partial-tensor role. The yielded value types must match the
    partial types carried by the produced `!ktdp.tile_future`.
  }];
  let arguments = (ins Variadic<AnyRankedTensor>:$values);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
  // Required so ensureTerminator() (called by SingleBlockImplicitTerminator
  // via the declarative format path) can construct a zero-operand terminator.
  let builders = [OpBuilder<(ins), [{ $_state.addOperands(ValueRange{}); }]>];
}

def KTDP_YieldReducedOp : Ktdp_Op<"yield_reduced",
    [Terminator, ReturnLike, Pure,
     ParentOneOf<["::mlir::ktdp::InterTileReduceOp"]>]> {
  let summary = "Terminator yielding the combined partials of a reducer region";
  let description = [{
    Terminates the reducer region of `ktdp.inter_tile_reduce` (and, in future,
    `ktdp.inter_tile_reduce_scatter`), yielding one combined value per
    partial-tensor role. The yielded value types must match the partial types
    `T_p_i` (not the rank-reduced result types).
  }];
  let arguments = (ins Variadic<AnyRankedTensor>:$values);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
  let builders = [OpBuilder<(ins), [{ $_state.addOperands(ValueRange{}); }]>];
}

def KTDP_InterTileProduceOp : Ktdp_Op<"inter_tile_produce",
    [SingleBlockImplicitTerminator<"::mlir::ktdp::YieldPartialOp">,
     RecursiveMemoryEffects]> {
  let summary = "Unified production op for inter-tile communication";
  let description = [{
    The `ktdp.inter_tile_produce` operation names **what partial results each
    tile contributes** to a cross-tile communication and returns a
    `!ktdp.tile_future` handle carrying per-tile availability signals.

    The producer region runs once per participating tile in that tile's SPMD
    execution. Its single block argument `%gid : index` is the index of the
    group the tile belongs to (the runtime binds tile `t` to the group `g`
    whose `producer_tiles_per_group(g)` contains `t`). The block terminates
    with `ktdp.yield_partial`, naming the per-tile contribution for each role.

    Attributes:
    - `producer_tiles_per_group` — parameterized affine integer set `(i)[g]`
      selecting which tiles produce per group.

    The `groups` integer set is embedded in the result `!ktdp.tile_future` type.
    The result `!ktdp.tile_future` must have exactly one use: the single
    delivery op that consumes it.

    Example:
    ```mlir
    %f = ktdp.inter_tile_produce
        producer_tiles_per_group = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
        -> !ktdp.tile_future<tensor<1x64xf16>, groups = affine_set<(g) : (g >= 0, -g + 7 >= 0)>>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial : tensor<1x64xf16>
    }
    ```
  }];

  let arguments = (ins
    Builtin_IntegerSetAttr:$producer_tiles_per_group);
  let results = (outs Ktdp_TileFutureType:$future);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat =
      "`producer_tiles_per_group` `=` $producer_tiles_per_group attr-dict"
      " `->` type($future) $body";
  let hasVerifier = 1;
  let hasRegionVerifier = 1;

  let extraClassDeclaration = [{
    /// Convenience: the partial types carried by the produced future.
    ::llvm::ArrayRef<::mlir::RankedTensorType> getPartialTypes() {
      return getFuture().getType().getPartialTypes();
    }
    /// The groups set, read from the result tile_future type.
    ::mlir::IntegerSet getGroups() {
      return getFuture().getType().getGroups().getValue();
    }
  }];
}

def KTDP_InterTileConsumeOp : Ktdp_Op<"inter_tile_consume",
    [RecursiveMemoryEffects,
     RangedTypesMatchWith<
         "result types must match future partial types",
         "future", "results",
         "llvm::cast<TileFutureType>($_self).getPartialTypes()">]> {
  let summary = "Copy delivery op (broadcast / routing)";
  let description = [{
    The `ktdp.inter_tile_consume` operation consumes the `!ktdp.tile_future`
    produced by `ktdp.inter_tile_produce` and delivers each partial unchanged
    to the consumer tiles. No combining occurs and there is no region.

    Each result `T_r_i` has the same type as the corresponding partial type
    `T_p_i` — `replicate` placement neither grows nor shrinks a partial.

    Attributes:
    - `consumer_tiles_per_group` — parameterized affine integer set `(i)[g]`
      selecting which tiles receive per group.
    - `producer_dependency_per_consumer` — optional affine integer set
      `(p)[c]` or `(p)[c, g]` naming, for each consumer tile, the producer it
      waits on and receives from. When absent, the consumer receives from
      every producer in its group, which is only meaningful when the group
      holds one.

    **Every consumer tile must have exactly one source.** A result has room
    for one contribution and there is no combiner to fold a second, so a
    consumer tile receiving from two producers has no defined value. With one
    producer per group that holds automatically and the attribute is a pure
    synchronization refinement — it changes when each consumer unblocks, not
    what it receives. That is broadcast.

    With **several** producers per group the attribute is required, and it
    names the sender: such a group is a *routing* pattern, several
    independent point-to-point deliveries sharing one produce op, rather than
    an all-producers delivery. A producer may serve several consumer tiles;
    it may not serve none.

    The `groups` integer set is inferred from the operand `!ktdp.tile_future`
    type.

    Example:
    ```mlir
    %r = ktdp.inter_tile_consume(%f)
        consumer_tiles_per_group = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
        : <(tensor<1x64xf16>), groups = affine_set<(g) : (g == 0)>>
          -> tensor<1x64xf16>
    ```
  }];

  let arguments = (ins
    Ktdp_TileFutureType:$future,
    Builtin_IntegerSetAttr:$consumer_tiles_per_group,
    OptionalAttr<Builtin_IntegerSetAttr>:$producer_dependency_per_consumer);
  let results = (outs Variadic<AnyRankedTensor>:$results);

  let assemblyFormat = [{
    `(` $future `)` `consumer_tiles_per_group` `=` $consumer_tiles_per_group
    (`,` `producer_dependency_per_consumer` `=` $producer_dependency_per_consumer^)?
    attr-dict `:` type($future) `->` type($results)
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Convenience: the partial types carried by the consumed future.
    ::llvm::ArrayRef<::mlir::RankedTensorType> getPartialTypes() {
      return getFuture().getType().getPartialTypes();
    }
    /// The groups set, read from the operand tile_future type.
    ::mlir::IntegerSet getGroups() {
      return getFuture().getType().getGroups().getValue();
    }
  }];
}

def KTDP_InterTileReduceOp : Ktdp_Op<"inter_tile_reduce",
    [SingleBlockImplicitTerminator<"::mlir::ktdp::YieldReducedOp">,
     RecursiveMemoryEffects,
     RangedTypesMatchWith<
         "result types must match future partial types",
         "future", "results",
         "llvm::cast<TileFutureType>($_self).getPartialTypes()">,
     TypesMatchWith<
         "identity types must match result types",
         "results", "identity", "$_self">]> {
  let summary = "Reduction delivery op (all-reduce / reduce)";
  let description = [{
    The `ktdp.inter_tile_reduce` operation consumes the `!ktdp.tile_future`
    produced by `ktdp.inter_tile_produce` and combines the partials of every
    producer tile in each group with an associative-commutative combiner,
    delivering the fully reduced result to the consumer tiles.

    The reducer region receives `2N` block arguments
    (`%lhs_1..%lhs_N, %rhs_1..%rhs_N`, each of partial type `T_p_i`) and
    terminates with `ktdp.yield_reduced`, yielding `N` combined values of the
    partial types. The combiner must be pure.

    Attributes:
    - `consumer_tiles_per_group` — tiles that receive the reduced result.
    - `producer_dependency_per_consumer` *(optional)* — per-tile dependency
      set; when absent, full-barrier semantics (all producers contribute).

    The `groups` integer set is inferred from the operand `!ktdp.tile_future` type.

    `identity` provides one identity tensor per role (matching `T_p_i`),
    hoisted before the op and shared across groups/tiles.

    Each result `T_r_i` has the same type as the corresponding partial type `T_p_i`.
    Every consumer tile in a group holds that group's fully reduced result
    (all-reduce when consumer set == producer set).

    Example:
    ```mlir
    %r = ktdp.inter_tile_reduce(%f)
        consumer_tiles_per_group = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>,
        identity(%add_id : tensor<1x64xf16>)
        : !ktdp.tile_future<tensor<1x64xf16>, groups = affine_set<(g) : (g >= 0, -g + 7 >= 0)>> -> tensor<1x64xf16>
    {
      ^bb0(%lhs: tensor<1x64xf16>, %rhs: tensor<1x64xf16>):
        %s = linalg.add ins(%lhs, %rhs : tensor<1x64xf16>, tensor<1x64xf16>)
                        outs(%lhs : tensor<1x64xf16>) -> tensor<1x64xf16>
        ktdp.yield_reduced %s : tensor<1x64xf16>
    }
    ```
  }];

  let arguments = (ins
    Ktdp_TileFutureType:$future,
    Variadic<AnyRankedTensor>:$identity,
    Builtin_IntegerSetAttr:$consumer_tiles_per_group,
    OptionalAttr<Builtin_IntegerSetAttr>:$producer_dependency_per_consumer);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$combiner);

  let assemblyFormat = [{
    `(` $future `)` `consumer_tiles_per_group` `=` $consumer_tiles_per_group `,`
    (`producer_dependency_per_consumer` `=` $producer_dependency_per_consumer^ `,`)?
    `identity` `(` ($identity^ `:` type($identity))? `)`
    attr-dict `:` type($future) `->` type($results) $combiner
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;

  let extraClassDeclaration = [{
    /// Convenience: the partial types carried by the consumed future.
    ::llvm::ArrayRef<::mlir::RankedTensorType> getPartialTypes() {
      return getFuture().getType().getPartialTypes();
    }
    /// The groups set, read from the operand tile_future type.
    ::mlir::IntegerSet getGroups() {
      return getFuture().getType().getGroups().getValue();
    }
  }];
}

def KTDP_GetComputeTileIdOp : Ktdp_Op<"get_compute_tile_id",
  [Pure]> {
    let summary = "Gets the id of the current compute tile";
    let description = [{
        The `ktdp.get_compute_tile_id` operation returns the identifier of the
          currently executing compute tile.

          This operation models a query to the execution environment that exposes
          the logical ID of the compute tile on which the operation is executed.
          The returned value uniquely identifies the tile within the device's
          execution grid or topology and can be used to specialize computation,
          index into distributed data structures, or select tile-specific memory
          regions.

          Example:
          ```
          %tile_id = ktdp.get_compute_tile_id : index
          ```
    }];

    let arguments = (ins);
    let results = (outs Variadic<Index>:$result);
    let assemblyFormat = [{
         attr-dict `:` type(results)
    }];

    let hasVerifier = 1;
}

// Create Tensor View
def KTDP_ConstructMemoryViewOp : Ktdp_Op<"construct_memory_view", [
      AttrSizedOperandSegments,
      MemRefsNormalizable,
      Pure
    ]> {
    let summary = "Operation to construct memory view.";
    let description = [{
      The `ktdp.construct_memory_view` operation creates a memref view that
      represents a tensor located in a specified memory space at a given base
      address, assuming allocation has already been done.

      This operation materializes a logical view from a raw memory
      location. The `offset` operand specifies the start address in the memory space,
      while the `sizes` and `strides` operands define the logical shape and layout of the view.
      Static components of sizes and strides may be encoded in the `static_sizes` and
      `static_strides` attributes, while dynamic components are supplied as SSA
      operands. Together, these describe a strided layout mapping logical tensor
      coordinates to memory addresses.

      The result type is a memref whose element type and memory space are
      determined by the result type annotation. The operation itself does not
      allocate memory; it only constructs a view over an existing region of
      memory.

      The `coordinate_set` attribute specifies the subset of logical
      coordinates associated with this view relative to a larger global or
      distributed tensor. This allows multiple memory views to represent
      disjoint or overlapping regions of a conceptual global tensor view when
      building distributed or partitioned views. When no special distribution
      semantics are required, the coordinate set typically corresponds directly
      to the full index space defined by `sizes`, representing a standard dense
      tensor view. When dimension sizes are symbolic, the symbols in the
      `coordinate_set` integer set are bound to variables in the `sizes` operand,
      from left-to-right in a one-to-one fashion.

      The `memory_space` attribute describes the logical memory space
      associated with return value (e.g., a memref view).

      Conceptually, this operation separates *memory interpretation* from
      *memory allocation*: it defines how a region of memory should be viewed as
      a tensor, but does not impose any constraints on how that memory was
      created or managed.

      Example:
      ```
        #set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 64 >= 0)>
        %A_view = ktdp.construct_memory_view %A_start_address, sizes: [32, 64], strides: [64, 1] {
            coordinate_set = #set, memory_space = #ktdp.memory_space<global>
        } : memref<32x64xf16>
      ```
    }];

  let arguments = (ins Index:$offset,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       DenseI64ArrayAttr:$static_sizes,
                       DenseI64ArrayAttr:$static_strides,
                       Ktdp_MemorySpaceAttr:$memory_space,
                       Builtin_IntegerSetAttr:$coordinate_set);
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $offset `` `,`  `sizes` `` `:`
    custom<DynamicIndexList>($sizes, $static_sizes)
    `` `,` `strides` `` `:`
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}

def KTDP_ConstructDistributedMemoryViewOp : Ktdp_Op<"construct_distributed_memory_view", [Pure]> {

    let summary = "Operation to construct distributed memory view over multiple memref objects";
    let description = [{
      The `ktdp.construct_distributed_memory_view` operation constructs a single
      logical memref view by composing multiple per-partition memory views, each
      typically produced by `ktdp.construct_memory_view`.

      Each input memref represents a view into a physical memory region, often
      residing in a particular memory space (e.g., per-tile scratchpad, HBM
      partition) and covering a subset of the coordinates of a
      conceptual global tensor. The association between an input memref and its
      covered region in the global index space is defined by the `coordinate_set`
      carried by the corresponding `construct_memory_view` that created it.

      This operation combines these individual views into a unified distributed
      view whose global coordinate domain is the union of the coordinate sets of
      the inputs. Conceptually, the result behaves like a single memref indexed
      over the global tensor dimensions, while its physical storage is distributed
      across the provided underlying memrefs.

      The operation does not allocate or move data. It only establishes a logical
      mapping from global coordinates to one of the constituent memref views. If
      coordinate sets overlap, the behavior is unspecified unless additionally
      constrained by dialect rules (e.g., requiring disjointness) or by program
      semantics.

      The result type determines the element type, global shape, and memory-space
      abstraction used to represent the distributed view. Lowering is expected to
      resolve global indexing into selection of the appropriate underlying memref
      and a local coordinate computation, potentially producing explicit address
      calculation and communication when required by the target architecture.

      This operation is intended as a construction primitive for representing
      globally indexed tensors whose storage is partitioned across multiple memory
      spaces, enabling explicit modeling of distributed scratchpads and other
      non-uniform memory organizations in the IR.

      Example:
      ```
      %A_dview = ktdp.construct_distributed_memory_view (%A0_view, %A1_view
        : memref<32x64xf16>, memref<32x64xf16>) : memref<64x64xf16>
      ```
    }];

    let arguments = (ins Variadic<AnyMemRef>:$memrefs);
    let results = (outs AnyMemRef);
    let assemblyFormat = [{
      `(` ($memrefs^ `:` type($memrefs))? `)`  attr-dict `:` type(results)
    }];

    let hasVerifier = 1;
}

// Access Tiles
def KTDP_ConstructAccessTilesOp : Ktdp_Op<"construct_access_tile",
        [AttrSizedOperandSegments, Pure]> {

  let summary = "Operation to construct access tiles over a tensor or memref objects.";
  let hasVerifier = 1;
    let description = [{
      The `ktdp.construct_access_tile` operation models a logical collection of coordinates
      derived from a source memref or tensor. It constructs an *access tile*,
      which represents a structured subset of indices into the underlying data
      object.

      The access tile is defined relative to a base coordinate computed from an affine
      map applied to a list of operands, which may include induction variables,
      symbols, or constants. This base coordinate acts as the anchor point for
      the access tile. The region of coordinates contained in the access tile is then specified
      by an IntegerSet attribute that describes the valid points relative to this
      base. Using an IntegerSet allows the access tile to represent general polyhedral
      regions--not just rectangular blocks--but also skewed, strided, triangular,
      or otherwise constrained coordinate sets.

      The `access_tile_set` IntegerSet may contain symbols. When symbols are present,
      the `symbol_operands` list provides SSA values that are bound to these symbols
      in left-to-right order. This allows the coordinate set (and hence the tile shape)
      to be parameterized by runtime values.

      Conceptually, this operation separates *address computation* from *data
      access*: it materializes a symbolic set of coordinates without performing
      any memory read or write. The resulting value is an access tile whose elements
      encode coordinates describing the selected region. Each element in the access tile
      is represented as an `index` typed value that corresponds to an encoded coordinate.
      This coordinate tile is intended to be consumed by subsequent operations (e.g., loads,
      stores, gathers, or scatters) that interpret it as an explicit access specification.

      Example:
      ```
        %A_access_tile = ktdp.construct_access_tile %A_view[%c0, %c0] {
            access_tile_set = #set
            access_tile_order = #map
        } : memref<32x64xf16> -> !ktdp.access_tile<32x64xindex>
      ```
    }];

  let arguments = (ins AnyTypeOf<[AnyMemRef,AnyTensor]>:$base,
        AffineMapAttr:$base_map,
        Variadic<Index>:$indices,
        Variadic<Index>:$symbol_operands,
        Builtin_IntegerSetAttr:$access_tile_set,
        AffineMapAttr:$access_tile_order);

  let results = (outs Ktdp_AccessTileType:$result);
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "AccessTileType":$resultType, "Value":$base, "mlir::AffineMap":$base_map,
      "ValueRange":$indices, "ValueRange":$symbol_operands, "IntegerSet":$access_tile_set,
      "mlir::AffineMap":$access_tile_order)>,
    OpBuilder<(ins "AccessTileType":$resultType, "Value":$base, "mlir::AffineMap":$base_map,
      "ValueRange":$indices, "IntegerSet":$access_tile_set, "mlir::AffineMap":$access_tile_order)>
  ];

  let extraClassDeclaration =  [{
    static StringRef getBaseMapAttrStrName() { return "base_map"; }
    static StringRef getAccessTileSetAttrStrName() { return "access_tile_set"; }
    static StringRef getAccessTileOrderAttrStrName() { return "access_tile_order"; }
  }];
}

def KTDP_ConstructIndirectAccessTilesOp : Ktdp_Op<"construct_indirect_access_tile",
        [AttrSizedOperandSegments,SingleBlockImplicitTerminator<"ktdp::RegionTerminatorOp">,
         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let summary = "Operation to construct access tiles over a tensor or memref objects.";
  let hasVerifier = 1;
    let description = [{
      The `ktdp.construct_indirect_access_tile` operation constructs an *access tile*
      for indirect (a.k.a. gather/scatter-style) indexing into a base memref/tensor.

      This operation specifies a multi-dimensional base memory view (`$base`) that is
      accessed via one or more indirect subscripts. There is an affine subscript expression
      for each dimension of the base memory view. Each subscript expression may be a direct
      access subscript or an indirect access subscript involving another memory view.
      Whether a subscript expression is direct or indirect is indicated via the boolean array
      attribute `per_dim_subscript_kinds`. A direct access subscript expression indexes
      directly into the corresponding dimension of `$base` memory view. An indirect access
      subscript expression indexes into a different memory view (selected from the list of
      memrefs provided by `$indirect_memrefs`) to load a final index into the corresponding
      dimension of the `$base` memory view.

      In order to facilitate parsing, an `ind(...)` marker is used to indicate that a subscript
      is an indirect access subscript. Direct subscripts are wrapped in parentheses.

      The subscript expressions may reference the `$captured_variables` operands (typically SSA
      values representing induction variables, symbols, or constants) prior to this operation,
      and may also reference *temporary/intermediate variables* defined in the operation's (hidden) region.
      The same common variables may be shared across multiple dimension subscripts,
      enabling coupled indexing patterns (e.g., both `IDX1[m,k]` and `IDX2[m,k]`
      using the same `(m,k)` iterators).

      The `$variables_space_set` integer set constrains the domain of all temporary
      variables used by the subscripts (introduced by the hidden region).
      It defines the set of legal values these intermediate variables may take while
      forming the access tile. Conceptually, the operation enumerates points in this
      variable space and, for each point, evaluates the per-dimension subscript
      expressions to produce a coordinate tuple into `$base` or one of `indirect_memrefs` views.
      For example `{(m, k) | 0 <= m < 2, 0 <= k < 64}` forms a 2x64 coordinate set.

      The `variables_space_set` IntegerSet may contain symbols. When symbols are present,
      the `symbol_operands` list provides SSA values that are bound to these symbols
      in left-to-right order. This allows the variable space (and hence the tile shape)
      to be parameterized by runtime values.

      The `$variables_space_order` attribute specifies the logical iteration ordering of the
      intermediate variable space. It is expressed as an AffineMap and defines the lexicographic
      traversal order over the dimensions of the variable space, where the rightmost output
      dimension corresponds to the innermost iteration dimension.
      Please note that this ordering applies to the enumeration of points in the intermediate
      variable space, and not to the ordering of components within an individual coordinate.
      Note that this map could have an impact on the shape of the access tile being constructed.
      For example using the 2x64 coordinate set above, an identity map would produce a 2x64 access
      tile while a transpose map (ie (d0, d1) -> (d1, d0)) would produce a 64x2 access tile.

      Indirect dimensions implicitly read from their associated index views: for a
      given variable-space point, the operation evaluates the subscript for that
      dimension to form an index into the corresponding memory view, loads the
      index value from that view, and uses it as the coordinate for the `$base`
      dimension. Direct dimensions bypass this indirection and use the evaluated
      subscript result directly as the coordinate.

      The result is an access-tile value (modeled as a tile of indices)
      that can be consumed by subsequent load/store operations. The operation itself
      does not perform any access to `$base`; it only materializes the coordinate
      collection describing where accesses should occur.
      Lowering is expected to resolve this into explicit address computation and
      the required loads from index views.

      Example: for `Y[m,k] = X[ IDX1[m,k], IDX2[m,k] ]`, the access tile is formed by
      iterating over `(m,k)` in the specified domain, loading `IDX1[m,k]` and
      `IDX2[m+1,k]`, and using the loaded values as the two-dimensional coordinates
      into `X`.

       ```

       %Y = ktdp.construct_indirect_access_tile intermediate_variables(%m, %k)
                      %X[ ind(%IDX1[%m, %k]), ind(%IDX2[%m + 1, %k]) ] {
                      variables_space_set = {(m, k) | 0 <= m < 2, 0 <= k < 64},
                      variables_space_order = (m,k) -> (m,k)
                    } : memref<64x64xf16>, memref<64x64xf16>, memref<64x64xf16> -> !ktdp.access_tile<2x64xindex>
       ```
    }];

  let arguments = (ins AnyTypeOf<[AnyMemRef,AnyTensor]>:$base,
                       ArrayAttr:$per_dim_subscript_kinds,
                       ArrayAttr:$per_dim_subscript_maps,
                       Variadic<AnyMemRef>:$indirect_memrefs,
                       Variadic<Index>:$captured_variables,
                       Variadic<Index>:$symbol_operands,
                       Builtin_IntegerSetAttr:$variables_space_set,
                       AffineMapAttr:$variables_space_order);

  let regions = (region AnyRegion:$region);

  let results = (outs Ktdp_AccessTileType:$result);
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "AccessTileType":$resultType, "Value":$base, "ArrayAttr":$per_dim_subscript_kinds,
    "ArrayAttr":$per_dim_subscript_maps, "ValueRange":$indirect_memrefs,
    "ValueRange":$captured_variables, "ValueRange":$symbol_operands,
    "IntegerSet":$variables_space_set, "AffineMap":$variables_space_order)>,
    OpBuilder<(ins "AccessTileType":$resultType, "Value":$base, "ArrayAttr":$per_dim_subscript_kinds,
    "ArrayAttr":$per_dim_subscript_maps, "ValueRange":$indirect_memrefs,
    "ValueRange":$captured_variables, "IntegerSet":$variables_space_set,
    "AffineMap":$variables_space_order)>
  ];

  let extraClassDeclaration =  [{
    static StringRef getPerDimSubscriptKindsAttrStrName() {return "per_dim_subscript_kinds"; }
    static StringRef getPerDimSubscriptMapsAttrStrName() { return "per_dim_subscript_maps"; }
    static StringRef getVariablesSpaceSetAttrStrName() { return "variables_space_set"; }
    static StringRef getVariablesSpaceOrderAttrStrName() { return "variables_space_order"; }
    ValueRange getIntermediateVariables() { return getRegion().getArguments(); }
  }];
}

// Load operation
def KTDP_LoadOp : Ktdp_Op<"load",
        [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let summary = "Operation to load data based on the coordinates from access_tile.";
    let description = [{
      The `load` performs a logical read from a source memref/tensor
      using an *access tile* that encodes a collection of coordinates.

      The access tile provides the index tuples to be read. Conceptually, the op
      iterates over the coordinates described by the access tile and gathers the
      corresponding elements from the source, producing a tensor containing the
      loaded values.

      The result is always a tensor. Its shape corresponds to the shape of the
      access tile's coordinate collection, and its element type matches the
      source element type.

      Note: This op is intended to consume the result of an access-tile
      construction op.

      Example:
      ```
        %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<32x64xindex> -> tensor<32x64xf16>
      ```
    }];

  let arguments = (ins Ktdp_AccessTileType:$access_tile);
  let results = (outs AnyTensor:$result);
    let assemblyFormat = [{
      $access_tile attr-dict `:` type($access_tile) `->` type(results)
    }];

  let builders = [
    // Builder that infers the result tensor shape from the access_tile
    OpBuilder<(ins "Value":$accessTile, "Type":$elementType)>
  ];

  let hasVerifier = 1;
}

// Store operation
def KTDP_StoreOp : Ktdp_Op<"store",
        [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let summary = "Operation to store data based on the coordinates from access_tile.";
    let description = [{
      The `store` operation performs a logical write into a destination
      memref/tensor using an *access tile* that encodes a collection of coordinates,
      and a *data tile* that provides the values to be written.

      The access tile (typically produced by `construct_access_tiles`) specifies the set of
      index tuples that define the write footprint. The data tile (typically
      produced by `tiled_load` and/or subsequent compute operations) supplies one
      value per coordinate in the access tile. Conceptually, the operation iterates
      over the coordinate collection described by the access tile and stores the
      corresponding element from the data tile into the destination at that
      coordinate.

      This operation models explicit data movement to memory: it has write effects
      on the destination. The destination element type must match the element type
      of the data tile. The shape of the data tile must match the logical iteration
      shape of the access tile (i.e., the number and structure of coordinate points
      described by the tile), establishing a one-to-one correspondence between
      coordinates and stored values.

      The `store` op does not define an ordering among writes beyond the
      dialect's semantics for the access tile.

      Example:
      ```
      ktdp.store %A_data_tile, %A_access_tile : tensor<32x64xf16>, !ktdp.access_tile<32x64xindex>
      ```
    }];

  let arguments = (ins AnyTensor:$data_tile, Ktdp_AccessTileType:$access_tile);
    let assemblyFormat = [{
      $data_tile `,` $access_tile attr-dict `:` type($data_tile) `,` type($access_tile)
    }];

  let hasVerifier = 1;
}

// Runtime argument extract operation
def KTDP_RuntimeArgExtractOp : Ktdp_Op<"runtime_arg_extract", [Pure]> {
  let summary = "Extract a value or attribute from a runtime_arg";
  let description = [{
    The `ktdp.runtime_arg_extract` operation extracts either the underlying value
    or one of the optional attributes (granularity or upperbound) from a runtime_arg type.

    The operation takes a runtime_arg operand and an extraction kind attribute that
    specifies what to extract:
    - `value`: Extracts the underlying SSA value with the runtime_arg's underlying type
    - `granularity`: Extracts the granularity attribute as an index value
    - `upperbound`: Extracts the upperbound attribute as an index value

    Examples:

    ```mlir
    %val = ktdp.runtime_arg_extract value from %M_sym : !ktdp.runtime_arg<index, granularity=3, upperbound=300> -> index
    %gran = ktdp.runtime_arg_extract granularity from %M_sym : !ktdp.runtime_arg<index, granularity=3, upperbound=300> -> index
    %ub = ktdp.runtime_arg_extract upperbound from %M_sym : !ktdp.runtime_arg<index, granularity=3, upperbound=300> -> index
    ```

    Note: Attempting to extract granularity or upperbound from a runtime_arg that
    doesn't have those attributes will result in a verification error.
  }];

  let arguments = (ins
    KTDP_RuntimeArgExtractKind:$kind,
    Ktdp_RuntimeArgType:$input
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $kind `from` $input `:` type($input) attr-dict `->` type($result)
  }];

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "RuntimeArgExtractKind":$kind, "Value":$input)>
  ];
}
#endif // KTDP_OPS_TD
