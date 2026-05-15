# Inter-tile Reductions in KTIR (all-reduce, reduce-scatter)

**Scope:** Two structured operations in the `ktdp` dialect for cross-tile
reductions over a partitioned set of compute tiles.

---

## 1. Overview

`ktdp.inter_tile_all_reduce` and `ktdp.inter_tile_reduce_scatter` express
cross-tile reductions across a participating set of compute tiles organized
into disjoint groups. The two ops share a common structure — operand list,
producer / reducer regions, value-typed result semantics, synchronization
model — and differ only in what each tile receives back:

- **`inter_tile_all_reduce`:** every tile in a group ends up holding that
  group's reduced result.
- **`inter_tile_reduce_scatter`:** each group's reduced result is split
  row-major along a named dimension across that group's tiles; each tile
  holds its own slice.

Both ops are value-typed — their results are tensor SSA values produced
in each participating tile's SPMD execution. There are no explicit
barriers in the IR; synchronization arises from SSA dataflow over tensor
values combined with the producer / reducer contract that defines what
each tile contributes and how partials are combined.

This document presents the two ops directly. Section 2 covers the
structure shared between them. Sections 3 and 4 give per-op semantics and
worked examples.

---

## 2. Common structure

The pieces below apply identically to both ops. Sections 3 and 4 add only
op-specific details (result-type rules, the scatter `dimension` attribute).

### 2.1 Operands and attributes

**`identity`** — variadic SSA operands, one per partial-tensor role. Each
identity is a tensor whose **shape and element type match the
corresponding partial type** (not the rank-reduced or scattered result
type). The identity is the reduction's neutral element: combining it
with any partial yields that partial. There is one identity per
partial-tensor role, **shared across all groups and all tiles** — a
single SSA value, hoisted before the op.

**`participating_tiles_per_group`** — a parameterized affine integer set
naming the participating tiles for a given group. The set has one
dimension (the tile id) and one symbol (`g`, the group index). Bound at
op-use time, the set selects the tiles that belong to group `g`. For
example, `affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>` selects
tile ids `4g .. 4g+3` for any group index `g`. An enumerated form (a
list of tile-id lists) is supported as a fallback when the per-group
membership is irregular and not affine-expressible.

**`groups`** — an affine integer set defining the range of group indices.
For example, `affine_set<(g) : (g >= 0, -g + 7 >= 0)>` defines 8 groups,
indexed `0..7`.

**Disjointness invariant.** For any two distinct group indices
`g_1 != g_2` in `groups`, `participating_tiles_per_group(g_1)` and
`participating_tiles_per_group(g_2)` must be disjoint. Equivalently,
every participating tile is in exactly one group. The verifier enforces
this. The motivation is unambiguous reduction membership: each tile
contributes to exactly one group's reduction and receives exactly one
group's result.

### 2.2 Producer region

The producer region indicates **what partial results each tile
contributes to the reduction**. It is the per-tile boundary that names
the SSA values participating in cross-tile combine. The region runs
once per participating tile, in that tile's SPMD execution.

**Region argument:** `^producer(%gid: index)` — the index of the group
this tile belongs to. The runtime binding is direct: tile `t` finds its
group by looking up which entry of `participating_tiles_per_group(g)`
contains `t`; that `g` is bound to `%gid` for tile `t`'s execution of
the producer.

The body of the producer region knows its tile id via
`ktdp.get_compute_tile_id` (the same way every SPMD KTIR body does) and
its group index via `%gid`.

**Termination:** the producer terminates with
`ktdp.yield_partial p_1, ..., p_N`, where each `p_i` has type
`T_partial_i` (the type of the i-th partial-tensor role declared in the
op's signature). The yielded values may reference SSA values from the
enclosing scope — typical use is a thin contribution marker:

```mlir
producer {
  ^producer(%gid: index):
    ktdp.yield_partial %my_partial : tensor<...>
}
```

with the per-tile compute that produced `%my_partial` living at function
scope (where it is naturally executed by every tile under SPMD). Richer
producer bodies are allowed when the contribution is awkward to hoist —
e.g., compute that depends only on `%gid`, or visibly local
"contribution preparation" the author wants to keep adjacent to the op.

### 2.3 Reducer region

The reducer region indicates **how to perform the reduction over two
partial results**. It is a pairwise combine — given two partials of the
same shape (either two per-tile contributions or intermediate combined
values), it produces a single combined value of the same shape. The
scheduler invokes the reducer as many times as needed (in tree, ring,
or linear order) to fold all of a group's partials into one.

**Region arguments:** `^bb0(%lhs_1, ..., %lhs_N, %rhs_1, ..., %rhs_N)` —
`2N` total, with each `lhs_i` and `rhs_i` of type `T_partial_i`.

**Purity.** The reducer must be pure — no memory effects, no calls to
side-effecting ops. Pure tensor ops (`tensor.empty`, `linalg` on
tensors, `arith.*`) are allowed. The verifier rejects reducers that
contain any operation with side effects.

**Per-group combine.** The reducer is shared by all groups, but each
combine only ever sees partials from tiles within the *same* group.
Different groups' reductions are independent and may be scheduled in
parallel.

**Combine ordering.** The combine ordering is unspecified within each
group: the combiner is associative-commutative (by user contract) and
the scheduler is free to choose tree, ring, linear, hardware-native, or
any other topology.

**Termination:** `ktdp.yield_reduced r_1, ..., r_N`, each `r_i` of type
`T_partial_i`.

### 2.4 Synchronization model

The op carries no explicit barriers. Synchronization is provided by:

1. **SSA value semantics over tensors.** The op's result is a tensor SSA
   value. The value cannot materialize until the relevant per-group
   reductions feeding it have completed. Standard MLIR dataflow
   ordering applies.
2. **The producer / reducer regions as the contribution contract.** In
   SPMD KTIR, a tile cannot observe other tiles' partials except through
   a dialect-defined boundary. The producer region is that boundary —
   it names the per-tile contribution. The reducer region defines how
   contributions combine. Together they delimit the cross-tile
   reduction's data-flow graph.

Lowering passes are responsible for inserting whatever target-specific
synchronization (hardware barriers, memory fences, interconnect sync) is
needed to honor the value-typed semantics. None of that machinery
appears in the source IR.

### 2.5 Result semantics

Both ops produce N variadic SSA values, one per partial-tensor role. The
values are *per-tile-valued* — each participating tile holds a result
value when the op completes:

- **For `inter_tile_all_reduce`:** every tile in group `g` holds an
  identical value — `g`'s reduced result. Tiles in different groups
  hold different values (each its own group's reduction).
- **For `inter_tile_reduce_scatter`:** each tile in group `g` holds its
  own row-major slice of `g`'s reduced result along the scatter
  dimension. Different tiles in the same group hold different slices
  whose concatenation is `g`'s full reduced result.

**Non-participating tiles.** A tile that is not in any group of the op
must not lexically encounter the op in its execution path. The
disjointness invariant ensures every participating tile is in exactly
one group; if the op appears in the SPMD body, every tile reaching it
must be in some group (i.e., the union over `groups` of
`participating_tiles_per_group(g)` must equal the set of tiles that
execute the op).

**Multi-tensor (variadic) reductions.** N ≥ 1 partials are supported.
Argmax-style reductions, where each partial is a correlated tuple of
tensors (values, indices), use N = 2: two identities, two yielded
partials, four reducer region arguments yielding two combined values,
two op results. The structure generalizes uniformly.

---

## 3. `ktdp.inter_tile_all_reduce`

### 3.1 Op signature and semantics

Common operands and regions per §2. No additional attributes.

**Result-type rule.** For each partial type `T_partial_i` and its
corresponding result type `T_result_i`, the result type must be
derivable from the partial type by removing one or more unit dimensions.
The removed dimensions are *the within-group tile axes* — the axes
along which different tiles' partials within a group are combined. At
least one tile axis must be removed; the same set of removed axes must
be consistent across all partials. Any unit dimensions of the partial
that are *not* removed in the result are *group axes* — they tag the
slice with its group's position in the larger conceptual aggregate and
are preserved through the op.

The verifier infers the within-group tile axes by comparing each
partial type against its corresponding result type.

### 3.2 Worked example: single-group all-reduce (96×64)

**Layout and partitioning.** `A` and `B` are `tensor<96x64xf16>` in HBM.
The kernel computes the column-wise sum of `A + B`, producing a
1-D `tensor<64xf16>`.

The 32 compute tiles form a single group. Tile `t` owns rows
`t*3 .. t*3+2` of `A` and `B` — a 3×64 slab each. The per-tile
contribution is the 3×64-summed row-reduction, expanded to 1×64 (with
the leading unit dim playing the within-group tile axis role). The op
collapses that unit dim, producing a `tensor<64xf16>` that every tile
holds identically.

There is one group covering all 32 tiles; the producer region's
`%gid` is always 0.

**Full IR.**

```mlir
#A_view_set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#AB_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 +  2 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#E_view_set  = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
#E_tile_set  = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
#identity_2d = affine_map<(d0, d1) -> (d0, d1)>

// One group containing all 32 tiles.
#group_tiles = affine_set<(i)[g] : (i - 32*g >= 0, -i + 32*(g+1) - 1 >= 0)>
#all_groups  = affine_set<(g) : (g == 0)>

module {
  func.func @inter_tile_all_reduce_single_group() {
    %c0 = arith.constant 0 : index
    %tile_size = arith.constant 3 : index
    %A_start = arith.constant 1024  : index
    %B_start = arith.constant 12288 : index
    %E_start = arith.constant 22528 : index

    %A_view = ktdp.construct_memory_view %A_start, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>
    %B_view = ktdp.construct_memory_view %B_start, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    // Identity for sum: tensor<1x64xf16> of zeros.
    %c_zero   = arith.constant 0.0 : f16
    %add_init = tensor.empty() : tensor<1x64xf16>
    %add_id   = linalg.fill ins(%c_zero : f16) outs(%add_init : tensor<1x64xf16>)
                  -> tensor<1x64xf16>

    // Per-tile compute (function-scope SPMD).
    %t = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %t, %tile_size : index

    %A_access = ktdp.construct_access_tile %A_view[%start_row, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_2d
    } : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>
    %B_access = ktdp.construct_access_tile %B_view[%start_row, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_2d
    } : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>

    %A_tile = ktdp.load %A_access : !ktdp.access_tile<3x64xindex> -> tensor<3x64xf16>
    %B_tile = ktdp.load %B_access : !ktdp.access_tile<3x64xindex> -> tensor<3x64xf16>

    %AB_init = tensor.empty() : tensor<3x64xf16>
    %AB_sum  = linalg.add ins(%A_tile, %B_tile : tensor<3x64xf16>, tensor<3x64xf16>)
                          outs(%AB_init : tensor<3x64xf16>) -> tensor<3x64xf16>

    %red_init   = tensor.empty() : tensor<64xf16>
    %red_filled = linalg.fill ins(%c_zero : f16) outs(%red_init : tensor<64xf16>)
                    -> tensor<64xf16>
    %partial_1d = linalg.reduce { arith.addf }
                    ins(%AB_sum : tensor<3x64xf16>)
                    outs(%red_filled : tensor<64xf16>)
                    dimensions = [0]
    %partial_2d = tensor.expand_shape %partial_1d [[0, 1]] output_shape [1, 64]
                    : tensor<64xf16> into tensor<1x64xf16>

    // Single-group all-reduce. The unit dim 0 is the within-group tile axis;
    // the op collapses it. Every tile holds the same %reduced.
    %reduced = ktdp.inter_tile_all_reduce
        identity(%add_id : tensor<64xf16>)
        participating_tiles_per_group = #group_tiles,
        groups = #all_groups
        : (tensor<1x64xf16>) -> tensor<64xf16>
    producer {
      ^producer(%gid: index):
        ktdp.yield_partial %partial_2d : tensor<1x64xf16>

    } reducer {
      ^bb0(%lhs: tensor<1x64xf16>, %rhs: tensor<1x64xf16>):
        %init = tensor.empty() : tensor<1x64xf16>
        %sum  = linalg.add ins(%lhs, %rhs : tensor<1x64xf16>, tensor<1x64xf16>)
                           outs(%init : tensor<1x64xf16>) -> tensor<1x64xf16>
        ktdp.yield_reduced %sum : tensor<1x64xf16>
    }

    // Post-reduction: every tile redundantly writes the same value to
    // a 1x64 HBM region. In a realistic kernel %reduced would feed
    // a downstream computation.
    %reduced_2d = tensor.expand_shape %reduced [[0, 1]] output_shape [1, 64]
                    : tensor<64xf16> into tensor<1x64xf16>

    %E_view = ktdp.construct_memory_view %E_start, sizes: [1, 64], strides: [64, 1] {
        coordinate_set = #E_view_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<1x64xf16>
    %E_access = ktdp.construct_access_tile %E_view[%c0, %c0] {
        access_tile_set = #E_tile_set, access_tile_order = #identity_2d
    } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>

    ktdp.store %reduced_2d, %E_access
              : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>

    return
  }
}
```

### 3.3 Worked example: multi-group all-reduce (128×8×12×64)

**Layout and partitioning.** `A` and `B` are `tensor<128x8x12x64xf16>` in
HBM. The four axes have distinct roles:

- Dim 0 (size 128): preserved through this op.
- Dim 1 (size 8): the **group axis** — 8 groups.
- Dim 2 (size 12): the **reduction axis** — within each group, 4 tiles
  cooperate over this axis.
- Dim 3 (size 64): vector / stick axis, preserved.

There are 32 compute tiles forming 8 groups of 4. For tile `t`,
the group index is `g = t / 4` and the within-group local index is
`l = t % 4`. Tile `(g, l)` reads slice `[*, g, l*3 : l*3+3, *]` of `A`
and `B` — shape `<128x1x3x64>` each.

The per-tile pipeline loads its slabs, adds them, locally reduces over
the 3-deep slab (the reduction axis), and expands a unit dim at
position 2 to mark the within-group cross-tile axis. The resulting
partial is `<128x1x1x64>`: dim 1 is the group axis (size 1, preserved),
dim 2 is the within-group tile axis (size 1, the op collapses it).

The op produces `<128x1x64>` per tile — group axis preserved,
within-group tile axis collapsed. All four tiles in a group hold
identical values (their group's reduction); different groups hold
different values.

**Full IR.**

```mlir
#A_view_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 11  >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#AB_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 2   >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#E_view_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 3   >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#E_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 == 0,
     d3 >= 0, -d3 + 63 >= 0)>

#identity_4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups  = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

module {
  func.func @inter_tile_all_reduce_multi_group() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %red_slab = arith.constant 3 : index   // 12 / 4

    %A_start = arith.constant 1024     : index
    %B_start = arith.constant 12583936 : index   // 1024 + 128*8*12*64*2
    %E_start = arith.constant 25166848 : index   // 128x8x4x64 output

    %A_view = ktdp.construct_memory_view %A_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<128x8x12x64xf16>
    %B_view = ktdp.construct_memory_view %B_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<128x8x12x64xf16>

    // Identity for the within-group sum: tensor<128x1x1x64xf16> of zeros.
    %c_zero  = arith.constant 0.0 : f16
    %id_init = tensor.empty() : tensor<128x1x1x64xf16>
    %add_id  = linalg.fill ins(%c_zero : f16) outs(%id_init : tensor<128x1x1x64xf16>)
                 -> tensor<128x1x1x64xf16>

    // Per-tile compute (function-scope SPMD).
    %t = ktdp.get_compute_tile_id : index
    %g = arith.divui %t, %c4 : index
    %l = arith.remui %t, %c4 : index
    %red_anchor = arith.muli %l, %red_slab : index

    %A_access = ktdp.construct_access_tile %A_view[%c0, %g, %red_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>
    %B_access = ktdp.construct_access_tile %B_view[%c0, %g, %red_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>

    %A_tile = ktdp.load %A_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>
    %B_tile = ktdp.load %B_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>

    %AB_init = tensor.empty() : tensor<128x1x3x64xf16>
    %AB_sum  = linalg.add ins(%A_tile, %B_tile
                              : tensor<128x1x3x64xf16>, tensor<128x1x3x64xf16>)
                          outs(%AB_init : tensor<128x1x3x64xf16>)
                          -> tensor<128x1x3x64xf16>

    %red_init   = tensor.empty() : tensor<128x1x64xf16>
    %red_filled = linalg.fill ins(%c_zero : f16)
                              outs(%red_init : tensor<128x1x64xf16>)
                              -> tensor<128x1x64xf16>
    %partial_3d = linalg.reduce { arith.addf }
                    ins(%AB_sum : tensor<128x1x3x64xf16>)
                    outs(%red_filled : tensor<128x1x64xf16>)
                    dimensions = [2]

    %partial_4d = tensor.expand_shape %partial_3d [[0], [1], [2, 3]]
                    output_shape [128, 1, 1, 64]
                    : tensor<128x1x64xf16> into tensor<128x1x1x64xf16>

    // Multi-group all-reduce. Within each group of 4 tiles, dim 2 of the
    // partial is the within-group tile axis (collapsed). Dim 1 is the
    // group axis (preserved). Each tile gets its group's reduced <128x1x64>.
    %my_group_result = ktdp.inter_tile_all_reduce
        identity(%add_id : tensor<128x1x64xf16>)
        participating_tiles_per_group = #group_tiles,
        groups = #all_groups
        : (tensor<128x1x1x64xf16>) -> tensor<128x1x64xf16>
    producer {
      ^producer(%gid: index):
        ktdp.yield_partial %partial_4d : tensor<128x1x1x64xf16>

    } reducer {
      ^bb0(%lhs: tensor<128x1x1x64xf16>, %rhs: tensor<128x1x1x64xf16>):
        %init = tensor.empty() : tensor<128x1x1x64xf16>
        %sum  = linalg.add ins(%lhs, %rhs
                               : tensor<128x1x1x64xf16>, tensor<128x1x1x64xf16>)
                           outs(%init : tensor<128x1x1x64xf16>)
                           -> tensor<128x1x1x64xf16>
        ktdp.yield_reduced %sum : tensor<128x1x1x64xf16>
    }

    // Post-reduction: each tile writes its group's reduced result to
    // slice [*, g, l, *] of a 128x8x4x64 output. The 4 tiles in a group
    // hold identical values; the redundancy is the all-reduce broadcast
    // made physically visible.
    %my_result_4d = tensor.expand_shape %my_group_result [[0], [1, 2], [3]]
                      output_shape [128, 1, 1, 64]
                      : tensor<128x1x64xf16> into tensor<128x1x1x64xf16>

    %E_view = ktdp.construct_memory_view %E_start, sizes: [128, 8, 4, 64],
        strides: [2048, 256, 64, 1] {
        coordinate_set = #E_view_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<128x8x4x64xf16>

    %E_access = ktdp.construct_access_tile %E_view[%c0, %g, %l, %c0] {
        access_tile_set = #E_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x4x64xf16> -> !ktdp.access_tile<128x1x1x64xindex>

    ktdp.store %my_result_4d, %E_access
              : tensor<128x1x1x64xf16>, !ktdp.access_tile<128x1x1x64xindex>

    return
  }
}
```

---

## 4. `ktdp.inter_tile_reduce_scatter`

### 4.1 Op signature and semantics

Common operands and regions per §2. One additional attribute.

**`dimension`** (i64) — names an axis of the *post-reduction shape*
(i.e., the partial type with the within-group tile axes collapsed)
along which scatter occurs. Within each group, the reduced result is
split row-major along this axis across that group's tiles. The size
along `dimension` must be divisible by the per-group tile count;
non-divisible cases are rejected by the verifier.

**Result-type rule.** Conceptually:

```
post_reduction_type   = partial_type with within-group tile axes removed
chunk_size_along_dim  = post_reduction_type[dimension] / |tiles per group|
result_type           = post_reduction_type with dimension's extent replaced
                        by chunk_size_along_dim
```

The verifier checks that the result type matches this construction
given the partial type, the within-group tile axes (inferred from
post-reduction shape), the `dimension` attribute, and the per-group
tile count.

**Per-tile slice.** For a tile with within-group local index `l`, the
slice it receives is
`reduced[l*chunk : (l+1)*chunk]` along `dimension`. The local index `l`
is determined by the tile's position in
`participating_tiles_per_group(g)`. For affine sets, this is the
ascending order of tile ids satisfying the set; for enumerated lists,
it is the list-element order.

**Per-group independence.** Different groups' reductions and scatters
proceed independently and may be scheduled in parallel.

### 4.2 Worked example: multi-group reduce-scatter (128×8×12×64)

**Layout and partitioning.** `A` and `B` are `tensor<128x8x12x64xf16>`
in HBM. The four axes have distinct roles:

- Dim 0 (size 128): the **scatter axis** — within each group, this axis
  is split across that group's 4 tiles.
- Dim 1 (size 8): the **group axis** — 8 groups.
- Dim 2 (size 12): the **reduction axis** — within each group, 4 tiles
  cooperate over this axis.
- Dim 3 (size 64): vector / stick axis, preserved.

Tile partitioning is the same as in §3.3: 32 tiles, 8 groups of 4. For
tile `t`, `g = t / 4` and `l = t % 4`. Tile `(g, l)` reads slice
`[*, g, l*3 : l*3+3, *]` of `A` and `B` — shape `<128x1x3x64>` each.

The per-tile pipeline through to the partial is identical to §3.3:
load, add, local reduce over the 3-deep slab, expand to
`<128x1x1x64>`. The op then:

- Reduces dim 2 (the within-group tile axis, size 1):
  `<128x1x1x64>` across 4 tiles → `<128x1x64>` per group.
- Scatters dim 0 (the 128 axis), 128 / 4 = 32: each tile receives
  `<32x1x64>`.

Tile `(g, l)` ends up with rows `[l*32 : (l+1)*32]` of group `g`'s
reduced `<128x1x64>`. The output `tensor<128x8x64>` in HBM is tiled
exactly by the 32 tiles' `<32x1x64>` chunks at slices
`[l*32 : (l+1)*32, g, *]`.

**Full IR.**

```mlir
#A_view_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 11  >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#AB_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 2   >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

// E view (post-scatter output): 128x8x64.
#E_view_set = affine_set<(d0, d1, d2) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 63  >= 0)>

// E access tile per writer: 32x1x64 anchored at [l*32, g, 0].
#E_tile_set = affine_set<(d0, d1, d2) :
    (d0 >= 0, -d0 + 31 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 63 >= 0)>

#identity_4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#identity_3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups  = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

module {
  func.func @inter_tile_reduce_scatter_multi_group() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %red_slab      = arith.constant 3  : index   // 12 / 4
    %scatter_chunk = arith.constant 32 : index   // 128 / 4

    %A_start = arith.constant 1024     : index
    %B_start = arith.constant 12583936 : index   // 1024 + 128*8*12*64*2
    %E_start = arith.constant 25166848 : index   // 128x8x64 output

    %A_view = ktdp.construct_memory_view %A_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<128x8x12x64xf16>
    %B_view = ktdp.construct_memory_view %B_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<128x8x12x64xf16>

    // Identity: tensor<128x1x1x64xf16> of zeros (matches the partial shape).
    %c_zero  = arith.constant 0.0 : f16
    %id_init = tensor.empty() : tensor<128x1x1x64xf16>
    %add_id  = linalg.fill ins(%c_zero : f16) outs(%id_init : tensor<128x1x1x64xf16>)
                 -> tensor<128x1x1x64xf16>

    // Per-tile compute (function-scope SPMD).
    %t = ktdp.get_compute_tile_id : index
    %g = arith.divui %t, %c4 : index
    %l = arith.remui %t, %c4 : index
    %red_anchor = arith.muli %l, %red_slab : index

    %A_access = ktdp.construct_access_tile %A_view[%c0, %g, %red_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>
    %B_access = ktdp.construct_access_tile %B_view[%c0, %g, %red_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>

    %A_tile = ktdp.load %A_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>
    %B_tile = ktdp.load %B_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>

    %AB_init = tensor.empty() : tensor<128x1x3x64xf16>
    %AB_sum  = linalg.add ins(%A_tile, %B_tile
                              : tensor<128x1x3x64xf16>, tensor<128x1x3x64xf16>)
                          outs(%AB_init : tensor<128x1x3x64xf16>)
                          -> tensor<128x1x3x64xf16>

    %red_init   = tensor.empty() : tensor<128x1x64xf16>
    %red_filled = linalg.fill ins(%c_zero : f16)
                              outs(%red_init : tensor<128x1x64xf16>)
                              -> tensor<128x1x64xf16>
    %partial_3d = linalg.reduce { arith.addf }
                    ins(%AB_sum : tensor<128x1x3x64xf16>)
                    outs(%red_filled : tensor<128x1x64xf16>)
                    dimensions = [2]

    %partial_4d = tensor.expand_shape %partial_3d [[0], [1], [2, 3]]
                    output_shape [128, 1, 1, 64]
                    : tensor<128x1x64xf16> into tensor<128x1x1x64xf16>

    // Multi-group reduce-scatter. Reduce dim 2 (within-group tile axis,
    // size 1). Scatter dim 0 (128) across each group's 4 tiles, chunk = 32.
    // Group axis (dim 1, size 1) preserved. Each tile receives <32x1x64>.
    %my_chunk = ktdp.inter_tile_reduce_scatter
        identity(%add_id : tensor<128x1x64xf16>)
        participating_tiles_per_group = #group_tiles,
        groups = #all_groups,
        dimension = 0
        : (tensor<128x1x1x64xf16>) -> tensor<32x1x64xf16>
    producer {
      ^producer(%gid: index):
        ktdp.yield_partial %partial_4d : tensor<128x1x1x64xf16>

    } reducer {
      ^bb0(%lhs: tensor<128x1x1x64xf16>, %rhs: tensor<128x1x1x64xf16>):
        %init = tensor.empty() : tensor<128x1x1x64xf16>
        %sum  = linalg.add ins(%lhs, %rhs
                               : tensor<128x1x1x64xf16>, tensor<128x1x1x64xf16>)
                           outs(%init : tensor<128x1x1x64xf16>)
                           -> tensor<128x1x1x64xf16>
        ktdp.yield_reduced %sum : tensor<128x1x1x64xf16>
    }

    // Post-reduction: tile (g, l) writes its <32x1x64> to slice
    // [l*32 : l*32+32, g, *] of the 128x8x64 HBM output. 32 tiles tile
    // the entire output exactly — no redundancy.
    %my_row_anchor = arith.muli %l, %scatter_chunk : index

    %E_view = ktdp.construct_memory_view %E_start, sizes: [128, 8, 64],
        strides: [512, 64, 1] {
        coordinate_set = #E_view_set,
        memory_space   = #ktdp.spyre_memory_space<HBM>
    } : memref<128x8x64xf16>

    %E_access = ktdp.construct_access_tile %E_view[%my_row_anchor, %g, %c0] {
        access_tile_set = #E_tile_set, access_tile_order = #identity_3d
    } : memref<128x8x64xf16> -> !ktdp.access_tile<32x1x64xindex>

    ktdp.store %my_chunk, %E_access
              : tensor<32x1x64xf16>, !ktdp.access_tile<32x1x64xindex>

    return
  }
}
```

### 4.3 Single-group reduce-scatter

A single-group reduce-scatter is a degenerate case of the multi-group
form: `groups` contains exactly one entry, and
`participating_tiles_per_group` selects all participating tiles for that
single entry. The producer region's `%gid` is always 0, the result is
scattered across all participating tiles in row-major order along
`dimension`, and the structure is otherwise identical to the multi-group
form. No separate worked example is provided.

---

## 5. Notes

- **Variadic / multi-tensor reductions.** Both ops support N ≥ 1
  partials. Argmax-style reductions use N = 2: two identities (e.g.,
  `tensor<...xf16>` filled with `-inf`, `tensor<...xi32>` filled with
  a sentinel like `-1`); two yielded partials; four reducer arguments
  yielding two combined values; two op results. Each result follows the
  same per-op rules independently.
