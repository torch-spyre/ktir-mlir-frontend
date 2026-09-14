// RUN: ktir-opt %s --ktir-check-legality | ktir-opt | FileCheck %s
//
// KT-07, half (a) — four raw contributions folded once.
//
// torch-spyre#4300 lx_relayout_workload_coverage.md §5 KT-07: "Raw
// contributions 1, 2, 4, 8 versus already-completed value 15.  Distinguish
// reduction from delivery of a completed sum".  Half (b) is the completed-sum
// delivery, in kt07b-completed-sum.mlir.  §1 of that document states the
// reference: "sum the independent contributions once.  If the producing matmul
// has already completed that sum, the following copy must not sum it again."
//
// OPERATION SEQUENCE.
//
//   LX-Load (addr_0) -> LocalReduce -> Expand -> Produce -> Reduce[ Add ]
//
//   addr_0 = the tile's own contribution slab.  `LocalReduce` is ordinary `linalg`
//   over the 128 rows, before any inter-tile op; `Reduce[ Add ]` is the cross-tile
//   fold with its combiner region.  The result is left unstored — a fold may travel
//   a different ring and need not land in LX per tile, unlike a copy delivery.
//
// ============================================================================
// TWO REDUCTIONS, AND ONLY ONE OF THEM IS THE INTER-TILE OP.
//
//   tensor<1x128x64xf16>   each tile's own slab
//        | linalg.reduce over dim 1        <- LOCAL, along an axis
//   tensor<1x64xf16>
//        | tensor.expand_shape
//   tensor<1x1x64xf16>     the partial
//        | ktdp.inter_tile_reduce          <- CROSS-TILE, elementwise
//   tensor<1x1x64xf16>     the result, same type
//
// The axis reduced is dim 1, extent 128 — a plain data axis, **not** the
// trailing stick-sized 64, which is carried through untouched.  That is the
// point of writing it this way: it shows what collapses and what does not.
//
// The inter-tile op reduces no axis at all.  `placement = replicate`, so by §4
// the result type equals the partial type; the fold runs across *tiles* and is
// applied elementwise at each of the 64 coordinates, which are never combined
// with one another.  Layout therefore plays no part in this case;
// kt02-receiving-only-cores.mlir is the example that carries the
// layout-verified physical form.
// ============================================================================
//
// Group structure.  One group, four tiles, every tile both producer and consumer.
//
//   group g   producers {4g .. 4g+3}   consumers {4g .. 4g+3}   contributions
//   g=0       {0, 1, 2, 3}             {0, 1, 2, 3}             1, 2, 4, 8 -> 15
//   ...       1 group only (g = 0)
//
// Fixture.
// The harness writes 2^t / 128 into every element of core t's own LX slab, so
// the local sum over the 128 rows is exactly 2^t and the four contributions
// entering the fold are 1, 2, 4, 8.  The fully reduced value is 15.
//
//   core 0 -> 1        expected at every core after the all-reduce: 15
//   core 1 -> 2
//   core 2 -> 4        a missing contribution gives 14, 13, 11 or 7
//   core 3 -> 8        a double-counted one gives 16 or more
//
// That is why the values are powers of two rather than a ramp: the result names
// exactly which contributions arrived, which is the bit-at-a-time scheme
// coverage §1 asks for.  Every value here is exact in fp16 — 2^-7 .. 2^-4, the
// partial sums k * 2^(t-7) for k <= 128, and 15.
//
// Why this half has no measured instance.  All 130 records of the pinned
// ownership catalog are `STCDPOpLx` relayouts, and every route class is
// copy-only: all_gather 26, grouped_all_gather_with_replication 65,
// replicate_or_owner_remap 32, permutation 6, general_relayout 1.  Not one is a
// reduction, which is what §9.3 of ../inter-tile-communication.md says of the 51
// and holds for all 130.  A genuine cross-core fold comes from split-K matmul
// instead — the pattern `inter_tile_reduce` was implemented for.  Half (b), by
// contrast, is what every one of those 130 records is.
//
// R13 and R14 are both satisfied: consumer set == producer set, so the consumers
// are producers (R13, `y` for `reduce`) and this is the all-reduce arm of R14.

#tiles     = affine_set<(i)[g] : (i - 4 * g >= 0, -i + 4 * g + 3 >= 0)>
#one_group = affine_set<(g) : (g == 0)>

// The whole of the tile's own slab.
#slab   = affine_set<(d0, d1, d2) : (
    d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#ident3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @kt07a_raw_contributions
module {
  func.func @kt07a_raw_contributions() {
    %c0   = arith.constant 0 : index
    %base = arith.constant 0 : index
    %zero = arith.constant 0.0 : f16

    // The contributions are input data, read from the tile's own LX.
    // `ct_local` with no ct_id is the executing tile's local memory, and the
    // slab is that tile's whole buffer, so no tile id enters the address.
    // Dense: lane 1, dim1 64, dim0 128*64 = 8192.
    %own = ktdp.construct_memory_view %base,
        sizes: [1, 128, 64], strides: [8192, 64, 1] {
        coordinate_set = #slab,
        memory_space   = #ktdp.memory_space<ct_local>
    } : memref<1x128x64xf16>

    %own_access = ktdp.construct_access_tile %own[%c0, %c0, %c0] {
        access_tile_set = #slab, access_tile_order = #ident3
    } : memref<1x128x64xf16> -> !ktdp.access_tile<1x128x64xindex>

    %slab = ktdp.load %own_access
        : !ktdp.access_tile<1x128x64xindex> -> tensor<1x128x64xf16>

    // LOCAL reduction, along dim 1 (128).  Ordinary linalg; the inter-tile ops
    // are not involved.  The trailing 64 survives.
    %red_init = tensor.empty() : tensor<1x64xf16>
    %red_zero = linalg.fill ins(%zero : f16) outs(%red_init : tensor<1x64xf16>)
                  -> tensor<1x64xf16>
    %local = linalg.reduce { arith.addf }
               ins(%slab : tensor<1x128x64xf16>)
               outs(%red_zero : tensor<1x64xf16>)
               dimensions = [1]

    // Restore the reduced axis as a unit dim so the partial keeps the rank of
    // the slab: dim 1 is now 1, the "already folded" axis.
    %partial = tensor.expand_shape %local [[0], [1, 2]]
                 output_shape [1, 1, 64]
                 : tensor<1x64xf16> into tensor<1x1x64xf16>

    // identity for the fold: 0.0, shaped as the partial type (R11).
    %id_init = tensor.empty() : tensor<1x1x64xf16>
    %identity = linalg.fill ins(%zero : f16) outs(%id_init : tensor<1x1x64xf16>)
                  -> tensor<1x1x64xf16>

    // CHECK: ktdp.inter_tile_produce
    %future = ktdp.inter_tile_produce
        producer_tiles_per_group = #tiles
        -> !ktdp.tile_future<(tensor<1x1x64xf16>), groups = #one_group>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial : tensor<1x1x64xf16>
    }

    // CROSS-TILE fold.  combine = fold, placement = replicate: no axis changes.
    // Writing the combiner region is what grants the scheduler permission to
    // re-associate (§3.5, "the associative-commutative contract is by user
    // agreement"), so tree, ring or linear are all legal.  Every consumer holds
    // the same value (§3.7).
    //
    // CHECK: ktdp.inter_tile_reduce
    %reduced = ktdp.inter_tile_reduce(%future)
        consumer_tiles_per_group = #tiles,
        identity(%identity : tensor<1x1x64xf16>)
        : !ktdp.tile_future<(tensor<1x1x64xf16>), groups = #one_group>
          -> tensor<1x1x64xf16>
    {
      ^bb0(%lhs: tensor<1x1x64xf16>, %rhs: tensor<1x1x64xf16>):
        %acc = tensor.empty() : tensor<1x1x64xf16>
        %sum = linalg.add
                 ins(%lhs, %rhs : tensor<1x1x64xf16>, tensor<1x1x64xf16>)
                 outs(%acc : tensor<1x1x64xf16>) -> tensor<1x1x64xf16>
        ktdp.yield_reduced %sum : tensor<1x1x64xf16>
    }

    // Expected: 15.0 at each of the 64 trailing coordinates, on all four cores.
    return
  }
}
