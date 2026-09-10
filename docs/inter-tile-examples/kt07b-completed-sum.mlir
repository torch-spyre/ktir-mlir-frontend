// KT-07, half (b) — an already-completed sum is moved, not folded again.
//
// torch-spyre#4300 lx_relayout_workload_coverage.md §1: "Reduction tests have a
// different reference: sum the independent contributions once.  **If the
// producing matmul has already completed that sum, the following copy must not
// sum it again.**"  Half (a), the genuine fold, is in
// kt07a-raw-contributions.mlir and it verifies.
//
// OPERATION SEQUENCE.
//
//   LX-Load (addr_0) -> LocalReduce -> Expand -> Produce -> Consume
//
//   Identical to kt07a up to the delivery, which is the point: the two differ only
//   in `Reduce[ Add ]` versus `Consume`, and the shapes do not separate them.
//
// ============================================================================
// THE ONLY DIFFERENCE FROM HALF (a) IS THE DELIVERY OP.
//
// Both halves read a slab from the tile's own LX and reduce it locally over dim
// 1 to a tensor<1x1x64xf16> partial.  That block is textually identical here and
// in half (a).  What differs:
//
//   half (a)   4 producers, partials 1, 2, 4, 8   ktdp.inter_tile_reduce  -> 15
//   half (b)   1 producer,  partial  15           ktdp.inter_tile_consume -> 15
//
// The producer set has to differ too — R8 gives a broadcast one producer per
// group — but nothing else does.  Reading this edge as half (a) would fold a
// completed sum a second time.
//
// WHAT 15 IS, AND WHAT 60 IS.  15 = 1 + 2 + 4 + 8, the value half (a) computes;
// it is used here so both halves have the same answer and can be read side by
// side.  With one producer the bit-per-tile property of half (a) does not carry
// over — 15 identifies nothing here, it is simply the completed sum.
//
// 60 is NOT a possible outcome of this fixture: only core 4g holds a value, so
// there is nothing to fold four of.  60 is what the *wrong model* of this edge
// produces — reading it as a fold over four holders — which is why the case
// exists.  For a single-producer set the shipped verifier rejects that model
// outright (see STATUS below), so here the mistake is caught rather than
// mis-valued.  On an edge where several cores really do hold the completed sum
// it would return 60 silently instead, and only the recorded expected values
// would catch it.
//
// Note the shapes do not separate the two readings.  `reduce` keeps the partial
// type (§4, replicate), so a completed sum being copied and a fresh fold being
// computed have the same result type.  Only the expected values separate them,
// which is why the coverage document's test contract records, per edge, whether
// the inputs are "independent tensor elements, unfinished sums, or
// already-completed sums".
// ============================================================================
//
// Group structure.  One group, four tiles, but only one of them produces.
//
//   group g   producers {4g}   consumers {4g .. 4g+3}   non-producers
//   g=0       {0}              {0, 1, 2, 3}             1, 2, 3
//   ...       1 group only (g = 0)
//
// Fixture.  The harness writes 15 / 128 into every
// element of core 4g's own LX slab, and the local reduction below sums the 128
// rows to exactly 15.  That local reduction stands in for the producing matmul:
// it is where the sum is completed, and the delivery must not repeat it.  Cores
// 4g+1, 4g+2 and 4g+3 hold nothing and must still end with 15.  15/128 and
// every partial sum k * 15 * 2^-7 for k <= 128 are exact in fp16.
//
// THIS HALF IS WHAT THE MEASURED CATALOG IS.  All 130 records of the pinned
// ownership catalog are `STCDPOpLx` relayouts and every route class is
// copy-only: all_gather 26, grouped_all_gather_with_replication 65,
// replicate_or_owner_remap 32, permutation 6, general_relayout 1.  Not one is a
// reduction.  So the risk this case guards against is one-sided: the mistake to
// avoid is reading a copy as a fold, and 130 of the 130 measured edges are
// copies whose consumer names (`mean_*`, `_safe_softmax-Sum`, `mm-BMM_1`) invite
// exactly that reading.
//
// ============================================================================
// STATUS: this form does NOT parse.  `ktdp.inter_tile_consume` is specified
// (§6.1 of ../inter-tile-communication.md) but absent from KTDP.td, which
// defines `inter_tile_produce` and `inter_tile_reduce` only.
//
// And it cannot be worked around with a degenerate `reduce` over one producer.
// Writing this delivery as `inter_tile_reduce` with
// `producer_tiles_per_group = {4g}` and `consumer_tiles_per_group = {4g..4g+3}`
// is rejected by the shipped legality pass:
//
//   $ ktir-opt broadcast_as_degenerate_reduce.mlir --ktir-check-legality
//   error: consumer_tiles_per_group for group 0 is not a subset of
//          producer_tiles_per_group (a consumer tile that did not produce is
//          unsupported; see open question Q1)
//
// That is R13 (`KTIRCheckLegality.cpp:107-117`), and R14's mode gate would
// reject it too: the consumer set neither equals the producer set nor is a
// single tile.  So half (b) needs `consume`; the fold op cannot stand in for it.
// The R13 message names the open question, which §10.1 now resolves as `n` for
// every op except `reduce` — but for `reduce` itself the check is correct to
// fire, because folding into a non-contributing tile is the case §10.1 leaves
// open.
// ============================================================================

#tiles     = affine_set<(i)[g] : (i - 4 * g >= 0, -i + 4 * g + 3 >= 0)>
// The completion owner: core 4g only.  Declaring it is the "declare completion
// owners" half of KT-07's requirement.
#completed = affine_set<(i)[g] : (i - 4 * g == 0)>
#one_group = affine_set<(g) : (g == 0)>

// The whole of the tile's own slab.
#slab   = affine_set<(d0, d1, d2) : (
    d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 127 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#ident3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module {
  func.func @kt07b_completed_sum() {
    %c0   = arith.constant 0 : index
    %base = arith.constant 0 : index
    %zero = arith.constant 0.0 : f16

    // ---- identical to half (a) from here ----
    //
    // One producer per group, so strictly these loads belong inside the produce
    // region: they must not run on the cores that hold nothing (§2.2).  They are
    // at function scope here so the block stays textually identical to half (a),
    // which is the comparison this pair is for.  A production form moves them
    // into the region, as §7.7.1 does for `scatter`.
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

    // LOCAL reduction, along dim 1 (128).  This is where the sum is completed.
    %red_init = tensor.empty() : tensor<1x64xf16>
    %red_zero = linalg.fill ins(%zero : f16) outs(%red_init : tensor<1x64xf16>)
                  -> tensor<1x64xf16>
    %local = linalg.reduce { arith.addf }
               ins(%slab : tensor<1x128x64xf16>)
               outs(%red_zero : tensor<1x64xf16>)
               dimensions = [1]

    %partial = tensor.expand_shape %local [[0], [1, 2]]
                 output_shape [1, 1, 64]
                 : tensor<1x64xf16> into tensor<1x1x64xf16>
    // ---- identical to half (a) up to here ----

    %future = ktdp.inter_tile_produce
        producer_tiles_per_group = #completed
        -> !ktdp.tile_future<(tensor<1x1x64xf16>), groups = #one_group>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial : tensor<1x1x64xf16>
    }

    // combine = none, placement = replicate.  No combiner region and no
    // identity: there is nothing to fold, which is the whole point.  Half (a)
    // has both at this position.  R8 is satisfied with one producer per group,
    // so no dependency attribute is needed and full-barrier and per-tile
    // synchronization coincide (§6.1).
    //
    // Ordering is trivial for `replicate`: every consumer receives the whole
    // value, so there is no `l` and no assembly order to get wrong — the other
    // half of KT-07's "declare completion owners and ordering".
    %value = ktdp.inter_tile_consume(%future)
        consumer_tiles_per_group = #tiles
        : !ktdp.tile_future<(tensor<1x1x64xf16>), groups = #one_group>
          -> tensor<1x1x64xf16>

    // Expected: 15.0 at each of the 64 trailing coordinates, on all four cores.
    // 60.0 is what a fold over four holders would give; see the note above on
    // why this fixture cannot produce it and where it could.
    return
  }
}
