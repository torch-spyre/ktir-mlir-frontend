// KT-04 — assembly along more than one axis.
//
// torch-spyre#4300 lx_relayout_workload_coverage.md §5 KT-04: "Four 1x1 tiles of a
// 2x2 logical matrix: source0=(0,0), source1=(1,0), source2=(0,1), source3=(1,1).
// Receiver needs `[[0,1],[2,3]]` when values encode row-major coordinates.  **Show
// multi-axis assembly, including exact result dimensions and placement.  A flat
// concatenation `[0,2,1,3]` is wrong.  Add the full multi-axis Granite shapes
// afterward.**"
//
// No new op: this is `inter_tile_gather` with **two** entries in
// `gather_dimensions` instead of one.  §9.1 row 4, the same row as kt01 and kt02.
// What multi-axis adds is that §4's flattening becomes load-bearing, and the two
// functions below disagree about whether that is enough.
//
// OPERATION SEQUENCE.
//
//   @kt04_measured   LX-Load (addr_0) -> Produce -> Gather -> LX-Store (addr_1)
//   @kt04_toy_2x2    LX-Load (addr_0) -> Produce -> Gather
//
//   addr_0 = own piece, addr_1 = the assembled region.  One Gather with two entries
//   in `gather_dimensions` — the sequence is kt02's, and only the attribute differs.
//   The toy leaves its result unstored; its point is what the Gather assembles.
//
// ============================================================================
// WHAT §4 FIXES, NORMATIVELY.
//
//   "A list of length n > 1 denotes the product space of those axes, linearized as
//    a row-major (mixed-radix odometer) order over the listed extents: **the first
//    entry is the slowest-varying and the last is the fastest-varying**."
//
//   "**The list is in ascending numerical order (R9).**  Entries must ascend ... It
//    removes a silent-miscompile class: `[2, 0]` and `[0, 2]` are both 'valid,
//    distinct, non-empty' and would flatten to *different* data orders, so a
//    reversed list passes every other check while meaning something else."
//
// R12 adds the per-axis obligation: the result flattened extent over the concat set
// is `P * E(D_concat)`, and that "requires every assembled producer to contribute
// the same extent along *each* listed axis — equal products alone would not give a
// well-defined multi-axis assembly, since the flattening of §4 depends on the
// individual extents".  "The same validity conditions as R9 apply to the list", so
// the ascending rule covers `gather_dimensions` and not only the split ops.
//
// Together those close the two *spelling* errors this case invites:
//
//   gather_dimensions = [1, 0]   descending, i.e. a column-major assembly
//                                -> rejected by R9's ascending rule
//   gather_dimensions = [1]      one axis multiplies where two must
//                                -> rejected by R12: 2 * E != the declared extent
//
// They do **not** close a third case, which is what the toy below is.
// ============================================================================

// ---------------------------------------------------------------------------
// MEASURED FIXTURE: GR-PF-121, `relayouts[120]` of the pinned catalog
// (sendnn_sdsc_lx_replay_manifest.json, sha256 4c8aca2e…c747d4).
//
//   consumer        cat_1-kvCacheScatter, input 0
//   route_class     all_gather
//   extents         {mb: 8, out: 128, x: 1, y: 1}   src == dst
//   word_length     2  (fp16)
//   16 source pieces  {mb: 1, out: 64, x: 1, y: 1}, one owner each
//    1 destination piece {mb: 8, out: 128, x: 1, y: 1}, owned by core 0
//   source_fragments_per_destination_piece = 16   -> P = 16
//   destination_pieces_per_source_piece    = 1
//
// Two axes grow, so both are concat axes and neither is split:
//
//   mb    1 -> 8     x8
//   out  64 -> 128   x2       8 * 2 = 16 = P
//
// GROUP STRUCTURE.  One group — there is one destination region.  Producers are the
// **even** cores, consumer is core 0:
//
//   group g   producers {2k, k=0..15}   consumers {0}   idle
//   g=0       {0, 2, 4, ..., 30}       {0}             odd cores 1, 3, ..., 31
//   ...       1 group only (g = 0)
//
// Core 0 is both a producer and the consumer, so R13 is satisfied here and this
// case does not depend on §10.1 — unlike kt02, which is the receiving-only case.
// The odd cores are neither, which is §10.2's "Replication versus idleness": with
// `prod(Nd(a)) = 1` against 32 cores the §9.1 guard sends this to row 4 (`gather`)
// rather than row 3, and the consumer set is the destination region's holders, `{0}`.
//
// THE OWNER ORDER AGREES WITH §4's ODOMETER, which is the point of using this
// record.  Sorting the 16 producers by ascending tile id and taking `l` as the
// position (§3.3):
//
//   l    owner    holds           odometer (mb = l/2, out = 64*(l mod 2))
//   0      0      mb=0 out=  0    (0,   0)
//   1      2      mb=0 out= 64    (0,  64)
//   2      4      mb=1 out=  0    (1,   0)
//   3      6      mb=1 out= 64    (1,  64)
//   ...   ...     ...             ...
//  14     28      mb=7 out=  0    (7,   0)
//  15     30      mb=7 out= 64    (7,  64)
//
// Checked for all 16: `l -> (mb = l floordiv 2, out = 64 * (l mod 2))` matches the
// ascending-owner order exactly, reading the `owners` field of each piece.  So `mb`
// is the slow axis and `out` the fast one, which is what `gather_dimensions = [0, 1]`
// means under §4 — ascending, first entry slowest.  **The measured data confirms
// §4's flattening rather than contradicting it**, so this function needs nothing
// beyond a plain two-axis gather.
//
// LAYOUT.  As in kt06, the piece is written in the record's logical axes; `x` and
// `y` are extent-1 and carry no data, and the catalog cannot settle the physical
// form because its byte fields are logical (see kt06's layout note).  Nothing here
// depends on it: both concat axes are data axes, and §10.3's physicalization
// inserts its chunk axis at the front of source and destination alike.
// ---------------------------------------------------------------------------

#producers = affine_set<(i)[g] : (i >= 0, -i + 31 >= 0, i mod 2 == 0)>
#consumer  = affine_set<(i)[g] : (i == 0)>
#groups    = affine_set<(g) : (g == 0)>

// One producer's piece, (mb, out) = (1, 64).
#piece  = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#region = affine_set<(d0, d1) : (
    d0 >= 0, -d0 + 7 >= 0,  d1 >= 0, -d1 + 127 >= 0)>
#ident2 = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @kt04_measured() {
    %c0       = arith.constant 0 : index
    %base     = arith.constant 0 : index
    %base_out = arith.constant 1024 : index

    // The producer's own piece.  `ct_local` with no ct_id is the executing tile's
    // own LX, so no tile id enters the address.
    %own = ktdp.construct_memory_view %base,
        sizes: [1, 64], strides: [64, 1] {
        coordinate_set = #piece,
        memory_space   = #ktdp.memory_space<ct_local>
    } : memref<1x64xf16>

    %own_access = ktdp.construct_access_tile %own[%c0, %c0] {
        access_tile_set = #piece, access_tile_order = #ident2
    } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>

    %piece_val = ktdp.load %own_access
        : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

    // The load sits outside the region, as in kt01 and kt02: every producer runs
    // it and the single consumer is itself a producer, so there is no
    // non-producing tile that must be kept out of it (§2.2).
    %future = ktdp.inter_tile_produce
        producer_tiles_per_group = #producers
        -> !ktdp.tile_future<(tensor<1x64xf16>), groups = #groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %piece_val : tensor<1x64xf16>
    }

    // combine = none, placement = concat, consumers = one (§1.2's `gather`).
    // TWO axes, ascending: axis 0 (`mb`) is the slow one, axis 1 (`out`) the fast
    // one, per §4.  R12: P * E(D) = 16 * (1*64) = 1024 = 8 * 128, and per axis
    // mb x8 and out x2 with 8*2 = 16 = P.
    %region_val = ktdp.inter_tile_gather(%future)
        consumer_tiles_per_group = #consumer,
        gather_dimensions        = [0, 1]
        : !ktdp.tile_future<(tensor<1x64xf16>), groups = #groups>
          -> tensor<8x128xf16>

    // Base 1024, not 0: the producer's own piece occupies elements 0..63, and the
    // assembled region must not be written over its own source.
    %out = ktdp.construct_memory_view %base_out,
        sizes: [8, 128], strides: [128, 1] {
        coordinate_set = #region,
        memory_space   = #ktdp.memory_space<ct_local>
    } : memref<8x128xf16>

    %out_at = ktdp.construct_access_tile %out[%c0, %c0] {
        access_tile_set = #region, access_tile_order = #ident2
    } : memref<8x128xf16> -> !ktdp.access_tile<8x128xindex>

    ktdp.store %region_val, %out_at
        : tensor<8x128xf16>, !ktdp.access_tile<8x128xindex>

    // Expected at core 0: the whole {mb: 8, out: 128} region, with core `2*(2m + n)`
    // supplying `mb = m, out = 64n..64n+63`.  Undefined at every other tile (§3.7).
    //
    // Values: identify each element by its (mb, out) coordinate one bit at a time,
    // per coverage §1's rule for low-precision tensors, so a misplaced piece names
    // the core it came from.
    //
    // Failures to catch, and which rule reaches each.  (1) `gather_dimensions = [1]`
    // — one axis where two are needed: R12, since 2 * 64 = 128 accounts for `out`
    // but leaves `mb` at 1 against the declared 8.  (2) `gather_dimensions = [1, 0]`
    // — R9's ascending rule, which exists precisely so this cannot be spelled.
    // (3) Swapping which core supplies which cell while keeping the piece count:
    // **no rule reaches it**, and it is the toy below.
    return
  }
}

// ---------------------------------------------------------------------------
// THE 2x2 TOY — SYNTHETIC, AND IT DOES NOT REDUCE TO A PLAIN GATHER.
//
// The requirement states the toy's ownership explicitly, and reading it as (row,
// column) — the reading under which its own wrong answer `[0,2,1,3]` is the one a
// gather actually produces — the coordinates are a transpose:
//
//   group g   producers {0, 1, 2, 3}   consumers {0}   assembled
//   g=0       {0, 1, 2, 3}             {0}             2x2, l -> (l/2, l mod 2)
//   ...       1 group only (g = 0)
//
//   producer   holds cell      row-major value at that cell
//   source0    (0, 0)          0
//   source1    (1, 0)          2
//   source2    (0, 1)          1
//   source3    (1, 1)          3
//
// The receiver needs `[[0,1],[2,3]]`, i.e. value 1 at cell (0,1) and value 2 at
// cell (1,0).
//
// Now apply §4 with `gather_dimensions = [0, 1]` over factors (2, 2).  Ascending
// tile order gives `l = 0,1,2,3` for source0..3, and the odometer sends
// `l -> (r = l floordiv 2, c = l mod 2)`:
//
//   l   producer   value   odometer cell   required cell
//   0   source0      0       (0, 0)          (0, 0)   ok
//   1   source1      2       (0, 1)          (1, 0)   WRONG
//   2   source2      1       (1, 0)          (0, 1)   WRONG
//   3   source3      3       (1, 1)          (1, 1)   ok
//
// The assembled result is `[[0,2],[1,3]]` — and read out in flat order that is
// `0,2,1,3`, the sequence the requirement names as wrong.  So "a flat concatenation
// `[0,2,1,3]` is wrong" has a second reading beyond the obvious one: it is not only
// that the result must be 2x2 rather than 1x4 (R12 gives that), it is that **the
// producers' tile order and the required cell order are a transpose of each other**,
// and the odometer cannot know.
//
// WHY NO RULE CATCHES IT.  Compare with the two spellings above.  Here
// `gather_dimensions = [0, 1]` is ascending, both axes are listed, every producer
// contributes the same 1x1 extent, `P = 4 = 2 * 2`, and the result type is
// `tensor<2x2xf16>`.  R9, R12, R5, R6 and R8 are all satisfied — by the correct
// assembly and by the transposed one alike, because they constrain extents and
// cardinalities and never the identity of what lands where.  Same shape, same
// element count, wrong answer.
//
// This is kt03's finding in its multi-axis form, and it should be read together
// with that file: an assembly's order comes from §3.3's ascending tile id, no
// attribute overrides it, and where the required order disagrees the fix is not in
// the delivery op.  Three routes, all outside `gather`:
//
//   (a) Assign tile ids upstream so that ascending order matches the odometer.
//       This is the real answer for a *planner*, and it is why the measured record
//       above has no problem: the work division already agrees with §4.
//   (b) Store in two or four pieces at swapped bases, as kt03 does — correct,
//       zero-copy, and invisible to the verifier.
//   (c) Permute the assembled `tensor` locally before use, for a live intermediate.
//
// So KT-04's verdict splits.  The two *spelling* errors are closed by R9 and R12,
// which is a stronger position than kt03's.  The *placement* error is not closed at
// all, and the toy is the fixture that shows it.  Measurement does not force the
// problem — `relayouts[120]` agrees with the odometer — exactly as kt03's 130 of 130
// do not contradict §3.3.
//
// NOT VERIFIED as a whole: `ktdp.inter_tile_gather` is specified (§6.4) but absent
// from KTDP.td, in both functions.  The `ktdp.inter_tile_produce` halves parse and
// round-trip, as do all the affine sets on their own.
// ---------------------------------------------------------------------------

#four   = affine_set<(i)[g] : (i >= 0, -i + 3 >= 0)>
#cell   = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0)>
#matrix = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 1 >= 0)>

module {
  func.func @kt04_toy_2x2() {
    %c0   = arith.constant 0 : index
    %base = arith.constant 0 : index

    // One 1x1 cell per producer.  Which logical cell it is comes from the work
    // division, not from anything in this function — which is the whole problem.
    %own = ktdp.construct_memory_view %base,
        sizes: [1, 1], strides: [1, 1] {
        coordinate_set = #cell,
        memory_space   = #ktdp.memory_space<ct_local>
    } : memref<1x1xf16>

    %own_access = ktdp.construct_access_tile %own[%c0, %c0] {
        access_tile_set = #cell, access_tile_order = #ident2
    } : memref<1x1xf16> -> !ktdp.access_tile<1x1xindex>

    %cell_val = ktdp.load %own_access
        : !ktdp.access_tile<1x1xindex> -> tensor<1x1xf16>

    %future = ktdp.inter_tile_produce
        producer_tiles_per_group = #four
        -> !ktdp.tile_future<(tensor<1x1xf16>), groups = #groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %cell_val : tensor<1x1xf16>
    }

    // Well-formed and still wrong: ascending list, both axes, uniform extents,
    // P = 4 = 2 * 2, result type exactly as §4 requires.  It assembles
    // [[0,2],[1,3]] from the toy's ownership, where [[0,1],[2,3]] is required.
    %matrix_val = ktdp.inter_tile_gather(%future)
        consumer_tiles_per_group = #consumer,
        gather_dimensions        = [0, 1]
        : !ktdp.tile_future<(tensor<1x1xf16>), groups = #groups>
          -> tensor<2x2xf16>

    // Expected, if the toy's ownership is taken as given: [[0,1],[2,3]].
    // Produced by this op: [[0,2],[1,3]].  The difference is a transpose and no
    // rule in §5 sees it — see the discussion above for the three places the fix
    // can live, none of which is this op.
    return
  }
}
