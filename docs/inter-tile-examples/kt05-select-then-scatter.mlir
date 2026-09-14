// KT-05 — select a row locally, then split it across the receivers.
//
// torch-spyre#4300 lx_relayout_workload_coverage.md §5 KT-05: "Proposal's
// `[512,32,64]` select: source core `4m+h` owns rows `64m:64m+64`, sticks
// `8h:8h+8`.  Destination core `d` needs row511, stick `d`. ... Source
// `28+floor(d/8)` supplies destination `d`.  Select locally, then split/send the
// selected row.  The original layout is not a no-op; fresh HBM loads at
// destinations would change the starting contract."  And §5's closing note:
// "Test the proposal's 32-stick example and Granite P14's 4096-column example
// separately; they have the same relation but different widths."
//
// Both are here: @kt05_granite_p14 is the measured one, @kt05_proposal_32_sticks
// the narrower one the proposal states.
//
// OPERATION SEQUENCE.  Both functions:
//
//   Produce[ LX-Load (addr_0) ] -> Scatter
//
//   addr_0 = the producer's own slab.  **The load is inside the produce region**,
//   unlike every other file here, because only one tile per group produces and the
//   group's eight consumers must not run it (§2.2).  The select is the access tile
//   that load uses, so it needs no op of its own.
//
// ============================================================================
// MEASURED FIXTURE: GR-PF-055, `relayouts[54]` of the pinned catalog
// (sendnn_sdsc_lx_replay_manifest.json, sha256 4c8aca2e…c747d4).
//
//   consumer        slice_161-Stcdp, input 0     (family P14)
//   tensor          mean_80-LayerNormNorm_out
//   extents         {mb: 512, out: 4096, y: 1}   src == dst
//   word_length     2  (fp16)
//   remote_required true
//   32 source pieces {mb:64, out:1024, y:1}, one owner each
//   32 destination pieces {mb:1, out:128, y:1}, one owner each
//
// Source piece k is owned by core k: rows 64*(k/4) upward, out columns 1024*(k%4)
// upward.  So the 32 source pieces form an 8 x 4 grid — eight row blocks of 64 by
// four out-slabs of 1024.  Every destination piece is a single row, **mb = 511**,
// and 128 out columns starting at out = 128*d, owned by core d.
//
// WHICH CORES SEND, AND TO WHOM.  Two observations settle it.
//
//   1. Every destination wants row 511, and row 511 lies in the **last row block**,
//      rows 448..511.  Only four cores hold that block: 28, 29, 30, 31.  The other
//      28 send nothing, which is what the catalog records as
//      `destination_pieces_per_source_piece = 0` for pieces 0..27.
//
//   2. Those four hold row 511 in four consecutive out-slabs of 1024:
//
//        core 28  out    0..1023        core 30  out 2048..3071
//        core 29  out 1024..2047        core 31  out 3072..4095
//
//      A destination is 128 columns wide, so **each slab is 1024/128 = 8
//      destinations wide**, and those destinations are consecutive in d:
//
//        core 28 -> d =  0.. 7          core 30 -> d = 16..23
//        core 29 -> d =  8..15          core 31 -> d = 24..31
//
// That is the whole structure: one producer per slab, eight consumers each.
// Coverage §5 states it as "source 28 + floor(d/8) supplies destination d".
//
// Group structure.  A group is one slab being distributed, so group g is core 28+g
// serving the eight consumers {8g .. 8g+7}.  Checked against the record: d = 0..7
// come from core 28, d = 8..15 from core 29, d = 16..23 from core 30, d = 24..31
// from core 31.
//
// The 8 in `8g+7` is also R9's `C` below — the slab's fan-out and the split count
// are one number, because the slab is what gets split.
//
//   group g   producers {28+g}   consumers {8g .. 8g+7}   out columns of row 511
//   g=0       {28}               {0, 1, ..., 7}           0..1023   -> 128 each
//   g=1       {29}               {8, 9, ..., 15}          1024..2047
//   g=2       {30}               {16, ..., 23}            2048..3071
//   g=3       {31}               {24, ..., 31}            3072..4095
//
// WHY THIS IS NOT A NO-OP.  Core 0 needs out 0..127 of row 511, and row 511 is
// owned by cores 28..31 — core 0 owns rows 0..63.  Coverage §5 makes the point
// with one coordinate: "Core0's required `(511,0,:)` starts on **source28**.
// That one coordinate is enough to disprove a no-communication interpretation."
// The record's `remote_required` is true.
//
// WHY A SELECT COMES FIRST — AND WHY THE GUARD THEN STOPS FIRING.
//
// Read as one relayout of the whole tensor — 32 owner slabs on one side, 32
// slivers of row 511 on the other — the pair fails §9.1's first guard row.  Not
// on the counts: `prod(Ns)` and `prod(Nd)` both equal their region counts.  It is
// the *coverage* clause that catches it, since the 32 distinct destination
// regions hold 32 * 128 = 4096 elements against the tensor's
// 512 * 4096 = 2097152 — a 512th of it.  §9.3 records this as the one
// "selection, not a partition" file.
//
// **After the select the guard is satisfied**, and §9.1 says so itself: the
// coverage clause is measured against "the value delivered ... after a select
// that is the selected sub-tensor and not the original — otherwise the
// select-then-deliver that repairs a selection would trip the guard it was meant
// to satisfy."  The value delivered here is the selected row, mb = 1 by
// out = 4096 — which is what `yield_partial` hands over below:
//
//   source regions   4 shards of out 1024    4 * 1024 = 4096   covers the row
//   destinations    32 chunks of out 128    32 *  128 = 4096   covers the row
//   Ns = {out: 4},  Nd = {out: 32}   ->   C = {},  R = {out}
//
// That is §9.1 row 2, `inter_tile_scatter` with `scatter_dimensions = R`.  So the
// guard's job is to reject the pre-select reading and force the select, not to
// say the record is inexpressible: the post-select pair classifies normally and
// yields exactly the op below, with `[1]` naming `out`.
//
// The selection itself needs no new op — an access tile over the selected
// sub-rectangle expresses it, which is what the produce region does.
//
// The catalog labels the record `route_class = permutation`.  That describes the
// pre-select pair (32 pieces to 32 pieces, one fragment each) and is not a KTIR
// op choice; how the label is computed is not in the JSON.  The op to emit comes
// from §9.1 applied to the post-select pair, as above.
//
// LAYOUT.  No assumption is needed for this record — unlike kt02.  A real SDSC
// run of this very logical shape (512 x 4096 x 1, fp16) reports
//
//   layoutDimOrder_ = ["mb", "out", "y"]   stickDimOrder_ = ["y"]
//   stickSize_      = [64]                 device_size = [1, 4096, 512, 64]
//
// and this record's axes are literally `mb`, `out`, `y`, so the measured order
// applies directly.  `y` is innermost and sticked, an extent-1 `y` is carried as
// 64 padded lanes, and the physical form is [1, <out>, <mb>, 64]:
//
//   producer's own piece   logical (mb=64, out=1024, y=1) -> [1, 1024, 64, 64]
//   after the select       logical (mb=1,  out=1024, y=1) -> [1, 1024,  1, 64]
//   per consumer           logical (mb=1,  out=128,  y=1) -> [1,  128,  1, 64]
//
// The split axis is `out`, physical dim 1 — not the stick axis, so a single
// index and no floordiv rule.  R9: E = 1024, C = 8, 1024 % 8 == 0.  In sticks
// that is 16 % 8 == 0, two sticks per consumer, so the split stays on stick
// boundaries (§4's stick-multiple reading of R9).
//
// R8 holds with one producer per group, so `scatter` takes no
// `producer_dependency_per_consumer` (§6.6) and full-barrier and per-tile
// synchronization coincide.  R13 is `n` for `scatter` (§6.6, §10.1) and it is
// needed: in group g=0 the producer is core 28 and the consumers are 0..7, which
// are disjoint from it.  Ordering is by consumer local index `l` (§3.3):
// consumer 8g+j takes chunk j.
//
// NOT VERIFIED as a whole: `ktdp.inter_tile_scatter` is specified (§6.6) but
// absent from KTDP.td.  The `ktdp.inter_tile_produce` half parses, including the
// in-region select.
// ============================================================================

// Producer of group g: core 28+g.
#producer  = affine_set<(i)[g] : (i - g - 28 == 0)>
// Consumers of group g: 8g .. 8g+7.
#consumers = affine_set<(i)[g] : (i - 8 * g >= 0, -i + 8 * g + 7 >= 0)>
#groups    = affine_set<(g) : (g >= 0, -g + 3 >= 0)>

// The producer's whole slab, and the single row selected out of it.  Both sets
// are relative to the access tile's base coordinate.
#slab     = affine_set<(d0, d1, d2, d3) : (
    d0 >= 0, -d0 >= 0,      d1 >= 0, -d1 + 1023 >= 0,
    d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#last_row = affine_set<(d0, d1, d2, d3) : (
    d0 >= 0, -d0 >= 0,      d1 >= 0, -d1 + 1023 >= 0,
    d2 >= 0, -d2 >= 0,      d3 >= 0, -d3 + 63 >= 0)>
#ident4   = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  func.func @kt05_granite_p14() {
    %c0   = arith.constant 0 : index
    %c63  = arith.constant 63 : index
    %base = arith.constant 0 : index

    %future = ktdp.inter_tile_produce
        producer_tiles_per_group = #producer
        -> !ktdp.tile_future<(tensor<1x1024x1x64xf16>), groups = #groups>
    {
      ^bb0(%gid: index):
        // The loads belong INSIDE the region: with one producer per group they
        // must not run on the group's eight non-producing consumers (§2.2, which
        // names this as the case where a richer body is normally required).
        //
        // `ct_local` with no ct_id is the executing tile's own LX.  Dense:
        // lane 1, mb 64, out 64*64 = 4096, and the extent-1 axis takes the rest.
        %own = ktdp.construct_memory_view %base,
            sizes: [1, 1024, 64, 64], strides: [4194304, 4096, 64, 1] {
            coordinate_set = #slab,
            memory_space   = #ktdp.memory_space<ct_local>
        } : memref<1x1024x64x64xf16>

        // THE SELECT.  Global row 511 is local row 63 of this producer's slab
        // (it owns rows 448..511), so the access tile is anchored at mb = 63 with
        // extent 1 there and full extent elsewhere.  No new op: an access tile
        // over a sub-rectangle is the selection §9.3 asks for.
        %sel = ktdp.construct_access_tile %own[%c0, %c0, %c63, %c0] {
            access_tile_set = #last_row, access_tile_order = #ident4
        } : memref<1x1024x64x64xf16> -> !ktdp.access_tile<1x1024x1x64xindex>

        %row = ktdp.load %sel
            : !ktdp.access_tile<1x1024x1x64xindex> -> tensor<1x1024x1x64xf16>

        ktdp.yield_partial %row : tensor<1x1024x1x64xf16>
    }

    // combine = none, placement = split.  The selected row is cut along `out`
    // into C = 8 chunks of 128, one per consumer in ascending local index (§3.3).
    // No combiner, no identity, and no dependency attribute (§6.6).
    %chunk = ktdp.inter_tile_scatter(%future)
        consumer_tiles_per_group = #consumers,
        scatter_dimensions       = [1]
        : !ktdp.tile_future<(tensor<1x1024x1x64xf16>), groups = #groups>
          -> tensor<1x128x1x64xf16>

    // Expected at core d: row 511, out columns 128d..128d+127, from source
    // 28 + d/8.  The consumers of a group hold **disjoint** ordered slices that
    // tile the selected row (§3.7, `split`) — unlike the gather cases, where all
    // consumers hold the same thing.  Undefined at no tile: every core consumes.
    //
    // Failure to catch: give consumer 8g+j chunk j+1, or read the selection from
    // HBM at each destination.  The second is the one coverage §5 warns about —
    // it would produce the right values while changing the starting ownership,
    // so the emitted memory accesses have to be checked as well.
    return
  }
}

// ---------------------------------------------------------------------------
// The proposal's narrower example, stated in coverage §5 as `[512, 32, 64]`:
// 512 rows, 32 sticks, 64 lanes.  Same core relationship, different width.
//
// Same structure as above, one slab narrower.  Row 511 is again in the last row
// block, so again only cores 28..31 send; they hold it in four slabs of 8 sticks,
// and a destination is one stick, so each slab is 8 destinations wide:
//
//   core 28  sticks  0.. 7  -> d =  0.. 7      core 30  sticks 16..23  -> d = 16..23
//   core 29  sticks  8..15  -> d =  8..15      core 31  sticks 24..31  -> d = 24..31
//
//   group g   producers {28+g}   consumers {8g .. 8g+7}   sticks of row 511
//   g=0       {28}               {0, ..., 7}              0..7   -> 1 each
//   g=1       {29}               {8, ..., 15}             8..15
//   g=2       {30}               {16, ..., 23}            16..23
//   g=3       {31}               {24, ..., 31}            24..31
//
// WHY IT IS WORTH TESTING SEPARATELY.  The width is what differs, and it lands
// R9 on its boundary: 8 sticks over 8 consumers is **one stick each**, where the
// Granite record gives two.  A split that went one step further would be
// sub-stick and R9 would have to reject it (§4: "a split that would drive the
// result sub-stick fails R9 rather than needing a rule of its own").  So this
// function is the tight case and the measured one is the slack case.
//
// Shapes are the proposal's own rank-3 spelling (rows, sticks, lanes) rather
// than the SDSC physical form used above; the two are not mixed in one function.
// ---------------------------------------------------------------------------

#slab3     = affine_set<(d0, d1, d2) : (
    d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 7 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#last_row3 = affine_set<(d0, d1, d2) : (
    d0 >= 0, -d0 >= 0,      d1 >= 0, -d1 + 7 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#ident3    = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module {
  func.func @kt05_proposal_32_sticks() {
    %c0   = arith.constant 0 : index
    %c63  = arith.constant 63 : index
    %base = arith.constant 0 : index

    %future = ktdp.inter_tile_produce
        producer_tiles_per_group = #producer
        -> !ktdp.tile_future<(tensor<1x8x64xf16>), groups = #groups>
    {
      ^bb0(%gid: index):
        %own = ktdp.construct_memory_view %base,
            sizes: [64, 8, 64], strides: [512, 64, 1] {
            coordinate_set = #slab3,
            memory_space   = #ktdp.memory_space<ct_local>
        } : memref<64x8x64xf16>

        // Global row 511 is local row 63 again.
        %sel = ktdp.construct_access_tile %own[%c63, %c0, %c0] {
            access_tile_set = #last_row3, access_tile_order = #ident3
        } : memref<64x8x64xf16> -> !ktdp.access_tile<1x8x64xindex>

        %row = ktdp.load %sel
            : !ktdp.access_tile<1x8x64xindex> -> tensor<1x8x64xf16>

        ktdp.yield_partial %row : tensor<1x8x64xf16>
    }

    // Split the stick axis 8 ways: E = 8, C = 8, one stick per consumer.  This
    // is R9 at its boundary.
    %chunk = ktdp.inter_tile_scatter(%future)
        consumer_tiles_per_group = #consumers,
        scatter_dimensions       = [1]
        : !ktdp.tile_future<(tensor<1x8x64xf16>), groups = #groups>
          -> tensor<1x1x64xf16>

    // Expected at core d: row 511, stick d, from source 28 + d/8.
    return
  }
}
