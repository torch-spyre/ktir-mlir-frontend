// KT-02 — receiving-only cores, on a measured Granite relayout.
//
// OPERATION SEQUENCE.
//
//   LX-Load (addr_0) -> Produce -> Gather
//
//   addr_0 = the producer's own piece in its LX.  The load is *outside* the produce
//   region because every tile here is a producer; contrast kt05.
//
// ============================================================================
// Fixture: GR-PF-002 of lx_relayout_granite_inventory.md, i.e. relayouts[1] of
// the pinned ownership catalog
//   sendnn_sdsc_lx_replay_manifest.json, sha256
//   4c8aca2e1989eefb76c4ee40b99a4a87800aac4134cbae68c41810ab98c747d4
// read directly rather than from the summary row, which cannot distinguish the
// divisions that produce the same `32 -> 8` counts.
//
//   consumer        mm-BMM_1, input 0        (family P09)
//   route_class     grouped_all_gather_with_replication
//   extents         {mb: 512, in: 4096, y: 1}   src == dst
//   word_length     2  (fp16)
//   remote_required true
//   16 source pieces  {mb: 32,  in: 4096, y: 1}, one owner each
//    8 destination pieces {mb: 64, in: 4096, y: 1}, four owners each
//
// Group structure, derived from the owner tables.  Source piece k is owned by
// core 2k and starts at mb = 32k, so group g (destination mb 64g..64g+63) is:
//
//   group g   producers {4g, 4g+2}   consumers {4g .. 4g+3}   non-producers
//   g=0       {0, 2}                 {0, 1, 2, 3}             1, 3
//   g=1       {4, 6}                 {4, 5, 6, 7}             5, 7
//   ...       8 groups, regular
//   g=7       {28, 30}               {28, 29, 30, 31}         29, 31
//
// WHAT IT SETTLES.  Cores 4g+1 and 4g+3 receive without producing, so R13 must
// be `n` for `gather` (§10.1 of ../inter-tile-communication.md).  This is not a
// corner case: 64 of the catalog's 130 records have destination owners that are
// not source owners, across three route classes
// (grouped_all_gather_with_replication, replicate_or_owner_remap, all_gather).
//
// It also exercises §3.3 directly.  The producers of a group are 4g and 4g+2 —
// NOT adjacent tile ids — so `l` cannot be the tile id.  Taken as a position in
// ascending tile-id order, l(4g) = 0 and l(4g+2) = 1, and since core 4g holds
// the lower mb half, ascending l reproduces ascending mb.  Using tile ids
// directly would leave slot 1 empty and slot 2 filled.
//
// ============================================================================
// Physical layout.  Taken from a real SDSC run of the same logical shape
// (512 x 4096 x 1, fp16), whose descriptor reports
//
//   layoutDimOrder_ = ["mb", "out", "y"]    stickDimOrder_ = ["y"]
//   stickSize_      = [64]
//   device_size     = [1, 4096, 512, 64]
//   device_coordinates = [0, c1, c0, 0]     c0 = mb(512), c1 = out(4096)
//   stride_map      = [1, -1, 4096, -1]     4096<->stride 1, 512<->stride 4096
//
// So: `y` is the innermost logical axis AND the stick axis, and an extent-1 `y`
// is carried as 64 padded lanes — only lane 0 holds data.  The physical form is
// one rank higher than the logical one, the two logical data axes appear in
// reverse order, and the stick is last:
//
//   physical = [1, <second logical axis>, <first logical axis>, 64]
//
// ASSUMPTION, the only one left: this record's layoutDimOrder_ is
// ["mb", "in", "y"].  Confirmed for the run above is ["mb", "out", "y"] — `y`
// innermost and sticked, `mb` first among the data axes — and `in` takes the
// position `out` held.  Only the numeric index in `gather_dimensions` depends
// on it; if the order were ["in", "mb", "y"] the gathered axis would move from
// physical 2 to physical 1.
//
//   whole tensor        logical (mb=512, in=4096, y=1) -> [1, 4096, 512, 64]
//   producer piece      logical (mb=32,  in=4096, y=1) -> [1, 4096,  32, 64]
//   destination region  logical (mb=64,  in=4096, y=1) -> [1, 4096,  64, 64]
//
// The gathered axis is `mb`, physical dim 2.  It is NOT the stick axis, so
// `gather_dimensions` is a single index and §4's floordiv rule does not apply:
// P = 2 multiplies the mb extent 32 -> 64 and nothing else moves.
//
// EXPECTED VALUES.  4096 x 512 fp16 cannot be checked with a ramp — a large
// float ramp rounds distinct coordinate IDs together (coverage §1).  Drive one
// logical coordinate at a time with an exactly representable 1.0 against 0.0
// and check it lands where the owner tables say.  Lanes 1..63 are `y` padding
// and must not be compared.
//
// NOT VERIFIED as a whole: `ktdp.inter_tile_gather` is specified (§6.4) but
// absent from KTDP.td.  The `ktdp.inter_tile_produce` half parses and
// round-trips through ktir-opt.
// ============================================================================

// Producers of group g: {4g, 4g+2}.  Even tile ids in [4g, 4g+2].
#producers = affine_set<(i)[g] : (i - 4 * g >= 0, -i + 4 * g + 2 >= 0,
                                 i mod 2 == 0)>
// Consumers of group g: {4g, 4g+1, 4g+2, 4g+3}, the destination piece's owners.
#consumers = affine_set<(i)[g] : (i - 4 * g >= 0, -i + 4 * g + 3 >= 0)>
#groups    = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// The whole of the producer's own piece: [1, 4096, 32, 64].
#piece  = affine_set<(d0, d1, d2, d3) : (
    d0 >= 0, -d0 >= 0,        d1 >= 0, -d1 + 4095 >= 0,
    d2 >= 0, -d2 + 31 >= 0,   d3 >= 0, -d3 + 63 >= 0)>
#ident4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  func.func @kt02_receiving_only_cores() {
    %c0   = arith.constant 0 : index
    %base = arith.constant 0 : index

    // `ct_local` with no ct_id is the executing tile's own LX.  The pieces
    // start distributed one per producer (source_owner_group_sizes = [1]), so a
    // fresh HBM load of the assembled region at one core would not reproduce
    // this starting ownership.  `remote_required` is true in the record.
    // Dense in its own buffer: lane 1, mb 64, in 32*64 = 2048, and the extent-1
    // axis takes the remaining product.
    %own = ktdp.construct_memory_view %base,
        sizes: [1, 4096, 32, 64], strides: [8388608, 2048, 64, 1] {
        coordinate_set = #piece,
        memory_space   = #ktdp.memory_space<ct_local>
    } : memref<1x4096x32x64xf16>

    %own_access = ktdp.construct_access_tile %own[%c0, %c0, %c0, %c0] {
        access_tile_set = #piece, access_tile_order = #ident4
    } : memref<1x4096x32x64xf16> -> !ktdp.access_tile<1x4096x32x64xindex>

    // Function scope: every producer reads only its own piece at offset 0, so
    // no tile id enters the address and the load need not sit inside the
    // region (§2.2).
    %piece_val = ktdp.load %own_access
        : !ktdp.access_tile<1x4096x32x64xindex> -> tensor<1x4096x32x64xf16>

    %future = ktdp.inter_tile_produce
        producer_tiles_per_group = #producers
        -> !ktdp.tile_future<(tensor<1x4096x32x64xf16>), groups = #groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %piece_val : tensor<1x4096x32x64xf16>
    }

    // combine = none, placement = concat, gathered axis = mb = physical dim 2.
    // P = 2, so mb goes 32 -> 64 (R12).  All four consumers of the group hold
    // the same assembled region (§3.7), which is the `with_replication` half of
    // the record's route_class.
    %region = ktdp.inter_tile_gather(%future)
        consumer_tiles_per_group = #consumers,
        gather_dimensions        = [2]
        : !ktdp.tile_future<(tensor<1x4096x32x64xf16>), groups = #groups>
          -> tensor<1x4096x64x64xf16>

    // Expected at cores 4g..4g+3: destination piece g, logical
    // (mb = 64g..64g+63, in = 0..4095, y = 0), with core 4g supplying the lower
    // mb half and core 4g+2 the upper.  Undefined elsewhere (§3.7).
    //
    // All four consumers hold the SAME assembled region, not a quarter each:
    // the record carries one destination rectangle with four owners, and §3.7
    // gives `concat` "the same assembled tensor" for tiles in one group.  So the
    // check runs per consumer, and all four must agree — including on order,
    // which is what makes coverage §5's "reverse two fragments" check bite.
    //
    // Counted per core the destination covers 8 * 64 * 4 = 2048 mb rows against
    // a 512-row tensor, 4x over. That is why §9.1's coverage clause is stated on
    // *distinct* regions: the 8 distinct ones sum to 512 and pass the guard.
    return
  }
}
