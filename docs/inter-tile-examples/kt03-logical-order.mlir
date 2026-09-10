// KT-03 — logical order disagreeing with tile-id order.
//
// torch-spyre#4300 lx_relayout_workload_coverage.md §5 KT-03: "Source core0 holds
// `[2,3]`, core1 holds `[0,1]`; receiver needs `[0,1,2,3]`.  **Show how logical
// order is preserved when it differs from core-ID order.**  Local
// selection/reordering is acceptable if fully expressed and stays on-chip."
//
// OPERATION SEQUENCE.
//
//   LX-Load (addr_0) -> Produce -> Gather
//     -> Slice -> LX-Store (addr_1, rows 32..63)
//     -> Slice -> LX-Store (addr_1, rows  0..31)
//
//   addr_0 = own piece, addr_1 = the assembled region.  **The two stores are the
//   whole point**: same destination view, swapped base rows, so the reorder is
//   carried by which base each slice goes to.  Both stores are landings the transfer
//   requires anyway, so the swap costs nothing.
//
// ============================================================================
// SYNTHETIC, WITH ONE THING CHANGED.
//
// Geometry, groups and piece types are GR-PF-002's, the same as kt01 and kt02:
//
//   group g   producers {4g, 4g+2}   consumers {4g .. 4g+3}   non-producers
//   g=0       {0, 2}                 {0, 1, 2, 3}             1, 3
//   g=1       {4, 6}                 {4, 5, 6, 7}             5, 7
//   ...       8 groups, regular
//   g=7       {28, 30}               {28, 29, 30, 31}         29, 31
//
// Only which core holds which half differs:
//
//   measured (kt02)   core 4g   holds mb 64g   .. 64g+31   (lower)
//                     core 4g+2 holds mb 64g+32.. 64g+63   (upper)
//   here              core 4g   holds the UPPER half
//                     core 4g+2 holds the LOWER half
//
// §3.3 orders an assembly by ascending tile id, so l(4g) = 0 and l(4g+2) = 1 and
// the gather produces [upper, lower].  The consumer needs [lower, upper].  No
// dependency set fixes this: a set carries no order, and the order comes from the
// tile id, which the work division fixed upstream.
//
// This shape does not occur in measurement.  Checked across all 130 records of
// the pinned catalog: on the gathered axis, the sources contributing to a
// destination piece are in ascending tile-id order in **130 of 130**.  So §3.3's
// rule is never contradicted by the measured data.  The case is still worth
// answering, because the answer is not where one looks for it: the reorder lives
// at the store, not on the delivery op.
// ============================================================================
//
// HOW THE REORDER IS EXPRESSED, AND WHAT IT COSTS.
//
// The gathered value is a `tensor` — no address, no memory space, and
// `!ktdp.tile_future` carries `tensor` rather than `memref`.  So the assembly is
// not "in LX in the wrong order": it is not in LX at all.  Memory identity begins
// at `ktdp.store`, and that is where the reorder goes.
//
// **A single access tile cannot do it.**  An access tile is a base coordinate
// plus a region relative to that base, so there is no way to permute positions
// *within* a dimension.  `access_tile_order` does not help: RFC 0682 says "the
// rightmost dimension in the output space corresponds to the innermost iteration
// dimension", which reads as a dimension nesting order — outer to inner — and a
// nesting order cannot reorder positions inside one dimension.  (Its next
// sentence, "the enumeration of points in the intermediate variable space", reads
// more generously, and the two are not obviously the same thing.  This file does
// not depend on the generous reading; see the aside at the end.)
//
// **Two stores do it, and avoid the problem rather than solving it.**  Each store
// writes a contiguous run at its own base, with identity order inside, and the
// swap is carried entirely by *which base each slice goes to*:
//
//   tensor rows  0..31  (upper, from core 4g)    -> memory rows 32..63
//   tensor rows 32..63  (lower, from core 4g+2)  -> memory rows  0..31
//
// Cost: **none.**  Not "two stores instead of one" — two stores instead of two.  For
// a copy delivery this hardware lands every received tile in LX before a compute unit
// can read it, so P landing stores are mandatory whatever the order is.  The two
// stores here *are* those landings, aimed at swapped bases.  Every element is written
// exactly once and no pass over the region is added.
//
// So a permutation of P pieces costing "up to P stores" is not a cost at all: P is
// exactly how many landings the transfer already requires, and P is 2, 4 or 8 in the
// measured divisions.  This is also why the rejected ordering attribute below would
// have bought no performance — there is no redundant write for it to remove.
//
// **The zero-copy case and the general case.**  Two stores are zero-copy, and
// that depends on the value being stored.  If the assembled value feeds `linalg`
// directly, as a live intermediate, there is no access tile to carry the reorder
// and it becomes an ordinary `tensor` permutation — still local, still on-chip,
// so still inside what the requirement allows, but with real data movement.
// Either way one intent is spread over P ops: the permutation becomes visible
// only after reading which slice goes to which base.
//
// THE REQUIREMENT IS SATISFIED.  Coverage §5 asks to "show how logical order is
// preserved when it differs from core-ID order", and states that "local
// selection/reordering is acceptable if fully expressed and stays on-chip".  Two
// stores are exactly that: fully expressed — every element's destination is read
// off the `%c32` / `%c0` bases — on-chip, and zero-copy.  **So this case needs no
// new capability.**
//
// WHAT IT COSTS IS VERIFICATION.  §3.3 fixes the assembly order, the two stores
// place the halves at swapped bases, and **no rule relates the second to the
// first.**  A verifier sees two well-formed stores that between them cover the
// region exactly once, which is all it is asked to see.  Storing the assembly
// verbatim instead has the same shape, the same element count, the same coverage
// — and the wrong answer.  So the correctness of the reorder is **outside
// inter-tile verification**, and the numerical check is its only guarantee.  That
// is why the expected values at the end of this file are its point, and the op
// list is not.
//
// AN ORDERING ATTRIBUTE WAS CONSIDERED AND REJECTED.  A `gather_order` affine map
// on the delivery op, redefining §3.3's `l`, would put the intent in one place
// instead of spreading it over P stores.  It is not worth it.  An attribute can be
// checked for well-formedness — is it a permutation of `0..P-1`? — but never for
// intent, so it would not move this case out of "numerical test only" and into the
// verifier.  It would only add a second statement of the ordering that can
// disagree with the stores.  Verifiability is the reason the explicit form is
// preferred at all, so a construct that adds surface without adding checkable
// content is a loss.
//
// Aside, independent of this case: RFC 0682 defines `access_tile_order` twice over
// — "the rightmost dimension in the output space corresponds to the innermost
// iteration dimension", and "the enumeration of points in the intermediate
// variable space".  The first reads as a dimension nesting order, the second as a
// sort key; only the second could reorder within a dimension.  This file
// deliberately does not depend on the generous reading, and `KTDP.td`'s op
// description carries neither sentence.  Worth settling in the dialect, but not a
// prerequisite for anything here.
//
// NOT VERIFIED as a whole: `ktdp.inter_tile_gather` is specified (§6.4) but absent
// from KTDP.td.  Everything else here — the produce, both slices and both
// stores — parses and round-trips.
// ============================================================================

#producers = affine_set<(i)[g] : (i - 4 * g >= 0, -i + 4 * g + 2 >= 0,
                                 i mod 2 == 0)>
#consumers = affine_set<(i)[g] : (i - 4 * g >= 0, -i + 4 * g + 3 >= 0)>
#groups    = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

#piece  = affine_set<(d0, d1, d2, d3) : (
    d0 >= 0, -d0 >= 0,        d1 >= 0, -d1 + 4095 >= 0,
    d2 >= 0, -d2 + 31 >= 0,   d3 >= 0, -d3 + 63 >= 0)>
#region = affine_set<(d0, d1, d2, d3) : (
    d0 >= 0, -d0 >= 0,        d1 >= 0, -d1 + 4095 >= 0,
    d2 >= 0, -d2 + 63 >= 0,   d3 >= 0, -d3 + 63 >= 0)>
#ident4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  func.func @kt03_logical_order() {
    %c0   = arith.constant 0 : index
    %c32  = arith.constant 32 : index
    %base = arith.constant 0 : index
    // Past the end of the producer's own piece (1*4096*32*64 = 8388608 elements), so
    // the assembly is not written over its own source.
    %dest = arith.constant 8388608 : index

    %own = ktdp.construct_memory_view %base,
        sizes: [1, 4096, 32, 64], strides: [8388608, 2048, 64, 1] {
        coordinate_set = #piece,
        memory_space   = #ktdp.memory_space<ct_local>
    } : memref<1x4096x32x64xf16>

    %own_access = ktdp.construct_access_tile %own[%c0, %c0, %c0, %c0] {
        access_tile_set = #piece, access_tile_order = #ident4
    } : memref<1x4096x32x64xf16> -> !ktdp.access_tile<1x4096x32x64xindex>

    %piece_val = ktdp.load %own_access
        : !ktdp.access_tile<1x4096x32x64xindex> -> tensor<1x4096x32x64xf16>

    %future = ktdp.inter_tile_produce
        producer_tiles_per_group = #producers
        -> !ktdp.tile_future<(tensor<1x4096x32x64xf16>), groups = #groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %piece_val : tensor<1x4096x32x64xf16>
    }

    // Assembles [upper, lower] — ascending tile id, per §3.3.  Nothing on this op
    // can say otherwise.
    %region_val = ktdp.inter_tile_gather(%future)
        consumer_tiles_per_group = #consumers,
        gather_dimensions        = [2]
        : !ktdp.tile_future<(tensor<1x4096x32x64xf16>), groups = #groups>
          -> tensor<1x4096x64x64xf16>

    %out = ktdp.construct_memory_view %dest,
        sizes: [1, 4096, 64, 64], strides: [16777216, 4096, 64, 1] {
        coordinate_set = #region,
        memory_space   = #ktdp.memory_space<ct_local>
    } : memref<1x4096x64x64xf16>

    // Upper half, tensor rows 0..31, to memory rows 32..63.  Note the base:
    // %c32.  The order inside is identity; nothing is permuted within a
    // dimension.
    %upper = tensor.extract_slice %region_val[0, 0, 0, 0] [1, 4096, 32, 64]
                                              [1, 1, 1, 1]
        : tensor<1x4096x64x64xf16> to tensor<1x4096x32x64xf16>
    %upper_at = ktdp.construct_access_tile %out[%c0, %c0, %c32, %c0] {
        access_tile_set = #piece, access_tile_order = #ident4
    } : memref<1x4096x64x64xf16> -> !ktdp.access_tile<1x4096x32x64xindex>
    ktdp.store %upper, %upper_at
        : tensor<1x4096x32x64xf16>, !ktdp.access_tile<1x4096x32x64xindex>

    // Lower half, tensor rows 32..63, to memory rows 0..31.  Base %c0.
    %lower = tensor.extract_slice %region_val[0, 0, 32, 0] [1, 4096, 32, 64]
                                              [1, 1, 1, 1]
        : tensor<1x4096x64x64xf16> to tensor<1x4096x32x64xf16>
    %lower_at = ktdp.construct_access_tile %out[%c0, %c0, %c0, %c0] {
        access_tile_set = #piece, access_tile_order = #ident4
    } : memref<1x4096x64x64xf16> -> !ktdp.access_tile<1x4096x32x64xindex>
    ktdp.store %lower, %lower_at
        : tensor<1x4096x32x64xf16>, !ktdp.access_tile<1x4096x32x64xindex>

    // Expected in memory at cores 4g..4g+3: mb 64g..64g+63 in logical order —
    // core 4g+2's half at rows 0..31 and core 4g's at 32..63.  The failure to
    // catch is storing the assembly verbatim: same shape, same element count,
    // wrong answer.  That is coverage §5's "reverse two fragments" check.
    return
  }
}
