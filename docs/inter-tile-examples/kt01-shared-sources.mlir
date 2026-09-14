// KT-01 — a source shared by several receivers, written as an explicit
// dependency.
//
// torch-spyre#4300 lx_relayout_workload_coverage.md §5 KT-01: "Four-way V toy
// above, default and explicit equal dependencies. **R5 must not reject valid
// sharing merely because different receiving cores use the same producer
// pieces.**  This concern is about explicit dependencies; default gather already
// describes an all-producer assembly."  And §4: "Check both the default
// all-producer dependency and an explicitly written equivalent dependency.  They
// should describe the same values."
//
// OPERATION SEQUENCE.  Both functions:
//
//   LX-Load (addr_0) -> Produce -> Gather
//
//   addr_0 = the producer's own piece in its LX.  The delivered region is left
//   unstored: this file is about the dependency attribute, not about placement.
//
// ============================================================================
// SAME FIXTURE AS kt02-receiving-only-cores.mlir — ONLY THE ATTRIBUTE IS ADDED.
//
// GR-PF-002, i.e. `relayouts[1]` of the pinned ownership catalog
// (sendnn_sdsc_lx_replay_manifest.json, sha256 4c8aca2e…c747d4), consumer
// `mm-BMM_1` input 0, route_class grouped_all_gather_with_replication.
//
// Group structure, from the owner tables.  Source piece k is owned by core 2k
// and starts at mb = 32k, so group g (destination mb 64g..64g+63) is:
//
//   group g   producers {4g, 4g+2}   consumers {4g .. 4g+3}   dep(c,g)   non-prod
//   g=0       {0, 2}                 {0, 1, 2, 3}             {0, 2}     1, 3
//   g=1       {4, 6}                 {4, 5, 6, 7}             {4, 6}     5, 7
//   g=2       {8, 10}                {8, 9, 10, 11}           {8, 10}    9, 11
//   ...       8 groups, regular
//   g=7       {28, 30}               {28, 29, 30, 31}         {28, 30}   29, 31
//
// Source piece {mb:32, in:4096, y:1}, one owner each; destination piece
// {mb:64, in:4096, y:1}, four owners each.
//
// kt02 leaves `producer_dependency_per_consumer` absent, which §3.4 defines as
// "the consumer waits on and receives from **all** producer tiles in the group".
// This file writes that same set out.  The two must describe the same values.
//
//   default form  (kt02)   attribute absent          -> P = |{4g, 4g+2}| = 2
//   explicit form (here)   dep(c, g) = {4g, 4g+2}    -> P = 2, for every c
//
// WHY kt02 IS NOT ENOUGH.  §4 asks to "check **both** the default all-producer
// dependency and an explicitly written equivalent dependency", so the pair is the
// requirement and neither file satisfies it alone.  More than that: **R5 cannot
// be exercised by kt02 at all** — the default form carries no dependency
// attribute, so there is nothing for the rule to apply to.  Coverage §5 says as
// much: "This concern is about explicit dependencies; default gather already
// describes an all-producer assembly."  Writing the set is what makes R5 able to
// fire, and it also pins P explicitly rather than by default, so a verifier that
// mis-derived P from the producer set would show up here.
// ============================================================================
//
// WHAT IT SETTLES.  All four consumers of a group name **the same two
// producers**, so their declared dependency sets are identical — not disjoint.
// Read as a blanket rule, R5's pairwise disjointness would reject this, and with
// it every all-gather (§6.4: the full group as consumer set, every consumer
// naming every producer) and every multicast source (R8: "a producer **may**
// serve several consumer tiles").  §5 now scopes the obligation to a
// *partitioning* use and says so directly: "A verifier must therefore not reject
// overlap as such."
//
// Well-definedness does not come from disjointness.  It comes from §3.3's rule
// that an assembling consumer's positions are taken from **its own** declared
// set: here that set is {4g, 4g+2}, so l(4g) = 0 and l(4g+2) = 1 and the two
// slabs land in mb order.  Whatever the other three consumers declare cannot
// disturb it.
//
// The other dependency rules are satisfied and worth checking against:
//   R3  dep(c, g) subset of producer_tiles_per_group(g)     {4g,4g+2} ⊆ {4g,4g+2}
//   R4  every producer named by some consumer               both, by all four
//   R6  uniform cardinality across consumers                |dep| = 2 for all
//   R13 consumers need not be producers                     4g+1, 4g+3 are not
//
// Note the producers' tile ids are **not adjacent** — 4g and 4g+2.  That is why
// §3.3 defines `l` as a position in ascending tile-id order rather than a tile
// id: taken literally, tile id 4g+2 would name slot 2 of a two-slot assembly.
//
// Geometry, expected values and the physical layout are as in kt02; see that
// file, which carries the SDSC-verified `layoutDimOrder_` / `stickDimOrder_`
// derivation and the one assumption left in it.
//
// ============================================================================
// TWO FUNCTIONS, BECAUSE THE MEASURED FORM CANNOT DISCRIMINATE §3.3.
//
// @kt01_shared_sources — measured.  GR-PF-002, explicit dependency naming both
// producers for every consumer.  Operationally identical to kt02: the attribute
// restates the default, so the two files must deliver the same values, and that
// agreement is the test coverage §4 asks for.  It exercises R5's scoping, since
// four identical sets are not disjoint.  It does NOT exercise §3.3's "which set"
// clause: with identical sets a producer lands at the same position under either
// reading, so the clause is invisible here.
//
// @kt01_overlapping_sets — SYNTHETIC.  Sets that overlap and differ, which is
// the only form that discriminates §3.3.  Searched for in the catalog and not
// found: 14 of 130 records have a source piece feeding several destination
// pieces, but in **0 of 130** do two consumers declare different overlapping
// sets — a shared source is always taken by consumers that declare the *same*
// set.  So §3.3's clause is currently unforced by measurement while R5's scoping
// is required by it, and both facts belong in the record.
//
// CONSISTENCY TO VERIFY.  A written dependency set is a second, independent
// statement of something the producer set already says, so the two can disagree.
// Four checks, in the order a verifier would reach them:
//
//   1. dep(c, g) subset of producer_tiles_per_group(g)                    (R3)
//   2. every producer named by at least one consumer                      (R4)
//   3. |dep(c, g)| the same for every consumer of the group               (R6)
//   4. **P taken from dep, not from the producer set**, and the declared
//      result type following from that P                            (§3.1, R12)
//
// Check 4 is the one with two possible answers.  §3.1 defines `P` as
// `|producer_tiles_per_group(g)|` when the attribute is **absent** and the
// per-consumer cardinality when it is **present**, so a verifier has to pick the
// right source.  The two functions differ in exactly this:
//
//   @kt01_shared_sources     producers/group 2, |dep| 2   -> cannot discriminate
//   @kt01_overlapping_sets   producers/group 4, |dep| 2   -> discriminates
//
// In the second, taking P from the producer set gives 4 and would make mb go
// 32 -> 128, contradicting the declared tensor<1x4096x64x64xf16>.  Taking it from
// dep gives 2 and 32 -> 64, which is what the file declares.  So the synthetic
// function is the discriminating test for the P derivation as well as for §3.3's
// position derivation — the same structure of problem, and the measured form is
// blind to both.
//
// The fifth check is not local to one file: **the explicit-all form must deliver
// what the default form does**, which means running kt02 and
// @kt01_shared_sources on the same input and comparing their outputs to each
// other, not only each to a reference (§4).
// ============================================================================
//
// NOT VERIFIED as a whole: `ktdp.inter_tile_gather` is specified (§6.4) but
// absent from KTDP.td.  The `ktdp.inter_tile_produce` half of both functions
// parses and round-trips, as do all the dependency sets on their own.

#producers = affine_set<(i)[g] : (i - 4 * g >= 0, -i + 4 * g + 2 >= 0,
                                 i mod 2 == 0)>
#consumers = affine_set<(i)[g] : (i - 4 * g >= 0, -i + 4 * g + 3 >= 0)>
#groups    = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// producer_dependency_per_consumer: `(p)[c, g]` per §3.4.  Every consumer `c` of
// group `g` names both producers, so `c` does not appear in the constraints —
// the mapping is the same for every consumer, which is exactly the sharing R5
// must not reject.  Written out rather than omitted so the two forms can be
// compared; omitting it is kt02.
#dep_all = affine_set<(p)[c, g] : (p - 4 * g >= 0, -p + 4 * g + 2 >= 0,
                                  p mod 2 == 0)>

// The whole of the producer's own piece: [1, 4096, 32, 64].
#piece  = affine_set<(d0, d1, d2, d3) : (
    d0 >= 0, -d0 >= 0,        d1 >= 0, -d1 + 4095 >= 0,
    d2 >= 0, -d2 + 31 >= 0,   d3 >= 0, -d3 + 63 >= 0)>
#ident4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  func.func @kt01_shared_sources() {
    %c0   = arith.constant 0 : index
    %base = arith.constant 0 : index

    // The producer's own LX piece, at offset 0.  `ct_local` with no ct_id is the
    // executing tile's local memory, so no tile id enters the address.
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

    // The only difference from kt02: producer_dependency_per_consumer is
    // written, naming both producers for every consumer.  P is unchanged at 2,
    // so the result type is unchanged and the delivered values are the same as
    // the default form's.  A verifier that rejected the identical sets here
    // would reject all-gather too.
    %region = ktdp.inter_tile_gather(%future)
        consumer_tiles_per_group         = #consumers,
        gather_dimensions                = [2],
        producer_dependency_per_consumer = #dep_all
        : !ktdp.tile_future<(tensor<1x4096x32x64xf16>), groups = #groups>
          -> tensor<1x4096x64x64xf16>

    // Expected at cores 4g..4g+3: destination piece g, logical
    // (mb = 64g..64g+63, in = 0..4095, y = 0), core 4g supplying the lower mb
    // half and core 4g+2 the upper.  All four hold the same assembled region
    // (§3.7), byte for byte identical to what kt02 delivers.  Undefined
    // elsewhere.
    //
    // The pair is the test: run kt02 and this file on the same input and compare
    // the two outputs, not just each against a reference.  "They should describe
    // the same values" is the requirement, so a difference between the two forms
    // is itself a failure even if both look plausible alone.
    return
  }
}

// ---------------------------------------------------------------------------
// SYNTHETIC.  Overlapping, differing dependency sets.
//
// Not a measured shape: no record in the catalog has two consumers declaring
// different overlapping sets (0 of 130).  Written because it is the only form
// that tells §3.3's two readings apart, so if the clause is ever load-bearing
// this is the shape that will show it.
//
// Deviation from GR-PF-002, stated so it is not mistaken for measurement: four
// producers per group instead of two, and three consumers instead of four, so
// that a two-wide sliding window fits inside a group.  The piece type is
// unchanged.
//
// Group structure, same layout as the table above.
//
//   group g   producers {4g .. 4g+3}   consumers {4g .. 4g+2}   produces only
//   g=0       {0, 1, 2, 3}             {0, 1, 2}                3
//   g=1       {4, 5, 6, 7}             {4, 5, 6}                7
//   g=2       {8, 9, 10, 11}           {8, 9, 10}               11
//   ...       8 groups, regular
//   g=7       {28, 29, 30, 31}         {28, 29, 30}             31
//
// dep(c, g) = {c, c+1}, so within group g the three consumers name
// 4g->{4g,4g+1}, 4g+1->{4g+1,4g+2}, 4g+2->{4g+2,4g+3}.  Tile 4g+3 produces but
// never consumes.
//
// Positions within each consumer's own set (§3.3):
//
//   consumer 4g+0 : {4g+0, 4g+1}    l(4g+0)=0, l(4g+1)=1
//   consumer 4g+1 : {4g+1, 4g+2}    l(4g+1)=0, l(4g+2)=1
//   consumer 4g+2 : {4g+2, 4g+3}    l(4g+2)=0, l(4g+3)=1
//
// PRODUCER 4g+1 LANDS AT SLOT 1 IN ONE ASSEMBLY AND SLOT 0 IN ANOTHER.  That is
// possible only if positions come from the consumer's own declared set, which is
// what §3.3 now states.  Taken from the group's producer set instead, 4g+1 would
// have one position and the two assemblies could not both be right.
//
// Rules, checked: R3 dep ⊆ producers; R4 all four producers named by some
// consumer (4g+0 by c=4g+0; 4g+3 by c=4g+2); R6 |dep| = 2 for every consumer, so
// P = 2 and the result type is single-valued; R5 scoped — the sets overlap and
// are not a partition, so the disjointness obligation does not apply.
//
// Expected values: the same bit-per-coordinate scheme as kt02.  The check that
// matters is that a coordinate supplied by producer 4g+1 appears in the *upper*
// mb half of consumer 4g+0's region and the *lower* half of consumer 4g+1's.
// Swapping those two is the failure this example exists to catch.
// ---------------------------------------------------------------------------

#producers4 = affine_set<(i)[g] : (i - 4 * g >= 0, -i + 4 * g + 3 >= 0)>
#consumers3 = affine_set<(i)[g] : (i - 4 * g >= 0, -i + 4 * g + 2 >= 0)>
// dep(c, g) = {c, c+1}.  `g` is unused: the rule is the same in every group.
#dep_window = affine_set<(p)[c, g] : (p - c >= 0, -p + c + 1 >= 0)>

module {
  func.func @kt01_overlapping_sets() {
    %c0   = arith.constant 0 : index
    %base = arith.constant 0 : index

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
        producer_tiles_per_group = #producers4
        -> !ktdp.tile_future<(tensor<1x4096x32x64xf16>), groups = #groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %piece_val : tensor<1x4096x32x64xf16>
    }

    // P = 2 from the declared subsets (§3.4), not 4 from the producer set, so
    // mb goes 32 -> 64 and each consumer assembles only its own two slabs.
    %region = ktdp.inter_tile_gather(%future)
        consumer_tiles_per_group         = #consumers3,
        gather_dimensions                = [2],
        producer_dependency_per_consumer = #dep_window
        : !ktdp.tile_future<(tensor<1x4096x32x64xf16>), groups = #groups>
          -> tensor<1x4096x64x64xf16>

    // Expected: consumer c holds producers c and c+1 in that order.  Results are
    // undefined at 4g+3, which consumes nothing here (§3.7).
    return
  }
}
