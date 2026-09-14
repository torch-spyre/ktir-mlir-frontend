// KT-06 — equal split counts, permuted core ownership.
//
// torch-spyre#4300 lx_relayout_workload_coverage.md §5 KT-06: "Gemma row-major
// versus column-major core order, described above.  **Equal split counts are
// insufficient.  Preserve or explicitly change ownership.**"
//
// And from the coverage document's Gemma discussion: "Both variants are required:
// if the next operation adopts the source order, zero-copy is correct; reading the
// old addresses with the new interpretation is not."
//
// Both variants are here.  They are not two spellings of one transfer — they are
// the two branches §9.1 row 1 already names, and which branch applies is a
// property of the *consumer*, not of this edge.
//
// OPERATION SEQUENCE.
//
//   @kt06_preserve_ownership   LX-Load (addr_0)
//   @kt06_relocate             LX-Load (addr_0) -> Produce -> Consume
//                                -> LX-Store (addr_0)
//
//   Variant A is one op long, and **the absence of a delivery is its claim** —
//   §9.1 row 1's "no op needed" branch.  Variant B loads and stores at the *same*
//   address: the permutation is entirely in the dependency attribute, and no local
//   address moves.
//
// ============================================================================
// MEASURED FIXTURE: GR-PF-052, `relayouts[51]` of the pinned catalog
// (sendnn_sdsc_lx_replay_manifest.json, sha256 4c8aca2e…c747d4).
//
//   consumer        bmm-wtAttnHeadBreak-VirtualReshape-Output-Restickify, input 0
//   route_class     permutation
//   extents         {j: 8, mb: 512, out: 128, x: 1, y: 1}    src == dst
//   word_length     2  (fp16)
//   32 source pieces  {j: 2, mb: 64, out: 128, x: 1, y: 1}, one owner each
//   32 destination pieces, **the same size**, one owner each
//   source_fragments_per_destination_piece = 1   -> P = 1
//   destination_pieces_per_source_piece    = 1   -> a bijection
//
// Two sibling records have the identical geometry and owner map: `relayouts[52]`
// (`bmm_2-…`, fold_factor 38, so the folded layer body) and `relayouts[53]`
// (`bmm_78-…`).  `relayouts[118]` and `[119]` are the decode counterparts.  So
// this edge occurs once per layer, 40 layers, and one KTIR form covers all five
// records.
//
// THE OWNER MAP.  Let J = j/2 in [0,4) and M = mb/64 in [0,8), so a region is
// (J, M) and there are 4*8 = 32 of them.  Reading the `owners` field of every
// piece on both sides:
//
//   source       owner k = 8*J + M     (M varies fastest — row-major in (J,M))
//   destination  owner m =   J + 4*M   (J varies fastest — column-major)
//
// Verified against all 32 pieces on both sides: `k == 8*(j/2) + (mb/64)` holds for
// every source piece and `m == (j/2) + 4*(mb/64)` for every destination piece.
//
// That is a transpose of a 4x8 index grid, which is exactly what coverage §5 calls
// "row-major versus column-major core order".  So
//
//   pi(k)      = (k floordiv 8) + 4 * (k mod 8)
//   pi^-1(m)   = 8 * (m mod 4) +     (m floordiv 4)
//
//   pi = [0, 4, 8,12,16,20,24,28,  1, 5, 9,13,17,21,25,29,
//         2, 6,10,14,18,22,26,30,  3, 7,11,15,19,23,27,31]
//
// It is a bijection, and **only owners 0 and 31 are fixed points**.  That matters
// for the test: an implementation that emits nothing at all is wrong on 30 of the
// 32 cores, so the identity is not a plausible near-miss here.
//
// Group structure, derived from the owner tables.  The transpose decomposes into
// **8 groups of 4 cores**.  Writing a consumer as `c = 4g + l` with `g = c/4` and
// `l = c mod 4`, its source is
//
//   pi^-1(4g + l) = 8*l + g
//
// so group `g` has consumers `{4g, 4g+1, 4g+2, 4g+3}` — four *consecutive* cores —
// drawing from producers `{g, g+8, g+16, g+24}` — four cores **spaced 8 apart**,
// i.e. exactly the cores congruent to `g` mod 8:
//
//   group g   producers {g, g+8, g+16, g+24}   consumers {4g .. 4g+3}
//   g=0       {0, 8, 16, 24}                   {0, 1, 2, 3}
//   g=1       {1, 9, 17, 25}                   {4, 5, 6, 7}
//   g=2       {2, 10, 18, 26}                  {8, 9, 10, 11}
//   ...       8 groups, regular
//   g=7       {7, 15, 23, 31}                  {28, 29, 30, 31}
//
// The pairing inside a group is `4g + l  <-  g + 8l` for l = 0..3, so group 0 is
// 0<-0, 1<-8, 2<-16, 3<-24 and group 1 is 4<-1, 5<-9, 6<-17, 7<-25.
//
// Checked against the record and not against the formula: consumer `4g + l`
// receives exactly what producer `g + 8*l` holds, for all 32 pairs, with both sides
// read out of the `owners` field of each piece.  So `#dep_transpose` below is
// consistent with the measured tables, not merely with the transpose that
// summarises them.
//
// Both families partition `0..31` — the consumer sets because they are consecutive
// blocks, the producer sets because they are the residue classes mod 8 — so R1 holds
// and is not vacuous.
//
// This is the semantically right grouping, and not only a tidier one: see "WHY 8
// GROUPS OF 4" below, where it turns out to catch strictly more than the
// alternatives.
//
// WHY 8 GROUPS OF 4.  Three groupings express this same permutation, and they are
// not equally checkable, because **R1, R4 and R5 are all obligations relative to a
// group.**  The grouping decides how much of the map a verifier ever compares.
//
//   32 singleton groups   one producer and one consumer each.  `pi^-1` is affine,
//                         so it is expressible — and **worthless**: with one
//                         consumer per group, R4 (every producer named) and R5
//                         (dependency sets disjoint) are vacuously true, and no
//                         rule ever relates the 32 pairs to each other.
//
//   one group of 32       R4 and R5 become surjectivity and injectivity of
//                         `pi^-1`, so the bijection is checked.  But R1 is trivial
//                         with a single group, and R3 (dep within the producer set)
//                         is nearly so, since every tile is in it.
//
//   8 groups of 4         R4 and R5 check four 4-element bijections instead of one
//                         32-element one — equally binding — and **R1 and R3 start
//                         working too.**  R1 has two partitions of `0..31` to
//                         verify; R3 rejects any dep that names a producer outside
//                         `{g, g+8, g+16, g+24}`.
//
// The last is strictly the strongest.  Take the error "consumer 0 reads core 1
// instead of core 0": under 8 groups it fails **R3** immediately, because 1 is not
// congruent to 0 mod 8 and so is not a producer of group 0.  Under one group of 32
// it slips past R3 and is caught only later and less directly, by R5, once some
// other consumer is found to name core 1 as well.  Under singleton groups it is not
// caught at all.
//
// What no grouping catches is a permutation *within* a group — swapping which of
// `{0, 8, 16, 24}` feeds consumers 0 and 1 keeps every rule satisfied.  So the
// rules narrow the error to "a valid permutation of the right four sources, but not
// this one", and the numerical check closes the rest.
//
// WHY NOT `all_to_all`.  A four-into-four exchange between whole pieces invites it,
// and §6.5 rules it out in a sentence: "one-to-one permutation of whole partials is
// already expressible as `consume` + a bijective dependency set (§7.4.2);
// `all_to_all` is only for the split-and-redistribute case, so the two mechanisms
// do not overlap."
//
// The reason is that §9.1 picks the op from two bits — does the edge split, does it
// concatenate.  `C = empty` and `R = empty` here, so it is row 1 (`consume`);
// `all_to_all` is row 3, which needs *both* non-empty.  Mechanically it also cannot
// be spelled: §6.5 requires both `split_dimensions` and `concat_dimensions`, and the
// result type is `T_p` with the split extent divided by `C` and the concat extent
// multiplied by `P`.  This case needs `T_c == T_p`, which forces `C = P = 1` and so
// 32 singleton groups — and then R9 still requires a non-empty split list, so one
// would be declaring a split into one chunk that does not happen, with `placement =
// permute` on a delivery that hands over an unmodified piece.
//
// The distinguishing test: `all_to_all` would be right if each consumer needed **a
// quarter of each of the four sources** instead of **the whole of one**.  §6.5 does
// allow what it calls a "pure ownership transpose along one axis set", but that
// transposes *within* the data and cuts every piece; this record cuts nothing.
//
// The near miss is in the same fixture family.  GR-PF-001 (`relayouts[0]`) also has
// `inputs per region = 2`, but `destination_pieces_per_source_piece = 4`, so its
// sources *are* cut — `mb` concatenated while `out` is split.  That one is a genuine
// `all_to_all`, and it is the KT-04 candidate.  In the catalog the two families are
// disjoint by construction: 37 records have neither axis set non-empty, 12 have both.
//
// ----------------------------------------------------------------------------
// HOW TO READ THE CATALOG FOR THIS RECORD — one trap, worth recording.
//
// `relayouts[*].source_pieces` is ordered lexicographically by its `key` string:
// p0, p1, p10, p11, ..., p19, p2, p20, ...  So **piece index is not owner order.**
// And `source_core_patterns` is a separate summary of the form
// `{cores: [...], pieces: N}`; it happens to have 32 entries for this record but
// it is not a per-piece record and carries no piece key.
//
// The authoritative owner is the **`owners` field on each piece**.  Deriving the
// map from `source_core_patterns[k]` alongside `source_pieces[k]` instead yields a
// table that is a bijection but not a transpose — an artifact of the lexicographic
// ordering, not a property of the record.  The transpose above only appears once
// the `owners` field is used.
// ----------------------------------------------------------------------------
//
// LAYOUT, AND WHY THIS CASE DOES NOT DEPEND ON IT.
//
// No SDSC run exists for this shape, unlike GR-PF-055 (see kt05).  And the catalog
// cannot supply one, because **its byte fields are logical**: for this record
// prod(extents) * word_length == logical_tensor_bytes exactly (1048576), and the
// same identity holds for GR-PF-055, where an SDSC run showed that a `y` extent of
// 1 is physically carried as 64 padded lanes.  Padding is therefore invisible in
// the catalog by construction, for every record.
//
// (A second naming trap: `source_piece_bytes` is the total over all pieces, not
// the size of one.  Here it is 1048576 = 32 * 32768.)
//
// This case is decidable anyway.  The permutation is a statement about **which
// core** holds a region, and §10.3's physicalization inserts the chunk-count axis
// at the front of *both* sides identically — it cannot change an owner map.  So
// the piece type below is written in the record's logical axes (`j`, `mb`, `out`,
// with the extent-1 `x` and `y` carrying no data), and this file makes no claim
// about the physical form.  It is not mixed with the SDSC physical spelling used
// in kt02 and kt05.  **KT-08 is where the layout question actually bites**, and it
// needs the SDSC run before it can be written.
// ============================================================================
//
// WHY THE PERMUTATION IS VERIFIABLE HERE, UNLIKE KT-03.
//
// kt03 is the same family of problem — a required order disagreeing with the order
// the IR supplies by default — and there the reorder had to go into the *store
// bases*, where no rule reaches it, so only a numerical check catches a mistake.
//
// Here the reorder is **across cores rather than within a region**, so it lives in
// `producer_dependency_per_consumer`, and the rules do reach it:
//
//   R1  groups pairwise exclusive over tiles       both families partition 0..31
//   R3  dep(c) subset of the group's producers     `g + 8*(c mod 4)` is in g mod 8
//   R4  every producer named by some consumer      surjective onto the 4 producers
//   R5  dependency sets pairwise disjoint          injective over the 4 consumers
//   R6  uniform cardinality                        |dep(c)| = 1 for every c
//   R8  exactly one source per consumer tile       dep is a function of (c, g)
//
// R5 applies here in the strong sense: this is a *partitioning* use — every producer
// is claimed by exactly one consumer of its group — so unlike kt01 the disjointness
// obligation is live and satisfied.  A dropped producer fails R4, a doubled one
// fails R5, and a source taken from the wrong group fails R3.  All of that depends
// on the grouping; see "WHY 8 GROUPS OF 4" above.
//
// What is *not* caught is a permutation within one group — swapping which of
// `{g, g+8, g+16, g+24}` feeds two of the group's consumers satisfies R1 through R8.
// So the rules confine the error to "a valid permutation of the right four sources,
// but not this one", and the numerical check closes the rest.  That is a strictly
// better position than kt03's, and the difference is only where the ordering
// information was allowed to live.
//
// NOT VERIFIED as a whole: `ktdp.inter_tile_consume` is specified (§6.1) but absent
// from KTDP.td.  The `ktdp.inter_tile_produce` half parses and round-trips, as do
// all three affine sets on their own — including the `floordiv` / `mod` in
// `#dep_transpose`, checked with ktir-opt.
// ============================================================================

// 8 groups of 4.  Producers of group `g` are the cores congruent to `g` mod 8,
// `{g, g+8, g+16, g+24}`; consumers are the consecutive block `{4g .. 4g+3}`.  Both
// families partition `0..31`, which is what makes R1 bite.
#producers = affine_set<(i)[g] : (i mod 8 - g == 0, i >= 0, -i + 31 >= 0)>
#consumers = affine_set<(i)[g] : (i - 4 * g >= 0, -i + 4 * g + 3 >= 0)>
#groups    = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// producer_dependency_per_consumer: `(p)[c, g]` per §3.4.  Consumer `c = 4g + l`
// waits for exactly one producer, `g + 8*l`, i.e. `g + 8*(c mod 4)`.
//
// Both `c` and `g` are required, as in §7.4.2's butterfly: `c` because each of the
// four consumers in a group has a different source, and `g` because the producer is
// offset by the group index.
#dep_transpose = affine_set<(p)[c, g] : (p - g - 8 * (c mod 4) == 0)>

// The whole of one owner's piece: (j, mb, out) = (2, 64, 128).
#piece  = affine_set<(d0, d1, d2) : (
    d0 >= 0, -d0 + 1 >= 0,  d1 >= 0, -d1 + 63 >= 0,  d2 >= 0, -d2 + 127 >= 0)>
#ident3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// ---------------------------------------------------------------------------
// VARIANT A — the consumer adopts the source order.  NO TRANSFER.
//
// §9.1 row 1 states this branch itself: with `C = empty` and `R = empty`, the
// regions are identical, and it is "`no op needed` if every core's region is its
// own".  Here every core's region *is* its own — the piece it already holds — so
// the correct emission for this variant is nothing at all.  Coverage §5's
// "preserve ... ownership" is this arm.
//
// The whole content of this function is therefore a load and the consumer, and
// **the absence of an inter-tile op is the claim being made.**
//
// THE PRECONDITION, WHICH IS NOT EXPRESSED ANYWHERE IN THE IR.  This is only
// correct if the consumer indexes its slab as k = 8*(j/2) + (mb/64) — the source
// map.  If it indexes as m = (j/2) + 4*(mb/64), the same addresses now name
// different data on 30 of the 32 cores, and nothing in this function changes:
// same ops, same types, same addresses, wrong answer.
//
// That is the failure coverage §5 warns about in the sentence "reading the old
// addresses with the new interpretation is not [correct]".  It is invisible to a
// verifier here for the same structural reason as kt03: the owner map has no
// syntactic home in this variant.  Which is the argument for making the consumer's
// expectation explicit somewhere — and, in variant B, it is.
// ---------------------------------------------------------------------------

module {
  func.func @kt06_preserve_ownership() {
    %c0   = arith.constant 0 : index
    %base = arith.constant 0 : index

    // The tile's own piece.  `ct_local` with no ct_id is the executing tile's own
    // LX, so no tile id enters the address.  Dense: out 1, mb 128, j 64*128.
    %own = ktdp.construct_memory_view %base,
        sizes: [2, 64, 128], strides: [8192, 128, 1] {
        coordinate_set = #piece,
        memory_space   = #ktdp.memory_space<ct_local>
    } : memref<2x64x128xf16>

    %own_access = ktdp.construct_access_tile %own[%c0, %c0, %c0] {
        access_tile_set = #piece, access_tile_order = #ident3
    } : memref<2x64x128xf16> -> !ktdp.access_tile<2x64x128xindex>

    %piece_val = ktdp.load %own_access
        : !ktdp.access_tile<2x64x128xindex> -> tensor<2x64x128xf16>

    // The consumer runs here, on the tile's own data, in the source order.
    // Expected at core k: region (J, M) with k = 8*J + M — that is, j in
    // [2*(k/8), 2*(k/8)+2) and mb in [64*(k%8), 64*(k%8)+64).  Unchanged from
    // what the core already held.
    return
  }
}

// ---------------------------------------------------------------------------
// VARIANT B — the consumer requires the destination order.  A RELOCATION.
//
// Same §9.1 row 1, other branch: regions identical but a core's destination region
// is *not* its own, so the op is `inter_tile_consume` — the row calls this "a
// **relocation**".  §6.1 calls the same shape a **routing** pattern rather than a
// broadcast: with several producers per group the dependency attribute names the
// sender, R8 requires it, and the group is "several independent point-to-point
// deliveries sharing one `produce` op, which is what lets `consume` express ...
// one-to-one permutation exchange (§7.4.2)".
//
// Coverage §5's "or explicitly change ownership" is this arm, and `#dep_transpose`
// is where the change is stated.
//
// LOCAL ADDRESSES DO NOT MOVE.  Both the load and the store are at base 0 in the
// executing tile's own LX: core k reads its own buffer, core m writes its own
// buffer.  What changes is only *which* core's value lands in which core's buffer.
// So the entire permutation is carried by the dependency set and nothing leaks
// into the addressing — the opposite of kt03, and the reason this case is
// verifiable while kt03 is not.
// ---------------------------------------------------------------------------

module {
  func.func @kt06_relocate() {
    %c0   = arith.constant 0 : index
    %base = arith.constant 0 : index

    %own = ktdp.construct_memory_view %base,
        sizes: [2, 64, 128], strides: [8192, 128, 1] {
        coordinate_set = #piece,
        memory_space   = #ktdp.memory_space<ct_local>
    } : memref<2x64x128xf16>

    %own_access = ktdp.construct_access_tile %own[%c0, %c0, %c0] {
        access_tile_set = #piece, access_tile_order = #ident3
    } : memref<2x64x128xf16> -> !ktdp.access_tile<2x64x128xindex>

    %piece_val = ktdp.load %own_access
        : !ktdp.access_tile<2x64x128xindex> -> tensor<2x64x128xf16>

    // Four producers per group, `{g, g+8, g+16, g+24}`, and every tile is a producer
    // of exactly one group.  The load sits outside the region here, as in kt01 and
    // kt02: no tile is a non-producer that must be kept from executing it (§2.2).
    // kt05 is the case where it must go inside.
    %future = ktdp.inter_tile_produce
        producer_tiles_per_group = #producers
        -> !ktdp.tile_future<(tensor<2x64x128xf16>), groups = #groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %piece_val : tensor<2x64x128xf16>
    }

    // combine = none, placement = replicate, |P(g)| = 4, |dep(c)| = 1 (§6.1).  No
    // dim attribute, no region, no identity — the result type is `T_p` unchanged,
    // because a relocation moves a piece without reshaping it.  With |P(g)| > 1 the
    // dependency attribute is *required* (§3.4), and it is the one attribute that
    // does the work: §6.1 calls this a routing pattern rather than a broadcast —
    // four independent point-to-point deliveries sharing one `produce`.
    %relocated = ktdp.inter_tile_consume(%future)
        consumer_tiles_per_group         = #consumers,
        producer_dependency_per_consumer = #dep_transpose
        : !ktdp.tile_future<(tensor<2x64x128xf16>), groups = #groups>
          -> tensor<2x64x128xf16>

    // Store into the consumer's own buffer, at base 0 — the same address the load
    // used.  See "LOCAL ADDRESSES DO NOT MOVE" above.
    %out = ktdp.construct_memory_view %base,
        sizes: [2, 64, 128], strides: [8192, 128, 1] {
        coordinate_set = #piece,
        memory_space   = #ktdp.memory_space<ct_local>
    } : memref<2x64x128xf16>

    %out_at = ktdp.construct_access_tile %out[%c0, %c0, %c0] {
        access_tile_set = #piece, access_tile_order = #ident3
    } : memref<2x64x128xf16> -> !ktdp.access_tile<2x64x128xindex>

    ktdp.store %relocated, %out_at
        : tensor<2x64x128xf16>, !ktdp.access_tile<2x64x128xindex>

    // Expected at core m: the region (J, M) with m = J + 4*M — that is, j in
    // [2*(m mod 4), 2*(m mod 4)+2) and mb in [64*(m/4), 64*(m/4)+64) — supplied by
    // core 8*(m mod 4) + (m floordiv 4).  Concretely for the first few:
    //
    //   core 0  <- core  0   j=0..1  mb=  0..63    (fixed point)
    //   core 1  <- core  8   j=2..3  mb=  0..63
    //   core 2  <- core 16   j=4..5  mb=  0..63
    //   core 3  <- core 24   j=6..7  mb=  0..63
    //   core 4  <- core  1   j=0..1  mb= 64..127
    //   core 31 <- core 31   j=6..7  mb=448..511   (fixed point)
    //
    // Values: identify each element by its (j, mb, out) coordinate one bit at a
    // time, per coverage §1's rule for low-precision tensors, so that a
    // misdelivered piece names the core it actually came from.
    //
    // Failures to catch.  (1) Emitting nothing — variant A's answer given variant
    // B's requirement — is wrong on 30 of 32 cores, since only 0 and 31 are fixed
    // points.  (2) Using pi instead of pi^-1: the dependency set must map a
    // *consumer* to its source, and pi maps the other way.  Both are bijections,
    // so R3 through R8 accept either and only the values distinguish them.  This is
    // the "wrong core map with identical split counts" check of coverage §5.
    return
  }
}
