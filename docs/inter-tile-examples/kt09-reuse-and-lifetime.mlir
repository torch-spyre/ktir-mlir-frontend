// KT-09 — many readers of one delivered value, and scratch reuse across iterations.
//
// torch-spyre#4300 lx_relayout_workload_coverage.md §5 KT-09: "Multiple consumers
// read one delivered X; then two expert/page iterations reuse scratch only after
// reads complete.  **Demonstrate reuse and lifetime without making every arithmetic
// reader initiate another transfer.  This does not require multiple delivery users
// of one future.**"
//
// That last sentence is the whole answer to the first half, and it is worth stating
// why: **R2 constrains the `!ktdp.tile_future`, not the delivery's result.**  The
// future must have exactly one use — the delivery op (`KTIRCheckLegality.cpp:80-85`).
// The delivery's *result* is an ordinary `tensor` SSA value with no use restriction,
// so any number of arithmetic readers may consume it and none of them is a transfer.
// One `produce` + one delivery + N readers is the natural spelling, not a workaround.
//
// OPERATION SEQUENCE.
//
//   @kt09_three_readers   LX-Load (addr_0) -> Produce -> Gather -> Mul x 3
//                           -> LX-Store x 3 (addr_1, addr_2, addr_3)
//
//   @kt09_scratch_reuse   Produce -> Gather -> LX-Store (addr_0)
//                           -> LX-Load x 2 (addr_0) -> Add -> LX-Store (addr_1)
//                         Produce -> Gather -> LX-Store (addr_0)
//                           -> LX-Load x 2 (addr_0) -> Add -> LX-Store (addr_2)
//
//   Half 1: addr_0 = own piece, addr_1..3 = the Q/K/V outputs.  One Gather feeds
//   three Muls — the delivered value stays a tensor and is never landed by the IR.
//
//   Half 2: addr_0 = the reused area, addr_1/addr_2 = per-iteration outputs.  The
//   second `LX-Store (addr_0)` is the reuse, and it is a write-after-read against
//   the two `LX-Load (addr_0)` above it.  No LX-Load before the first Produce: the
//   partials are computed, so addr_0 is the only area in the function.
//
// ============================================================================
// MEASURED FIXTURE: the same relayout recorded three times, once per reader.
//
// `mean-LayerNormNorm_out` (prefill) is the source tensor of **three** catalog
// records, all input 0 of a matmul:
//
//   relayouts[ 1]   consumer mm-BMM_1      (this is GR-PF-002, kt01's and kt02's)
//   relayouts[ 8]   consumer mm_1-BMM_1
//   relayouts[16]   consumer mm_2-BMM_1
//
// One layer-norm output feeding three projections — Q, K and V.  Checked: the three
// records are **identical** in extents, source and destination pieces, owner tables,
// route class and fragment counts.  They are not three relayouts; they are one
// relayout that three consumers need.
//
//   extents      {in: 4096, mb: 512, y: 1}
//   16 source pieces {in: 4096, mb: 32, y: 1}, one owner each
//    8 destination pieces {in: 4096, mb: 64, y: 1}, four owners each
//   remote_destination_bytes  12582912  (12 MiB), the same in all three records
//
// THE COST OF GETTING IT WRONG IS MEASURED, NOT ARGUED.  Emitting one delivery per
// reader moves 3 * 12 MiB = 36 MiB where 12 MiB suffices — **exactly 3x** the ring
// traffic for the same result.  The catalog invites the mistake by construction: it
// is indexed by (consumer, input), so a shared source appears as several records and
// a reader that walks the records one at a time will emit one transfer each.  Eight
// of the 120 distinct tensors in the catalog are shared this way.
//
// Group structure — GR-PF-002's, the same as kt01 and kt02.  Source piece k is owned
// by core 2k and starts at mb = 32k, so group g (destination mb 64g..64g+63) is:
//
//   group g   producers {4g, 4g+2}   consumers {4g .. 4g+3}   non-producers
//   g=0       {0, 2}                 {0, 1, 2, 3}             1, 3
//   g=1       {4, 6}                 {4, 5, 6, 7}             5, 7
//   ...       8 groups, regular
//   g=7       {28, 30}               {28, 29, 30, 31}         29, 31
//
// LAYOUT.  As kt02 derives it: an SDSC run of this logical shape reports
// `layoutDimOrder_ = ["mb", "out", "y"]`, `stickDimOrder_ = ["y"]`, `stickSize_ = 64`,
// so `y` is innermost and sticked, an extent-1 `y` is 64 padded lanes, and the
// physical form is one rank higher with the data axes reversed.  The one assumption
// kt02 leaves standing — that this record's order is `["mb", "in", "y"]` — is
// unchanged here and does not affect what this file is about.
// ============================================================================
//
// WHAT THE TWO HALVES OF KT-09 COST, AND WHY THEY DIFFER.
//
// **Half 1, many readers: expressible, and checked.**  R2 is a rule and it is
// implemented, so a spelling that gave each reader its own future would be *rejected*
// if it tried to reuse one produce — and if it duplicated the produce instead, the
// duplication is visible as N `inter_tile_produce` ops in the IR.  Either way the
// mistake is not silent.  This is the one half of one KT case so far that the
// verifier and the eye both reach.
//
// **Half 2, lifetime: expressible, but only if the buffer is named.**  Two things are
// being asked for and KTIR treats them differently.
//
//   (a) A named area's ordering IS expressed, and needs no inter-tile rule.
//       Iteration 1 must not store into `%shared` before iteration 0's readers have
//       loaded from it.  Both are memory effects on one `memref`, so ordinary MLIR
//       carries it: a write-after-read on one memref is a dependence no pass may
//       reorder.  Half 2 is that, and nothing more.
//
//   (b) An *unnamed* delivered value has no live range to reason about.  The
//       delivery returns a `tensor` — no address, no memory space (§2.2's future
//       carries `tensor`, not `memref`).  So half 1's three readers read no buffer;
//       the memref read ended at `ktdp.load`, before the produce, and **those
//       readers cannot extend a live range because there is no live range to
//       extend.**  For a copy delivery the buffer nevertheless exists physically —
//       the hardware lands every received tile in LX — so the question is never
//       whether there is one, only whether the IR says so.  Name it and the lifetime
//       is checkable; leave it unnamed and coverage §1's "buffers preserved until
//       last reader" lands on the emitted program instead.
//
// The root cause of (b) is the same "no address" fact kt03 leaned on to put its
// reorder in store bases, and the same one behind Table 1's note that a
// dist-mem-view cannot declare ownership for an intermediate.  Three cases, one
// root cause — but note the sign differs.  For kt03 the absence of an address is
// what makes the reorder free; here it is what makes the lifetime invisible.
//
// NOT VERIFIED as a whole: `ktdp.inter_tile_gather` is specified (§6.4) but absent
// from KTDP.td, in both functions.  The `ktdp.inter_tile_produce` halves, the three
// readers and the scratch ordering all parse and round-trip.
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

// ---------------------------------------------------------------------------
// HALF 1 — one delivery, three readers.
//
// The three `linalg.mul` ops stand for the Q, K and V projections; what they compute
// is not the point and a real kernel would put matmuls here.  The point is what sits
// *above* them: one `inter_tile_produce`, one `inter_tile_gather`, and `%region_val`
// used three times.  Reading a `tensor` three times is free — there is no second
// transfer to elide, because a transfer is an op and there is only one.
// ---------------------------------------------------------------------------

module {
  func.func @kt09_three_readers() {
    %c0   = arith.constant 0 : index
    %base = arith.constant 0 : index
    %one  = arith.constant 1.0 : f16
    %two  = arith.constant 2.0 : f16
    %four = arith.constant 4.0 : f16

    // Three distinct destination buffers.  The source piece occupies
    // 1*4096*32*64 = 8388608 elements from 0, and each assembled region is
    // 1*4096*64*64 = 16777216, so the three bases are laid end to end after it.
    %base_q = arith.constant  8388608 : index
    %base_k = arith.constant 25165824 : index
    %base_v = arith.constant 41943040 : index

    %own = ktdp.construct_memory_view %base,
        sizes: [1, 4096, 32, 64], strides: [8388608, 2048, 64, 1] {
        coordinate_set = #piece,
        memory_space   = #ktdp.memory_space<ct_local>
    } : memref<1x4096x32x64xf16>

    %own_access = ktdp.construct_access_tile %own[%c0, %c0, %c0, %c0] {
        access_tile_set = #piece, access_tile_order = #ident4
    } : memref<1x4096x32x64xf16> -> !ktdp.access_tile<1x4096x32x64xindex>

    // The memref read ends HERE.  Everything downstream is value semantics.
    %piece_val = ktdp.load %own_access
        : !ktdp.access_tile<1x4096x32x64xindex> -> tensor<1x4096x32x64xf16>

    %future = ktdp.inter_tile_produce
        producer_tiles_per_group = #producers
        -> !ktdp.tile_future<(tensor<1x4096x32x64xf16>), groups = #groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %piece_val : tensor<1x4096x32x64xf16>
    }

    // ONE delivery.  `%future` has exactly one use, which is R2 satisfied.
    %region_val = ktdp.inter_tile_gather(%future)
        consumer_tiles_per_group = #consumers,
        gather_dimensions        = [2]
        : !ktdp.tile_future<(tensor<1x4096x32x64xf16>), groups = #groups>
          -> tensor<1x4096x64x64xf16>

    // THREE readers of the one result.  R2 says nothing about this — it is a rule on
    // the future, and `%region_val` is a plain tensor.
    %wq_e = tensor.empty() : tensor<1x4096x64x64xf16>
    %wq   = linalg.fill ins(%one : f16) outs(%wq_e : tensor<1x4096x64x64xf16>)
              -> tensor<1x4096x64x64xf16>
    %q_e  = tensor.empty() : tensor<1x4096x64x64xf16>
    %q    = linalg.mul ins(%region_val, %wq : tensor<1x4096x64x64xf16>,
                                              tensor<1x4096x64x64xf16>)
              outs(%q_e : tensor<1x4096x64x64xf16>) -> tensor<1x4096x64x64xf16>

    %wk_e = tensor.empty() : tensor<1x4096x64x64xf16>
    %wk   = linalg.fill ins(%two : f16) outs(%wk_e : tensor<1x4096x64x64xf16>)
              -> tensor<1x4096x64x64xf16>
    %k_e  = tensor.empty() : tensor<1x4096x64x64xf16>
    %k    = linalg.mul ins(%region_val, %wk : tensor<1x4096x64x64xf16>,
                                              tensor<1x4096x64x64xf16>)
              outs(%k_e : tensor<1x4096x64x64xf16>) -> tensor<1x4096x64x64xf16>

    %wv_e = tensor.empty() : tensor<1x4096x64x64xf16>
    %wv   = linalg.fill ins(%four : f16) outs(%wv_e : tensor<1x4096x64x64xf16>)
              -> tensor<1x4096x64x64xf16>
    %v_e  = tensor.empty() : tensor<1x4096x64x64xf16>
    %v    = linalg.mul ins(%region_val, %wv : tensor<1x4096x64x64xf16>,
                                              tensor<1x4096x64x64xf16>)
              outs(%v_e : tensor<1x4096x64x64xf16>) -> tensor<1x4096x64x64xf16>

    // Store the three results to three distinct destinations, so the readers are
    // visibly independent rather than a chain.  The three `coordinate_set`s are the
    // same `#region` on purpose: Q, K and V are three *different* tensors and each
    // core holds the same region of each, so what has to differ is the base address,
    // not the coordinate set.
    %out_q = ktdp.construct_memory_view %base_q,
        sizes: [1, 4096, 64, 64], strides: [16777216, 4096, 64, 1] {
        coordinate_set = #region,
        memory_space   = #ktdp.memory_space<ct_local>
    } : memref<1x4096x64x64xf16>
    %at_q = ktdp.construct_access_tile %out_q[%c0, %c0, %c0, %c0] {
        access_tile_set = #region, access_tile_order = #ident4
    } : memref<1x4096x64x64xf16> -> !ktdp.access_tile<1x4096x64x64xindex>
    ktdp.store %q, %at_q
        : tensor<1x4096x64x64xf16>, !ktdp.access_tile<1x4096x64x64xindex>

    %out_k = ktdp.construct_memory_view %base_k,
        sizes: [1, 4096, 64, 64], strides: [16777216, 4096, 64, 1] {
        coordinate_set = #region,
        memory_space   = #ktdp.memory_space<ct_local>
    } : memref<1x4096x64x64xf16>
    %at_k = ktdp.construct_access_tile %out_k[%c0, %c0, %c0, %c0] {
        access_tile_set = #region, access_tile_order = #ident4
    } : memref<1x4096x64x64xf16> -> !ktdp.access_tile<1x4096x64x64xindex>
    ktdp.store %k, %at_k
        : tensor<1x4096x64x64xf16>, !ktdp.access_tile<1x4096x64x64xindex>

    %out_v = ktdp.construct_memory_view %base_v,
        sizes: [1, 4096, 64, 64], strides: [16777216, 4096, 64, 1] {
        coordinate_set = #region,
        memory_space   = #ktdp.memory_space<ct_local>
    } : memref<1x4096x64x64xf16>
    %at_v = ktdp.construct_access_tile %out_v[%c0, %c0, %c0, %c0] {
        access_tile_set = #region, access_tile_order = #ident4
    } : memref<1x4096x64x64xf16> -> !ktdp.access_tile<1x4096x64x64xindex>
    ktdp.store %v, %at_v
        : tensor<1x4096x64x64xf16>, !ktdp.access_tile<1x4096x64x64xindex>

    // Expected: `%q`, `%k` and `%v` each hold the assembled region scaled by 1, 2
    // and 4, so a reader that received the wrong region shows up as the wrong
    // multiple.  The three destination extents are 16777216 elements apart, so the
    // stores do not alias and a later reader of Q cannot see V's result.
    //
    // Failures to catch.  (1) Three deliveries instead of one: the result is
    // identical, so **only the emitted transfer count separates them** — 36 MiB
    // against 12 MiB.  Coverage §1's "check the emitted memory accesses as well as
    // numerical output" is the check that bites, and no numerical test can.
    // (2) Duplicating the produce as well as the delivery: visible in the IR as
    // three `inter_tile_produce` ops, and each core then computes its partial three
    // times.
    return
  }
}

// ---------------------------------------------------------------------------
// HALF 2 — one LX area reused by two iterations, with several readers in between.
//
// SYNTHETIC in its iteration structure: Granite 3.3 is dense, so it has no expert
// loop, and the pinned catalog holds no two-iteration reuse fixture.  A page loop
// over `cat_*-kvCacheScatter` would be the measured shape to attach here once its
// descriptors are available.  The delivery itself is the measured one above.
//
// **The received tile is stored, and that store is not overhead.** For an LX-to-LX
// transfer this hardware lands every received tile in LX before a compute unit can
// read it, so a landing store exists whether or not the IR names one.  Half 1 does
// not name it — the delivered `tensor` feeds `linalg` directly and the backend
// invents the buffer.  Half 2 names it, and that is the whole difference between
// them: naming costs nothing extra and buys a live range that the rules can see.
//
// **This is a fact about copy delivery, not about `reduce`.**  A reduction is not
// confined to the LX-to-LX path: a compute unit can send a tile out over a different
// ring, so a fold need not land each partial in LX on the way.  So the landing store
// is implied by `consume`, `gather`, `scatter` and `all_to_all` — the ops that move
// data unchanged — and not by `inter_tile_reduce`, which is why kt07a can leave its
// result unstored without that being a shortcut.
//
// So the two halves are a genuine choice, not a good form and a bad one:
//
//   delivered value kept as a `tensor`   no store in the IR; the staging buffer is
//   (half 1, and kt03/kt04/kt06)         the backend's, and its lifetime with it
//
//   delivered value stored into LX       the landing is explicit; readers read the
//   (half 2)                             memref, so the live range is in the IR
//
// WHAT THIS DEMONSTRATES.  `%shared` is the reused area.  Iteration 0 stores the
// delivered region into it and two readers load from it; iteration 1 then stores its
// own delivered region into the same address.  That second store is the reuse, and it
// is a **write-after-read** against both of iteration 0's loads — conflicting memory
// effects on one `memref`, which no pass may reorder.  Coverage §5's "reuse scratch
// only after reads complete" is therefore discharged by ordinary MLIR memory
// semantics, and needs no inter-tile rule at all.
//
// Note which readers hold the buffer.  In half 1 the three readers consume a
// `tensor` and hold nothing, so they cannot keep a buffer alive.  Here they consume
// the `memref`, and they can.  That is the same distinction from the other side.
//
// The partials are computed rather than loaded, so `%shared` is the only memref in
// this function and the reuse cannot be confused with a source buffer's traffic.
// ---------------------------------------------------------------------------

module {
  func.func @kt09_scratch_reuse() {
    %c0    = arith.constant 0 : index
    %v0    = arith.constant 1.0 : f16
    %v1    = arith.constant 8.0 : f16

    // The reused LX area, and the two per-iteration outputs after it.  One region is
    // 1*4096*64*64 = 16777216 elements.
    %base_shared = arith.constant        0 : index
    %base_out0   = arith.constant 16777216 : index
    %base_out1   = arith.constant 33554432 : index

    %shared = ktdp.construct_memory_view %base_shared,
        sizes: [1, 4096, 64, 64], strides: [16777216, 4096, 64, 1] {
        coordinate_set = #region,
        memory_space   = #ktdp.memory_space<ct_local>
    } : memref<1x4096x64x64xf16>
    %shared_at = ktdp.construct_access_tile %shared[%c0, %c0, %c0, %c0] {
        access_tile_set = #region, access_tile_order = #ident4
    } : memref<1x4096x64x64xf16> -> !ktdp.access_tile<1x4096x64x64xindex>

    // ---- iteration 0 ----
    %e0 = tensor.empty() : tensor<1x4096x32x64xf16>
    %p0 = linalg.fill ins(%v0 : f16) outs(%e0 : tensor<1x4096x32x64xf16>)
            -> tensor<1x4096x32x64xf16>

    %fut0 = ktdp.inter_tile_produce
        producer_tiles_per_group = #producers
        -> !ktdp.tile_future<(tensor<1x4096x32x64xf16>), groups = #groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %p0 : tensor<1x4096x32x64xf16>
    }

    %region_0 = ktdp.inter_tile_gather(%fut0)
        consumer_tiles_per_group = #consumers,
        gather_dimensions        = [2]
        : !ktdp.tile_future<(tensor<1x4096x32x64xf16>), groups = #groups>
          -> tensor<1x4096x64x64xf16>

    // THE LANDING, named.  On hardware this store happens either way; writing it
    // here is what gives `%shared` a live range the IR can reason about.
    ktdp.store %region_0, %shared_at
        : tensor<1x4096x64x64xf16>, !ktdp.access_tile<1x4096x64x64xindex>

    // TWO readers, both reading the LX area rather than a tensor.  These are what
    // iteration 1's store must wait for.
    %a0 = ktdp.load %shared_at
        : !ktdp.access_tile<1x4096x64x64xindex> -> tensor<1x4096x64x64xf16>
    %b0 = ktdp.load %shared_at
        : !ktdp.access_tile<1x4096x64x64xindex> -> tensor<1x4096x64x64xf16>

    %s0 = tensor.empty() : tensor<1x4096x64x64xf16>
    %c0v = linalg.add ins(%a0, %b0 : tensor<1x4096x64x64xf16>,
                                     tensor<1x4096x64x64xf16>)
             outs(%s0 : tensor<1x4096x64x64xf16>) -> tensor<1x4096x64x64xf16>

    %out0 = ktdp.construct_memory_view %base_out0,
        sizes: [1, 4096, 64, 64], strides: [16777216, 4096, 64, 1] {
        coordinate_set = #region,
        memory_space   = #ktdp.memory_space<ct_local>
    } : memref<1x4096x64x64xf16>
    %at0 = ktdp.construct_access_tile %out0[%c0, %c0, %c0, %c0] {
        access_tile_set = #region, access_tile_order = #ident4
    } : memref<1x4096x64x64xf16> -> !ktdp.access_tile<1x4096x64x64xindex>
    ktdp.store %c0v, %at0
        : tensor<1x4096x64x64xf16>, !ktdp.access_tile<1x4096x64x64xindex>

    // ---- iteration 1, reusing %shared ----
    %e1 = tensor.empty() : tensor<1x4096x32x64xf16>
    %p1 = linalg.fill ins(%v1 : f16) outs(%e1 : tensor<1x4096x32x64xf16>)
            -> tensor<1x4096x32x64xf16>

    %fut1 = ktdp.inter_tile_produce
        producer_tiles_per_group = #producers
        -> !ktdp.tile_future<(tensor<1x4096x32x64xf16>), groups = #groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %p1 : tensor<1x4096x32x64xf16>
    }

    %region_1 = ktdp.inter_tile_gather(%fut1)
        consumer_tiles_per_group = #consumers,
        gather_dimensions        = [2]
        : !ktdp.tile_future<(tensor<1x4096x32x64xf16>), groups = #groups>
          -> tensor<1x4096x64x64xf16>

    // THE REUSE.  This store overwrites iteration 0's landing, and it is a
    // write-after-read against `%a0` and `%b0` above.  Hoisting it over either of
    // them would give iteration 0's readers iteration 1's data.
    ktdp.store %region_1, %shared_at
        : tensor<1x4096x64x64xf16>, !ktdp.access_tile<1x4096x64x64xindex>

    %a1 = ktdp.load %shared_at
        : !ktdp.access_tile<1x4096x64x64xindex> -> tensor<1x4096x64x64xf16>
    %b1 = ktdp.load %shared_at
        : !ktdp.access_tile<1x4096x64x64xindex> -> tensor<1x4096x64x64xf16>

    %s1 = tensor.empty() : tensor<1x4096x64x64xf16>
    %c1v = linalg.add ins(%a1, %b1 : tensor<1x4096x64x64xf16>,
                                     tensor<1x4096x64x64xf16>)
             outs(%s1 : tensor<1x4096x64x64xf16>) -> tensor<1x4096x64x64xf16>

    %out1 = ktdp.construct_memory_view %base_out1,
        sizes: [1, 4096, 64, 64], strides: [16777216, 4096, 64, 1] {
        coordinate_set = #region,
        memory_space   = #ktdp.memory_space<ct_local>
    } : memref<1x4096x64x64xf16>
    %at1 = ktdp.construct_access_tile %out1[%c0, %c0, %c0, %c0] {
        access_tile_set = #region, access_tile_order = #ident4
    } : memref<1x4096x64x64xf16> -> !ktdp.access_tile<1x4096x64x64xindex>
    ktdp.store %c1v, %at1
        : tensor<1x4096x64x64xf16>, !ktdp.access_tile<1x4096x64x64xindex>

    // Expected.  Every producer contributes 1.0 in iteration 0, so the assembled
    // region is 1.0 everywhere and `%out0` holds 1.0 + 1.0 = 2.0.  Iteration 1
    // contributes 8.0, so `%out1` holds 16.0.  Both exact in fp16.
    //
    // The failures this fixture exists to catch:
    //
    //   `%out0` reading 16.0   iteration 1's store was hoisted over `%a0` or `%b0` —
    //                          the reuse happened before the reads completed, which
    //                          is exactly coverage §5's requirement.
    //   `%out1` reading  2.0   iteration 1's store was elided as dead, since a naive
    //                          pass may see `%shared` written twice and keep the
    //                          first.
    //   either reading  1.0    one of the two loads was folded away, so the add ran
    //                          against a single reader and the buffer's live range
    //                          was shorter than the source says.
    //
    // Coverage §1's "repeat after poison" applies directly: poison `%shared` between
    // the iterations and `%out1` must still be 16.0.
    return
  }
}
