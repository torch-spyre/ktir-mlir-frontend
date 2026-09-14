# Minimal inter-tile examples

Smallest KTIR forms for the cases
[torch-spyre#4300](https://github.com/torch-spyre/torch-spyre/pull/4300) asks
this PR to settle first. Its `lx_relayout_workload_coverage.md` §7 names KT-01/02
(shared sources and receiving-only cores), KT-05 (selection) and KT-07 (completed
sums) as the ones that "decide semantics rather than implementation polish".

Each file carries its fixture, its expected values, which rule it exercises, and
its verification status. Fixtures are read from the **pinned ownership catalog**
rather than from the inventory's summary rows, because those rows cannot
distinguish divisions that produce the same piece counts — `32 -> 8` with 4
inputs per region matches two different §9.3 rows of
[`inter-tile-communication.md`](../inter-tile-communication.md).

## Status

| File | Case | Parses | Legality | Blocked on |
|---|---|:---:|:---:|---|
| [`kt07a-raw-contributions.mlir`](kt07a-raw-contributions.mlir) | KT-07 (a) | **yes** | **yes** | — |
| [`kt07b-completed-sum.mlir`](kt07b-completed-sum.mlir) | KT-07 (b) | no | — | `ktdp.inter_tile_consume` |
| [`kt01-shared-sources.mlir`](kt01-shared-sources.mlir) | KT-01 | no | — | `ktdp.inter_tile_gather` |
| [`kt02-receiving-only-cores.mlir`](kt02-receiving-only-cores.mlir) | KT-02 | no | — | `ktdp.inter_tile_gather` |
| [`kt05-select-then-scatter.mlir`](kt05-select-then-scatter.mlir) | KT-05 | no | — | `ktdp.inter_tile_scatter` |
| [`kt03-logical-order.mlir`](kt03-logical-order.mlir) | KT-03 | no | — | `ktdp.inter_tile_gather` |
| [`kt06-owner-permutation.mlir`](kt06-owner-permutation.mlir) | KT-06 | variant A **yes** | variant A **yes** | variant B: `ktdp.inter_tile_consume` |
| [`kt04-multi-axis-assembly.mlir`](kt04-multi-axis-assembly.mlir) | KT-04 | no | — | `ktdp.inter_tile_gather` |
| [`kt09-reuse-and-lifetime.mlir`](kt09-reuse-and-lifetime.mlir) | KT-09 | no | — | `ktdp.inter_tile_gather` |

The four cases §7 asks to settle first are all here, plus KT-03, KT-04, KT-06 and
KT-09. `kt01`, `kt04`, `kt05`, `kt06` and `kt09` carry two functions each. In `kt01`,
`kt04` and `kt09` the second is one the measured data cannot supply, marked as such
in the file; in `kt06` both are measured and the pair *is* the requirement. `kt03` is
synthetic throughout.

Only `inter_tile_produce` and `inter_tile_reduce` exist today, so KT-07 (a) — an
all-reduce — is the only *delivery* that can be checked end to end. Its `produce`
half is shared with the others, and that half does parse; the files that fail do so
at their delivery op and nowhere else. `kt06`'s variant A also passes legality, for
the opposite reason: it is the branch §9.1 row 1 resolves to "no op needed", so
there is no delivery op in it to be missing.

## Operation sequences

Each file states its own sequence in a header block. Collected here, because the
structural differences are what separate the cases:

| File | Sequence |
|---|---|
| `kt01`, `kt02` | `LX-Load (a0) -> Produce -> Gather` |
| `kt03` | `LX-Load (a0) -> Produce -> Gather -> Slice -> LX-Store (a1, rows 32..63) -> Slice -> LX-Store (a1, rows 0..31)` |
| `kt04` | `LX-Load (a0) -> Produce -> Gather -> LX-Store (a1)` |
| `kt05` | `Produce[ LX-Load (a0) ] -> Scatter` |
| `kt06` A | `LX-Load (a0)` |
| `kt06` B | `LX-Load (a0) -> Produce -> Consume -> LX-Store (a0)` |
| `kt07a` | `LX-Load (a0) -> LocalReduce -> Expand -> Produce -> Reduce[ Add ]` |
| `kt07b` | `LX-Load (a0) -> LocalReduce -> Expand -> Produce -> Consume` |
| `kt09` 1 | `LX-Load (a0) -> Produce -> Gather -> Mul x 3 -> LX-Store x 3 (a1, a2, a3)` |
| `kt09` 2 | `Produce -> Gather -> LX-Store (a0) -> LX-Load x 2 (a0) -> Add -> LX-Store (a1)`, twice |

Six things are visible only in this view:

- **`kt05` is the only file whose load is *inside* the produce region.** Everywhere
  else every tile produces, so the load can sit outside; there only one tile per
  group produces and the group's eight consumers must not run it (§2.2).
- **`kt06` variant A is one op long, and the missing delivery is the claim** — §9.1
  row 1's "no op needed" branch, which is also why it passes legality.
- **`kt06` variant B loads and stores at the same address.** The permutation is
  entirely in the dependency attribute; no local address moves.
- **`kt03`'s two stores go to one view at swapped base rows.** That is where its
  reorder lives, and since a copy delivery must land every received tile in LX
  anyway, those two stores are the mandatory landings rather than extra work.
- **`kt09` half 1 is the only sequence with one delivery and several readers**, and
  the only one where a wrong version has an identical shape *and* identical values —
  three deliveries differ from one only in emitted transfer count.
- **`kt09` half 2 is the only sequence that lands the delivered value and reads it
  back.** Everywhere else the delivered value stays a `tensor`, so the landing store
  the hardware performs is not named in the IR and its live range is not either.

`Reduce[ Add ]` and `Produce[ … ]` denote a region. `LocalReduce` is ordinary
`linalg` over the tile's own rows, before any inter-tile op.

Execution and application verdicts are not attempted. §1 of the coverage document
requires all three separately; only the expression column is addressed here.

**Spelling.** These files use the *implemented* spelling:
`!ktdp.tile_future<(T_p), groups = affine_set<...>>` — partial types
parenthesised, `groups` a keyword — and `inter_tile_produce` with no operand
types before `->`. §7 of `inter-tile-communication.md` writes
`!ktdp.tile_future<T_p, #groups>` and `produce ... : T_p -> ...`, which matches
neither `KTDPTypes.td:222` nor `KTDP.td`'s `assemblyFormat`. Checked:

```
$ ktir-opt spec_form.mlir
spec_form.mlir:5:63: error: expected '->'
```

So §7 as printed is not parser input. Worth reconciling separately from the
semantics this directory is about.

## What the examples establish

**Receiving-only cores are the measured norm, not a corner case.** 64 of the
catalog's 130 records have destination owners that are not source owners, across
three route classes (`grouped_all_gather_with_replication`,
`replicate_or_owner_remap`, `all_gather`). R13 = `n` for the copy-only ops
(§10.1) is required by measurement, not only by argument. `kt02` is
GR-PF-002 (`relayouts[1]`), where each group has producers `{4g, 4g+2}` and
consumers `{4g..4g+3}` — and the producers' tile ids are **not adjacent**, which
is precisely why §3.3 defines `l` as a position rather than a tile id.

**The catalog contains no reduction at all.** All 130 records are `STCDPOpLx`
relayouts and every route class is copy-only: `all_gather` 26,
`grouped_all_gather_with_replication` 65, `replicate_or_owner_remap` 32,
`permutation` 6, `general_relayout` 1. So KT-07's two halves come from different
places: (b) is what every one of the 130 is, while (a) — a genuine cross-core
fold — comes from split-K matmul, the pattern `inter_tile_reduce` was implemented
for. The risk KT-07 guards against is therefore one-sided: the mistake to avoid
is reading a copy as a fold, and the measured consumer names (`mean_*`,
`_safe_softmax-Sum`, `mm-BMM_1`) invite exactly that.

**A fold op cannot stand in for a copy op.** Writing KT-07 (b)'s broadcast as a
degenerate `inter_tile_reduce` over one producer is rejected by the shipped
legality pass:

```
error: consumer_tiles_per_group for group 0 is not a subset of
       producer_tiles_per_group (a consumer tile that did not produce is
       unsupported; see open question Q1)
```

That is R13 (`KTIRCheckLegality.cpp:107-117`); R14's mode gate would reject it
too. `consume` is needed, and this is evidence rather than argument.

**The physical layout is not guessable from the inventory.** Its Source shape
column is alphabetically sorted, not `layoutDimOrder_` order. A real SDSC run of
the same logical shape (512 x 4096 x 1, fp16) reports

```
layoutDimOrder_ = ["mb", "out", "y"]   stickDimOrder_ = ["y"]   stickSize_ = [64]
device_size = [1, 4096, 512, 64]       device_coordinates = [0, c1, c0, 0]
```

so `y` is both the innermost logical axis and the stick axis, an extent-1 `y` is
carried as 64 padded lanes, the physical form is one rank higher than the logical
one, and the two data axes appear in reverse order. `kt02` uses this; its one
remaining assumption is that the record's order is `["mb", "in", "y"]`, `in`
taking the position `out` held.

**Overlapping-but-differing dependency sets do not occur in measurement.** 14 of
the 130 records have a source piece feeding several destination pieces, but in
**0 of 130** do two consumers declare different overlapping sets — a shared
source is always taken by consumers declaring the *same* set. So R5's scoping is
required by measurement while §3.3's "which set" clause is not yet forced by it.
`kt01` therefore has two functions: the measured form, which cannot tell §3.3's
two readings apart, and a synthetic one that can, since a shared producer lands
at different positions in two consumers' assemblies. The same function
discriminates the `P` derivation of §3.1, where the measured form again cannot:
its producer count and its `|dep|` are both 2.

**The §9.1 guard forces a select rather than forbidding the record.** `kt05` is
GR-PF-055, whose destination regions are 32 slivers of row 511 — a 512th of the
tensor, so the coverage clause rejects the pre-select pair. After the select the
guard is satisfied, which §9.1 states itself ("the value delivered ... after a
select that is the selected sub-tensor and not the original"), and the post-select
pair classifies as row 2, `inter_tile_scatter` with `scatter_dimensions = R`. The
op in the file is derived, not chosen. This record also needs no layout
assumption: its axes are literally `mb`, `out`, `y`, matching the measured
`layoutDimOrder_`.

**KT-03 is satisfied by two stores; what it costs is verifiability.** On the
gathered axis, the sources contributing to a destination piece are in ascending
tile-id order in **130 of 130** records, so §3.3's ordering rule is never
contradicted by measurement — `kt03` is synthetic for that reason. Where the order
does disagree, the reorder is expressible today because the assembled value has no
memory identity until it is stored: two stores at swapped base coordinates carry
it, with identity order inside each and no extra data movement. A single access
tile cannot, since it is a base plus a region relative to that base and offers no
way to permute positions *within* a dimension. Coverage §5 permits exactly this
form — "local selection/reordering is acceptable if fully expressed and stays
on-chip" — so **the requirement needs no new capability.** For a live intermediate
that is never stored, the reorder becomes an ordinary `tensor` permutation: still
local and on-chip, so still allowed, but no longer zero-copy.

What it does cost is verification. **No rule relates the store bases to §3.3's
assembly order.** A verifier sees two well-formed stores that between them cover
the region exactly once, which is all it is asked to see; storing the assembly
verbatim has the same shape, the same element count, the same coverage, and the
wrong answer. So the correctness of the reorder falls **outside inter-tile
verification, and the numerical check is its only guarantee.** An ordering
attribute on the delivery op redefining §3.3's `l` was considered and rejected on
that basis: an attribute can be checked for well-formedness but never for intent,
so it would not move the case into the verifier — it would only add a second
statement of the ordering that can disagree with the stores.

**One delivery can have many readers, and R2 does not stand in the way.** `kt09` is
the case where the catalog's own indexing invites the mistake. `mean-LayerNormNorm_out`
is the source tensor of **three** records — `relayouts[1, 8, 16]`, the Q/K/V
projections off one layer norm — and the three are **identical** in extents, pieces,
owner tables and route class, each recording 12 MiB of remote traffic. They are one
relayout that three consumers need. Since the catalog is indexed by (consumer, input),
a reader that walks records one at a time emits one transfer each and moves **36 MiB
where 12 suffices, exactly 3×**. The spelling that avoids it is the natural one: R2
constrains the `tile_future`, not the delivery's *result*, so one `produce` + one
delivery + N readers of a plain `tensor` is legal and is what the file shows. Eight of
the catalog's 120 distinct tensors are shared this way.

Note what separates right from wrong here: **nothing numerical.** Three deliveries
give the same values as one. Only the emitted transfer count distinguishes them, which
is coverage §1's "check the emitted memory accesses as well as numerical output".

**Lifetime splits into an expressible half and an inexpressible one.** The *source*
buffer's anti-dependence is carried by ordinary MLIR: iteration 1's `ktdp.store` into
the scratch cannot be hoisted above iteration 0's `ktdp.load` from it, because those
are conflicting memory effects on one `memref`. No inter-tile rule is needed. But the
*delivered value's* buffer lifetime cannot be expressed at all — the delivery returns
a `tensor`, so its readers touch no memref and the memref read ended at `ktdp.load`,
before the produce. **At this level a reader cannot extend a live range because there
is no live range to extend**, so coverage §1's "buffers preserved until last reader"
lands on the emitted program. That is the same "no address" fact `kt03` used to put its
reorder in store bases and that Table 1 records for dist-mem-view intermediates —
three cases, one root cause.

**Multi-axis assembly needs no new op, and its verdict splits.** `kt04` is
`inter_tile_gather` with two entries in `gather_dimensions` rather than one — §9.1
row 4, the same row as `kt01` and `kt02`. What multi-axis adds is that §4's
flattening becomes load-bearing, and two of the three errors it invites are closed
by rule. `gather_dimensions = [1, 0]`, a column-major assembly, is rejected by R9's
ascending-order requirement, which §4 says exists for exactly that reason: `[2, 0]`
and `[0, 2]` "would flatten to *different* data orders, so a reversed list passes
every other check while meaning something else". `gather_dimensions = [1]`, one axis
where two are needed, is rejected by R12, whose per-axis obligation is explicit that
"equal products alone would not give a well-defined multi-axis assembly". R12 also
confirms the ascending rule reaches `gather_dimensions` and not only the split ops.

The third error is closed by nothing, and the requirement's own toy is that fixture.
Read its ownership as (row, column) — the reading under which its stated wrong
answer `[0,2,1,3]` is what a gather actually produces — and `source1` holds cell
`(1,0)` while sitting at ascending position `l = 1`, which §4's odometer sends to
`(0,1)`. The assembly comes out `[[0,2],[1,3]]` where `[[0,1],[2,3]]` is required.
Every rule is satisfied: ascending list, both axes present, uniform extents,
`P = 4 = 2 × 2`, and the declared `tensor<2x2xf16>`. **Same shape, same element
count, transposed answer.** This is `kt03`'s finding in multi-axis form, and the fix
is again outside the delivery op — assign tile ids upstream to match the odometer,
store in pieces at swapped bases, or permute the assembled tensor locally.

Measurement does not force the problem. `relayouts[120]` (`cat_1-kvCacheScatter`,
16 pieces → 1, `P = 16`, concat `mb` ×8 and `out` ×2) **agrees with the odometer for
all 16 producers** — ascending owner position `l` maps to `(mb = l/2, out = 64·(l mod
2))`, checked off the `owners` fields. So the measured function needs nothing beyond
a plain two-axis gather, exactly as `kt03`'s 130 of 130 never contradict §3.3.

**An owner permutation is verifiable; a within-region reorder is not.** `kt06` is
the same family of problem as `kt03` — a required order disagreeing with the order
the IR supplies — with the opposite outcome, and the difference is only where the
ordering information is allowed to live. GR-PF-052 (`relayouts[51]`, plus four
siblings with identical geometry) has 32 pieces going to 32 pieces of **the same
size**, `P = 1`, a bijection. Reading the `owners` field of every piece: with
`J = j/2` and `M = mb/64`, the source owner is `8J + M` and the destination owner is
`J + 4M` — a transpose of a 4×8 grid, which is exactly what coverage §5 means by
"row-major versus column-major core order". Because the reorder is **across cores
rather than within a region**, it lives in `producer_dependency_per_consumer`, where
R3 through R8 all reach it: a dropped producer fails R4, a doubled one fails R5.
What the rules cannot catch is a bijection that is the *wrong* bijection — using
`π` where `π⁻¹` belongs — which is coverage §5's "wrong core map with identical
split counts", and only the values distinguish it. `kt03`, by contrast, has to put
its ordering in store bases where no rule reaches it at all.

**The two variants are the two branches of §9.1 row 1, not two spellings.** With
`C = ∅` and `R = ∅`, that row resolves to "`no op needed` if every core's region is
its own, else `inter_tile_consume` — a **relocation**". Coverage §5's "preserve or
explicitly change ownership" names the same pair, and which branch applies is a
property of the *consumer*, not of the edge. So `kt06`'s variant A contains no
inter-tile op — and that absence is the claim, which is also why it is the one file
besides `kt07a` that passes legality today. Its precondition is unexpressed
anywhere in the IR, which is the failure coverage §5 warns about: same ops, same
types, same addresses, wrong answer on 30 of 32 cores, because only owners 0 and 31
are fixed points of the transpose.

**Two traps in reading the catalog, found while deriving that map.** First,
`source_pieces` is ordered lexicographically by its `key` string (`p0, p1, p10,
p11, … p2, p20`), so **piece index is not owner order**, and `source_core_patterns`
is a separate summary carrying no piece key. The authoritative owner is the
`owners` field on each piece; deriving the map from index position instead yields a
bijection that is *not* a transpose — an artifact of the ordering. Second, the
catalog's byte fields are **logical**: `prod(extents) × word_length ==
logical_tensor_bytes` exactly, including for GR-PF-055 where an SDSC run showed a
`y` extent of 1 is physically 64 padded lanes. So padding is invisible in the
catalog by construction, and no record can answer a layout question on its own.
(`source_piece_bytes` is also the total over all pieces, not the size of one.)

**An aside on `access_tile_order`, independent of the above.** RFC 0682 defines it
twice over, as "the rightmost dimension in the output space corresponds to the
innermost iteration dimension" and as "the enumeration of points in the
intermediate variable space". The first reads as a dimension nesting order, the
second as a sort key; only the second could reorder within a dimension. `kt03`
deliberately does not depend on the generous reading, and `KTDP.td`'s op
description carries neither sentence. Worth settling in the dialect, but nothing
here waits on it.

## Not written yet

| Case | What it needs |
|---|---|
| KT-08 | Stick indivisibility. Measured, in four shapes: `y` 32→64 in `relayouts[39..50]` (each source holds half a stick), `y` 4→2 in `relayouts[57,64,73]` (a split inside one stick), `y` 1→4 in `[56,63,71]`, `y` 24→192 in `[117]`. **Needs an SDSC run first** — the catalog's byte fields are logical, so it cannot say whether `y` is sticked here or at what size (see the layout note in `kt06`). |
| KT-10 | Deliberate bad variants: remove a required piece, duplicate a piece inside one destination, corrupt an owner, supply a wrong output shape. Only `kt07a`'s and `kt06` variant A's negatives can actually be run today. |

The negative variants of KT-10 matter for every file here: coverage §5 requires
that reversing two fragments, omitting one, or reusing a wrong core map with
identical split counts all **fail**. None of that is written yet.

The production-shaped four-way V case (source `mb:32`, destination
`mb:8, qpk:4`) is also absent: it needs the saved physical descriptors, which are
not in this repository.
