# Inter-tile communications in KTIR

**Scope:** Seven ops — one production op, `ktdp.inter_tile_produce`, and
six delivery ops: `ktdp.inter_tile_consume`, `ktdp.inter_tile_reduce`,
`ktdp.inter_tile_reduce_scatter`, `ktdp.inter_tile_gather`,
`ktdp.inter_tile_all_to_all`, and `ktdp.inter_tile_scatter`. Together they
cover the six inter-core communication patterns: broadcast, all-reduce,
reduce-scatter, gather, all-to-all, and scatter.

**Organization.** The delivery ops share almost all of their machinery.
That machinery is stated once, in §3 (operand, consumer set, local index,
dependency attribute, combiner, synchronization, result semantics), §4
(type rules), and §5 (verification rules). §6 then defines each op by
what is *only* true of it.

**Rule numbering.** The verification rules are numbered R1–R14 and
collected in §5, which is their single point of definition. They are cited
as `(Rn)` at the place the attribute they constrain is introduced, so a
citation like `(R1)` in §2.1 means "§5 states this rule; here is the
attribute it applies to."

Sections are normative except §8 (implementation status) and §9 (measured
backend patterns). §9 doubles as the evidence for *which* ops a lowering
must actually emit: §9.3 reads the requirement off 51 measured relayouts.

---

## 1. Motivation and the three-property decomposition

Inter-tile communication involves three separate concerns:

1. **Production** — which tiles contribute data and what they contribute.
2. **Delivery** — how the contributed data is mapped onto the receiving
   tiles' results.
3. **Synchronization granularity** — whether each consumer tile waits
   for *all* producer tiles in its group to complete (full-barrier mode),
   or only for the specific producers whose data it requires (per-tile
   mode). Per-tile mode allows a consumer to begin as soon as its
   individual dependencies are satisfied, reducing stall time when
   producers finish at different times.

Separating production from delivery keeps each op single-purpose and
enables any combination: one production op plus a choice of delivery op.
The pairing is **one-to-one** — a production op is consumed by exactly one
delivery op (R2, §2.3). A pattern needing two deliveries therefore needs
two `ktdp.inter_tile_produce` ops. Allowing several deliveries per future
would let them share one production, but it makes R4 (coverage) and R5
(disjointness) non-local — they would have to union the dependency sets across
every use of the SSA value — so the restriction stands until a use case needs
it.

### 1.1 Semantics matrix

The six delivery ops differ in exactly three independent properties.
"Property" rather than "axis" throughout: in this document *axis* always
means a tensor or tile axis.

- **combine** — `none` | `fold` (combiner region + identity operand).
- **placement** — how producer contributions map onto consumer results:
  `replicate` | `concat` | `permute` | `split`.
- **cardinality** — producer tiles per group × consumer tiles per group.

**Semantics matrix.** One row per delivery op.

| Op | combine | placement | producers/grp | consumers/grp | dim attrs | region | identity |
|---|---|---|---|---|---|---|---|
| `consume` | none | replicate | 1 per consumer | free | — | — | — |
| `reduce` | fold | replicate | all | free | — | combiner | yes |
| `reduce_scatter` | fold | split | all | free | `scatter_dimensions` | combiner | yes |
| `gather` | none | concat | all | free | `gather_dimensions` | — | — |
| `all_to_all` | none | permute | all | all | `split_dimensions`, `concat_dimensions` | — | — |
| `scatter` | none | split | 1 per group | free | `scatter_dimensions` | — | — |

`all_to_all` is listed before `scatter` because it shares the
all-producers cardinality cell with `gather` and `reduce_scatter`, and
because its relationship to the two copy-only placements is structural:
**permute = split + concat in one step**, which is why it carries both dim
attributes and no new ones. `all_to_all` names them `split_dimensions` and
`concat_dimensions` — the same two roles `scatter_dimensions` and
`gather_dimensions` play on the single-role ops, renamed because on this
op both are present at once and *scatter* / *gather* would then name
neither the op nor a unique role.

**Every dim attribute is a list of axis indices** into `T_p` — an
`i64` array, not a single `i64` — flattened in list order per §4. §9.3
contains a measured pattern whose concat is three axes wide, so the
list-valued form is required by a named pattern rather than reserved for
a corner case.

Three things this matrix makes visible:

- **`placement` takes only four values.** The per-op type rules are four
  formulas (§4), not six.
- **The empty cells are principled.** `none` × `replicate` with all
  producers is undefined (which producer's value wins?), and `fold` ×
  `concat` / `fold` × `permute` is meaningless (fold what, then shuffle
  what?).
- **`all_to_all` is the fourth placement value, not a special case.**

### 1.2 Pattern coverage

The "all-" prefixed patterns are not separate ops: an op whose
`consumers/grp` cell is `free` already subsumes its all-tiles case by
widening `consumer_tiles_per_group`. The consumer set is therefore a
column here, since it is what distinguishes gather from all-gather and
all-to-all from scatter.

**Coverage table.** One row per named collective pattern.

| Pattern | Producers/grp | Consumers/grp | Delivery op | Result per consumer |
|---------|---------------|---------------|-------------|---------------------|
| Broadcast | 1 | free | `inter_tile_consume` | full copy |
| Reduce-to-one | all | 1 | `inter_tile_reduce` | fully reduced |
| All-reduce | all | all | `inter_tile_reduce` | fully reduced |
| Reduce-scatter | all | free | `inter_tile_reduce_scatter` | 1/C slice of reduced |
| Gather | all | 1 | `inter_tile_gather` | full assembled tensor |
| All-gather | all | all | `inter_tile_gather` | full assembled tensor |
| All-to-all | all | all | `inter_tile_all_to_all` | one slice from every producer |
| Scatter | 1 | free | `inter_tile_scatter` | 1/C slice of full |

`inter_tile_scatter` and `inter_tile_consume` have no natural "all-"
variant: R8 (§5) gives each consumer tile exactly one source, so there is no
all-producers case to widen to. `consume` still admits a group holding several
producers, but only as a **routing** pattern in which the dependency attribute
pairs each consumer tile with one of them (§6.1) — several point-to-point
deliveries sharing one `produce`, not an all-producers delivery.

### 1.3 The future value

`ktdp.inter_tile_produce` returns a `!ktdp.tile_future<T_p, #groups>` SSA
value. The group set `#groups` is carried as a parameter of the future
type rather than repeated as a separate `groups` attribute on both the
production and delivery ops. Each delivery op therefore infers the groups
from its operand type, and a group mismatch between production and
delivery is inexpressible — the def-use edge already requires the operand
type to equal the result type, so the type system rejects it structurally
rather than a verifier catching it after the fact.

The def-use edge from production to delivery encodes the happens-before
ordering with no explicit barriers in the IR. The synchronization
granularity — full-barrier or per-tile — is controlled by the
`producer_dependency_per_consumer` attribute on the delivery op (§3.4).
Corresponding production and delivery ops are expected to be adjacent in
a single basic block to avoid deadlocks.

---

## 2. `ktdp.inter_tile_produce` — unified production op

### 2.1 Attributes

**`producer_tiles_per_group`** — parameterized affine integer set `(i)[g]`
selecting which tiles produce per group. The set has one dimension (the tile
id) and one symbol (`g`, the group index). For example,
`affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>` selects tile ids
`4g .. 4g+3` for any group index `g`. An enumerated form (a list of tile-id
lists) is supported as a fallback when per-group membership is irregular.
Which cardinality each delivery op requires of this set is given by the
`producers/grp` column of §1.1 and enforced by R8 (§5).

**Disjointness invariant (R1).** For any two distinct group indices
`g_1 != g_2` in `groups`, `producer_tiles_per_group(g_1)` and
`producer_tiles_per_group(g_2)` must be disjoint. Every producing tile is
in exactly one group. The motivation is unambiguous group membership: each
tile contributes to exactly one group's production.

**`groups`** — affine integer set defining the range of valid group indices.
For example, `affine_set<(g) : (g >= 0, -g + 7 >= 0)>` defines 8 groups,
indexed `0..7`. It bounds the range of the `g` symbol used by
`producer_tiles_per_group`. This set is **not** a standalone attribute: it
is carried as the trailing parameter of the result
`!ktdp.tile_future<..., #groups>` type, and every delivery op infers it
from its operand type (§1.3).

### 2.2 Producer region

The producer region indicates **what partial results each tile contributes**.
It is the per-tile boundary that names the SSA values entering the cross-tile
communication. The block runs once per participating tile, in that tile's
SPMD execution.

**Block argument:** `%gid: index` — the index of the group this tile
belongs to. The runtime binding is direct: tile `t` finds its group by
looking up which entry of `producer_tiles_per_group(g)` contains `t`;
that `g` is bound to `%gid` for tile `t`'s execution of the block.

The body knows its tile id via `ktdp.get_compute_tile_id` (the same way
every SPMD KTIR body does) and its group index via `%gid`.

**Termination:** the block terminates with
`ktdp.yield_partial %val_1, ..., %val_N : T_p_1, ..., T_p_N`, yielding
one value per partial-tensor role. The yielded values may reference SSA
values from the enclosing scope — typical use is a thin contribution
marker:

```mlir
{
  ^bb0(%gid: index):
    ktdp.yield_partial %my_partial_1, ..., %my_partial_N
                       : T_p_1, ..., T_p_N
}
```

with the per-tile compute that produced `%my_partial` living at function
scope (where it is naturally executed by every tile under SPMD). Richer
bodies are allowed when the contribution is awkward to hoist — e.g.,
compute that depends only on `%gid`, or visibly local "contribution
preparation" the author wants to keep adjacent to the op. A
single-producer-per-group op (`consume`, `scatter`) is the case where a
richer body is normally *required*: the loads that feed the partial must
not run on the group's non-producing tiles, so they belong inside the
region rather than at function scope (§7.7.1).

### 2.3 Op signature

```mlir
%future = ktdp.inter_tile_produce
    producer_tiles_per_group = <affine-set>
    : T_p_1, ..., T_p_N -> !ktdp.tile_future<T_p_1, ..., T_p_N, #groups>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %val_1, ..., %val_N : T_p_1, ..., T_p_N
}
```

`%future` is a workgroup-visible handle carrying per-tile availability
signals. Each producer tile's contribution becomes independently
observable the moment that tile executes `ktdp.yield_partial`.

**Single-use invariant (R2).** `%future` must have exactly one use — the
single delivery op that consumes it. If two delivery ops need to
communicate with the same set of producers, they must each have their own
`ktdp.inter_tile_produce` (see §1).

---

## 3. Shared delivery semantics

Everything in this section holds for **every** delivery op. §6 states
only per-op deltas; where §6 is silent, this section governs.

### 3.1 Notation

**Symbol table.** These names are used unqualified throughout.

| Symbol | Meaning |
|---|---|
| `T_p_i` | the partial type of role `i`, as yielded by `ktdp.yield_partial` |
| `N` | number of partial-tensor roles (variadic arity), `N >= 1` |
| `P` | number of producer tiles a given consumer assembles from / waits on |
| `C` | number of consumer tiles per group, `\|consumer_tiles_per_group(g)\|` |
| `l` | within-group local index (§3.3) |

`P` is `|producer_tiles_per_group(g)|` when
`producer_dependency_per_consumer` is absent, and the (common, by R6)
cardinality of the per-consumer producer set when it is present.

### 3.2 Operand and consumer set

**Operand:** `!ktdp.tile_future<T_p_1, ..., T_p_N, #groups>` — the future
returned by the corresponding `ktdp.inter_tile_produce`. The def-use edge
is the ordering constraint, and the `#groups` parameter of this type
supplies the group set. There is no separate `groups` attribute; a group
mismatch with production is inexpressible (§1.3).

**`consumer_tiles_per_group`** — affine integer set, of the same
`(i)[g]` form as `producer_tiles_per_group`, selecting the tiles that
receive a result per group. Its permitted cardinality per op is the
`consumers/grp` column of §1.1.

The operations that use a delivery op's result are performed only by the
tiles in `consumer_tiles_per_group`. This ownership constraint is carried
by the def-use chain from the result: any use of the result is reachable
only by consumer tiles. No block is needed on any delivery op for
post-delivery computation — that is ordinary function-scope SPMD code
consuming the SSA value.

### 3.3 Within-group local index — normative

**`l` is a tile's position, counting from 0 in ascending tile-id order,
among the relevant set within its group** — the producer set for `concat` placement (and for
`permute`'s `concat_dimensions`), the consumer set for `split` placement
(and for `permute`'s `split_dimensions`).

This definition is what makes ordered placement well-defined. Without it,
concatenation and split orders are pinned down only by contiguous-tile-id
coincidence and break silently under non-monotone tile assignments. Every
"ascending local-index order" in this document means exactly this
position — never a tile id, and never an offset in the textual order of an
enumerated set. ("Position" rather than "rank": in this document *rank*
always means a tensor's number of dimensions.)

### 3.4 `producer_dependency_per_consumer` *(optional)*

Affine integer set `(p)[c, g]` over producer tile IDs `p`, parameterized
by consumer tile `c` and group index `g`. For consumer tile `c` in group
`g`, only the producer tiles satisfying this set are waited on and
received. If absent, the consumer waits on and receives from **all**
producer tiles in the group (full-barrier semantics).

The attribute has two distinct effects, depending on placement:

- For `replicate` placement it selects *which* producer a consumer reads
  and *when* it unblocks — a synchronization refinement only.
- For `concat` and `permute` placements it additionally narrows the set
  of contributions assembled, yielding a partial (segmented) gather over
  the declared subset; `P` and hence the result type follow from it.
- For `fold` placement it makes the result a partial reduction over the
  declared subset: contributions from the remaining producers are treated
  as the identity.

`scatter` is the one op that does not accept the attribute (§6.6).

Its verification obligations are R3–R7 (§5); they are stated there and
not restated here.

Not every symbol needs to appear in a given instantiation:

- **`g` may be omitted** when the mapping is the same relative rule for
  every group (group-independent mapping). Example: a fixed per-tile
  pairing, `(p)[c] : (p - c + 2 == 0)`. Note: because groups are
  disjoint and integer division is not affine, `g` cannot be derived
  from `c` alone; it must appear explicitly whenever the constraint
  involves group-relative addresses such as `4*g`.
- **Both `c` and `g` are needed** when the mapping varies by both
  consumer identity and group. Example: a butterfly mirror exchange,
  `(p)[c, g] : (p + c - 8*g - 3 == 0)`, where the sum `p + c` differs
  for each group.

### 3.5 Combiner region and `identity` — `fold` placement only

The two `fold` ops (`reduce`, `reduce_scatter`) carry a combiner region
and an `identity` operand list. The four copy-only ops carry neither:
they place contributions by position, so there is nothing to fold and no
identity element to supply.

**Region.** A single block receiving `2N` arguments —
`%lhs_1, ..., %lhs_N, %rhs_1, ..., %rhs_N` with each `%lhs_i` and
`%rhs_i` of type `T_p_i` — terminated by
`ktdp.yield_reduced %val_1, ..., %val_N : T_p_1, ..., T_p_N`.

**Purity (R10).** The combiner must be pure — no memory effects, no calls
to side-effecting ops. Pure tensor ops (`tensor.empty`, `linalg` on
tensors, `arith.*`) are allowed.

**Combine ordering.** The associative-commutative contract is by user
agreement; the scheduler is free to combine in tree, ring, linear, or any
hardware-native topology. Different groups' reductions are independent
and may be scheduled in parallel. This freedom is what distinguishes
`fold` from the copy-only placements, whose ordered placement by `l`
(§3.3) is deterministic and requires no commutativity.

**`identity` (R11).** `N` variadic SSA operands, one per role. Each
identity tensor's shape and element type must match the corresponding
partial type `T_p_i` — *not* the result type. The identities are hoisted
before the op and shared across all groups and all tiles. Combining any
identity with its corresponding partial yields that partial.

### 3.6 Synchronization model

No explicit barriers appear in the IR. The
`!ktdp.tile_future<T_p, #groups>` SSA value carries **per-tile
availability signals** rather than a monolithic group barrier:

1. Each producer tile's contribution becomes independently observable as
   soon as that tile executes `ktdp.yield_partial` in the production
   block.
2. A delivery op cannot use a producer tile's contribution until that
   tile's signal is set in `%future`.
3. The producer tiles a given consumer tile waits for are declared by
   `producer_dependency_per_consumer` (§3.4):

   - **Absent (default) — full-barrier mode:** consumer tile `c` in group
     `g` waits for every producer tile in `producer_tiles_per_group(g)`
     before the delivery op executes. This maps directly to a hardware
     group barrier and preserves the simplest safety guarantee.
   - **Present — per-tile mode:** consumer tile `c` waits only for the
     producer tiles `p` satisfying
     `producer_dependency_per_consumer(p)[c, g]`. The consumer unblocks
     as soon as those specific tiles have completed, without waiting for
     unrelated producers. Different consumer tiles may declare different
     dependency sets, enabling fine-grained producer–consumer pipelining.

A multi-producer wait is therefore a **per-consumer AND-join over
existing per-tile signals**, not a new primitive. This is why the
all-producers ops (`reduce`, `reduce_scatter`, `gather`, `all_to_all`)
introduce no synchronization machinery beyond what a single-producer op
already needs: they differ only in how many signals the join covers.

In SPMD KTIR, a tile cannot observe other tiles' partials except through
a dialect-defined boundary. The `ktdp.inter_tile_produce` block is that
boundary — it names the per-tile contribution and exposes it via
`%future`. The delivery op's result tensor is an SSA value that cannot
materialize until the declared dependencies are satisfied; standard MLIR
dataflow ordering applies.

Lowering inserts target-specific hardware synchronization: a group
barrier for full-barrier mode, and point-to-point ready/wait signals for
per-tile mode.

### 3.7 Result semantics

Every delivery op produces `N` variadic SSA values, one per
partial-tensor role. The values are **per-tile-valued**: each consumer
tile holds its own result value when the op completes. Whether tiles in
the same group hold the *same* value is a property of the placement.

**Sharing table.** One row per placement value.

| placement | tiles in one group hold | tiles in different groups hold |
|---|---|---|
| `replicate` | the same value | their own group's value |
| `concat` | the same assembled tensor | their own group's assembly |
| `split` | disjoint ordered slices that tile the whole | slices of their own group's tensor |
| `permute` | different assemblies (one slice per producer) | their own group's exchange |

**Non-participating tiles.** Results are undefined for tiles not in
`consumer_tiles_per_group`.

**Multi-tensor (variadic) delivery.** `N >= 1` roles are supported by
every op, and all roles share the same attributes (`scatter_dimensions`,
`gather_dimensions`, `P`, `C`) — only the types differ. Argmax-style reductions,
where each contribution is a correlated tuple of tensors (values,
indices), use `N = 2`: two identities, two yielded partials, four
combiner arguments yielding two combined values, two op results. Each
role's result type follows the §4 rule independently.

---

## 4. Placement algebra and type rules

Result types are a function of the placement value alone. There are four
formulas, applied per role `i` to `T_p_i`.

**Type-rule table.** One row per placement value.

| placement | result type derived from `T_p` |
|---|---|
| `replicate` | `T_p` unchanged — no rank reduction |
| `concat` | extents along `gather_dimensions` multiplied by `P` in total |
| `split` | extents along `scatter_dimensions` divided by `C` in total |
| `permute` | extents along `scatter_dimensions` divided by `C` in total, **and** extents along `gather_dimensions` multiplied by `P` in total |

`reduce_scatter` is `fold` + `split`: the `split` formula applies directly
to `T_p`, with no collapse first.

**No rank reduction anywhere.** All four formulas keep `T_p`'s rank. The
same reasoning that settled it for `reduce` (§6.2) applies to `concat` and
`split`: the axis the op concatenates along or splits is an axis `T_p`
already has, so there is nothing to collapse and no rank to restore. A
`concat` result differs from `T_p` only in the extent along the listed axes,
a `split` result likewise — never in rank. This keeps every result type a
per-axis extent rewrite of the partial, which is what lets §10.3 reason
about layout transparency one axis at a time.

**Axis sets and flattening — normative.** Each dim attribute is a *list*
of axis indices into `T_p`, not a single axis. A list of length `n > 1`
denotes the product space of those axes, linearized as a row-major
(mixed-radix odometer) order over the listed extents: **the first entry is
the slowest-varying and the last is the fastest-varying**. Write
`E(D) = prod(T_p[d] for d in D)` for the flattened extent of axis set `D`.
The single-axis case is `n == 1`, where `E(D) = T_p[d]` and every formula
below reduces to its familiar form; `n == 0` is invalid for an op that
carries the attribute.

**The list is in ascending numerical order (R9).** Entries must ascend, so
the slowest-to-fastest flattening above coincides with ascending axis index
and the attribute has exactly one legal spelling per axis set. Two reasons
this is a rule and not a convention. It removes a silent-miscompile class:
`[2, 0]` and `[0, 2]` are both "valid, distinct, non-empty" and would flatten
to *different* data orders, so a reversed list passes every other check while
meaning something else. And it makes attribute equality a list comparison —
which §4's conservation case below depends on, since `all_to_all` decides
whether `T_c == T_p` by testing `split_dimensions == concat_dimensions`.

Entries need not be **adjacent**: `[0, 2]` over a rank-3 partial is legal and
is exactly what physicalization produces (§10.3).

**Split and concat apply to the floordiv axis — normative.** When a listed
axis is a **sticked** axis — one that a stick layout has split into a
`floordiv` (chunk-count) axis and a `mod` (within-stick) axis — the `÷ C` or
`× P` applies to the **floordiv axis only**. The `mod` axis is invariant: its
extent is the stick size, and changing it would redefine what a stick is.

This settles what "`E(D)` divided by `C`" alone leaves open, since a flattened
extent does not say which listed axis absorbs the factor. For a partial
`[2, 16, 32]` (logical `[16, 64]`, stick 32) with `gather_dimensions = [0, 2]`
and `P = 4`, the result is `[8, 16, 32]` — the chunk count goes `2 → 8` and
the stick axis stays `32`, which is exactly the physicalization of the logical
result `[16, 256]`. Absorbing into the `mod` axis instead would give
`[2, 16, 128]`: the same flattened extent, the wrong tensor.

A useful consequence: **R9 applied to the floordiv axis is the stick-multiple
check.** `E(floordiv) % C == 0` holds exactly when the logical result extent
is a whole multiple of the stick, so a split that would drive the result
sub-stick fails R9 rather than needing a rule of its own. On the partial
above, `C = 2` gives `2 % 2 == 0` and a result of `[1, 16, 32]`; `C = 4` gives
`2 % 4 ≠ 0` and is rejected — correctly, since the logical result `[16, 16]`
is half a stick and unrepresentable in that layout.

Fixing this order is a requirement, not a convenience: §9.3 contains a
measured three-axis concat, so the flattening must be well-defined over
more than two axes for a *named* pattern rather than only a corner case.

**Which slice a tile gets.** For `split`, the consumer with local index
`l` (§3.3) receives `[l*chunk : (l+1)*chunk]` of the flattened
`scatter_dimensions` space, where `chunk = E(scatter_dimensions) / C`.
For `concat`, the producer with local index `l` occupies
`[l*chunk : (l+1)*chunk]` of the flattened `gather_dimensions` space,
where `chunk = E(gather_dimensions)` is that producer's own flattened
extent over those axes. For `permute`, both hold simultaneously: consumer
`l_c` receives, from each producer `l_p`, that producer's
`scatter_dimensions` slice `l_c`, placed at `gather_dimensions` position
`l_p`.

A multi-axis split or concat therefore needs no special case in the
decision procedure of §9.1: the dimension attributes are the axis *sets*
themselves, in this order.

**Conservation in the square case.** Whenever `P == C`, the `permute`
result has the same element count as `T_p` — one axis is divided and
another multiplied by the same factor — so a square all-to-all is a pure
redistribution of ownership. If additionally `split_dimensions == concat_dimensions`,
the result *type* equals `T_p`: the distributed transpose. That equal-type
case is not what the SDSC backend emits today — every measured all-to-all
splits and concats *different* axes (§9.3), so `P == C` conserves the
element count while the type still changes.

**Why `split` divides an honest data axis.** Every splitting op divides an
extent of an axis the partial already has, so the types stay honest:
`<128x1x64>` → `<32x1x64>`, never `<1x...>`. No op manufactures a unit
dimension for a collapse to consume, and none removes one.

---

## 5. Verification rules

Principle: **each rule has exactly one owner and one statement;
applicability is a column, not a restatement.** "Owner" is the op that
carries the attribute the rule constrains.

**Verification matrix.** One row per rule, one column per delivery op.

| Rule | Owner | consume | reduce | red_scat | gather | all_to_all | scatter |
|---|---|---|---|---|---|---|---|
| R1 group disjointness (§2.1) | produce | y | y | y | y | y | y |
| R2 single-use future (§2.3) | produce | y | y | y | y | y | y |
| R3 dep set subset of producers | delivery | y | y | y | y | y | n/a |
| R4 every producer covered by some consumer | delivery | y | y | y | y | y | n/a |
| R5 dep sets pairwise disjoint | delivery | — | — | — | y | y | n/a |
| R6 uniform dep-set cardinality | delivery | — | — | — | y | y | n/a |
| R7 uniform producer cardinality across groups | delivery | — | — | — | y | y | n/a |
| R8 single-source delivery | delivery | y | — | — | — | — | y |
| R9 flattened split extent divisible by `C` | delivery | — | — | y | — | y | y |
| R10 combiner purity (§3.5) | delivery | — | y | y | — | — | — |
| R11 identity shape matches `T_p` (§3.5) | delivery | — | y | y | — | — | — |
| R12 flattened concat extent × `P` well-defined | delivery | — | — | — | y | y | — |
| R13 consumer set subset of producer set | delivery | — | y | ? | ? | ? | n |
| R14 reduce mode gate: `C == P` or `\|C\| == 1` | delivery | — | y | ? | — | — | — |

Statements:

- **R3 — subset.** The declared dependency set must be a subset of
  `producer_tiles_per_group`; referencing a non-producer tile is an
  error.

  ```text
  { p | ∃ c, g : producer_dependency_per_consumer(p)[c, g] }
    ⊆
  { p | ∃ g : p ∈ producer_tiles_per_group(g) }
  ```

- **R4 — coverage.** For every group `g` and every producer `p` in
  `producer_tiles_per_group(g)`, at least one consumer `c` in
  `consumer_tiles_per_group(g)` must satisfy
  `producer_dependency_per_consumer(p)[c, g]`. An uncovered producer
  yields a value no consumer reads, risking deadlock in push-based
  lowerings.

  ```text
  ∀ g, ∀ p ∈ producer_tiles_per_group(g) :
      ∃ c ∈ consumer_tiles_per_group(g) :
          producer_dependency_per_consumer(p)[c, g]
  ```

- **R5 — pairwise disjointness.** For the assembling placements
  (`concat`, `permute`), distinct consumers' declared dependency sets
  must be disjoint. R4 alone requires only that each producer be claimed
  by *at least one* consumer, which combined with R6 admits declared sets
  that double-count producers — and a double-counted producer has no
  well-defined position in the assembly.
- **R6 — uniform dep-set cardinality.** All consumers in a group must
  declare the same number of producers, so `P` is a single number and the
  op has one static result type.
- **R7 — uniform producer cardinality across groups.**
  `producer_tiles_per_group` is a parameterized affine set over `g` and
  nothing otherwise requires equal cardinality per group. Since the op
  result is a single static tensor type, unequal groups yield no
  expressible result type for the assembling placements.
- **R8 — single-source delivery.** For the ops whose `producers/grp` cell is
  `1` — `inter_tile_consume` and `inter_tile_scatter` — every consumer tile
  must receive from exactly one producer tile:

  ```text
  ∀ g, ∀ c ∈ consumer_tiles_per_group(g) : |dep(c, g)| == 1
  ```

  where `dep(c, g)` is the producer set `producer_dependency_per_consumer`
  declares for consumer tile `c` (§3.4). `inter_tile_scatter` takes no such
  attribute (§6.6), so for it the rule reduces to its simplest form: exactly
  one producer tile per group.

  **Why per consumer tile rather than per group.** These two ops deliver into a
  result no larger than one contribution — unchanged for `replicate`, a `1/C`
  slice for `split` — with no combiner and nowhere to put a second value. A
  consumer tile holding two contributions is therefore the undefined cell of
  §1.1, "which producer's value wins?". What the op needs is not that the
  *group* hold one producer, but that each *consumer tile* have a single
  source. The two coincide when `|P(g)| == 1`, the common case (broadcast,
  §7.1), which needs no attribute at all.

  **For `inter_tile_consume` with `|P(g)| > 1` the attribute is required.**
  There is no meaningful default, because receiving from every producer is
  exactly that undefined cell. With the attribute, such a group is a
  **routing** pattern — several independent point-to-point deliveries sharing
  one `produce` op, as in §7.4.1 and §7.4.2 — and no consumer tile ever sees
  two values. A group with `|P(g)| > 1` and no attribute is rejected.

  A producer **may** serve several consumer tiles (multicast within the
  group); it may not serve none, which R4 already requires. So `dep` need not
  be injective — only single-valued per consumer tile, and total over
  producers.

- **R9 — split divisibility.** `E(D_split) % C == 0`, where `D_split` is
  the op's split axis set (`scatter_dimensions`, or `split_dimensions` for
  `all_to_all`) and `E` is the flattened extent of §4. Stating the rule on the
  *flattened* extent is what lets one rule cover all three splitting ops
  and every arity: a multi-axis split need only divide in the product,
  not axis by axis.

  Every axis index in the list must be a valid, distinct axis of `T_p`; the
  list must be non-empty; and the entries must be in **ascending numerical
  order** (§4). Repeated indices would double-count an extent in `E`, and an
  out-of-order list would silently denote a different flattening.
- **R11 and the shipped constraint.** R11 pins `identity` to `T_p`, while
  the implemented `reduce` ties it to *results* (`KTDP.td:172-174`). With no
  rank reduction (§4) these coincide for `reduce`, since its result *is*
  `T_p`. They diverge for `reduce_scatter`, whose result is `T_p` split by
  `C`: R11's `T_p` is the correct one there, since the identity is combined
  with partials before the split. A verifier generalizing the shipped
  constraint to `reduce_scatter` must therefore retarget it from results to
  the future's partial types (§10.3).

- **R12 — concat well-definedness.** The result flattened extent over the
  concat axis set `D_concat` (`gather_dimensions`, or `concat_dimensions`
  for `all_to_all`) is `P × E(D_concat)`, which requires every assembled
  producer to contribute the same extent along *each* listed axis — equal
  products alone would not give a well-defined multi-axis assembly, since
  the flattening of §4 depends on the individual extents. The same
  validity conditions as R9 apply to the list. For the square
  `all_to_all` case the divisibility follows from R7 + R9, but it must be
  stated independently for the non-square case, which §9.3 shows is
  measured and not hypothetical.
- **R13 — consumer set subset of producer set.** Every consumer tile in a
  group must also be a producer in that group, i.e.
  `consumer_tiles_per_group(g) ⊆ producer_tiles_per_group(g)`. Whether
  this should hold is §10.1; it is currently enforced for `reduce` only.
- **R14 — reduce mode gate.** For `reduce`, the consumer set must either
  equal the producer set (all-reduce) or be a single tile
  (reduce-to-one); a strict multi-tile subset — reduce-to-subset — is
  rejected. This is a present implementation restriction, not a design
  conclusion (§10.1).

---

## 6. The delivery ops

Each subsection states only what is specific to that op: its cells from
§1.1, its result type from §4, its signature, and any op-specific
argument. Shared machinery is §3; rules are §5.

### 6.1 `ktdp.inter_tile_consume` — broadcast

`combine = none`, `placement = replicate`, one producer per consumer tile
(R8), consumer set free, no dim attribute, no region, no identity.

**Result type.** `T_p_i` unchanged (§4, `replicate` + `none`).

**Semantics.** No combining occurs. The value produced by the group's
producer tile is delivered unchanged to every consumer tile in that
group — broadcast.

```mlir
%result_1, ..., %result_N = ktdp.inter_tile_consume(%future)
    consumer_tiles_per_group         = <affine-set>,
    producer_dependency_per_consumer = <affine-set>   // optional; default: all producers
    : !ktdp.tile_future<T_p_1, ..., T_p_N, #groups> -> T_p_1, ..., T_p_N
```

With one producer per group the attribute is a pure synchronization
refinement (§3.4): there is only one value to receive, so it changes when
each consumer unblocks and nothing else. With **several** producers per
group it also names the sender, and R8 then requires it — each consumer
tile must be paired with exactly one producer. Such a group is a
**routing** pattern rather than a broadcast: several independent
point-to-point deliveries sharing one `produce` op, which is what lets
`consume` express per-tile pairing (§7.4.1) and one-to-one permutation
exchange (§7.4.2). Delivering to a consumer tile from more than one
producer is never legal here — with no combiner and a result the size of
one contribution, there would be nowhere to put the second value (§1.1).

### 6.2 `ktdp.inter_tile_reduce` — reduction

`combine = fold`, `placement = replicate`, all tiles produce, consumer
set free, no dim attribute, combiner region and `identity` per §3.5.

**Result type.** `T_r_i == T_p_i` — no rank reduction. An earlier draft
collapsed the within-group tile axes; the implementation deliberately does
not (`KTDP.td:197`), because the partial already carries that axis and
keeping it makes the op simpler: result, partial and `identity` are then one
type, tied declaratively (`KTDP.td:168-174`) rather than by a shape
computation. This is also what makes `reduce` transparent under
physicalization (§10.3).

```mlir
%r_1, ..., %r_N = ktdp.inter_tile_reduce(%future)
    consumer_tiles_per_group         = <affine-set>,
    producer_dependency_per_consumer = <affine-set>,   // optional; default: all producers
    identity(%id_1 : T_p_1, ..., %id_N : T_p_N)
    : !ktdp.tile_future<T_p_1, ..., T_p_N, #groups> -> T_r_1, ..., T_r_N
{
  ^bb0(%lhs_1: T_p_1, ..., %lhs_N: T_p_N,
       %rhs_1: T_p_1, ..., %rhs_N: T_p_N):
    ktdp.yield_reduced %val_1, ..., %val_N : T_p_1, ..., T_p_N
}
```

Consumer set = producer set is all-reduce; a single consumer per group is
reduce-to-one. Both are supported today; a strict multi-tile subset is
not (R14).

### 6.3 `ktdp.inter_tile_reduce_scatter` — reduction then split

`combine = fold`, `placement = split`, all tiles produce, consumer set
free, `scatter_dimensions`, combiner region and `identity` per §3.5.

**`scatter_dimensions`** (`i64` array) — axes of `T_p` along which the
reduced result is split row-major across the consumer tiles (R9).

**Result type.** `T_r_i` is `T_p_i` with the flattened extent over
`scatter_dimensions` divided by `C` — no rank reduction (§4). The same axes
and the same split apply to all roles.

```mlir
%chunk_1, ..., %chunk_N = ktdp.inter_tile_reduce_scatter(%future)
    consumer_tiles_per_group         = <affine-set>,
    scatter_dimensions               = <i64-array>,
    producer_dependency_per_consumer = <affine-set>,   // optional; default: all producers
    identity(%id_1 : T_p_1, ..., %id_N : T_p_N)
    : !ktdp.tile_future<T_p_1, ..., T_p_N, #groups> -> T_r_1, ..., T_r_N
{
  ^bb0(%lhs_1: T_p_1, ..., %lhs_N: T_p_N,
       %rhs_1: T_p_1, ..., %rhs_N: T_p_N):
    ktdp.yield_reduced %val_1, ..., %val_N : T_p_1, ..., T_p_N
}
```

### 6.4 `ktdp.inter_tile_gather` — ordered assembly

`combine = none`, `placement = concat`, all tiles produce, consumer set
free, `gather_dimensions`, no region, no identity.

**`gather_dimensions`** (`i64` array) — axes of `T_p` along which the
producers' partials are concatenated, in ascending producer local-index
order (§3.3). A multi-axis set concatenates in the flattened space of §4,
listed axes ordered slowest- to fastest-varying; §9.3's all-gather
patterns supply a measured three-axis case.

**Result type.** `T_g_i` is `T_p_i` with the flattened extent over
`gather_dimensions` multiplied by `P` (R12).

```mlir
%gathered_1, ..., %gathered_N = ktdp.inter_tile_gather(%future)
    consumer_tiles_per_group         = <affine-set>,
    gather_dimensions                = <i64-array>,
    producer_dependency_per_consumer = <affine-set>   // optional; default: all producers
    : !ktdp.tile_future<T_p_1, ..., T_p_N, #groups> -> T_g_1, ..., T_g_N
```

One consumer per group is a plain gather; the full group as consumer set
is all-gather — the same op with a wider set (§1.2), not a separate op.
With `producer_dependency_per_consumer` present the assembly is a partial
(segmented) gather over each consumer's declared subset, subject to
R5–R7.

### 6.5 `ktdp.inter_tile_all_to_all` — split and reassemble

`combine = none`, `placement = permute`, all tiles produce, all tiles
consume, both `split_dimensions` and `concat_dimensions`, no region, no
identity.

**Attributes.** `split_dimensions` (`i64` array) — axes each producer
splits into `C` chunks (R9). `concat_dimensions` (`i64` array) — axes
along which each consumer concatenates the chunks it received, in
ascending producer local-index order (R12). Both are flattened in list
order per §4. The two sets may be equal (pure ownership transpose along
one axis set) or different (reshape-transpose, e.g. split heads and
regather sequence); §9.3 shows both readings occur in measurement.

**Result type.** `T_c_i` is `T_p_i` with the flattened `split_dimensions`
extent divided by `C` and the flattened `concat_dimensions` extent
multiplied by `P`. In the square case (`P == C` and
`split_dimensions == concat_dimensions`) `T_c_i == T_p_i` (§4).

**Semantics.** Consumer with local index `l_c` receives, from each
producer `l_p`, the slice `[l_c*chunk : (l_c+1)*chunk]` of that
producer's tensor in the flattened `split_dimensions` space
(`chunk = E(split_dimensions) / C`), and concatenates those `P` slices in
the flattened `concat_dimensions` space in ascending `l_p` order.

```mlir
%out_1, ..., %out_N = ktdp.inter_tile_all_to_all(%future)
    consumer_tiles_per_group         = <affine-set>,
    split_dimensions                 = <i64-array>,
    concat_dimensions                = <i64-array>,
    producer_dependency_per_consumer = <affine-set>   // optional; default: all producers
    : !ktdp.tile_future<T_p_1, ..., T_p_N, #groups> -> T_c_1, ..., T_c_N
```

**Why it is a first-class op rather than a composition.** All-to-all
requires every tile to be simultaneously a producer of `C` distinct
slices and a consumer of `P` distinct slices:

```text
tile 0: A[0][0..3]     tile 0: A[0][0] A[1][0] A[2][0] A[3][0]
tile 1: A[1][0..3]     tile 1: A[0][1] A[1][1] A[2][1] A[3][1]
tile 2: A[2][0..3] --> tile 2: A[0][2] A[1][2] A[2][2] A[3][2]
tile 3: A[3][0..3]     tile 3: A[0][3] A[1][3] A[2][3] A[3][3]
```

Neither existing copy-only op admits this. `gather` delivers the *same*
assembled tensor to every consumer (§3.7) and cannot give consumers
different content. `scatter` permits exactly one producer per group (R8)
and cannot have every tile contribute. Composing them materializes the
full concatenation on every tile — wrong data volume and wrong
communication pattern.

The only faithful composition is `C` separate `produce`+`scatter` pairs
(one per source tile, forced by R2) followed by a per-consumer `concat`
in ordinary SPMD code: `C×` the ops, `C×` the produce handles, and
reassembly pushed out of the inter-tile layer. That composition is the
useful *reference lowering* — it is why `all_to_all` needs no new
synchronization (§3.6) and no new verification beyond `scatter` ∪
`gather` (R9 and R5–R7/R12 respectively) — but it is the wrong surface
form. Note also that **one-to-one permutation** of whole partials is
already expressible as `consume` + a bijective dependency set (§7.4.2);
`all_to_all` is only for the split-and-redistribute case, so the two
mechanisms do not overlap.

**Generalizing `scatter` to `P > 1` is not an alternative.** Adding a
concat axis set and lifting R8 on `scatter` *is* `all_to_all` under another
name; it hides the multi-producer wait inside `scatter` and gives that op
two regimes. A separate op keeps one-op-one-pattern and leaves
`scatter`'s `P == 1` contract clean.

**Why not `inter_tile_shuffle`.** The SDSC backend calls the primitive
`shuffle`, but `shuffle` is the *lowering* of several patterns rather than this
one alone (§9), so `all_to_all` is used as the established collective term.

### 6.6 `ktdp.inter_tile_scatter` — ordered split

`combine = none`, `placement = split`, one producer per group (R8),
consumer set free, `scatter_dimensions`, no region, no identity.

**`scatter_dimensions`** (`i64` array) — axes of `T_p` along which the
single producer's tensor is partitioned into `C` equal chunks (R9), one
per consumer in ascending consumer local-index order (§3.3), flattened in
list order per §4.

**Result type.** `T_s_i` is `T_p_i` with the flattened extent over
`scatter_dimensions` divided by `C`.

```mlir
%scattered_1, ..., %scattered_N = ktdp.inter_tile_scatter(%future)
    consumer_tiles_per_group = <affine-set>,
    scatter_dimensions       = <i64-array>
    : !ktdp.tile_future<T_p_1, ..., T_p_N, #groups> -> T_s_1, ..., T_s_N
```

**No `producer_dependency_per_consumer`.** With a single producer per
group there is exactly one producer to wait for, so full-barrier and
per-tile synchronization collapse to the same thing; the attribute would
be degenerate. R3–R7 are therefore `n/a` for this op (§5).

**Consumers need not be producers.** A consumer tile that does not appear
in `producer_tiles_per_group` simply receives its slice; unlike a partial
gather or a reduce there is nothing for a non-producing consumer to
contribute or miss, so no coverage obligation arises. For a pure split
the consumer set is unconstrained relative to the producer set — which
resolves §10.1 for `scatter`, and only for `scatter`.

---

## 7. Pattern instantiation

### 7.1 Broadcast  →  `inter_tile_produce` + `inter_tile_consume`

```mlir
// 4 tiles, 1 group: tile 0 loads W; all 4 tiles compute.
#tile_0          = affine_set<(i)[g] : (i - 4*g == 0)>
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#single_group    = affine_set<(g) : (g == 0)>

%W_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #tile_0
    : tensor<64x128xf16> -> !ktdp.tile_future<tensor<64x128xf16>, #single_group>
{
  ^bb0(%gid: index):
    %W = ktdp.load ...
    ktdp.yield_partial %W : tensor<64x128xf16>
}

// Every consumer tile extracts its copy; no combiner → value passes through.
// Groups are inferred from the future's #single_group parameter.
%W_tile = ktdp.inter_tile_consume(%W_future)
    consumer_tiles_per_group = #all_group_tiles
    : !ktdp.tile_future<tensor<64x128xf16>, #single_group> -> tensor<64x128xf16>

// Post-delivery SPMD compute — owned by consumer_tiles_per_group.
// Ownership verified by traversing the def-use chain from %W_tile.
%A = ktdp.load ...
%C = linalg.matmul ins(%A, %W_tile ...) ...
ktdp.store %C, ...
```

### 7.2 Reduce  →  `inter_tile_produce` + `inter_tile_reduce`

```mlir
// 4 tiles per group, 8 groups (32 tiles total).
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// All tiles contribute a partial; future carries all partials.
%partial_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #all_group_tiles
    : tensor<1x64xf16> -> !ktdp.tile_future<tensor<1x64xf16>, #all_groups>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %partial_2d : tensor<1x64xf16>
}

// Reduce all partials; every consumer tile receives the same reduced value.
%reduced = ktdp.inter_tile_reduce(%partial_future)
    consumer_tiles_per_group = #all_group_tiles,
    identity(%add_id : tensor<1x64xf16>)
    : !ktdp.tile_future<tensor<1x64xf16>, #all_groups> -> tensor<1x64xf16>
{
  ^bb0(%lhs: tensor<1x64xf16>, %rhs: tensor<1x64xf16>):
    %sum = linalg.add ins(%lhs, %rhs ...) ...
    ktdp.yield_reduced %sum : tensor<1x64xf16>
}
```

#### 7.2.1 Full IR — single-group reduce (96×64)

**Layout and partitioning.** `A` and `B` are `tensor<96x64xf16>` in global memory.
The kernel computes the column-wise sum of `A + B`, producing a
`tensor<1x64xf16>` (the leading unit dim is the within-group tile axis,
preserved by the op per §4).

The 32 compute tiles form a single group. Tile `t` owns rows
`t*3 .. t*3+2` of `A` and `B` — a 3×64 slab each. The per-tile
contribution is the row-reduced partial expanded to `tensor<1x64xf16>`,
whose leading unit dimension is the within-group tile axis. The op preserves
it (§4), so every tile holds the same `%reduced : tensor<1x64xf16>`
(all-reduce case: consumer set = producer set).

```mlir
#A_view_set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#AB_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 +  2 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#E_view_set  = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
#E_tile_set  = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
#identity_2d = affine_map<(d0, d1) -> (d0, d1)>

// One group containing all 32 tiles.
#group_tiles = affine_set<(i)[g] : (i - 32*g >= 0, -i + 32*(g+1) - 1 >= 0)>
#all_groups  = affine_set<(g) : (g == 0)>

module {
  func.func @inter_tile_reduce_single_group() {
    %c0 = arith.constant 0 : index
    %tile_size = arith.constant 3 : index
    %A_start = arith.constant 1024  : index
    %B_start = arith.constant 12288 : index
    %E_start = arith.constant 22528 : index

    %A_view = ktdp.construct_memory_view %A_start, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<96x64xf16>
    %B_view = ktdp.construct_memory_view %B_start, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    // Identity: tensor<1x64xf16> of zeros — matches partial type T_p.
    %c_zero   = arith.constant 0.0 : f16
    %add_init = tensor.empty() : tensor<1x64xf16>
    %add_id   = linalg.fill ins(%c_zero : f16) outs(%add_init : tensor<1x64xf16>)
                  -> tensor<1x64xf16>

    // Per-tile compute (function-scope SPMD).
    %t = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %t, %tile_size : index

    %A_access = ktdp.construct_access_tile %A_view[%start_row, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_2d
    } : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>
    %B_access = ktdp.construct_access_tile %B_view[%start_row, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_2d
    } : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>

    %A_tile = ktdp.load %A_access : !ktdp.access_tile<3x64xindex> -> tensor<3x64xf16>
    %B_tile = ktdp.load %B_access : !ktdp.access_tile<3x64xindex> -> tensor<3x64xf16>

    %AB_init = tensor.empty() : tensor<3x64xf16>
    %AB_sum  = linalg.add ins(%A_tile, %B_tile : tensor<3x64xf16>, tensor<3x64xf16>)
                          outs(%AB_init : tensor<3x64xf16>) -> tensor<3x64xf16>

    %red_init   = tensor.empty() : tensor<64xf16>
    %red_filled = linalg.fill ins(%c_zero : f16) outs(%red_init : tensor<64xf16>)
                    -> tensor<64xf16>
    %partial_1d = linalg.reduce { arith.addf }
                    ins(%AB_sum : tensor<3x64xf16>)
                    outs(%red_filled : tensor<64xf16>)
                    dimensions = [0]
    %partial_2d = tensor.expand_shape %partial_1d [[0, 1]] output_shape [1, 64]
                    : tensor<64xf16> into tensor<1x64xf16>

    // Produce: every tile contributes its partial_2d to the future.
    %partial_future = ktdp.inter_tile_produce
        producer_tiles_per_group = #group_tiles
        : tensor<1x64xf16> -> !ktdp.tile_future<tensor<1x64xf16>, #all_groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial_2d : tensor<1x64xf16>
    }

    // Reduce: unit dim 0 is the within-group tile axis; the op preserves it.
    // Every tile holds the same %reduced : tensor<1x64xf16> (all-reduce case).
    %reduced = ktdp.inter_tile_reduce(%partial_future)
        consumer_tiles_per_group = #group_tiles,
        identity(%add_id : tensor<1x64xf16>)
        : !ktdp.tile_future<tensor<1x64xf16>, #all_groups> -> tensor<1x64xf16>
    {
      ^bb0(%lhs: tensor<1x64xf16>, %rhs: tensor<1x64xf16>):
        %init = tensor.empty() : tensor<1x64xf16>
        %sum  = linalg.add ins(%lhs, %rhs : tensor<1x64xf16>, tensor<1x64xf16>)
                           outs(%init : tensor<1x64xf16>) -> tensor<1x64xf16>
        ktdp.yield_reduced %sum : tensor<1x64xf16>
    }

    // Post-reduction: every tile redundantly writes the same value.
    // No expand_shape needed — the result already carries the unit dim.

    %E_view = ktdp.construct_memory_view %E_start, sizes: [1, 64], strides: [64, 1] {
        coordinate_set = #E_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<1x64xf16>
    %E_access = ktdp.construct_access_tile %E_view[%c0, %c0] {
        access_tile_set = #E_tile_set, access_tile_order = #identity_2d
    } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>

    ktdp.store %reduced, %E_access
              : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>

    return
  }
}
```

#### 7.2.2 Full IR — multi-group reduce (128×8×12×64)

**Layout and partitioning.** `A` and `B` are `tensor<128x8x12x64xf16>` in
global memory. The four axes have distinct roles:

- Dim 0 (size 128): preserved through this op.
- Dim 1 (size 8): the **group axis** — 8 groups.
- Dim 2 (size 12): the **reduction axis** — within each group, 4 tiles
  cooperate over this axis.
- Dim 3 (size 64): vector / stick axis, preserved.

There are 32 compute tiles forming 8 groups of 4. For tile `t`,
`g = t / 4` and `l = t % 4`. Tile `(g, l)` reads slice
`[*, g, l*3 : l*3+3, *]` of `A` and `B` — shape `<128x1x3x64>` each.

The partial is `<128x1x1x64>`: dim 1 is the group axis and dim 2 the
within-group tile axis, both preserved, so the result is `<128x1x1x64>`
too (§4). All four tiles in a group hold identical values; different groups
hold different values.

```mlir
#A_view_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 11  >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#AB_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 2   >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#E_view_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 3   >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#E_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 == 0,
     d3 >= 0, -d3 + 63 >= 0)>

#identity_4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups  = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

module {
  func.func @inter_tile_reduce_multi_group() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %red_slab = arith.constant 3 : index   // 12 / 4

    %A_start = arith.constant 1024     : index
    %B_start = arith.constant 12583936 : index
    %E_start = arith.constant 25166848 : index

    %A_view = ktdp.construct_memory_view %A_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x12x64xf16>
    %B_view = ktdp.construct_memory_view %B_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x12x64xf16>

    // Identity: tensor<128x1x1x64xf16> of zeros — matches partial type T_p.
    %c_zero  = arith.constant 0.0 : f16
    %id_init = tensor.empty() : tensor<128x1x1x64xf16>
    %add_id  = linalg.fill ins(%c_zero : f16) outs(%id_init : tensor<128x1x1x64xf16>)
                 -> tensor<128x1x1x64xf16>

    // Per-tile compute (function-scope SPMD).
    %t = ktdp.get_compute_tile_id : index
    %g = arith.divui %t, %c4 : index
    %l = arith.remui %t, %c4 : index
    %red_anchor = arith.muli %l, %red_slab : index

    %A_access = ktdp.construct_access_tile %A_view[%c0, %g, %red_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>
    %B_access = ktdp.construct_access_tile %B_view[%c0, %g, %red_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>

    %A_tile = ktdp.load %A_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>
    %B_tile = ktdp.load %B_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>

    %AB_init = tensor.empty() : tensor<128x1x3x64xf16>
    %AB_sum  = linalg.add ins(%A_tile, %B_tile
                              : tensor<128x1x3x64xf16>, tensor<128x1x3x64xf16>)
                          outs(%AB_init : tensor<128x1x3x64xf16>)
                          -> tensor<128x1x3x64xf16>

    %red_init   = tensor.empty() : tensor<128x1x64xf16>
    %red_filled = linalg.fill ins(%c_zero : f16)
                              outs(%red_init : tensor<128x1x64xf16>)
                              -> tensor<128x1x64xf16>
    %partial_3d = linalg.reduce { arith.addf }
                    ins(%AB_sum : tensor<128x1x3x64xf16>)
                    outs(%red_filled : tensor<128x1x64xf16>)
                    dimensions = [2]

    %partial_4d = tensor.expand_shape %partial_3d [[0], [1], [2, 3]]
                    output_shape [128, 1, 1, 64]
                    : tensor<128x1x64xf16> into tensor<128x1x1x64xf16>

    // Produce: every tile contributes its partial_4d to the future.
    %partial_future = ktdp.inter_tile_produce
        producer_tiles_per_group = #group_tiles
        : tensor<128x1x1x64xf16>
          -> !ktdp.tile_future<tensor<128x1x1x64xf16>, #all_groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial_4d : tensor<128x1x1x64xf16>
    }

    // Multi-group reduce: no rank reduction — dims 1 and 2 both preserved.
    // Each tile gets its group's <128x1x1x64>.
    %my_group_result = ktdp.inter_tile_reduce(%partial_future)
        consumer_tiles_per_group = #group_tiles,
        identity(%add_id : tensor<128x1x1x64xf16>)
        : !ktdp.tile_future<tensor<128x1x1x64xf16>, #all_groups>
          -> tensor<128x1x1x64xf16>
    {
      ^bb0(%lhs: tensor<128x1x1x64xf16>, %rhs: tensor<128x1x1x64xf16>):
        %init = tensor.empty() : tensor<128x1x1x64xf16>
        %sum  = linalg.add ins(%lhs, %rhs
                               : tensor<128x1x1x64xf16>, tensor<128x1x1x64xf16>)
                           outs(%init : tensor<128x1x1x64xf16>)
                           -> tensor<128x1x1x64xf16>
        ktdp.yield_reduced %sum : tensor<128x1x1x64xf16>
    }

    // Post-reduction: each tile writes its group's result to slice [*, g, l, *].
    // No expand_shape needed — the result already carries both unit dims.

    %E_view = ktdp.construct_memory_view %E_start, sizes: [128, 8, 4, 64],
        strides: [2048, 256, 64, 1] {
        coordinate_set = #E_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x4x64xf16>

    %E_access = ktdp.construct_access_tile %E_view[%c0, %g, %l, %c0] {
        access_tile_set = #E_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x4x64xf16> -> !ktdp.access_tile<128x1x1x64xindex>

    ktdp.store %my_group_result, %E_access
              : tensor<128x1x1x64xf16>, !ktdp.access_tile<128x1x1x64xindex>

    return
  }
}
```

### 7.3 Reduce-scatter  →  `inter_tile_produce` + `inter_tile_reduce_scatter`

```mlir
// 4 tiles per group, 8 groups (32 tiles total).
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// All tiles contribute a partial.
%partial_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #all_group_tiles
    : tensor<128x1x1x64xf16> -> !ktdp.tile_future<tensor<128x1x1x64xf16>, #all_groups>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %partial_4d : tensor<128x1x1x64xf16>
}

// Reduce and scatter; each tile receives its own slice along dim 0.
// scatter_dimensions = [0] → 128-row axis split across 4 tiles; each gets
// <32x1x1x64> (rank preserved, §4).
%my_chunk = ktdp.inter_tile_reduce_scatter(%partial_future)
    consumer_tiles_per_group = #all_group_tiles,
    scatter_dimensions              = [0],
    identity(%add_id : tensor<128x1x1x64xf16>)
    : !ktdp.tile_future<tensor<128x1x1x64xf16>, #all_groups>
      -> tensor<32x1x1x64xf16>
{
  ^bb0(%lhs: tensor<128x1x1x64xf16>, %rhs: tensor<128x1x1x64xf16>):
    %sum = linalg.add ins(%lhs, %rhs ...) ...
    ktdp.yield_reduced %sum : tensor<128x1x1x64xf16>
}
// Each tile holds a different slice — ownership explicit via SSA result.
```

#### 7.3.1 Full IR — multi-group reduce-scatter (128×8×12×64)

**Layout and partitioning.** `A` and `B` are `tensor<128x8x12x64xf16>`
in global memory. The four axes have distinct roles:

- Dim 0 (size 128): the **scatter axis** — within each group, this axis
  is split across that group's 4 tiles.
- Dim 1 (size 8): the **group axis** — 8 groups.
- Dim 2 (size 12): the **reduction axis** — within each group, 4 tiles
  cooperate over this axis.
- Dim 3 (size 64): vector / stick axis, preserved.

32 tiles, 8 groups of 4. `g = t / 4`, `l = t % 4`. Tile `(g, l)` reads
slice `[*, g, l*3 : l*3+3, *]` — shape `<128x1x3x64>`. The per-tile
pipeline through to `%partial_4d` (shape `<128x1x1x64>`) is identical
to §7.2.2.

The op reduces across the group and scatters dim 0 (128 / 4 = 32 rows per
tile), preserving rank (§4). Tile `(g, l)` ends up with rows
`[l*32 : (l+1)*32]` of group `g`'s reduced `<128x1x1x64>`.

```mlir
#A_view_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 11  >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#AB_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 2   >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

// E view (post-scatter output): 128x8x64.
#E_view_set = affine_set<(d0, d1, d2) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 63  >= 0)>

// E access tile per writer: 32x1x64 in E's 3-D memref, anchored at [l*32, g, 0].
#E_tile_set = affine_set<(d0, d1, d2) :
    (d0 >= 0, -d0 + 31 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 63 >= 0)>

#identity_4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#identity_3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups  = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

module {
  func.func @inter_tile_reduce_scatter_multi_group() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %red_slab      = arith.constant 3  : index   // 12 / 4
    %scatter_chunk = arith.constant 32 : index   // 128 / 4

    %A_start = arith.constant 1024     : index
    %B_start = arith.constant 12583936 : index
    %E_start = arith.constant 25166848 : index

    %A_view = ktdp.construct_memory_view %A_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x12x64xf16>
    %B_view = ktdp.construct_memory_view %B_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x12x64xf16>

    // Identity: tensor<128x1x1x64xf16> of zeros — matches partial type T_p.
    %c_zero  = arith.constant 0.0 : f16
    %id_init = tensor.empty() : tensor<128x1x1x64xf16>
    %add_id  = linalg.fill ins(%c_zero : f16) outs(%id_init : tensor<128x1x1x64xf16>)
                 -> tensor<128x1x1x64xf16>

    // Per-tile compute (function-scope SPMD).
    %t = ktdp.get_compute_tile_id : index
    %g = arith.divui %t, %c4 : index
    %l = arith.remui %t, %c4 : index
    %red_anchor = arith.muli %l, %red_slab : index

    %A_access = ktdp.construct_access_tile %A_view[%c0, %g, %red_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>
    %B_access = ktdp.construct_access_tile %B_view[%c0, %g, %red_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>

    %A_tile = ktdp.load %A_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>
    %B_tile = ktdp.load %B_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>

    %AB_init = tensor.empty() : tensor<128x1x3x64xf16>
    %AB_sum  = linalg.add ins(%A_tile, %B_tile
                              : tensor<128x1x3x64xf16>, tensor<128x1x3x64xf16>)
                          outs(%AB_init : tensor<128x1x3x64xf16>)
                          -> tensor<128x1x3x64xf16>

    %red_init   = tensor.empty() : tensor<128x1x64xf16>
    %red_filled = linalg.fill ins(%c_zero : f16)
                              outs(%red_init : tensor<128x1x64xf16>)
                              -> tensor<128x1x64xf16>
    %partial_3d = linalg.reduce { arith.addf }
                    ins(%AB_sum : tensor<128x1x3x64xf16>)
                    outs(%red_filled : tensor<128x1x64xf16>)
                    dimensions = [2]

    %partial_4d = tensor.expand_shape %partial_3d [[0], [1], [2, 3]]
                    output_shape [128, 1, 1, 64]
                    : tensor<128x1x64xf16> into tensor<128x1x1x64xf16>

    // Produce: every tile contributes its partial_4d to the future.
    %partial_future = ktdp.inter_tile_produce
        producer_tiles_per_group = #group_tiles
        : tensor<128x1x1x64xf16>
          -> !ktdp.tile_future<tensor<128x1x1x64xf16>, #all_groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial_4d : tensor<128x1x1x64xf16>
    }

    // Reduce across the group, then scatter dim 0 (chunk = 32).
    // No rank reduction: dims 1 and 2 preserved. Each tile receives <32x1x1x64>.
    %my_chunk = ktdp.inter_tile_reduce_scatter(%partial_future)
        consumer_tiles_per_group = #group_tiles,
        scatter_dimensions              = [0],
        identity(%add_id : tensor<128x1x1x64xf16>)
        : !ktdp.tile_future<tensor<128x1x1x64xf16>, #all_groups>
          -> tensor<32x1x1x64xf16>
    {
      ^bb0(%lhs: tensor<128x1x1x64xf16>, %rhs: tensor<128x1x1x64xf16>):
        %init = tensor.empty() : tensor<128x1x1x64xf16>
        %sum  = linalg.add ins(%lhs, %rhs
                               : tensor<128x1x1x64xf16>, tensor<128x1x1x64xf16>)
                           outs(%init : tensor<128x1x1x64xf16>)
                           -> tensor<128x1x1x64xf16>
        ktdp.yield_reduced %sum : tensor<128x1x1x64xf16>
    }

    // Post-scatter: tile (g, l) writes rows [l*32 : l*32+32] of group g's result.
    %my_row_anchor = arith.muli %l, %scatter_chunk : index

    %E_view = ktdp.construct_memory_view %E_start, sizes: [128, 8, 64],
        strides: [512, 64, 1] {
        coordinate_set = #E_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x64xf16>

    %E_access = ktdp.construct_access_tile %E_view[%my_row_anchor, %g, %c0] {
        access_tile_set = #E_tile_set, access_tile_order = #identity_3d
    } : memref<128x8x64xf16> -> !ktdp.access_tile<32x1x64xindex>

    // Rank reduction now lives in ordinary code, not the op: collapse the
    // within-group tile axis to match E's 3-D layout.
    %my_chunk_3d = tensor.collapse_shape %my_chunk [[0], [1, 2], [3]]
                     : tensor<32x1x1x64xf16> into tensor<32x1x64xf16>

    ktdp.store %my_chunk_3d, %E_access
              : tensor<32x1x64xf16>, !ktdp.access_tile<32x1x64xindex>

    return
  }
}
```

### 7.4 Per-tile synchronization  →  `inter_tile_consume` with `producer_dependency_per_consumer`

#### 7.4.1 Per-tile pairing within a single group

Four tiles per group: tiles `4g` and `4g+1` are producers, tiles `4g+2`
and `4g+3` are consumers. Each consumer depends on its dedicated producer
(`4g+2` ← `4g`, `4g+3` ← `4g+1`), so the pairing is `p = c - 2` — a
constant relative offset that does not depend on the group index `g`.

**Dependency table** for group 0:

| group | producer | consumer |
|-------|----------|----------|
| 0     | 0        | 2        |
| 0     | 1        | 3        |

```mlir
// Producers: tiles 4g, 4g+1.  Consumers: tiles 4g+2, 4g+3.
#producer_tiles  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 1 >= 0)>
#consumer_tiles  = affine_set<(i)[g] : (i - 4*g - 2 >= 0, -i + 4*g + 3 >= 0)>
#single_group    = affine_set<(g) : (g == 0)>

// The pairing p = c - 2 is group-independent, so g is not needed as a symbol.
#dep_per_consumer = affine_set<(p)[c] : (p - c + 2 == 0)>

%data_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #producer_tiles
    : tensor<64xf16> -> !ktdp.tile_future<tensor<64xf16>, #single_group>
{
  ^bb0(%gid: index):
    %data = ktdp.load ...
    ktdp.yield_partial %data : tensor<64xf16>
}

// Each consumer unblocks independently as its assigned producer finishes.
%my_data = ktdp.inter_tile_consume(%data_future)
    consumer_tiles_per_group         = #consumer_tiles,
    producer_dependency_per_consumer = #dep_per_consumer
    : !ktdp.tile_future<tensor<64xf16>, #single_group> -> tensor<64xf16>
```

Without `producer_dependency_per_consumer`, both consumers stall until
both producers finish. With it, each consumer stalls only for its own
producer, halving the worst-case wait when the two producers finish at
different times.

#### 7.4.2 Butterfly mirror exchange across multiple groups

Eight groups of 4 tiles; all 4 tiles in each group both produce and
consume. Tile `c = 4g + l` waits only for its mirror partner
`p = 4g + (3 - l)`, equivalent to `p + c = 8g + 3`. This models a
butterfly-style partner exchange.

Both `c` and `g` are required: `c` identifies which specific consumer is
asking (different consumers within the group have different mirrors), and
`g` anchors the equation to the group (the target sum `8g + 3` is `3`,
`11`, `19`, ... for groups `0`, `1`, `2`, ..., so `g` cannot be
eliminated).

**Dependency table**, first two groups:

| group | producer | consumer |
|-------|----------|----------|
| 0     | 0        | 3        |
| 0     | 1        | 2        |
| 0     | 2        | 1        |
| 0     | 3        | 0        |
| 1     | 4        | 7        |
| 1     | 5        | 6        |
| 1     | 6        | 5        |
| 1     | 7        | 4        |
| …     | …        | …        |

```mlir
// 8 groups of 4 tiles; every tile is both producer and consumer.
#all_group_tiles  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups       = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// Tile c = 4g+l waits for mirror tile p = 4g+(3-l), i.e. p + c = 8g + 3.
// c is needed: different consumers have different mirrors within a group.
// g is needed: the sum p + c = 8g + 3 is a different value for each group.
#dep_per_consumer = affine_set<(p)[c, g] : (p + c - 8*g - 3 == 0)>

%data_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #all_group_tiles
    : tensor<64xf16> -> !ktdp.tile_future<tensor<64xf16>, #all_groups>
{
  ^bb0(%gid: index):
    %data = ktdp.load ...
    ktdp.yield_partial %data : tensor<64xf16>
}

// Each tile unblocks as soon as its single mirror partner has yielded,
// without waiting for the other two tiles in the group.
%partner_data = ktdp.inter_tile_consume(%data_future)
    consumer_tiles_per_group         = #all_group_tiles,
    producer_dependency_per_consumer = #dep_per_consumer
    : !ktdp.tile_future<tensor<64xf16>, #all_groups> -> tensor<64xf16>
```

### 7.5 Gather  →  `inter_tile_produce` + `inter_tile_gather`

```mlir
// 4 tiles per group, 8 groups (32 tiles total).
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#group_consumer  = affine_set<(i)[g] : (i - 4*g == 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// All tiles contribute a partial slab.
%partial_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #all_group_tiles
    : tensor<128x1x3x64xf16> -> !ktdp.tile_future<tensor<128x1x3x64xf16>, #all_groups>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %partial_4d : tensor<128x1x3x64xf16>
}

// Gather along dim 2; one consumer per group (tile 4g) assembles the four
// 3-wide slabs. No combiner, no identity — placement is by within-group
// local index. gather_dimensions = [2] → 3 * 4 = 12; consumer gets <128x1x12x64>.
%assembled = ktdp.inter_tile_gather(%partial_future)
    consumer_tiles_per_group = #group_consumer,
    gather_dimensions               = [2]
    : !ktdp.tile_future<tensor<128x1x3x64xf16>, #all_groups> -> tensor<128x1x12x64xf16>
// The consumer holds the full assembled tensor — ownership via SSA result.
```

#### 7.5.1 Full IR — multi-group gather (128×8×12×64)

**Layout and partitioning.** `A` and `B` are `tensor<128x8x12x64xf16>` in global
memory. The four axes have distinct roles:

- Dim 0 (size 128): preserved through this op.
- Dim 1 (size 8): the **group axis** — 8 groups.
- Dim 2 (size 12): the **gather axis** — within each group, 4 tiles each own
  a 3-wide slab that gather concatenates back into the full 12.
- Dim 3 (size 64): vector / stick axis, preserved.

32 tiles, 8 groups of 4. `g = t / 4`, `l = t % 4`. Tile `(g, l)` reads
slice `[*, g, l*3 : l*3+3, *]` — shape `<128x1x3x64>`. Each tile's partial
is the summed slab `A + B` over its own columns (no reduction across tiles).
Gather along dim 2 places tile `(g, l)`'s slab at columns `[l*3 : l*3+3]` of
the assembled `<128x1x12x64>`, which one consumer per group (tile `4g`)
writes back to `E[*, g, *, *]`.

```mlir
#A_view_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 11  >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#AB_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 2   >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

// E access tile for the consumer: 128x1x12x64 anchored at [0, g, 0, 0].
#E_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 11  >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

#identity_4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#group_tiles    = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#group_consumer = affine_set<(i)[g] : (i - 4*g == 0)>
#all_groups     = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

module {
  func.func @inter_tile_gather_multi_group() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %col_slab = arith.constant 3 : index   // 12 / 4

    %A_start = arith.constant 1024     : index
    %B_start = arith.constant 12583936 : index
    %E_start = arith.constant 25166848 : index

    %A_view = ktdp.construct_memory_view %A_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x12x64xf16>
    %B_view = ktdp.construct_memory_view %B_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x12x64xf16>

    // Per-tile compute (function-scope SPMD).
    %t = ktdp.get_compute_tile_id : index
    %g = arith.divui %t, %c4 : index
    %l = arith.remui %t, %c4 : index
    %col_anchor = arith.muli %l, %col_slab : index

    %A_access = ktdp.construct_access_tile %A_view[%c0, %g, %col_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>
    %B_access = ktdp.construct_access_tile %B_view[%c0, %g, %col_anchor, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x3x64xindex>

    %A_tile = ktdp.load %A_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>
    %B_tile = ktdp.load %B_access
                : !ktdp.access_tile<128x1x3x64xindex> -> tensor<128x1x3x64xf16>

    // No reduction — the summed slab is this tile's partial; gather will
    // concatenate the four slabs along dim 2.
    %AB_init = tensor.empty() : tensor<128x1x3x64xf16>
    %partial_4d = linalg.add ins(%A_tile, %B_tile
                                 : tensor<128x1x3x64xf16>, tensor<128x1x3x64xf16>)
                             outs(%AB_init : tensor<128x1x3x64xf16>)
                             -> tensor<128x1x3x64xf16>

    // Produce: every tile contributes its 3-wide slab to the future.
    %partial_future = ktdp.inter_tile_produce
        producer_tiles_per_group = #group_tiles
        : tensor<128x1x3x64xf16>
          -> !ktdp.tile_future<tensor<128x1x3x64xf16>, #all_groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial_4d : tensor<128x1x3x64xf16>
    }

    // Gather dim 2: 4 producers x 3 = 12. One consumer (tile 4g) per group
    // assembles the full <128x1x12x64>. No combiner region, no identity.
    %assembled = ktdp.inter_tile_gather(%partial_future)
        consumer_tiles_per_group = #group_consumer,
        gather_dimensions               = [2]
        : !ktdp.tile_future<tensor<128x1x3x64xf16>, #all_groups>
          -> tensor<128x1x12x64xf16>

    // Post-gather: the consumer tile 4g writes its group's full slab to
    // E[*, g, *, *]. Ownership is explicit via the def-use chain of %assembled.
    %E_view = ktdp.construct_memory_view %E_start, sizes: [128, 8, 12, 64],
        strides: [6144, 768, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x12x64xf16>

    %E_access = ktdp.construct_access_tile %E_view[%c0, %g, %c0, %c0] {
        access_tile_set = #E_tile_set, access_tile_order = #identity_4d
    } : memref<128x8x12x64xf16> -> !ktdp.access_tile<128x1x12x64xindex>

    ktdp.store %assembled, %E_access
              : tensor<128x1x12x64xf16>, !ktdp.access_tile<128x1x12x64xindex>

    return
  }
}
```

### 7.6 All-to-all  →  `inter_tile_produce` + `inter_tile_all_to_all`

```mlir
// 4 tiles per group, 8 groups (32 tiles total).
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// Sequence-parallel production: every tile owns a 128-row shard of all 4 heads.
%partial_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #all_group_tiles
    : tensor<128x1x4x64xf16> -> !ktdp.tile_future<tensor<128x1x4x64xf16>, #all_groups>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %partial_4d : tensor<128x1x4x64xf16>
}

// Head-parallel consumption: split dim 2 (heads) across the 4 consumers,
// regather dim 0 (sequence) from the 4 producers.
// split_dimensions = [2] → 4 / 4 = 1;  concat_dimensions = [0] → 128 * 4 = 512.
%relaid = ktdp.inter_tile_all_to_all(%partial_future)
    consumer_tiles_per_group = #all_group_tiles,
    split_dimensions                = [2],
    concat_dimensions               = [0]
    : !ktdp.tile_future<tensor<128x1x4x64xf16>, #all_groups> -> tensor<512x1x1x64xf16>
// Every tile is both producer and consumer; P == C == 4, so the element count
// is conserved (128*4 = 512*1) even though the type changes.
```

#### 7.6.1 Full IR — sequence-parallel to head-parallel (512×8×4×64)

**Layout and partitioning.** `A`, `B`, and `E` are `tensor<512x8x4x64xf16>`
in global memory. The four axes have distinct roles:

- Dim 0 (size 512): the **gather axis** — sequence. Sharded 4 ways before
  the op, whole after it.
- Dim 1 (size 8): the **group axis** — 8 groups.
- Dim 2 (size 4): the **scatter axis** — heads. Whole before the op,
  sharded 4 ways after it.
- Dim 3 (size 64): vector / stick axis, preserved.

32 tiles, 8 groups of 4. `g = t / 4`, `l = t % 4`. Before the op, tile
`(g, l)` owns sequence shard `l`: it reads `[l*128 : l*128+128, g, *, *]`,
shape `<128x1x4x64>`, and its partial is `A + B` over those rows. After the
op, tile `(g, l)` owns head `l` for the whole sequence, shape
`<512x1x1x64>`, and writes it back to `E[*, g, l, *]`.

This is the pattern a sequence-parallel prefill hands to a head-parallel
attention: the ownership axis moves from dim 0 to dim 2 in one collective,
with no tile ever holding more than its `1/4` share.

```mlir
#A_view_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 511 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 3   >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

// A/B access tile for the producer: 128x1x4x64 anchored at [l*128, g, 0, 0].
#AB_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 3   >= 0,
     d3 >= 0, -d3 + 63  >= 0)>

// E access tile for the consumer: 512x1x1x64 anchored at [0, g, l, 0].
#E_tile_set = affine_set<(d0, d1, d2, d3) :
    (d0 >= 0, -d0 + 511 >= 0,
     d1 == 0,
     d2 == 0,
     d3 >= 0, -d3 + 63  >= 0)>

#identity_4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups  = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

module {
  func.func @inter_tile_all_to_all_relayout() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %row_shard = arith.constant 128 : index   // 512 / 4

    %A_start = arith.constant 1024    : index
    %B_start = arith.constant 2098176 : index
    %E_start = arith.constant 4195328 : index

    %A_view = ktdp.construct_memory_view %A_start, sizes: [512, 8, 4, 64],
        strides: [2048, 256, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<512x8x4x64xf16>
    %B_view = ktdp.construct_memory_view %B_start, sizes: [512, 8, 4, 64],
        strides: [2048, 256, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<512x8x4x64xf16>

    // Per-tile compute (function-scope SPMD).
    %t = ktdp.get_compute_tile_id : index
    %g = arith.divui %t, %c4 : index
    %l = arith.remui %t, %c4 : index
    %row_anchor = arith.muli %l, %row_shard : index

    %A_access = ktdp.construct_access_tile %A_view[%row_anchor, %g, %c0, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<512x8x4x64xf16> -> !ktdp.access_tile<128x1x4x64xindex>
    %B_access = ktdp.construct_access_tile %B_view[%row_anchor, %g, %c0, %c0] {
        access_tile_set = #AB_tile_set, access_tile_order = #identity_4d
    } : memref<512x8x4x64xf16> -> !ktdp.access_tile<128x1x4x64xindex>

    %A_tile = ktdp.load %A_access
                : !ktdp.access_tile<128x1x4x64xindex> -> tensor<128x1x4x64xf16>
    %B_tile = ktdp.load %B_access
                : !ktdp.access_tile<128x1x4x64xindex> -> tensor<128x1x4x64xf16>

    // No reduction — the summed sequence shard is this tile's partial; the
    // all-to-all redistributes it from sequence-sharded to head-sharded.
    %AB_init = tensor.empty() : tensor<128x1x4x64xf16>
    %partial_4d = linalg.add ins(%A_tile, %B_tile
                                 : tensor<128x1x4x64xf16>, tensor<128x1x4x64xf16>)
                             outs(%AB_init : tensor<128x1x4x64xf16>)
                             -> tensor<128x1x4x64xf16>

    // Produce: every tile contributes its sequence shard to the future.
    %partial_future = ktdp.inter_tile_produce
        producer_tiles_per_group = #group_tiles
        : tensor<128x1x4x64xf16>
          -> !ktdp.tile_future<tensor<128x1x4x64xf16>, #all_groups>
    {
      ^bb0(%gid: index):
        ktdp.yield_partial %partial_4d : tensor<128x1x4x64xf16>
    }

    // All-to-all: consumer l takes head slice l (split_dimensions = [2], 4 / 4 = 1)
    // from each of the 4 producers, and concatenates them along the sequence
    // axis (concat_dimensions = [0], 128 * 4 = 512) in ascending producer local-index
    // order. The producer's local index picks the destination row block;
    // the consumer's local index picks the head. No combiner, no identity.
    %relaid = ktdp.inter_tile_all_to_all(%partial_future)
        consumer_tiles_per_group = #group_tiles,
        split_dimensions                = [2],
        concat_dimensions               = [0]
        : !ktdp.tile_future<tensor<128x1x4x64xf16>, #all_groups>
          -> tensor<512x1x1x64xf16>

    // Post-exchange: tile (g, l) now owns head l for the whole sequence and
    // writes it to E[*, g, l, *]. Every tile is a consumer, so unlike the
    // gather example there is no idle tile after the collective.
    %E_view = ktdp.construct_memory_view %E_start, sizes: [512, 8, 4, 64],
        strides: [2048, 256, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<512x8x4x64xf16>

    %E_access = ktdp.construct_access_tile %E_view[%c0, %g, %l, %c0] {
        access_tile_set = #E_tile_set, access_tile_order = #identity_4d
    } : memref<512x8x4x64xf16> -> !ktdp.access_tile<512x1x1x64xindex>

    ktdp.store %relaid, %E_access
              : tensor<512x1x1x64xf16>, !ktdp.access_tile<512x1x1x64xindex>

    return
  }
}
```

### 7.7 Scatter  →  `inter_tile_produce` + `inter_tile_scatter`

```mlir
// 4 tiles per group, 8 groups (32 tiles total).
#group_producer  = affine_set<(i)[g] : (i - 4*g == 0)>
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// One producer per group (tile 4g) holds the whole 128-row tensor.
%whole_future = ktdp.inter_tile_produce
    producer_tiles_per_group = #group_producer
    : tensor<128x1x64xf16> -> !ktdp.tile_future<tensor<128x1x64xf16>, #all_groups>
{
  ^bb0(%gid: index):
    ktdp.yield_partial %whole : tensor<128x1x64xf16>
}

// Scatter along dim 0; the four tiles per group each receive one 32-row
// chunk. No combiner, no identity — placement is by within-group local
// index. scatter_dimensions = [0] → 128 / 4 = 32; each consumer gets <32x1x64>.
%chunk = ktdp.inter_tile_scatter(%whole_future)
    consumer_tiles_per_group = #all_group_tiles,
    scatter_dimensions              = [0]
    : !ktdp.tile_future<tensor<128x1x64xf16>, #all_groups> -> tensor<32x1x64xf16>
// Each consumer holds its own 32-row slice — ownership via SSA result.
```

#### 7.7.1 Full IR — multi-group scatter (128×8×64)

**Layout and partitioning.** `A` and `B` are `tensor<128x8x64xf16>` in global
memory. The three axes have distinct roles:

- Dim 0 (size 128): the **scatter axis** — the producer's 128 rows are
  split into 4 chunks of 32, one per consumer tile.
- Dim 1 (size 8): the **group axis** — 8 groups.
- Dim 2 (size 64): vector / stick axis, preserved.

32 tiles, 8 groups of 4. `g = t / 4`, `l = t % 4`. Per group, the single
producer tile `4g` reads its group's whole slab `A[*, g, *]` /
`B[*, g, *]` — shape `<128x1x64>` — sums them, and produces the summed
tensor. Scatter along dim 0 delivers chunk `[l*32 : l*32+32, *, *]` to the
consumer with within-group local index `l`, which writes its `<32x1x64>`
slice back to `E[l*32 : l*32+32, g, *]`.

**Why the loads live inside the produce region.** Unlike the other full-IR
examples, which hoist their `ktdp.load`s to function scope, this one keeps
them inside `ktdp.inter_tile_produce`. That is deliberate, and it follows
from single-producer cardinality (§2.2): only tile `4g` may read the
group's whole slab, so hoisting the loads would make every tile in the
group execute them. The other examples have every tile produce, so
function-scope loads are correct there.

```mlir
#A_view_set = affine_set<(d0, d1, d2) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 >= 0, -d1 + 7   >= 0,
     d2 >= 0, -d2 + 63  >= 0)>

// Producer partial: the whole 128-row slab of one group, anchored at [0, g, 0].
#whole_tile_set = affine_set<(d0, d1, d2) :
    (d0 >= 0, -d0 + 127 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 63  >= 0)>

// Consumer chunk: 32 rows, anchored at [l*32, g, 0].
#chunk_tile_set = affine_set<(d0, d1, d2) :
    (d0 >= 0, -d0 + 31 >= 0,
     d1 == 0,
     d2 >= 0, -d2 + 63 >= 0)>

#identity_3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#group_producer  = affine_set<(i)[g] : (i - 4*g == 0)>
#all_group_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#all_groups      = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

module {
  func.func @inter_tile_scatter_multi_group() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %row_chunk = arith.constant 32 : index   // 128 / 4

    %A_start = arith.constant 1024    : index
    %B_start = arith.constant 1049600 : index
    %E_start = arith.constant 2098176 : index

    %A_view = ktdp.construct_memory_view %A_start, sizes: [128, 8, 64],
        strides: [512, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x64xf16>
    %B_view = ktdp.construct_memory_view %B_start, sizes: [128, 8, 64],
        strides: [512, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x64xf16>

    %t = ktdp.get_compute_tile_id : index
    %g = arith.divui %t, %c4 : index
    %l = arith.remui %t, %c4 : index

    // Produce: only the group's producer tile (4g) runs this region; it
    // reads and sums its group's whole 128-row slab.
    %whole_future = ktdp.inter_tile_produce
        producer_tiles_per_group = #group_producer
        : tensor<128x1x64xf16>
          -> !ktdp.tile_future<tensor<128x1x64xf16>, #all_groups>
    {
      ^bb0(%gid: index):
        %A_access = ktdp.construct_access_tile %A_view[%c0, %gid, %c0] {
            access_tile_set = #whole_tile_set, access_tile_order = #identity_3d
        } : memref<128x8x64xf16> -> !ktdp.access_tile<128x1x64xindex>
        %B_access = ktdp.construct_access_tile %B_view[%c0, %gid, %c0] {
            access_tile_set = #whole_tile_set, access_tile_order = #identity_3d
        } : memref<128x8x64xf16> -> !ktdp.access_tile<128x1x64xindex>

        %A_tile = ktdp.load %A_access
                    : !ktdp.access_tile<128x1x64xindex> -> tensor<128x1x64xf16>
        %B_tile = ktdp.load %B_access
                    : !ktdp.access_tile<128x1x64xindex> -> tensor<128x1x64xf16>

        %AB_init = tensor.empty() : tensor<128x1x64xf16>
        %whole = linalg.add ins(%A_tile, %B_tile
                                : tensor<128x1x64xf16>, tensor<128x1x64xf16>)
                            outs(%AB_init : tensor<128x1x64xf16>)
                            -> tensor<128x1x64xf16>
        ktdp.yield_partial %whole : tensor<128x1x64xf16>
    }

    // Scatter dim 0: 128 / 4 = 32. Each of the four consumer tiles per group
    // receives one 32-row chunk. No combiner region, no identity.
    %chunk = ktdp.inter_tile_scatter(%whole_future)
        consumer_tiles_per_group = #all_group_tiles,
        scatter_dimensions              = [0]
        : !ktdp.tile_future<tensor<128x1x64xf16>, #all_groups>
          -> tensor<32x1x64xf16>

    // Post-scatter: consumer (g, l) writes its 32-row chunk to
    // E[l*32 : l*32+32, g, *]. Ownership is explicit via the def-use chain.
    %row_anchor = arith.muli %l, %row_chunk : index

    %E_view = ktdp.construct_memory_view %E_start, sizes: [128, 8, 64],
        strides: [512, 64, 1] {
        coordinate_set = #A_view_set,
        memory_space   = #ktdp.memory_space<global>
    } : memref<128x8x64xf16>

    %E_access = ktdp.construct_access_tile %E_view[%row_anchor, %g, %c0] {
        access_tile_set = #chunk_tile_set, access_tile_order = #identity_3d
    } : memref<128x8x64xf16> -> !ktdp.access_tile<32x1x64xindex>

    ktdp.store %chunk, %E_access
              : tensor<32x1x64xf16>, !ktdp.access_tile<32x1x64xindex>

    return
  }
}
```

---

## 8. Implementation status

Where the rules of §5 stand in the verifier today. Non-normative: this
section records the current state, not an obligation.

The legality pass (`lib/Conversion/ConvertToKTIR/KTIRCheckLegality.cpp`)
walks `InterTileProduceOp`, `InterTileConsumeOp` and `InterTileReduceOp`:

**Implemented-rule table.** One row per check that exists today.

| Rule | Op | Check | Location |
|---|---|---|---|
| R2 | `inter_tile_produce` | `future.hasOneUse()` | `KTIRCheckLegality.cpp:128–132` |
| R13 | `inter_tile_reduce` | `C ⊆ P` per group | `KTIRCheckLegality.cpp:155–164` |
| R14 | `inter_tile_reduce` | `C == P` or `\|C\| == 1` | `KTIRCheckLegality.cpp:167–175` |
| R3 | `reduce`, `consume` | declared dep `p ∈ P(g)` | `KTIRCheckLegality.cpp:82–89` |
| R4 | `reduce`, `consume` | every `p` covered by some dep | `KTIRCheckLegality.cpp:92–97` |
| R8 | `inter_tile_consume` | one source per consumer tile; attribute required when `\|P(g)\| > 1` | `KTIRCheckLegality.cpp:192–234` |

**Not yet implemented:** R1, R5, R6, R7, R9, R10, R11, R12, and R13/R14 for
every op other than `reduce`. Of the ops §9.3 shows the backend requires,
`consume` now has a verifier; `gather`, `all_to_all` and `scatter` do not, and
R9/R12 in particular are stated over *flattened* multi-axis extents (§4), so
implementing them means validating an axis list, not a single index. R5 and R7 are enforced in
the Torch-Spyre SDSC planner (`_compatible_partitions`) but are absent
from the KTIR verifier entirely — the gap exists at both the spec and the
implementation level.

**Dependency-set arity.** §3.4's one-symbol spelling `(p)[c]` is accepted as
of `KTDPInterTileHelpers.cpp:69–100` and `KTIRCheckLegality.cpp:135–142`.
Before that, `depTilesOf` always bound two symbols and the pass rejected any
set whose symbol count was not exactly 2, so the group-independent form this
document documents — and uses in §7.4.1 — was unusable in practice. The symbol
count now selects how many values are bound, and 3-or-more is diagnosed.

**Two asymmetries the verification matrix (§5) forces into the open.**

1. R8 is a verifier obligation for both `consume` and `scatter`, but it
   bites differently: `scatter` takes no dependency attribute, so one
   producer per group is the whole rule, whereas `consume` admits a
   multi-producer group whenever the attribute pairs each consumer tile with
   exactly one producer (§5). Neither is implemented yet.
2. R13 and R14 are implemented for `reduce` only, and R13 is the
   implementation of open question §10.1 (must a consumer also be a
   producer?) for that one op. The `?` cells in that matrix are exactly
   that question, unresolved: for `scatter` the answer is **no** (§6.6), for
   `reduce` the current answer is **yes** (enforced), and for
   `reduce_scatter` / `gather` / `all_to_all` it is undecided. R14's
   mode gate is likewise a current implementation restriction, not a
   design conclusion.

---

## 9. Backend pattern catalogue — non-normative

Relayout patterns measured in the Torch-Spyre backend, and which op of §6
each one needs. Descriptive, not normative: its purpose is to establish
**which ops a lowering must actually emit**, and with what attribute
arity.

Every relayout compiles to one SDSC entry with `opfunc = "shuffle"`, whose
payload is a pair of per-core ownership tables — what each core owns before
the movement and after. A table maps `core_id → {axis: slice_index}`; a
core's **region** is the intersection of its per-axis ranges, and an axis
absent from an entry is uncut. Classifying a pattern means deciding, from
those two tables, which delivery op expresses the same movement.

**Coarsened and refined.** Let `Ns(a)` and `Nd(a)` be the number of
**distinct slices** axis `a` carries in the source and destination tables —
counted from the slices a table actually contains, never from slices it does
not: dividing the axis extent by one slice's extent counts pieces no core
owns. An axis absent from an entry counts 1. Then `a` is **coarsened** when
`Ns(a) > Nd(a)` — fewer, larger pieces after the move, so data must be
*assembled* along it — and **refined** when `Nd(a) > Ns(a)`, so data must be
*split* along it. Write

```
C = {a : Ns(a) > Nd(a)}    # coarsened — assemble
R = {a : Nd(a) > Ns(a)}    # refined   — split
```

These two sets are what select the op. Everything else the classification
needs is a product down the axes: `len(src_regions) = prod(Ns(a))`,
`len(dst_regions) = prod(Nd(a))`, and `components = prod(gcd(Ns(a), Nd(a)))`
— a **component** being a maximal set of source and destination regions
that exchange only among themselves, which for `all_to_all` is exactly a
permute group.

Two cautions. Region counts classify; **core** counts mislead, since
several cores may hold one region and a region count need not divide the
core count. And axes must be aligned by **physical axis, not by label** —
labels differ between the two sides, and this is the only place a wrong
answer can enter.

**Where the two readings differ.** §9.2 works a pattern forward from slice
counts already given; this pair shows where those counts come from, on the
one measured file where counting and dividing disagree. The tensor is
`512 × 32 × 64` elements, `out` counting sticks. The source divides it 8
ways on `mb` and 4 ways on `out`; the destination keeps **one** `mb` index —
the last, `mb[511]` — and spreads that single row over all 32 cores, one
stick each.

**Ownership tables.** One row per side; core ids run row-major with `mb`
outermost.

| | slice counts | core 0 | core 28 | per-core type |
|---|---|---|---|---|
| src | `{mb:8, out:4}` | `mb[0:64] × out[0:8]` | `mb[448:512] × out[0:8]` | `tensor<64x8x64xf16>` |
| dst | `{mb:1, out:32}` | `mb[511:512] × out[0:1]` | `mb[511:512] × out[28:29]` | `tensor<1x1x64xf16>` |

Read off the destination table, `mb` carries one distinct slice — every core
names `mb[511:512]` — so `Nd(mb) = 1` against `Ns(mb) = 8`, and `mb` is
coarsened; `out` goes the other way, `4` against `32`, so it is refined.
Divide instead, extent `512` by the slice's extent `1`, and `Nd(mb) = 512`:
`mb` would come out *refined*, on the strength of 511 pieces the table never
mentions. Only the first reading is a fact about the tables.

**Axis names versus axis indices.** Axis sets are written below with the
backend's symbol names (`mb`, `in`, `out`, …), the vocabulary the ownership
tables speak. The op attributes of §6 are `i64` arrays of *axis indices*
into `T_p`; a lowering resolves each named axis to its position in the
producer tile type, preserving list order, which §4 fixes as slowest- to
fastest-varying.

### 9.1 The decision table

| # | condition | result |
|---|---|---|
| — | `prod(Ns(a)) != len(src_regions)` or `prod(Nd(a)) != len(dst_regions)`, or either side's distinct regions do not cover the tensor | **not a work-division pair** — enumerate regions instead. Check first |
| — | any axis ragged (non-uniform overlap) | *insufficient information* — no single op has uniform dependency-set cardinality (R6) |
| 1 | `C = ∅` and `R = ∅` | regions identical: `no op needed` if every core's region is its own, else `inter_tile_consume` — a **relocation**, or a **broadcast** where destination regions are shared |
| 2 | `C = ∅`, `R ≠ ∅` | `inter_tile_scatter`, `scatter_dimensions = R` |
| 3 | `C ≠ ∅`, `R ≠ ∅`, `prod(Nd(a)) == num_cores` | `inter_tile_all_to_all`, `split_dimensions = R`, `concat_dimensions = C`; one group per component |
| 4 | `C ≠ ∅` otherwise | `inter_tile_gather`, `gather_dimensions = C`; consumers per group = the destination region's holders |

The dimension attributes are **the axis sets themselves**, in the order §4
fixes — which is why they must be list-valued (§1.1). Rows 2–4 return
*insufficient information* when a source region has several holders, since
R8 gives each consumer tile exactly one source and the tables do not say which
holder transmits (§10.2).

**The coverage clause of the first guard row.** Its other two clauses are
region counts; this one is a volume — each **distinct** region's element
count, summed over a side, against the element count of the value being
delivered. Both qualifications matter. **Distinct**, because summed per
*core* an all-gather exceeds the tensor by design: the file whose one
destination region is held by 28 cores (§9.3) would overshoot 28 times over.
And **the value delivered**, because after a select that is the selected
sub-tensor and not the original — otherwise the select-then-deliver that
repairs a selection would trip the guard it was meant to satisfy.

The clause is also the only test in §9 that reads slice **sizes** and the
tensor shape rather than slice counts, which is why the count clauses cannot
replace it. On the pair tabled above, `prod(Nd(a)) = 1 × 32 = 32` equals the
destination region count and `prod(Ns(a)) = 8 × 4 = 32` equals the source's,
so **both count clauses pass**. Coverage is what fails: the 32 distinct
destination regions hold one stick each, `32 × 64 = 2048` elements against
the tensor's `512 × 32 × 64 = 1048576` — a 512th of it.

Row 3's `prod(Nd(a)) == num_cores` is the one irreducibly global test:
holding the division fixed and varying the core count changes the op, so no
per-axis quantity can see it. It is also weaker than asking whether any
destination region is shared — the two part company on every row-4 output
under an idle-core reading (§10.2), which is why the guard rows run first.

For `all_to_all`, group sizes follow from the component count — each tile
contributes `M` slices and receives `K`:

```python
M = len(dst_regions) / components      # consumers per group
K = len(src_regions) / components      # producers per group
```

Note the sides: `M` counts *destination* regions. Inverting them is
invisible on a square exchange and wrong on every other.

### 9.2 Worked example

**A real all-to-all** — `{mb:8, out:4} → {mb:32}` on 32 cores.

| axis | `Ns(a)` | `Nd(a)` | relation | `gcd` |
|---|---|---|---|---|
| `mb` | 8 | 32 | refined | 8 |
| `out` | 4 | 1 | coarsened | 1 |

32 regions a side, `components = 8`. Both sets non-empty and
`prod(Nd(a)) = 32 = num_cores`, so row 3: **`inter_tile_all_to_all`,
`split_dimensions = [mb]`, `concat_dimensions = [out]`**, with **8 groups
and `M = K = 4`** — eight independent 4-way exchanges, not one 32-way.

By contrast, source and destination cutting *different* axes,
`{Lk:32} → {H:8}`, makes both sets non-empty yet `prod(Nd(a)) = 8 ≠ 32`, so
row 4 gathers — the only shape where the global comparison does the work,
and unattested in the measurements.

### 9.3 Measured use cases

51 measured relayouts, each `opfunc = "shuffle"` on 32 cores, carrying
explicit regions and per-region core sets — so replication versus idleness,
which a work division can never settle, is read directly.

| use case | n | src | dst | C | R | KTIR op |
|---|---|---|---|---|---|---|
| all-gather to every core | 3 | `{in:2, out:8, x:2}` | `{}` | `in`,`out`,`x` | — | `inter_tile_gather`, `gather_dimensions = [in, out, x]` |
| all-gather, 4 cores idle | 1 | `{in:32}` | `{}` | `in` | — | `inter_tile_gather`, `gather_dimensions = [in]` |
| grouped gather, drop an axis | 6 | `{mb:8, in:4}` | `{mb:8}` | `in` | — | `inter_tile_gather`, `gather_dimensions = [in]` |
| grouped gather, coarsen one axis | 15 | `{mb:32}` | `{mb:8}` | `mb` | — | `inter_tile_gather`, `gather_dimensions = [mb]` |
| grouped gather, coarsen one axis | 3 | `{mb:16}` | `{mb:8}` | `mb` | — | `inter_tile_gather`, `gather_dimensions = [mb]` |
| all-to-all, square | 6 | `{mb:8, out:4}` | `{mb:32}` | `out` | `mb` | `inter_tile_all_to_all`, split `[mb]` / concat `[out]`; 8 groups, `M=K=4` |
| all-to-all, square | 3 | `{x:8, mb:4}` | `{x:32}` | `mb` | `x` | `inter_tile_all_to_all`, split `[x]` / concat `[mb]`; 8 groups, `M=K=4` |
| all-to-all, **non-square** | 1 | `{mb:16}` | `{mb:8, out:4}` | `mb` | `out` | `inter_tile_all_to_all`, split `[out]` / concat `[mb]`; 8 groups, **`M=4, K=2`** |
| pure split | 12 | `{y:16}` | `{y:32}` | — | `y` | `inter_tile_scatter`, `scatter_dimensions = [y]` |
| broadcast | — | `{h:8}` on 8 cores | `{h:8}` × 4 cores | — | — | `inter_tile_consume`, consumer set widened to the 4 holders |
| selection, not a partition | 1 | `{mb:8, out:4}` | *selection* | — | — | **not a work-division pair** — guard row |

Divisions are the **measured** ones, in `layoutDimOrder_` order. The
broadcast row comes from separate broadcast work (PR #4061), not the 51.

**Which ops this requires.** Four of the six delivery ops, with these
arities:

| Op | measured files | arity needed |
|---|---|---|
| `inter_tile_gather` | 28 | up to **3 axes** |
| `inter_tile_all_to_all` | 10 | 1 axis each side, but **non-square** `M ≠ K` |
| `inter_tile_scatter` | 12 | 1 axis |
| `inter_tile_consume` | broadcast work | — |

`inter_tile_reduce` and `inter_tile_reduce_scatter` are exercised by none
of the 51 — expected, since a relayout moves ownership without combining
values. They stay required by §7.2 and §7.3, which are not relayouts.

Two consequences for implementation order: `gather` carries the most
measured weight *and* the widest arity, so its multi-axis path cannot be
deferred; and `all_to_all`'s non-square case is measured, not
hypothetical, so `P == C` is not a safe simplifying assumption.

**What the measurements also establish.**

- **A three-axis concat exists**, so §4's flattening order must be fixed
  over three axes — list-valued attributes are a requirement of a named
  pattern, not a corner case.
- **Idleness is the norm, replicated sources are rare.** Every source
  region in the 51 has one holder; 16 files have fewer source regions than
  cores and all resolve to single holders plus idle cores. One destination
  region is held by 28 cores with 4 idle.
- **The contiguity assumption is false** — four-core destination groups are
  contiguous in 9 files and strided in 15, so the core-to-region map is not
  a function of the division. This is why §3.3 defines position by
  ascending tile id within the set rather than by contiguity, and why
  `producer_tiles_per_group` / `consumer_tiles_per_group` must come from
  the tables rather than the axis counts.
- **The stick-level assumption holds** — 20 files coarsen or refine the
  stick axis and every piece size on it is an exact stick multiple.
- **One file is a selection, not a partition** — 1/512 coverage, which is
  what the §9.1 validity guard is for. A selection is not a delivery: it
  needs a select op before one.

**Still unmeasured.** Two recorded patterns match no file: one needs a side
with 8 active cores (measured counts are 1, 16, 28, 32), and one is an axis
transpose `{A:4, B:8} → {A:8, B:4}`. The transpose classifies under row 3 —
`C = {B}`, `R = {A}`, 16 components, `M = K = 2` — but with nothing coarsened
or refined in the *region* sense it could equally be read as row 1, and only
divisions keyed by physical axis settle which. Uniformity holds on all 51
measured files, so R6/R7 are so far confirmed rather than assumed.

**Fused relayout is deferred.** Relayout stays a separate preceding op and
fusion is a lowering concern. The backend structurally cannot fuse them
today: restickify is a separate pass that runs before LX planning, and
restickified weights are explicitly barred as shuffle sources.

---

## 10. Open questions

### 10.1 Must a consumer also be a producer?

**What turns on it:** whether the verifier rejects a delivery op whose consumer
set is not contained in its producer set. That check exists and runs today — R13
for `reduce` (`KTIRCheckLegality.cpp:107–117`) — so whoever implements
`gather`, `all_to_all` or `reduce_scatter` must decide whether to extend it,
and the answer changes which programs are legal.

**Why it is unresolved:** `reduce`'s *yes* is one op's implementation choice,
made when it was the only delivery op. `scatter`'s *no* is settled and
argued (§6.6) — a consumer that receives a slice contributes nothing, so
there is nothing to miss. Neither generalizes: `gather` and `all_to_all`
assemble, so a non-producing consumer is coherent for them in a way it is not
for a reduction. The `?` cells in §5 are exactly the ops still to decide, and
R14's mode gate (all-reduce or reduce-to-one, no strict multi-tile subset) is
a present restriction on `reduce` awaiting the same call.

### 10.2 Two things a work division cannot settle

Both are §9.1 escape hatches, and both need the per-region core sets rather
than the division.

**Producer election.** Rows 2–4 return *insufficient information* when a
source region has several holders: R8 requires each consumer tile to have
exactly one source, and the tables record who *holds* a region, not who
*transmits* it. Unforced by
measurement — every source region across the 51 files has a single holder — so
the choice between electing a canonical producer (lowest tile id), requiring
the frontend to pick, and rejecting replicated sources can wait.

**Replication versus idleness.** Row 3's `prod(Nd(a)) == num_cores` is weaker
than asking whether a destination region is shared, and the two part company on
every row-4 output: a region held by several cores is either genuine
replication or one consumer plus idle cores. Both occur in measurement — one
region held by 28 cores with 4 idle, and the broadcast work genuinely
replicating — so the distinction is real. What is open is whether the op
surface should mark it, or whether `consumer_tiles_per_group` naming the actual
holders suffices. This is why §9.1's membership step runs before the
core-count test.

### 10.3 Physicalization: which ops are layout-transparent

Raised by Triton issue #92. **Physicalization** rewrites a tensor to a stick
layout, splitting one axis by the stick size with the chunk count at the front
and the within-stick extent at the back:

```
logical [16, 64], stick on the 64 axis, stick = 32
     →  physical [64/32, 16, 32] = [2, 16, 32]
```

Rank grows by one and the logical stick axis becomes **two non-adjacent
physical axes**. Nothing in this repository represents a stick layout today, so
what follows is a design obligation, not current behaviour.

**Why today's `reduce` is transparent.** It carries no axis-index attribute and
pins results to partials (`KTDP.td:168-171`), so physicalizing the input carries
the result along with no op knowledge — the "elementwise" property. Issue #92's
failure is adjacent: the `identity` operand is tied to results
(`KTDP.td:172-174`) but materialized at logical rank before any layout pass
runs. That is a *propagation* bug, and since the identity is a splat,
re-materializing it at the right type is shape-agnostic by construction.

**The split follows §1.1 exactly**, because §4 makes result type a function of
`placement` alone and `replicate` is the only placement naming no axis set:

| Op | placement | Axis attrs | Result vs partial | Transparent? |
|---|---|---|---|---|
| `consume` | replicate | — | identical | **yes** |
| `reduce` | replicate | — | identical | **yes** |
| `reduce_scatter` | split | `scatter_dimensions` | ÷ `C` | no — attrs, shape, identity |
| `gather` | concat | `gather_dimensions` | × `P` | no — attrs, shape |
| `all_to_all` | permute | `split_`/`concat_dimensions` | ÷ `C` and × `P` | no — attrs, shape |
| `scatter` | split | `scatter_dimensions` | ÷ `C` | no — attrs, shape |

`consume` joins `reduce`. `scatter` does **not**, despite being copy-only: it
divides an extent and names the axis it divides. No rank reduction (§4) is
load-bearing here — a collapse is an axis-*position* operation, so a `reduce`
that collapsed would not be transparent either.

**What §4's rules already settle.** A dim attribute naming a sticked axis
becomes *two* indices (`[1]` → `[0, 2]`), which only the list-valued form can
express, and §4's slowest-to-fastest order is exactly what the stick layout
produces — physical `(c, m, s)` holds logical `n = c*32 + s`. The floordiv rule
fixes which axis absorbs the ×`P` or ÷`C`, and R9 on the floordiv axis is then
precisely the stick-multiple check: `scatter` with `C = 4` on a 2-chunk axis
fails `2 % 4`, correctly rejecting a logical result of `[16,16]` that is half a
stick. `E(D)` itself is invariant (`2 × 32 = 64`), so R9 and R12 cannot change
verdict on the flattened extent — provided a rewrite lists *both* halves of a
split axis; listing one half is simply the wrong rewrite, and R9 catches it.

**Axis indices shift, and physicalization is where that is handled.** The
chunk-count axis is inserted at the *front*, so logical axis 0 of `[16,64]`
becomes physical axis 1: no dim attribute survives untouched, including one
naming an axis physicalization never split. Left unshifted,
`gather_dimensions = [0]` names the chunk axis instead — a valid, distinct
index, so R9/R12 pass and the op is silently wrong.

The remedy needs no new mechanism. Physicalization **is** the logical-to-physical
mapping, so the pass that applies it already knows which logical axis was split,
the stick size, and where every logical axis landed — exactly the information a
dim attribute needs. The attributes name logical axes as authored, and the pass
rewrites them in the same step it retypes the tensors: `[1]` → `[0, 2]` for the
split axis, `[0]` → `[1]` for the shifted one. Nothing downstream re-derives it,
and the ops stay layout-agnostic, which matches §9's framing where axes are
backend symbol names until lowering.

This is not the shape of issue #92. There the `identity` was missed because
`retypeChain` walks forward along operand 0 and never reaches a sibling
operand — an incompleteness in *which values* the pass visits. Attributes sit on
the op the pass is already rewriting, so they are in reach by construction; what
is required is that the mapping be applied to them, not that it be discovered
somewhere else.

**What is still open.**

1. **R12's per-axis clause gains teeth.** Single-axis lists make it trivial;
   `[0,2]` makes it two checks. Stick size depends on element type (32 for f32,
   64 for f16), so variadic roles with mixed types can have equal products and
   unequal per-axis extents — reachable, since §3.7 requires all roles to share
   one axis set.
2. **`reduce_scatter`'s identity.** Its identity must match `T_p` while its
   result is `T_p` split by `C`, so issue #92's fix is needed there in a harder
   form — and the shipped constraint must be retargeted from results to partials
   (R11, §5).
3. **The floordiv rule against a sticked multi-axis pattern**, once one is
   measured. §9.3's three-axis concat has no sticked axis, so it does not test it.

---

## Appendix A. Relationship to what exists today

**Two of the seven ops exist.** `include/ktir/Dialect/KTDP/KTDP.td` defines
`ktdp.inter_tile_produce` and `ktdp.inter_tile_reduce`, plus the
`ktdp.yield_partial` and `ktdp.yield_reduced` terminators. That is all —
the other five delivery ops are new work, not revisions of existing ops.

**Migration table.** One row per op of this design, with its state in
`KTDP.td` today.

| Op | Status today | This design |
|---|---|---|
| `inter_tile_produce` | exists (`KTDP.td:107`) | already matches: carries `producer_tiles_per_group` and no consumer set, returns a future |
| `inter_tile_reduce` | exists (`KTDP.td:165`) | already matches: consumes the future, carries `consumer_tiles_per_group` and a reducer region only |
| `inter_tile_consume` | exists (`KTDP.td:165`) | implemented per §6.1: no region, no identity, results pinned to partials |
| `inter_tile_reduce_scatter` | **not implemented** | new (§6.3) |
| `inter_tile_gather` | **not implemented** | new (§6.4) |
| `inter_tile_all_to_all` | **not implemented** | new (§6.5) |
| `inter_tile_scatter` | **not implemented** | new (§6.6) |

`inter_tile_reduce_scatter` appears in the current tree only as prose:
`KTDPTypes.td:235` lists it among the delivery ops the future type is
*intended* to serve, but it has no op definition. So this document remains a
specification for four new ops.

Cross-checking against §9.3: of those four, `gather`, `all_to_all` and
`scatter` are required by measured relayouts, while `reduce_scatter` is
required only by the reduction patterns of §7.3.

**The `!ktdp.tile_future<T, #groups>` type** already exists
(`KTDPTypes.td:231`) and is shared across all ops; its `#groups` parameter
carries the group set (§1.3).

**The earlier single-op draft.** A `ktdp.inter_tile` op carrying producer
and optional combiner regions in one op, with `consumer_tiles_per_group`
determining the delivery mode, was drafted but never landed. Splitting
production from delivery makes the mode a choice of op rather than an
inference over attribute combinations — which is what lets §3 state the
shared machinery once and §6 reduce each op to its own cells.
