// RUN: ktir-opt --ktir-check-legality --split-input-file --verify-diagnostics %s

// Single-use invariant (§2.3): future must have exactly one use.

#su_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#su_grp   = affine_set<(g) : (g == 0)>

func.func @bad_future_two_uses(%p: tensor<64xf16>, %id: tensor<64xf16>)
    -> (tensor<64xf16>, tensor<64xf16>) {
  // expected-error @below {{future result must have exactly one use}}
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #su_tiles
      -> <(tensor<64xf16>), groups = #su_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<64xf16> }
  %r1 = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #su_tiles,
      identity(%id : tensor<64xf16>)
      : <(tensor<64xf16>), groups = #su_grp> -> tensor<64xf16>
  { ^bb0(%l: tensor<64xf16>, %rr: tensor<64xf16>):
      ktdp.yield_reduced %l : tensor<64xf16> }
  %r2 = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #su_tiles,
      identity(%id : tensor<64xf16>)
      : <(tensor<64xf16>), groups = #su_grp> -> tensor<64xf16>
  { ^bb0(%l: tensor<64xf16>, %rr: tensor<64xf16>):
      ktdp.yield_reduced %l : tensor<64xf16> }
  return %r1, %r2 : tensor<64xf16>, tensor<64xf16>
}

// -----

// C⊆P check: C = {4g+2..4g+5}, P = {4g..4g+3} — tiles 4g+4,4g+5 not in P.

#q1_prod = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#q1_cons = affine_set<(i)[g] : (i - 4*g - 2 >= 0, -i + 4*g + 5 >= 0)>
#q1_grp  = affine_set<(g) : (g == 0)>

func.func @bad_consumer_not_subset(%p: tensor<64xf16>, %id: tensor<64xf16>)
    -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #q1_prod
      -> <(tensor<64xf16>), groups = #q1_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<64xf16> }
  // expected-error @below {{consumer_tiles_per_group for group 0 is not a subset of producer_tiles_per_group}}
  %r = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #q1_cons,
      identity(%id : tensor<64xf16>)
      : <(tensor<64xf16>), groups = #q1_grp> -> tensor<64xf16>
  { ^bb0(%l: tensor<64xf16>, %rr: tensor<64xf16>):
      ktdp.yield_reduced %l : tensor<64xf16> }
  return %r : tensor<64xf16>
}

// -----

// Mode gate: reduce-to-subset (|C|=2, C ⊊ P) unsupported.
// C = {4g, 4g+1}, P = {4g..4g+3}.

#rs_prod = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#rs_cons = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 1 >= 0)>
#rs_grp  = affine_set<(g) : (g == 0)>

func.func @bad_reduce_to_subset(%p: tensor<64xf16>, %id: tensor<64xf16>)
    -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #rs_prod
      -> <(tensor<64xf16>), groups = #rs_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<64xf16> }
  // expected-error @below {{reduce-to-subset is unsupported; only all-reduce and reduce-to-one are supported}}
  %r = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #rs_cons,
      identity(%id : tensor<64xf16>)
      : <(tensor<64xf16>), groups = #rs_grp> -> tensor<64xf16>
  { ^bb0(%l: tensor<64xf16>, %rr: tensor<64xf16>):
      ktdp.yield_reduced %l : tensor<64xf16> }
  return %r : tensor<64xf16>
}

// -----

// A dependency set may carry one symbol (c) or two (c, g); three is not a
// legal spelling.

#ds_prod = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#ds_grp  = affine_set<(g) : (g == 0)>
#ds_bad  = affine_set<(p)[c, g, x] : (p - c == 0, x >= 0)>

func.func @bad_dep_symbol_count(%p: tensor<64xf16>, %id: tensor<64xf16>)
    -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #ds_prod
      -> <(tensor<64xf16>), groups = #ds_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<64xf16> }
  // expected-error @below {{`producer_dependency_per_consumer` must have one symbol (c) or two symbols (c, g)}}
  %r = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #ds_prod,
      producer_dependency_per_consumer = #ds_bad,
      identity(%id : tensor<64xf16>)
      : <(tensor<64xf16>), groups = #ds_grp> -> tensor<64xf16>
  { ^bb0(%l: tensor<64xf16>, %rr: tensor<64xf16>):
      ktdp.yield_reduced %l : tensor<64xf16> }
  return %r : tensor<64xf16>
}

// -----

// R8: several producers per group, so the dependency attribute is required to
// name each consumer tile's single source.

#mp_prod = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#mp_grp  = affine_set<(g) : (g == 0)>

func.func @bad_consume_multi_producer_no_dep(%d: tensor<64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #mp_prod
      -> <(tensor<64xf16>), groups = #mp_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %d : tensor<64xf16> }
  // expected-error @below {{has 4 producers, so producer_dependency_per_consumer is required to name each consumer tile's single source}}
  %r = ktdp.inter_tile_consume(%f) consumer_tiles_per_group = #mp_prod
      : <(tensor<64xf16>), groups = #mp_grp> -> tensor<64xf16>
  return %r : tensor<64xf16>
}

// -----

// R8: a consumer tile depending on two producers has no defined value --
// `replicate` has no combiner and room for one contribution.

#ts_prod = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#ts_grp  = affine_set<(g) : (g == 0)>
#ts_dep  = affine_set<(p)[c] : (p >= 0, -p + 1 >= 0)>

func.func @bad_consume_two_sources(%d: tensor<64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #ts_prod
      -> <(tensor<64xf16>), groups = #ts_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %d : tensor<64xf16> }
  // expected-error @below {{depends on 2 producers; inter_tile_consume delivers one contribution per consumer tile, so exactly one is required}}
  %r = ktdp.inter_tile_consume(%f) consumer_tiles_per_group = #ts_prod,
      producer_dependency_per_consumer = #ts_dep
      : <(tensor<64xf16>), groups = #ts_grp> -> tensor<64xf16>
  return %r : tensor<64xf16>
}

// -----

// R4: a producer serving no consumer is an error, even though every consumer
// has exactly one source.

#ip_prod = affine_set<(i)[g] : (i >= 0, -i + 1 >= 0)>
#ip_cons = affine_set<(i)[g] : (i >= 0, -i + 3 >= 0)>
#ip_grp  = affine_set<(g) : (g == 0)>
#ip_dep  = affine_set<(p)[c] : (p == 0)>

func.func @bad_consume_idle_producer(%d: tensor<64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #ip_prod
      -> <(tensor<64xf16>), groups = #ip_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %d : tensor<64xf16> }
  // expected-error @below {{does not cover producer tile 1 (no consumer has it as a dependency)}}
  %r = ktdp.inter_tile_consume(%f) consumer_tiles_per_group = #ip_cons,
      producer_dependency_per_consumer = #ip_dep
      : <(tensor<64xf16>), groups = #ip_grp> -> tensor<64xf16>
  return %r : tensor<64xf16>
}

// -----

// R2 applies to consume-terminated chains too: the future has exactly one use.

#su_prod = affine_set<(i)[g] : (i - 4*g == 0)>
#su_cons = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#su_grp  = affine_set<(g) : (g == 0)>

func.func @bad_consume_future_two_uses(%d: tensor<64xf16>) -> tensor<64xf16> {
  // expected-error @below {{future result must have exactly one use}}
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #su_prod
      -> <(tensor<64xf16>), groups = #su_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %d : tensor<64xf16> }
  %r1 = ktdp.inter_tile_consume(%f) consumer_tiles_per_group = #su_cons
      : <(tensor<64xf16>), groups = #su_grp> -> tensor<64xf16>
  %r2 = ktdp.inter_tile_consume(%f) consumer_tiles_per_group = #su_cons
      : <(tensor<64xf16>), groups = #su_grp> -> tensor<64xf16>
  return %r1 : tensor<64xf16>
}
