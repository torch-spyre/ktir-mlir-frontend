// RUN: ktir-opt "%s" -split-input-file -verify-diagnostics

// Note: -split-input-file parses each chunk independently, so affine-set
// aliases are declared per chunk.

// -----
// yield_partial arity must match the future's partial count.
#g  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#ag = affine_set<(g) : (g == 0)>
func.func @bad_produce_arity(%p: tensor<1x64xf16>) {
  // expected-error @below {{yield_partial yields 1 values but the future carries 2 partial type(s)}}
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #g
      -> !ktdp.tile_future<(tensor<1x64xf16>, tensor<1x64xf16>), groups = #ag>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<1x64xf16> }
  return
}

// -----
// identity type must match the partial type.
#g  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#ag = affine_set<(g) : (g == 0)>
func.func @bad_identity_type(%p: tensor<1x64xf16>, %id: tensor<2x64xf16>) -> tensor<1x64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #g
      -> !ktdp.tile_future<(tensor<1x64xf16>), groups = #ag>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<1x64xf16> }
  // expected-error @below {{failed to verify that identity types must match result types}}
  %r = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #g, identity(%id : tensor<2x64xf16>)
      : !ktdp.tile_future<(tensor<1x64xf16>), groups = #ag> -> tensor<1x64xf16>
  { ^bb0(%l: tensor<1x64xf16>, %rr: tensor<1x64xf16>): ktdp.yield_reduced %l : tensor<1x64xf16> }
  return %r : tensor<1x64xf16>
}

// -----
// result type must match the partial type (no rank reduction).
#g  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#ag = affine_set<(g) : (g == 0)>
func.func @bad_result_type(%p: tensor<1x64xf16>, %id: tensor<1x64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #g
      -> !ktdp.tile_future<(tensor<1x64xf16>), groups = #ag>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<1x64xf16> }
  // expected-error @below {{failed to verify that result types must match future partial types}}
  %r = ktdp.inter_tile_reduce(%f) consumer_tiles_per_group = #g, identity(%id : tensor<1x64xf16>)
      : !ktdp.tile_future<(tensor<1x64xf16>), groups = #ag> -> tensor<64xf16>
  { ^bb0(%l: tensor<1x64xf16>, %rr: tensor<1x64xf16>): ktdp.yield_reduced %l : tensor<1x64xf16> }
  return %r : tensor<64xf16>
}



// -----
// tile_future partial types must be ranked tensors.
// expected-error @below {{invalid kind of type specified: expected builtin.tensor, but found 'index'}}
// expected-error @below {{failed to parse KTDP_TileFutureType parameter 'partialTypes'}}
func.func @bad_future_scalar(%a: !ktdp.tile_future<(index), groups = affine_set<(g) : (g == 0)>>) { return }

// -----
// producer tile sets for distinct groups must be disjoint (§2.1).
// tiles 3g..3g+3 over groups 0,1: g=0 -> {0,1,2,3}, g=1 -> {3,4,5,6}; tile 3 overlaps.
#prod_overlap = affine_set<(i)[g] : (i - 3*g >= 0, -i + 3*g + 3 >= 0)>
#two_groups   = affine_set<(g) : (g >= 0, -g + 1 >= 0)>
func.func @bad_producer_overlap(%p: tensor<1x64xf16>) {
  // expected-error @below {{producer_tiles_per_group for groups 0 and 1 are not disjoint}}
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #prod_overlap
      -> !ktdp.tile_future<(tensor<1x64xf16>), groups = #two_groups>
  { ^bb0(%gid: index): ktdp.yield_partial %p : tensor<1x64xf16> }
  return
}

// -----
// tile_future groups must have exactly one dimension (not zero dims).
// expected-error @below {{tile_future `groups` must have exactly one dimension (g)}}
func.func @bad_future_groups_zero_dims(%a: !ktdp.tile_future<(tensor<1x64xf16>), groups = affine_set<() : (0 == 0)>>) { return }

// -----
// tile_future groups must have exactly one dimension (not two dims).
// expected-error @below {{tile_future `groups` must have exactly one dimension (g)}}
func.func @bad_future_groups_two_dims(%a: !ktdp.tile_future<(tensor<1x64xf16>), groups = affine_set<(g, h) : (g == 0, h == 0)>>) { return }

// -----
// tile_future groups must have no symbols.
// expected-error @below {{tile_future `groups` must have no symbols}}
func.func @bad_future_groups_has_symbol(%a: !ktdp.tile_future<(tensor<1x64xf16>), groups = affine_set<(g)[s] : (g == s)>>) { return }
