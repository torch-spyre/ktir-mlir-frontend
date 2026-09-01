// RUN: ktir-opt "%s" -split-input-file -verify-diagnostics

// Note: -split-input-file parses each chunk independently, so affine-set
// aliases are declared per chunk.

// -----
// Result type must equal the future's partial type: `replicate` placement
// neither grows nor shrinks a partial.

#tile_0    = affine_set<(i)[g] : (i - 4*g == 0)>
#all_4     = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#one_group = affine_set<(g) : (g == 0)>

func.func @bad_result_type(%d: tensor<1x64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #tile_0
      -> <(tensor<1x64xf16>), groups = #one_group>
  { ^bb0(%gid: index): ktdp.yield_partial %d : tensor<1x64xf16> }
  // expected-error @below {{failed to verify that result types must match future partial types}}
  %r = ktdp.inter_tile_consume(%f) consumer_tiles_per_group = #all_4
      : <(tensor<1x64xf16>), groups = #one_group> -> tensor<64xf16>
  return %r : tensor<64xf16>
}

// -----
// Result arity must match the number of roles.

#tile_0    = affine_set<(i)[g] : (i - 4*g == 0)>
#all_4     = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#one_group = affine_set<(g) : (g == 0)>

func.func @bad_result_arity(%v: tensor<128xf32>, %i: tensor<128xi32>)
    -> tensor<128xf32> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #tile_0
      -> <(tensor<128xf32>, tensor<128xi32>), groups = #one_group>
  { ^bb0(%gid: index):
      ktdp.yield_partial %v, %i : tensor<128xf32>, tensor<128xi32> }
  // expected-error @below {{failed to verify that result types must match future partial types}}
  %r = ktdp.inter_tile_consume(%f) consumer_tiles_per_group = #all_4
      : <(tensor<128xf32>, tensor<128xi32>), groups = #one_group>
        -> tensor<128xf32>
  return %r : tensor<128xf32>
}

// -----
// The consumer set's single symbol is the group index.

#tile_0    = affine_set<(i)[g] : (i - 4*g == 0)>
#no_symbol = affine_set<(i) : (i >= 0, -i + 3 >= 0)>
#one_group = affine_set<(g) : (g == 0)>

func.func @bad_consumer_no_symbol(%d: tensor<64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #tile_0
      -> <(tensor<64xf16>), groups = #one_group>
  { ^bb0(%gid: index): ktdp.yield_partial %d : tensor<64xf16> }
  // expected-error @below {{`consumer_tiles_per_group` must have exactly one symbol (the group index g)}}
  %r = ktdp.inter_tile_consume(%f) consumer_tiles_per_group = #no_symbol
      : <(tensor<64xf16>), groups = #one_group> -> tensor<64xf16>
  return %r : tensor<64xf16>
}

// -----
// The consumer set has one dimension, the tile id.

#tile_0    = affine_set<(i)[g] : (i - 4*g == 0)>
#two_dims  = affine_set<(i, j)[g] : (i - 4*g >= 0, j >= 0)>
#one_group = affine_set<(g) : (g == 0)>

func.func @bad_consumer_two_dims(%d: tensor<64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #tile_0
      -> <(tensor<64xf16>), groups = #one_group>
  { ^bb0(%gid: index): ktdp.yield_partial %d : tensor<64xf16> }
  // expected-error @below {{`consumer_tiles_per_group` must have exactly one dimension (the tile id i)}}
  %r = ktdp.inter_tile_consume(%f) consumer_tiles_per_group = #two_dims
      : <(tensor<64xf16>), groups = #one_group> -> tensor<64xf16>
  return %r : tensor<64xf16>
}

// -----
// A dependency set carries one symbol (c) or two (c, g); three is not legal.

#tile_0    = affine_set<(i)[g] : (i - 4*g == 0)>
#all_4     = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#one_group = affine_set<(g) : (g == 0)>
#dep_3sym  = affine_set<(p)[c, g, x] : (p - c == 0, x >= 0)>

func.func @bad_dep_three_symbols(%d: tensor<64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #tile_0
      -> <(tensor<64xf16>), groups = #one_group>
  { ^bb0(%gid: index): ktdp.yield_partial %d : tensor<64xf16> }
  // expected-error @below {{`producer_dependency_per_consumer` must have one symbol (c) or two symbols (c, g)}}
  %r = ktdp.inter_tile_consume(%f) consumer_tiles_per_group = #all_4,
      producer_dependency_per_consumer = #dep_3sym
      : <(tensor<64xf16>), groups = #one_group> -> tensor<64xf16>
  return %r : tensor<64xf16>
}
