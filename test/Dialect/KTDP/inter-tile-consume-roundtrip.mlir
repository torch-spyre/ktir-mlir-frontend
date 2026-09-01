// RUN: ktir-opt %s | ktir-opt | FileCheck %s

// CHECK lines generated with mlir/utils/generate-test-checks.py. This file is
// parsed as a single module, so affine-set aliases are declared once and shared
// across the functions below.


#tile_0    = affine_set<(i)[g] : (i - 4*g == 0)>
#all_4     = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#one_group = affine_set<(g) : (g == 0)>
#eight_grp = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

#r_prod = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 1 >= 0)>
#r_cons = affine_set<(i)[g] : (i - 4*g - 2 >= 0, -i + 4*g + 3 >= 0)>
#r_dep  = affine_set<(p)[c] : (p - c + 2 == 0)>

#b_grps = affine_set<(g) : (g >= 0, -g + 1 >= 0)>
#b_dep  = affine_set<(p)[c, g] : (p + c - 8*g - 3 == 0)>

// Broadcast: a single producer per group, so no dependency attribute is
// needed and none is printed.

// CHECK: #[[$ATTR_0:.+]] = affine_set<(d0)[s0] : (d0 - s0 * 4 == 0)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0) : (d0 == 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0)[s0] : (d0 - s0 * 4 >= 0, -d0 + s0 * 4 + 3 >= 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0)[s0] : (d0 - s0 * 4 >= 0, -d0 + s0 * 4 + 1 >= 0)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0)[s0] : (d0 - s0 * 4 - 2 >= 0, -d0 + s0 * 4 + 3 >= 0)>
// CHECK: #[[$ATTR_6:.+]] = affine_set<(d0)[s0] : (d0 - s0 + 2 == 0)>
// CHECK: #[[$ATTR_7:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>
// CHECK: #[[$ATTR_8:.+]] = affine_set<(d0)[s0, s1] : (d0 + s0 - s1 * 8 - 3 == 0)>
// CHECK-LABEL:   func.func @consume_broadcast(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<64x128xf16>) -> tensor<64x128xf16> {
// CHECK:           %[[INTER_TILE_PRODUCE_0:.*]] = ktdp.inter_tile_produce producer_tiles_per_group = #[[$ATTR_0]] -> <(tensor<64x128xf16>), groups = #[[$ATTR_1]]> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: index):
// CHECK:             ktdp.yield_partial %[[ARG0]] : tensor<64x128xf16>
// CHECK:           }
// CHECK:           %[[INTER_TILE_CONSUME_0:.*]] = ktdp.inter_tile_consume(%[[INTER_TILE_PRODUCE_0]]) consumer_tiles_per_group = #[[$ATTR_2]] : <(tensor<64x128xf16>), groups = #[[$ATTR_1]]> -> tensor<64x128xf16>
// CHECK:           return %[[INTER_TILE_CONSUME_0]] : tensor<64x128xf16>
// CHECK:         }
func.func @consume_broadcast(%data: tensor<64x128xf16>) -> tensor<64x128xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #tile_0
      -> <(tensor<64x128xf16>), groups = #one_group>
  { ^bb0(%gid: index): ktdp.yield_partial %data : tensor<64x128xf16> }
  %r = ktdp.inter_tile_consume(%f) consumer_tiles_per_group = #all_4
      : <(tensor<64x128xf16>), groups = #one_group> -> tensor<64x128xf16>
  return %r : tensor<64x128xf16>
}

// Eight groups, one producer each: the group set rides on the future type.

// CHECK-LABEL:   func.func @consume_multi_group(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x64xf16>) -> tensor<1x64xf16> {
// CHECK:           %[[INTER_TILE_PRODUCE_0:.*]] = ktdp.inter_tile_produce producer_tiles_per_group = #[[$ATTR_0]] -> <(tensor<1x64xf16>), groups = #[[$ATTR_3]]> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: index):
// CHECK:             ktdp.yield_partial %[[ARG0]] : tensor<1x64xf16>
// CHECK:           }
// CHECK:           %[[INTER_TILE_CONSUME_0:.*]] = ktdp.inter_tile_consume(%[[INTER_TILE_PRODUCE_0]]) consumer_tiles_per_group = #[[$ATTR_2]] : <(tensor<1x64xf16>), groups = #[[$ATTR_3]]> -> tensor<1x64xf16>
// CHECK:           return %[[INTER_TILE_CONSUME_0]] : tensor<1x64xf16>
// CHECK:         }
func.func @consume_multi_group(%data: tensor<1x64xf16>) -> tensor<1x64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #tile_0
      -> <(tensor<1x64xf16>), groups = #eight_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %data : tensor<1x64xf16> }
  %r = ktdp.inter_tile_consume(%f) consumer_tiles_per_group = #all_4
      : <(tensor<1x64xf16>), groups = #eight_grp> -> tensor<1x64xf16>
  return %r : tensor<1x64xf16>
}

// Two roles: variadic results print as %r:2 and each keeps its partial type.

// CHECK-LABEL:   func.func @consume_two_roles(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<128xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<128xi32>) -> (tensor<128xf32>, tensor<128xi32>) {
// CHECK:           %[[INTER_TILE_PRODUCE_0:.*]] = ktdp.inter_tile_produce producer_tiles_per_group = #[[$ATTR_0]] -> <(tensor<128xf32>, tensor<128xi32>), groups = #[[$ATTR_1]]> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: index):
// CHECK:             ktdp.yield_partial %[[ARG0]], %[[ARG1]] : tensor<128xf32>, tensor<128xi32>
// CHECK:           }
// CHECK:           %[[INTER_TILE_CONSUME_0:.*]]:2 = ktdp.inter_tile_consume(%[[INTER_TILE_PRODUCE_0]]) consumer_tiles_per_group = #[[$ATTR_2]] : <(tensor<128xf32>, tensor<128xi32>), groups = #[[$ATTR_1]]> -> tensor<128xf32>, tensor<128xi32>
// CHECK:           return %[[INTER_TILE_CONSUME_0]]#0, %[[INTER_TILE_CONSUME_0]]#1 : tensor<128xf32>, tensor<128xi32>
// CHECK:         }
func.func @consume_two_roles(%v: tensor<128xf32>, %i: tensor<128xi32>)
    -> (tensor<128xf32>, tensor<128xi32>) {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #tile_0
      -> <(tensor<128xf32>, tensor<128xi32>), groups = #one_group>
  { ^bb0(%gid: index):
      ktdp.yield_partial %v, %i : tensor<128xf32>, tensor<128xi32> }
  %a, %b = ktdp.inter_tile_consume(%f) consumer_tiles_per_group = #all_4
      : <(tensor<128xf32>, tensor<128xi32>), groups = #one_group>
        -> tensor<128xf32>, tensor<128xi32>
  return %a, %b : tensor<128xf32>, tensor<128xi32>
}

// Routing: two producers per group, each consumer paired with exactly one via
// a one-symbol dependency set -- the pairing p = c - 2 does not depend on g.

// CHECK-LABEL:   func.func @consume_routing_one_symbol_dep(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<64xf16>) -> tensor<64xf16> {
// CHECK:           %[[INTER_TILE_PRODUCE_0:.*]] = ktdp.inter_tile_produce producer_tiles_per_group = #[[$ATTR_4]] -> <(tensor<64xf16>), groups = #[[$ATTR_1]]> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: index):
// CHECK:             ktdp.yield_partial %[[ARG0]] : tensor<64xf16>
// CHECK:           }
// CHECK:           %[[INTER_TILE_CONSUME_0:.*]] = ktdp.inter_tile_consume(%[[INTER_TILE_PRODUCE_0]]) consumer_tiles_per_group = #[[$ATTR_5]], producer_dependency_per_consumer = #[[$ATTR_6]] : <(tensor<64xf16>), groups = #[[$ATTR_1]]> -> tensor<64xf16>
// CHECK:           return %[[INTER_TILE_CONSUME_0]] : tensor<64xf16>
// CHECK:         }
func.func @consume_routing_one_symbol_dep(%d: tensor<64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #r_prod
      -> <(tensor<64xf16>), groups = #one_group>
  { ^bb0(%gid: index): ktdp.yield_partial %d : tensor<64xf16> }
  %r = ktdp.inter_tile_consume(%f) consumer_tiles_per_group = #r_cons,
      producer_dependency_per_consumer = #r_dep
      : <(tensor<64xf16>), groups = #one_group> -> tensor<64xf16>
  return %r : tensor<64xf16>
}

// Butterfly mirror exchange: the pairing varies by both consumer and group, so
// the dependency set needs both symbols.

// CHECK-LABEL:   func.func @consume_butterfly_two_symbol_dep(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<64xf16>) -> tensor<64xf16> {
// CHECK:           %[[INTER_TILE_PRODUCE_0:.*]] = ktdp.inter_tile_produce producer_tiles_per_group = #[[$ATTR_2]] -> <(tensor<64xf16>), groups = #[[$ATTR_7]]> {
// CHECK:           ^bb0(%[[VAL_0:.*]]: index):
// CHECK:             ktdp.yield_partial %[[ARG0]] : tensor<64xf16>
// CHECK:           }
// CHECK:           %[[INTER_TILE_CONSUME_0:.*]] = ktdp.inter_tile_consume(%[[INTER_TILE_PRODUCE_0]]) consumer_tiles_per_group = #[[$ATTR_2]], producer_dependency_per_consumer = #[[$ATTR_8]] : <(tensor<64xf16>), groups = #[[$ATTR_7]]> -> tensor<64xf16>
// CHECK:           return %[[INTER_TILE_CONSUME_0]] : tensor<64xf16>
// CHECK:         }
func.func @consume_butterfly_two_symbol_dep(%d: tensor<64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #all_4
      -> <(tensor<64xf16>), groups = #b_grps>
  { ^bb0(%gid: index): ktdp.yield_partial %d : tensor<64xf16> }
  %r = ktdp.inter_tile_consume(%f) consumer_tiles_per_group = #all_4,
      producer_dependency_per_consumer = #b_dep
      : <(tensor<64xf16>), groups = #b_grps> -> tensor<64xf16>
  return %r : tensor<64xf16>
}
