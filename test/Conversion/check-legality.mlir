// RUN: ktir-opt --ktir-check-legality %s | FileCheck %s

#group_tiles = affine_set<(i)[g] : (i - 32*g >= 0, -i + 32*(g+1) - 1 >= 0)>
#all_groups  = affine_set<(g) : (g == 0)>

// CHECK-LABEL: func.func @reduce_single_role
func.func @reduce_single_role(%partial: tensor<64xf16>,
                              %add_id: tensor<64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce
      producer_tiles_per_group = #group_tiles
      -> <(tensor<64xf16>), groups = #all_groups>
  { ^bb0(%gid: index): ktdp.yield_partial %partial : tensor<64xf16> }
  %r = ktdp.inter_tile_reduce(%f)
      consumer_tiles_per_group = #group_tiles,
      identity(%add_id : tensor<64xf16>)
      : <(tensor<64xf16>), groups = #all_groups> -> tensor<64xf16>
  { ^bb0(%lhs: tensor<64xf16>, %rhs: tensor<64xf16>):
      %s = linalg.add ins(%lhs, %rhs : tensor<64xf16>, tensor<64xf16>)
                      outs(%lhs : tensor<64xf16>) -> tensor<64xf16>
      ktdp.yield_reduced %s : tensor<64xf16> }
  return %r : tensor<64xf16>
}

// A group-independent producer/consumer pairing may be spelled with a single
// symbol, `(p)[c]`, instead of the general `(p)[c, g]` (§3.4). Here P == C
// (all-reduce) and each consumer depends on exactly one producer, p == c.

#dep1_tiles = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#dep1_grp   = affine_set<(g) : (g >= 0, -g + 7 >= 0)>
#dep1_pair  = affine_set<(p)[c] : (p - c == 0)>

// CHECK-LABEL: func.func @reduce_dep_one_symbol
func.func @reduce_dep_one_symbol(%partial: tensor<64xf16>,
                                 %add_id: tensor<64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce
      producer_tiles_per_group = #dep1_tiles
      -> <(tensor<64xf16>), groups = #dep1_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %partial : tensor<64xf16> }
  // CHECK: producer_dependency_per_consumer
  %r = ktdp.inter_tile_reduce(%f)
      consumer_tiles_per_group = #dep1_tiles,
      producer_dependency_per_consumer = #dep1_pair,
      identity(%add_id : tensor<64xf16>)
      : <(tensor<64xf16>), groups = #dep1_grp> -> tensor<64xf16>
  { ^bb0(%lhs: tensor<64xf16>, %rhs: tensor<64xf16>):
      %s = linalg.add ins(%lhs, %rhs : tensor<64xf16>, tensor<64xf16>)
                      outs(%lhs : tensor<64xf16>) -> tensor<64xf16>
      ktdp.yield_reduced %s : tensor<64xf16> }
  return %r : tensor<64xf16>
}

// Broadcast: one producer per group, so R8 holds with no dependency attribute.

#bc_tile_0 = affine_set<(i)[g] : (i - 4*g == 0)>
#bc_all    = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#bc_grp    = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

// CHECK-LABEL: func.func @consume_broadcast
func.func @consume_broadcast(%data: tensor<64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #bc_tile_0
      -> <(tensor<64xf16>), groups = #bc_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %data : tensor<64xf16> }
  // CHECK: ktdp.inter_tile_consume
  %r = ktdp.inter_tile_consume(%f) consumer_tiles_per_group = #bc_all
      : <(tensor<64xf16>), groups = #bc_grp> -> tensor<64xf16>
  return %r : tensor<64xf16>
}

// Routing: two producers per group, each consumer paired with exactly one.
// A producer may serve several consumer tiles, so long as none serves nobody.

#rt_prod = affine_set<(i)[g] : (i >= 0, -i + 1 >= 0)>
#rt_cons = affine_set<(i)[g] : (i >= 0, -i + 3 >= 0)>
#rt_grp  = affine_set<(g) : (g == 0)>
#rt_dep  = affine_set<(p)[c] : (c - 2*p >= 0, -c + 2*p + 1 >= 0)>

// CHECK-LABEL: func.func @consume_routing_multicast
func.func @consume_routing_multicast(%data: tensor<64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #rt_prod
      -> <(tensor<64xf16>), groups = #rt_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %data : tensor<64xf16> }
  // CHECK: ktdp.inter_tile_consume
  %r = ktdp.inter_tile_consume(%f) consumer_tiles_per_group = #rt_cons,
      producer_dependency_per_consumer = #rt_dep
      : <(tensor<64xf16>), groups = #rt_grp> -> tensor<64xf16>
  return %r : tensor<64xf16>
}

// A dependency set with no static upper bound on the producer tile cannot be
// enumerated, so coverage (R4) is deferred rather than reported against a
// producer some unreadable consumer may claim. Nothing is diagnosed here.

#ub_prod = affine_set<(i)[g] : (i >= 0, -i + 1 >= 0)>
#ub_grp  = affine_set<(g) : (g == 0)>
#ub_dep  = affine_set<(p)[c] : (p - c >= 0)>

// CHECK-LABEL: func.func @consume_unbounded_dep_defers_coverage
func.func @consume_unbounded_dep_defers_coverage(%d: tensor<64xf16>) -> tensor<64xf16> {
  %f = ktdp.inter_tile_produce producer_tiles_per_group = #ub_prod
      -> <(tensor<64xf16>), groups = #ub_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %d : tensor<64xf16> }
  // CHECK: ktdp.inter_tile_consume
  %r = ktdp.inter_tile_consume(%f) consumer_tiles_per_group = #ub_prod,
      producer_dependency_per_consumer = #ub_dep
      : <(tensor<64xf16>), groups = #ub_grp> -> tensor<64xf16>
  return %r : tensor<64xf16>
}
