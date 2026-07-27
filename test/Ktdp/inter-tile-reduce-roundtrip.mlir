// RUN: ktir-opt %s | ktir-opt | FileCheck %s



// CHECK: #[[$ATTR_0:.+]] = affine_set<(d0)[s0] : (d0 - s0 * 32 >= 0, -d0 + (s0 + 1) * 32 - 1 >= 0)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0) : (d0 == 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0)[s0] : (d0 - s0 * 4 >= 0, -d0 + s0 * 4 + 3 >= 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0)[s0] : (d0 - s0 * 4 == 0)>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 7 >= 0)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0)[s0, s1] : (d0 + s0 - s1 * 8 - 3 == 0)>

// CHECK-LABEL:   func.func @reduce_single_role(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<1x64xf16>, %[[VAL_1:.*]]: tensor<1x64xf16>) -> tensor<1x64xf16> {
// CHECK-NEXT:     %[[VAL_2:.*]] = ktdp.inter_tile_produce producer_tiles_per_group = #[[$ATTR_0]] -> <(tensor<1x64xf16>), groups = #[[$ATTR_1]]> {
// CHECK-NEXT:     ^bb0(%[[VAL_3:.*]]: index):
// CHECK-NEXT:       ktdp.yield_partial %[[VAL_0]] : tensor<1x64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[VAL_4:.*]] = ktdp.inter_tile_reduce(%[[VAL_2]]) consumer_tiles_per_group = #[[$ATTR_0]], identity(%[[VAL_1]] : tensor<1x64xf16>) : <(tensor<1x64xf16>), groups = #[[$ATTR_1]]> -> tensor<1x64xf16> {
// CHECK-NEXT:     ^bb0(%[[VAL_5:.*]]: tensor<1x64xf16>, %[[VAL_6:.*]]: tensor<1x64xf16>):
// CHECK-NEXT:       %[[VAL_7:.*]] = linalg.add ins(%[[VAL_5]], %[[VAL_6]] : tensor<1x64xf16>, tensor<1x64xf16>) outs(%[[VAL_5]] : tensor<1x64xf16>) -> tensor<1x64xf16>
// CHECK-NEXT:       ktdp.yield_reduced %[[VAL_7]] : tensor<1x64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_4]] : tensor<1x64xf16>
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @reduce_to_one(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<1x64xf16>, %[[VAL_1:.*]]: tensor<1x64xf16>) -> tensor<1x64xf16> {
// CHECK-NEXT:     %[[VAL_2:.*]] = ktdp.inter_tile_produce producer_tiles_per_group = #[[$ATTR_2]] -> <(tensor<1x64xf16>), groups = #[[$ATTR_1]]> {
// CHECK-NEXT:     ^bb0(%[[VAL_3:.*]]: index):
// CHECK-NEXT:       ktdp.yield_partial %[[VAL_0]] : tensor<1x64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[VAL_4:.*]] = ktdp.inter_tile_reduce(%[[VAL_2]]) consumer_tiles_per_group = #[[$ATTR_3]], identity(%[[VAL_1]] : tensor<1x64xf16>) : <(tensor<1x64xf16>), groups = #[[$ATTR_1]]> -> tensor<1x64xf16> {
// CHECK-NEXT:     ^bb0(%[[VAL_5:.*]]: tensor<1x64xf16>, %[[VAL_6:.*]]: tensor<1x64xf16>):
// CHECK-NEXT:       %[[VAL_7:.*]] = linalg.add ins(%[[VAL_5]], %[[VAL_6]] : tensor<1x64xf16>, tensor<1x64xf16>) outs(%[[VAL_5]] : tensor<1x64xf16>) -> tensor<1x64xf16>
// CHECK-NEXT:       ktdp.yield_reduced %[[VAL_7]] : tensor<1x64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_4]] : tensor<1x64xf16>
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @reduce_multi_group(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<128x1x1x64xf16>, %[[VAL_1:.*]]: tensor<128x1x1x64xf16>) -> tensor<128x1x1x64xf16> {
// CHECK-NEXT:     %[[VAL_2:.*]] = ktdp.inter_tile_produce producer_tiles_per_group = #[[$ATTR_2]] -> <(tensor<128x1x1x64xf16>), groups = #[[$ATTR_4]]> {
// CHECK-NEXT:     ^bb0(%[[VAL_3:.*]]: index):
// CHECK-NEXT:       ktdp.yield_partial %[[VAL_0]] : tensor<128x1x1x64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[VAL_4:.*]] = ktdp.inter_tile_reduce(%[[VAL_2]]) consumer_tiles_per_group = #[[$ATTR_2]], identity(%[[VAL_1]] : tensor<128x1x1x64xf16>) : <(tensor<128x1x1x64xf16>), groups = #[[$ATTR_4]]> -> tensor<128x1x1x64xf16> {
// CHECK-NEXT:     ^bb0(%[[VAL_5:.*]]: tensor<128x1x1x64xf16>, %[[VAL_6:.*]]: tensor<128x1x1x64xf16>):
// CHECK-NEXT:       %[[VAL_7:.*]] = linalg.add ins(%[[VAL_5]], %[[VAL_6]] : tensor<128x1x1x64xf16>, tensor<128x1x1x64xf16>) outs(%[[VAL_5]] : tensor<128x1x1x64xf16>) -> tensor<128x1x1x64xf16>
// CHECK-NEXT:       ktdp.yield_reduced %[[VAL_7]] : tensor<128x1x1x64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_4]] : tensor<128x1x1x64xf16>
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @reduce_argmax(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<1x64xf32>, %[[VAL_1:.*]]: tensor<1x64xi32>, %[[VAL_2:.*]]: tensor<1x64xf32>, %[[VAL_3:.*]]: tensor<1x64xi32>) -> (tensor<1x64xf32>, tensor<1x64xi32>) {
// CHECK-NEXT:     %[[VAL_4:.*]] = ktdp.inter_tile_produce producer_tiles_per_group = #[[$ATTR_2]] -> <(tensor<1x64xf32>, tensor<1x64xi32>), groups = #[[$ATTR_4]]> {
// CHECK-NEXT:     ^bb0(%[[VAL_5:.*]]: index):
// CHECK-NEXT:       ktdp.yield_partial %[[VAL_0]], %[[VAL_1]] : tensor<1x64xf32>, tensor<1x64xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[VAL_6:.*]]:2 = ktdp.inter_tile_reduce(%[[VAL_4]]) consumer_tiles_per_group = #[[$ATTR_2]], producer_dependency_per_consumer = #[[$ATTR_5]], identity(%[[VAL_2]], %[[VAL_3]] : tensor<1x64xf32>, tensor<1x64xi32>) : <(tensor<1x64xf32>, tensor<1x64xi32>), groups = #[[$ATTR_4]]> -> tensor<1x64xf32>, tensor<1x64xi32> {
// CHECK-NEXT:     ^bb0(%[[VAL_7:.*]]: tensor<1x64xf32>, %[[VAL_8:.*]]: tensor<1x64xi32>, %[[VAL_9:.*]]: tensor<1x64xf32>, %[[VAL_10:.*]]: tensor<1x64xi32>):
// CHECK-NEXT:       %[[VAL_11:.*]] = arith.cmpf ogt, %[[VAL_7]], %[[VAL_9]] : tensor<1x64xf32>
// CHECK-NEXT:       %[[VAL_12:.*]] = arith.maxnumf %[[VAL_7]], %[[VAL_9]] : tensor<1x64xf32>
// CHECK-NEXT:       %[[VAL_13:.*]] = arith.select %[[VAL_11]], %[[VAL_8]], %[[VAL_10]] : tensor<1x64xi1>, tensor<1x64xi32>
// CHECK-NEXT:       ktdp.yield_reduced %[[VAL_12]], %[[VAL_13]] : tensor<1x64xf32>, tensor<1x64xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_14:.*]]#0, %[[VAL_14]]#1 : tensor<1x64xf32>, tensor<1x64xi32>
// CHECK-NEXT:   }



// CHECK-LABEL:   func.func @reduce_same_rank(
// CHECK-SAME:  %[[VAL_0:.*]]: tensor<1x64xf16>, %[[VAL_1:.*]]: tensor<1x64xf16>) -> tensor<1x64xf16> {
// CHECK-NEXT:     %[[VAL_2:.*]] = ktdp.inter_tile_produce producer_tiles_per_group = #[[$ATTR_0]] -> <(tensor<1x64xf16>), groups = #[[$ATTR_1]]> {
// CHECK-NEXT:     ^bb0(%[[VAL_3:.*]]: index):
// CHECK-NEXT:       ktdp.yield_partial %[[VAL_0]] : tensor<1x64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[VAL_4:.*]] = ktdp.inter_tile_reduce(%[[VAL_2]]) consumer_tiles_per_group = #[[$ATTR_0]], identity(%[[VAL_1]] : tensor<1x64xf16>) : <(tensor<1x64xf16>), groups = #[[$ATTR_1]]> -> tensor<1x64xf16> {
// CHECK-NEXT:     ^bb0(%[[VAL_5:.*]]: tensor<1x64xf16>, %[[VAL_6:.*]]: tensor<1x64xf16>):
// CHECK-NEXT:       %[[VAL_7:.*]] = linalg.add ins(%[[VAL_5]], %[[VAL_6]] : tensor<1x64xf16>, tensor<1x64xf16>) outs(%[[VAL_5]] : tensor<1x64xf16>) -> tensor<1x64xf16>
// CHECK-NEXT:       ktdp.yield_reduced %[[VAL_7]] : tensor<1x64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[VAL_4]] : tensor<1x64xf16>
// CHECK-NEXT:   }




// Inter-tile all-reduce: ktdp.inter_tile_produce + ktdp.inter_tile_reduce.
// See docs/inter-tile-communication.md (§8.2).

// All four functions share one module after parsing, so affine sets are
// deduplicated into a single top-level namespace.


// -----
// Minimal single-role reduce: collapse the leading unit within-group tile axis.
// -----

#group_tiles = affine_set<(i)[g] : (i - 32*g >= 0, -i + 32*(g+1) - 1 >= 0)>
#all_groups  = affine_set<(g) : (g == 0)>

func.func @reduce_single_role(%partial: tensor<1x64xf16>,
                              %add_id: tensor<1x64xf16>) -> tensor<1x64xf16> {
  %f = ktdp.inter_tile_produce
      producer_tiles_per_group = #group_tiles
      -> <(tensor<1x64xf16>), groups = #all_groups>
  {
    ^bb0(%gid: index):
      ktdp.yield_partial %partial : tensor<1x64xf16>
  }
  %r = ktdp.inter_tile_reduce(%f)
      consumer_tiles_per_group = #group_tiles,
      identity(%add_id : tensor<1x64xf16>)
      : <(tensor<1x64xf16>), groups = #all_groups> -> tensor<1x64xf16>
  {
    ^bb0(%lhs: tensor<1x64xf16>, %rhs: tensor<1x64xf16>):
      %s = linalg.add ins(%lhs, %rhs : tensor<1x64xf16>, tensor<1x64xf16>)
                      outs(%lhs : tensor<1x64xf16>) -> tensor<1x64xf16>
      ktdp.yield_reduced %s : tensor<1x64xf16>
  }
  return %r : tensor<1x64xf16>
}

// -----
// Reduce-to-one: 4 producer tiles per group, only tile 4g consumes the result.
// Supported (|C| == 1, C subset of P); see §4.1 / open question Q1.
// -----

#r2o_prod = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#r2o_cons = affine_set<(i)[g] : (i - 4*g == 0)>
#r2o_grp  = affine_set<(g) : (g == 0)>

func.func @reduce_to_one(%partial: tensor<1x64xf16>,
                         %add_id: tensor<1x64xf16>) -> tensor<1x64xf16> {
  %f = ktdp.inter_tile_produce
      producer_tiles_per_group = #r2o_prod
      -> <(tensor<1x64xf16>), groups = #r2o_grp>
  { ^bb0(%gid: index): ktdp.yield_partial %partial : tensor<1x64xf16> }
  %r = ktdp.inter_tile_reduce(%f)
      consumer_tiles_per_group = #r2o_cons,
      identity(%add_id : tensor<1x64xf16>)
      : <(tensor<1x64xf16>), groups = #r2o_grp> -> tensor<1x64xf16>
  { ^bb0(%lhs: tensor<1x64xf16>, %rhs: tensor<1x64xf16>):
      %s = linalg.add ins(%lhs, %rhs : tensor<1x64xf16>, tensor<1x64xf16>)
                      outs(%lhs : tensor<1x64xf16>) -> tensor<1x64xf16>
      ktdp.yield_reduced %s : tensor<1x64xf16> }
  return %r : tensor<1x64xf16>
}

// -----
// Multi-group reduce (collapse interior unit axis, preserve the group axis).
// docs/inter-tile-communication.md §8.2.2.
// -----

#mg_tiles  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#mg_groups = affine_set<(g) : (g >= 0, -g + 7 >= 0)>

func.func @reduce_multi_group(%partial: tensor<128x1x1x64xf16>,
                              %add_id: tensor<128x1x1x64xf16>) -> tensor<128x1x1x64xf16> {
  %f = ktdp.inter_tile_produce
      producer_tiles_per_group = #mg_tiles
      -> <(tensor<128x1x1x64xf16>), groups = #mg_groups>
  {
    ^bb0(%gid: index):
      ktdp.yield_partial %partial : tensor<128x1x1x64xf16>
  }
  %r = ktdp.inter_tile_reduce(%f)
      consumer_tiles_per_group = #mg_tiles,
      identity(%add_id : tensor<128x1x1x64xf16>)
      : <(tensor<128x1x1x64xf16>), groups = #mg_groups> -> tensor<128x1x1x64xf16>
  {
    ^bb0(%lhs: tensor<128x1x1x64xf16>, %rhs: tensor<128x1x1x64xf16>):
      %s = linalg.add ins(%lhs, %rhs : tensor<128x1x1x64xf16>, tensor<128x1x1x64xf16>)
                      outs(%lhs : tensor<128x1x1x64xf16>) -> tensor<128x1x1x64xf16>
      ktdp.yield_reduced %s : tensor<128x1x1x64xf16>
  }
  return %r : tensor<128x1x1x64xf16>
}

// -----
// Multi-role (argmax-style, N=2) reduce with per-tile dependency set.
// -----

#bf_tiles  = affine_set<(i)[g] : (i - 4*g >= 0, -i + 4*g + 3 >= 0)>
#bf_groups = affine_set<(g) : (g >= 0, -g + 7 >= 0)>
#bf_dep    = affine_set<(p)[c, g] : (p + c - 8*g - 3 == 0)>

func.func @reduce_argmax(%pv: tensor<1x64xf32>, %pi: tensor<1x64xi32>,
                         %iv: tensor<1x64xf32>, %ii: tensor<1x64xi32>)
    -> (tensor<1x64xf32>, tensor<1x64xi32>) {
  %f = ktdp.inter_tile_produce
      producer_tiles_per_group = #bf_tiles
      -> <(tensor<1x64xf32>, tensor<1x64xi32>), groups = #bf_groups>
  {
    ^bb0(%gid: index):
      ktdp.yield_partial %pv, %pi : tensor<1x64xf32>, tensor<1x64xi32>
  }
  %rv, %ri = ktdp.inter_tile_reduce(%f)
      consumer_tiles_per_group = #bf_tiles,
      producer_dependency_per_consumer = #bf_dep,
      identity(%iv, %ii : tensor<1x64xf32>, tensor<1x64xi32>)
      : <(tensor<1x64xf32>, tensor<1x64xi32>), groups = #bf_groups>
        -> tensor<1x64xf32>, tensor<1x64xi32>
  {
    ^bb0(%lv: tensor<1x64xf32>, %li: tensor<1x64xi32>,
         %rv2: tensor<1x64xf32>, %ri2: tensor<1x64xi32>):
      // 2-adic argmax combiner: keep the larger value and its index.
      %take_lhs = arith.cmpf ogt, %lv, %rv2 : tensor<1x64xf32>
      %max_v = arith.maxnumf %lv, %rv2 : tensor<1x64xf32>
      %max_i = arith.select %take_lhs, %li, %ri2 : tensor<1x64xi1>, tensor<1x64xi32>
      ktdp.yield_reduced %max_v, %max_i : tensor<1x64xf32>, tensor<1x64xi32>
  }
  return %rv, %ri : tensor<1x64xf32>, tensor<1x64xi32>
}

// -----
// Same-rank result: result type equals the partial type (no rank reduction).
// -----

func.func @reduce_same_rank(%partial: tensor<1x64xf16>,
                             %add_id: tensor<1x64xf16>) -> tensor<1x64xf16> {
  %f = ktdp.inter_tile_produce
      producer_tiles_per_group = #group_tiles
      -> <(tensor<1x64xf16>), groups = #all_groups>
  {
    ^bb0(%gid: index):
      ktdp.yield_partial %partial : tensor<1x64xf16>
  }
  %r = ktdp.inter_tile_reduce(%f)
      consumer_tiles_per_group = #group_tiles,
      identity(%add_id : tensor<1x64xf16>)
      : <(tensor<1x64xf16>), groups = #all_groups> -> tensor<1x64xf16>
  {
    ^bb0(%lhs: tensor<1x64xf16>, %rhs: tensor<1x64xf16>):
      %s = linalg.add ins(%lhs, %rhs : tensor<1x64xf16>, tensor<1x64xf16>)
                      outs(%lhs : tensor<1x64xf16>) -> tensor<1x64xf16>
      ktdp.yield_reduced %s : tensor<1x64xf16>
  }
  return %r : tensor<1x64xf16>
}
