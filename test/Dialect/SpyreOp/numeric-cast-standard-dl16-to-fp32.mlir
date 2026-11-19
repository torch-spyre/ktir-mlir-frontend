// SPDX-License-Identifier: Apache-2.0
// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s" --implicit-check-not=spyreop.identity
// One core: floor -> Standard DL16 [3,2,64] to Standard FP32 [6,2,32] -> exp.
// DL16 is represented by f16. Both memory views have ordinary dense strides.
// The cast produces Staggered FP32; exp reads it and writes Standard FP32.
// Access tiles use identity order and perform no element relayout.

// CHECK-DAG: #[[IDENTITY:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[STAGGERED:.*]] = affine_map<(d0, d1, d2) -> (d0 * 2 + (d2 mod 8) floordiv 4, d1, (d2 floordiv 8) * 4 + d2 mod 4)>
// CHECK-DAG: #[[RELAYOUT:.*]] = affine_map<(d0, d1, d2) -> ((d0 floordiv 2) * 2 + (d2 mod 8) floordiv 4, d1, (d0 mod 2) * 16 + (d2 floordiv 8) * 4 + d2 mod 4)>
// CHECK-LABEL: func.func @standard_dl16_to_standard_fp32(
// CHECK-SAME: %[[SRC_ADDR:.*]]: index, %[[DST_ADDR:.*]]: index)
// CHECK-SAME: grid = [1]
// CHECK: %[[SRC:.*]] = ktdp.construct_memory_view %[[SRC_ADDR]], sizes: [3, 2, 64], strides: [128, 64, 1]
// CHECK-SAME: memory_space = #ktdp.memory_space<global>
// CHECK-SAME: memref<3x2x64xf16>
// CHECK: %[[DST:.*]] = ktdp.construct_memory_view %[[DST_ADDR]], sizes: [6, 2, 32], strides: [64, 32, 1]
// CHECK-SAME: memory_space = #ktdp.memory_space<global>
// CHECK-SAME: memref<6x2x32xf32>
// CHECK: %[[IN_TILE:.*]] = ktdp.construct_access_tile %[[SRC]][0, 0, 0] {access_tile_order = #[[IDENTITY]],
// CHECK: %[[INPUT:.*]] = ktdp.load %[[IN_TILE]] {{.*}} -> tensor<3x2x64xf16>
// CHECK: %[[FLOOR:.*]] = linalg.generic {indexing_maps = [#[[IDENTITY]], #[[IDENTITY]]]
// CHECK-SAME: ins(%[[INPUT]] : tensor<3x2x64xf16>)
// CHECK: ^bb0(%[[FLOOR_ARG:.*]]: f16, %{{.*}}: f16):
// CHECK: %[[FLOOR_VALUE:.*]] = math.floor %[[FLOOR_ARG]] : f16
// CHECK: linalg.yield %[[FLOOR_VALUE]] : f16
// CHECK: %[[STAGGERED_VALUE:.*]] = linalg.generic {indexing_maps = [#[[IDENTITY]], #[[STAGGERED]]], iterator_types = ["parallel", "parallel", "parallel"]}
// CHECK-SAME: ins(%[[FLOOR]] : tensor<3x2x64xf16>)
// CHECK-SAME: outs(%{{.*}} : tensor<6x2x32xf32>)
// CHECK-NEXT: ^bb0(%[[CAST_ARG:.*]]: f16, %{{.*}}: f32):
// CHECK-NEXT: %[[CAST_VALUE:.*]] = spyreop.numeric_cast %[[CAST_ARG]] {kind = "dl16tofp32"} : f16 -> f32
// CHECK-NEXT: linalg.yield %[[CAST_VALUE]] : f32
// CHECK: %[[EXP:.*]] = linalg.generic {indexing_maps = [#[[RELAYOUT]], #[[IDENTITY]]]
// CHECK-SAME: ins(%[[STAGGERED_VALUE]] : tensor<6x2x32xf32>)
// CHECK: ^bb0(%[[EXP_ARG:.*]]: f32, %{{.*}}: f32):
// CHECK: %[[EXP_VALUE:.*]] = spyreop.exp %[[EXP_ARG]] : f32
// CHECK: linalg.yield %[[EXP_VALUE]] : f32
// CHECK: %[[OUT_TILE:.*]] = ktdp.construct_access_tile %[[DST]][0, 0, 0] {access_tile_order = #[[IDENTITY]],
// CHECK: ktdp.store %[[EXP]], %[[OUT_TILE]] : tensor<6x2x32xf32>

#dl16_domain = affine_set<(g, r, l) :
    (g >= 0, 2 - g >= 0, r >= 0, 1 - r >= 0, l >= 0, 63 - l >= 0)>
#fp32_domain = affine_set<(k, r, e) :
    (k >= 0, 5 - k >= 0, r >= 0, 1 - r >= 0, e >= 0, 31 - e >= 0)>
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// S: Standard DL16 iteration coordinates -> Staggered FP32 coordinates.
// g: 64-element group (0..2); r: row (0..1); l: DL16 lane (0..63).
// Each group of eight DL16 lanes contributes four lanes to each packet.
#dl16_to_staggered_fp32 = affine_map<(g, r, l) -> (
    g * 2 + (l mod 8) floordiv 4, r, (l floordiv 8) * 4 + l mod 4)>

// M: Standard FP32 (k,r,e) -> Staggered FP32 coordinates of the same value.
// k: packet (0..5); e: FP32 lane (0..31); r: row (0..1).
#standard_to_staggered_fp32 = affine_map<(k, r, e) -> (
    (k floordiv 2) * 2 + (e mod 8) floordiv 4,
    r, (k mod 2) * 16 + (e floordiv 8) * 4 + e mod 4)>

func.func @standard_dl16_to_standard_fp32(
    %src_addr: index, %dst_addr: index) attributes {grid = [1]} {
  %src = ktdp.construct_memory_view %src_addr,
      sizes: [3, 2, 64], strides: [128, 64, 1] {
    coordinate_set = #dl16_domain, memory_space = #ktdp.memory_space<global>
  } : memref<3x2x64xf16>
  %dst = ktdp.construct_memory_view %dst_addr,
      sizes: [6, 2, 32], strides: [64, 32, 1] {
    coordinate_set = #fp32_domain, memory_space = #ktdp.memory_space<global>
  } : memref<6x2x32xf32>

  %input_tile = ktdp.construct_access_tile %src[0, 0, 0] {
    access_tile_set = #dl16_domain, access_tile_order = #identity
  } : memref<3x2x64xf16> -> !ktdp.access_tile<3x2x64xindex>
  %input = ktdp.load %input_tile
      : !ktdp.access_tile<3x2x64xindex> -> tensor<3x2x64xf16>

  %floor_init = tensor.empty() : tensor<3x2x64xf16>
  %rounded = linalg.generic {
    indexing_maps = [#identity, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%input : tensor<3x2x64xf16>) outs(%floor_init : tensor<3x2x64xf16>) {
  ^bb0(%value: f16, %unused: f16):
    %floor = math.floor %value : f16
    linalg.yield %floor : f16
  } -> tensor<3x2x64xf16>
  // staggered[S(g,r,l)] = cast(rounded[g,r,l]).
  %init = tensor.empty() : tensor<6x2x32xf32>
  %staggered = linalg.generic {
    indexing_maps = [#identity, #dl16_to_staggered_fp32],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%rounded : tensor<3x2x64xf16>) outs(%init : tensor<6x2x32xf32>) {
  ^bb0(%value: f16, %unused: f32):
    %cast = spyreop.numeric_cast %value {kind = "dl16tofp32"} : f16 -> f32
    linalg.yield %cast : f32
  } -> tensor<6x2x32xf32>

  %exp_init = tensor.empty() : tensor<6x2x32xf32>
  // output[k,r,e] = exp(staggered[M(k,r,e)]): relayout is explicit in exp.
  %output = linalg.generic {
    indexing_maps = [#standard_to_staggered_fp32, #identity],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%staggered : tensor<6x2x32xf32>) outs(%exp_init : tensor<6x2x32xf32>) {
  ^bb0(%value: f32, %unused: f32):
    %exp = spyreop.exp %value : f32
    linalg.yield %exp : f32
  } -> tensor<6x2x32xf32>

  %output_tile = ktdp.construct_access_tile %dst[0, 0, 0] {
    access_tile_set = #fp32_domain, access_tile_order = #identity
  } : memref<6x2x32xf32> -> !ktdp.access_tile<6x2x32xindex>
  ktdp.store %output, %output_tile
      : tensor<6x2x32xf32>, !ktdp.access_tile<6x2x32xindex>
  return
}
