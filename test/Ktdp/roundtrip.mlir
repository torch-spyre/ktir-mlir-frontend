// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"

// -----
// Types
// -----

// CHECK-LABEL: func.func @test_access_tile_type
// CHECK-SAME: %{{.*}}: !ktdp.access_tile<32x64xindex>
func.func @test_access_tile_type(%arg0: !ktdp.access_tile<32x64xindex>) -> !ktdp.access_tile<32x64xindex> {
  return %arg0 : !ktdp.access_tile<32x64xindex>
}

// CHECK-LABEL: func.func @test_access_tile_dynamic
// CHECK-SAME: %{{.*}}: !ktdp.access_tile<?x64xindex>
func.func @test_access_tile_dynamic(%arg0: !ktdp.access_tile<?x64xindex>) -> !ktdp.access_tile<?x64xindex> {
  return %arg0 : !ktdp.access_tile<?x64xindex>
}

// CHECK-LABEL: func.func @test_access_tile_scalar
// CHECK-SAME: %{{.*}}: !ktdp.access_tile<index>
func.func @test_access_tile_scalar(%arg0: !ktdp.access_tile<index>) -> !ktdp.access_tile<index> {
  return %arg0 : !ktdp.access_tile<index>
}

// CHECK-LABEL: func.func @test_runtime_arg_basic
// CHECK-SAME: %{{.*}}: !ktdp.runtime_arg<index>
func.func @test_runtime_arg_basic(%arg0: !ktdp.runtime_arg<index>) -> !ktdp.runtime_arg<index> {
  return %arg0 : !ktdp.runtime_arg<index>
}

// CHECK-LABEL: func.func @test_runtime_arg_granularity
// CHECK-SAME: %{{.*}}: !ktdp.runtime_arg<index, granularity=3>
func.func @test_runtime_arg_granularity(%arg0: !ktdp.runtime_arg<index, granularity=3>) -> !ktdp.runtime_arg<index, granularity=3> {
  return %arg0 : !ktdp.runtime_arg<index, granularity=3>
}

// CHECK-LABEL: func.func @test_runtime_arg_full
// CHECK-SAME: %{{.*}}: !ktdp.runtime_arg<i32, granularity=4, upperbound=1024>
func.func @test_runtime_arg_full(%arg0: !ktdp.runtime_arg<i32, granularity=4, upperbound=1024>) -> !ktdp.runtime_arg<i32, granularity=4, upperbound=1024> {
  return %arg0 : !ktdp.runtime_arg<i32, granularity=4, upperbound=1024>
}

// -----
// Attributes
// -----

// CHECK-LABEL: func.func @test_spyre_memory_space_hbm
// CHECK: memory_space = #ktdp.spyre_memory_space<HBM>
#set0 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>
func.func @test_spyre_memory_space_hbm(%addr: index) -> memref<32x64xf16> {
  %view = ktdp.construct_memory_view %addr, sizes: [32, 64], strides: [64, 1] {
    coordinate_set = #set0, memory_space = #ktdp.spyre_memory_space<HBM>
  } : memref<32x64xf16>
  return %view : memref<32x64xf16>
}

// CHECK-LABEL: func.func @test_spyre_memory_space_lx_core
// CHECK: memory_space = #ktdp.spyre_memory_space<LX, core = 7>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>
func.func @test_spyre_memory_space_lx_core(%addr: index) -> memref<32x64xf16> {
  %view = ktdp.construct_memory_view %addr, sizes: [32, 64], strides: [64, 1] {
    coordinate_set = #set1, memory_space = #ktdp.spyre_memory_space<LX, core = 7>
  } : memref<32x64xf16>
  return %view : memref<32x64xf16>
}

// -----
// Operations
// -----

// CHECK-LABEL: func.func @test_get_compute_tile_id
// CHECK: ktdp.get_compute_tile_id : index
func.func @test_get_compute_tile_id() -> index {
  %0 = ktdp.get_compute_tile_id : index
  return %0 : index
}

// CHECK-LABEL: func.func @test_get_compute_tile_id_multi
// CHECK: ktdp.get_compute_tile_id : index, index
func.func @test_get_compute_tile_id_multi() -> (index, index) {
  %0:2 = ktdp.get_compute_tile_id : index, index
  return %0#0, %0#1 : index, index
}

// CHECK-LABEL: func.func @test_construct_memory_view
// CHECK: ktdp.construct_memory_view
// CHECK-SAME: sizes: [32, 64], strides: [64, 1]
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>
func.func @test_construct_memory_view(%addr: index) -> memref<32x64xf16> {
  %view = ktdp.construct_memory_view %addr, sizes: [32, 64], strides: [64, 1] {
    coordinate_set = #set2, memory_space = #ktdp.spyre_memory_space<HBM>
  } : memref<32x64xf16>
  return %view : memref<32x64xf16>
}

// CHECK-LABEL: func.func @test_construct_distributed_memory_view
// CHECK: ktdp.construct_distributed_memory_view
func.func @test_construct_distributed_memory_view(%a: memref<32x64xf16>, %b: memref<32x64xf16>) -> memref<64x64xf16> {
  %dv = ktdp.construct_distributed_memory_view (%a, %b : memref<32x64xf16>, memref<32x64xf16>) : memref<64x64xf16>
  return %dv : memref<64x64xf16>
}

// CHECK-LABEL: func.func @test_construct_access_tile
// CHECK: ktdp.construct_access_tile
// CHECK-SAME: -> !ktdp.access_tile<32x64xindex>
#set3 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @test_construct_access_tile(%view: memref<32x64xf16>, %c0: index) -> !ktdp.access_tile<32x64xindex> {
  %at = ktdp.construct_access_tile %view[%c0, %c0] {
    access_tile_set = #set3,
    access_tile_order = #map0
  } : memref<32x64xf16> -> !ktdp.access_tile<32x64xindex>
  return %at : !ktdp.access_tile<32x64xindex>
}

// CHECK-LABEL: func.func @test_construct_indirect_access_tile
// CHECK: ktdp.construct_indirect_access_tile intermediate_variables
// CHECK-SAME: -> !ktdp.access_tile<2x64xindex>
#set_iat = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#map_iat = affine_map<(d0, d1) -> (d0, d1)>
func.func @test_construct_indirect_access_tile(%base: memref<64x64xf16>, %idx: memref<2x64xi32>) -> !ktdp.access_tile<2x64xindex> {
  %at = ktdp.construct_indirect_access_tile intermediate_variables(%m, %k)
    %base[ind(%idx[%m, %k]), (%k)] {
      variables_space_set = #set_iat,
      variables_space_order = #map_iat
    } : memref<64x64xf16>, memref<2x64xi32> -> !ktdp.access_tile<2x64xindex>
  return %at : !ktdp.access_tile<2x64xindex>
}

// CHECK-LABEL: func.func @test_load
// CHECK: ktdp.load %{{.*}} : <32x64xindex> -> tensor<32x64xf16>
func.func @test_load(%at: !ktdp.access_tile<32x64xindex>) -> tensor<32x64xf16> {
  %data = ktdp.load %at : !ktdp.access_tile<32x64xindex> -> tensor<32x64xf16>
  return %data : tensor<32x64xf16>
}

// CHECK-LABEL: func.func @test_store
// CHECK: ktdp.store %{{.*}}, %{{.*}} : tensor<32x64xf16>, <32x64xindex>
func.func @test_store(%data: tensor<32x64xf16>, %at: !ktdp.access_tile<32x64xindex>) {
  ktdp.store %data, %at : tensor<32x64xf16>, !ktdp.access_tile<32x64xindex>
  return
}

// CHECK-LABEL: func.func @test_runtime_arg_extract_value
// CHECK: ktdp.runtime_arg_extract value from %{{.*}} : <index, granularity=3, upperbound=300> -> index
func.func @test_runtime_arg_extract_value(%arg0: !ktdp.runtime_arg<index, granularity=3, upperbound=300>) -> index {
  %val = ktdp.runtime_arg_extract value from %arg0 : !ktdp.runtime_arg<index, granularity=3, upperbound=300> -> index
  return %val : index
}

// CHECK-LABEL: func.func @test_runtime_arg_extract_granularity
// CHECK: ktdp.runtime_arg_extract granularity from %{{.*}} : <index, granularity=3, upperbound=300> -> index
func.func @test_runtime_arg_extract_granularity(%arg0: !ktdp.runtime_arg<index, granularity=3, upperbound=300>) -> index {
  %gran = ktdp.runtime_arg_extract granularity from %arg0 : !ktdp.runtime_arg<index, granularity=3, upperbound=300> -> index
  return %gran : index
}

// CHECK-LABEL: func.func @test_runtime_arg_extract_upperbound
// CHECK: ktdp.runtime_arg_extract upperbound from %{{.*}} : <index, granularity=3, upperbound=300> -> index
func.func @test_runtime_arg_extract_upperbound(%arg0: !ktdp.runtime_arg<index, granularity=3, upperbound=300>) -> index {
  %ub = ktdp.runtime_arg_extract upperbound from %arg0 : !ktdp.runtime_arg<index, granularity=3, upperbound=300> -> index
  return %ub : index
}

// CHECK-LABEL: func.func @test_runtime_arg_extract_value_i32
// CHECK: ktdp.runtime_arg_extract value from %{{.*}} : <i32, granularity=4, upperbound=1024> -> i32
func.func @test_runtime_arg_extract_value_i32(%arg0: !ktdp.runtime_arg<i32, granularity=4, upperbound=1024>) -> i32 {
  %val = ktdp.runtime_arg_extract value from %arg0 : !ktdp.runtime_arg<i32, granularity=4, upperbound=1024> -> i32
  return %val : i32
}

// CHECK-LABEL: func.func @test_runtime_arg_extract_value_no_optional
// CHECK: ktdp.runtime_arg_extract value from %{{.*}} : <index> -> index
func.func @test_runtime_arg_extract_value_no_optional(%arg0: !ktdp.runtime_arg<index>) -> index {
  %val = ktdp.runtime_arg_extract value from %arg0 : !ktdp.runtime_arg<index> -> index
  return %val : index
}
