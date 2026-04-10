// RUN: ktir-opt "%s" -split-input-file -verify-diagnostics

// -----

// Verify: load shape mismatch
func.func @bad_load_shape(%at: !ktdp.access_tile<32x64xindex>) -> tensor<16x64xf16> {
  // expected-error @below {{'ktdp.load' op access tile shape must match result tensor shape}}
  %data = ktdp.load %at : !ktdp.access_tile<32x64xindex> -> tensor<16x64xf16>
  return %data : tensor<16x64xf16>
}

// -----

// Verify: store shape mismatch
func.func @bad_store_shape(%data: tensor<16x64xf16>, %at: !ktdp.access_tile<32x64xindex>) {
  // expected-error @below {{'ktdp.store' op data tile shape must match access tile shape}}
  ktdp.store %data, %at : tensor<16x64xf16>, !ktdp.access_tile<32x64xindex>
  return
}

// -----

// Verify: distributed memory view element type mismatch
func.func @bad_distributed_elem_type(%a: memref<32x64xf16>, %b: memref<32x64xf32>) -> memref<64x64xf16> {
  // expected-error @below {{'ktdp.construct_distributed_memory_view' op all input memrefs must have the same element type as the result}}
  %dv = ktdp.construct_distributed_memory_view (%a, %b : memref<32x64xf16>, memref<32x64xf32>) : memref<64x64xf16>
  return %dv : memref<64x64xf16>
}

// -----

// Verify: distributed memory view rank mismatch
func.func @bad_distributed_rank(%a: memref<32x64xf16>, %b: memref<64xf16>) -> memref<64x64xf16> {
  // expected-error @below {{'ktdp.construct_distributed_memory_view' op all input memrefs must have the same rank as the result}}
  %dv = ktdp.construct_distributed_memory_view (%a, %b : memref<32x64xf16>, memref<64xf16>) : memref<64x64xf16>
  return %dv : memref<64x64xf16>
}

// -----

// Verify: SpyreMemorySpaceAttr core affinity on HBM (invalid)
#set_bad = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>
func.func @bad_core_affinity_hbm(%addr: index) -> memref<32x64xf16> {
  %view = ktdp.construct_memory_view %addr, sizes: [32, 64], strides: [64, 1] {
    // expected-error @+1 {{core affinity is only valid for LX or L0 memory spaces}}
    coordinate_set = #set_bad, memory_space = #ktdp.spyre_memory_space<HBM, core = 3>
  } : memref<32x64xf16>
  return %view : memref<32x64xf16>
}

// -----

// Verify: construct_memory_view rank mismatch
#set_rank = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>
func.func @bad_memory_view_rank(%addr: index) -> memref<32x64x8xf16> {
  // expected-error @below {{'ktdp.construct_memory_view' op result memref rank does not match sizes/strides length}}
  %view = ktdp.construct_memory_view %addr, sizes: [32, 64], strides: [64, 1] {
    coordinate_set = #set_rank, memory_space = #ktdp.spyre_memory_space<HBM>
  } : memref<32x64x8xf16>
  return %view : memref<32x64x8xf16>
}

// -----

// Verify: access_tile element type must be index
// expected-error @below {{tile element type must be 'index'}}
func.func @bad_access_tile_elem(%arg0: !ktdp.access_tile<32x64xf16>) -> !ktdp.access_tile<32x64xf16> {
  return %arg0 : !ktdp.access_tile<32x64xf16>
}

// -----

// Verify: runtime_arg granularity must be positive
// expected-error @below {{runtime_arg granularity must be positive}}
func.func @bad_runtime_arg_granularity(%arg0: !ktdp.runtime_arg<index, granularity=0>) -> !ktdp.runtime_arg<index, granularity=0> {
  return %arg0 : !ktdp.runtime_arg<index, granularity=0>
}

// -----

// Verify: runtime_arg upperbound must be positive
// expected-error @below {{runtime_arg upperbound must be positive}}
func.func @bad_runtime_arg_upperbound(%arg0: !ktdp.runtime_arg<index, upperbound=-1>) -> !ktdp.runtime_arg<index, upperbound=-1> {
  return %arg0 : !ktdp.runtime_arg<index, upperbound=-1>
}

// -----

// Verify: runtime_arg_extract value type mismatch
func.func @bad_extract_value_type(%arg0: !ktdp.runtime_arg<i32, granularity=4, upperbound=1024>) -> index {
  // expected-error @below {{'ktdp.runtime_arg_extract' op result type (}}
  %val = ktdp.runtime_arg_extract value from %arg0 : !ktdp.runtime_arg<i32, granularity=4, upperbound=1024> -> index
  return %val : index
}

// -----

// Verify: runtime_arg_extract granularity from runtime_arg without granularity
func.func @bad_extract_granularity_missing(%arg0: !ktdp.runtime_arg<index>) -> index {
  // expected-error @below {{'ktdp.runtime_arg_extract' op cannot extract granularity from runtime_arg that doesn't have one}}
  %gran = ktdp.runtime_arg_extract granularity from %arg0 : !ktdp.runtime_arg<index> -> index
  return %gran : index
}

// -----

// Verify: runtime_arg_extract upperbound from runtime_arg without upperbound
func.func @bad_extract_upperbound_missing(%arg0: !ktdp.runtime_arg<index>) -> index {
  // expected-error @below {{'ktdp.runtime_arg_extract' op cannot extract upperbound from runtime_arg that doesn't have one}}
  %ub = ktdp.runtime_arg_extract upperbound from %arg0 : !ktdp.runtime_arg<index> -> index
  return %ub : index
}

// -----

// Verify: runtime_arg_extract granularity result must be index
func.func @bad_extract_granularity_type(%arg0: !ktdp.runtime_arg<index, granularity=3>) -> i32 {
  // expected-error @below {{'ktdp.runtime_arg_extract' op result type must be index when extracting granularity}}
  %gran = ktdp.runtime_arg_extract granularity from %arg0 : !ktdp.runtime_arg<index, granularity=3> -> i32
  return %gran : i32
}

// -----

// Verify: runtime_arg_extract upperbound result must be index
func.func @bad_extract_upperbound_type(%arg0: !ktdp.runtime_arg<index, granularity=3, upperbound=300>) -> i32 {
  // expected-error @below {{'ktdp.runtime_arg_extract' op result type must be index when extracting upperbound}}
  %ub = ktdp.runtime_arg_extract upperbound from %arg0 : !ktdp.runtime_arg<index, granularity=3, upperbound=300> -> i32
  return %ub : i32
}
