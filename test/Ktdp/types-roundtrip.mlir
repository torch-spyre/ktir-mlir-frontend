// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"

// CHECK-LABEL: func.func @access_tile_static
// CHECK-SAME: %arg0: !ktdp.access_tile<4x8xindex>
// CHECK-SAME: -> !ktdp.access_tile<4x8xindex>
func.func @access_tile_static(%arg0: !ktdp.access_tile<4x8xindex>) -> !ktdp.access_tile<4x8xindex> {
  return %arg0 : !ktdp.access_tile<4x8xindex>
}

// CHECK-LABEL: func.func @access_tile_dynamic
// CHECK-SAME: %arg0: !ktdp.access_tile<?x?xindex>
func.func @access_tile_dynamic(%arg0: !ktdp.access_tile<?x?xindex>) -> !ktdp.access_tile<?x?xindex> {
  return %arg0 : !ktdp.access_tile<?x?xindex>
}

// CHECK-LABEL: func.func @access_tile_mixed
// CHECK-SAME: %arg0: !ktdp.access_tile<?x64xindex>
func.func @access_tile_mixed(%arg0: !ktdp.access_tile<?x64xindex>) -> !ktdp.access_tile<?x64xindex> {
  return %arg0 : !ktdp.access_tile<?x64xindex>
}

// CHECK-LABEL: func.func @access_tile_scalar
// CHECK-SAME: %arg0: !ktdp.access_tile<index>
func.func @access_tile_scalar(%arg0: !ktdp.access_tile<index>) -> !ktdp.access_tile<index> {
  return %arg0 : !ktdp.access_tile<index>
}

// CHECK-LABEL: func.func @runtime_arg_basic
// CHECK-SAME: %arg0: !ktdp.runtime_arg<index>
func.func @runtime_arg_basic(%arg0: !ktdp.runtime_arg<index>) -> !ktdp.runtime_arg<index> {
  return %arg0 : !ktdp.runtime_arg<index>
}

// CHECK-LABEL: func.func @runtime_arg_granularity
// CHECK-SAME: %arg0: !ktdp.runtime_arg<index, granularity=3>
func.func @runtime_arg_granularity(%arg0: !ktdp.runtime_arg<index, granularity=3>) -> !ktdp.runtime_arg<index, granularity=3> {
  return %arg0 : !ktdp.runtime_arg<index, granularity=3>
}

// CHECK-LABEL: func.func @runtime_arg_upperbound
// CHECK-SAME: %arg0: !ktdp.runtime_arg<index, upperbound=300>
func.func @runtime_arg_upperbound(%arg0: !ktdp.runtime_arg<index, upperbound=300>) -> !ktdp.runtime_arg<index, upperbound=300> {
  return %arg0 : !ktdp.runtime_arg<index, upperbound=300>
}

// CHECK-LABEL: func.func @runtime_arg_full
// CHECK-SAME: %arg0: !ktdp.runtime_arg<index, granularity=3, upperbound=300>
func.func @runtime_arg_full(%arg0: !ktdp.runtime_arg<index, granularity=3, upperbound=300>) -> !ktdp.runtime_arg<index, granularity=3, upperbound=300> {
  return %arg0 : !ktdp.runtime_arg<index, granularity=3, upperbound=300>
}

// CHECK-LABEL: func.func @runtime_arg_i32
// CHECK-SAME: %arg0: !ktdp.runtime_arg<i32, granularity=4, upperbound=1024>
func.func @runtime_arg_i32(%arg0: !ktdp.runtime_arg<i32, granularity=4, upperbound=1024>) -> !ktdp.runtime_arg<i32, granularity=4, upperbound=1024> {
  return %arg0 : !ktdp.runtime_arg<i32, granularity=4, upperbound=1024>
}

// Verify non-canonical keyword order normalizes to granularity-first.
// CHECK-LABEL: func.func @runtime_arg_reversed_order
// CHECK-SAME: %arg0: !ktdp.runtime_arg<index, granularity=3, upperbound=300>
func.func @runtime_arg_reversed_order(%arg0: !ktdp.runtime_arg<index, upperbound=300, granularity=3>) -> !ktdp.runtime_arg<index, upperbound=300, granularity=3> {
  return %arg0 : !ktdp.runtime_arg<index, upperbound=300, granularity=3>
}
