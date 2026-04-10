// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"

// CHECK-LABEL: func.func @memspace_hbm
// CHECK-SAME: memref<64xf32, #ktdp.spyre_memory_space<HBM>>
func.func @memspace_hbm(%arg0: memref<64xf32, #ktdp.spyre_memory_space<HBM>>) -> memref<64xf32, #ktdp.spyre_memory_space<HBM>> {
  return %arg0 : memref<64xf32, #ktdp.spyre_memory_space<HBM>>
}

// CHECK-LABEL: func.func @memspace_lx
// CHECK-SAME: memref<64xf32, #ktdp.spyre_memory_space<LX>>
func.func @memspace_lx(%arg0: memref<64xf32, #ktdp.spyre_memory_space<LX>>) -> memref<64xf32, #ktdp.spyre_memory_space<LX>> {
  return %arg0 : memref<64xf32, #ktdp.spyre_memory_space<LX>>
}

// CHECK-LABEL: func.func @memspace_l0
// CHECK-SAME: memref<64xf32, #ktdp.spyre_memory_space<L0>>
func.func @memspace_l0(%arg0: memref<64xf32, #ktdp.spyre_memory_space<L0>>) -> memref<64xf32, #ktdp.spyre_memory_space<L0>> {
  return %arg0 : memref<64xf32, #ktdp.spyre_memory_space<L0>>
}

// CHECK-LABEL: func.func @memspace_unspecified
// CHECK-SAME: memref<64xf32, #ktdp.spyre_memory_space<unspecified>>
func.func @memspace_unspecified(%arg0: memref<64xf32, #ktdp.spyre_memory_space<unspecified>>) -> memref<64xf32, #ktdp.spyre_memory_space<unspecified>> {
  return %arg0 : memref<64xf32, #ktdp.spyre_memory_space<unspecified>>
}

// CHECK-LABEL: func.func @memspace_lx_core
// CHECK-SAME: memref<64xf32, #ktdp.spyre_memory_space<LX, core = 7>>
func.func @memspace_lx_core(%arg0: memref<64xf32, #ktdp.spyre_memory_space<LX, core = 7>>) -> memref<64xf32, #ktdp.spyre_memory_space<LX, core = 7>> {
  return %arg0 : memref<64xf32, #ktdp.spyre_memory_space<LX, core = 7>>
}

// CHECK-LABEL: func.func @memspace_l0_core
// CHECK-SAME: memref<64xf32, #ktdp.spyre_memory_space<L0, core = 3>>
func.func @memspace_l0_core(%arg0: memref<64xf32, #ktdp.spyre_memory_space<L0, core = 3>>) -> memref<64xf32, #ktdp.spyre_memory_space<L0, core = 3>> {
  return %arg0 : memref<64xf32, #ktdp.spyre_memory_space<L0, core = 3>>
}

// CHECK-LABEL: func.func @memspace_lx_core_zero
// CHECK-SAME: memref<64xf32, #ktdp.spyre_memory_space<LX, core = 0>>
func.func @memspace_lx_core_zero(%arg0: memref<64xf32, #ktdp.spyre_memory_space<LX, core = 0>>) -> memref<64xf32, #ktdp.spyre_memory_space<LX, core = 0>> {
  return %arg0 : memref<64xf32, #ktdp.spyre_memory_space<LX, core = 0>>
}
