// RUN: ktir-opt "%s" -split-input-file -verify-diagnostics

// Core affinity on HBM should fail.
func.func @memspace_hbm_core_invalid(%arg0: memref<64xf32,
    // expected-error @+1 {{core affinity is only valid for LX or L0 memory spaces, but got memory space 'HBM' with core = 2}}
    #ktdp.spyre_memory_space<HBM, core = 2>>) {
  return
}

// -----

// Core affinity on unspecified should fail.
func.func @memspace_unspecified_core_invalid(%arg0: memref<64xf32,
    // expected-error @+1 {{core affinity is only valid for LX or L0 memory spaces, but got memory space 'unspecified' with core = 0}}
    #ktdp.spyre_memory_space<unspecified, core = 0>>) {
  return
}
