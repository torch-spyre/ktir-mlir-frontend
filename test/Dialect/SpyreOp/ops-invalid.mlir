// RUN: ktir-opt %s -split-input-file -verify-diagnostics

func.func @softplus_beta_zero(%arg0: f16) {
  // expected-error@+1 {{attribute 'beta' failed to satisfy constraint: 32-bit float attribute which is non-zero}}
  spyreop.softplus %arg0 beta 0.0 threshold 0.0 : f16
  return
}
