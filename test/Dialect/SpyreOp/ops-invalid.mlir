// RUN: ktir-opt "%s" -split-input-file -verify-diagnostics

func.func @softplus_beta_zero(%arg0: f16) {
  // expected-error@+1 {{attribute 'beta' failed to satisfy constraint: 32-bit float attribute which is non-zero}}
  spyreop.softplus %arg0 beta 0.0 threshold 0.0 : f16
  return
}

// -----

func.func @layernormnorm_integer(%arg0: i32) {
  // expected-error@+1 {{operand #0 must be 16-bit float or IBM df16 float or 32-bit float, but got 'i32'}}
  spyreop.layernormnorm %arg0 squares %arg0 scale %arg0 weight %arg0 bias %arg0 : i32
  return
}

// -----

func.func @exx2_integer(%arg0: i32) {
  // expected-error@+1 {{operand #0 must be 16-bit float or IBM df16 float or 32-bit float, but got 'i32'}}
  spyreop.exx2 %arg0 : i32
  return
}

// -----

func.func @exx2_fused_plain_result(%arg0: f16) {
  // expected-error@+1 {{result #0 must be A pair of 16-bit floats held as one value or A pair of 32-bit floats held as one value, but got 'f16'}}
  %0 = spyreop.exx2_fused %arg0 : f16 -> f16
  return
}

// -----

func.func @layernormscale_fused_plain_operand(%arg0: f16) {
  // expected-error@+1 {{operand #0 must be A pair of 16-bit floats held as one value or A pair of 32-bit floats held as one value, but got 'f16'}}
  %0 = spyreop.layernormscale_fused %arg0 : f16 -> f16
  return
}
