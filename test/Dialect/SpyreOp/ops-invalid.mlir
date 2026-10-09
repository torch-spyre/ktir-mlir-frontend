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
  // expected-error@+1 {{result #0 must be A pair of df16 floats held as one value or A pair of 16-bit floats held as one value or A pair of 32-bit floats held as one value, but got 'f16'}}
  %0 = spyreop.exx2_fused %arg0 : f16 -> f16
  return
}

// -----

func.func @layernormscale_fused_plain_operand(%arg0: f16) {
  // expected-error@+1 {{operand #0 must be A pair of df16 floats held as one value or A pair of 16-bit floats held as one value or A pair of 32-bit floats held as one value, but got 'f16'}}
  %0 = spyreop.layernormscale_fused %arg0 : f16 -> f16
  return
}

// -----

// The result of the fused form is the pair of the operand, not a pair of
// something else.
func.func @exx2_fused_wrong_pair(%arg0: f16) {
  // expected-error@+2 {{inferred type(s) '!spyreop.fp16_fused' are incompatible with return type(s) of operation '!spyreop.fp32_fused'}}
  // expected-error@+1 {{failed to infer returned types}}
  %0 = spyreop.exx2_fused %arg0 : f16 -> !spyreop.fp32_fused
  return
}

// -----

// And what comes out of the fused form is what the operand holds a pair of.
func.func @layernormscale_fused_wrong_scalar(%arg0: !spyreop.fp16_fused) {
  // expected-error@+2 {{inferred type(s) 'f16' are incompatible with return type(s) of operation 'f32'}}
  // expected-error@+1 {{failed to infer returned types}}
  %0 = spyreop.layernormscale_fused %arg0 : !spyreop.fp16_fused -> f32
  return
}

// -----

func.func @constant_integer() {
  // expected-error@+1 {{result #0 must be 16-bit float or 32-bit float, but got 'i32'}}
  %c = spyreop.constant 0 : i32
  return
}

// -----

func.func @constant_index() {
  // expected-error@+1 {{result #0 must be 16-bit float or 32-bit float, but got 'index'}}
  %c = spyreop.constant 0 : index
  return
}

// -----

func.func @constant_double() {
  // expected-error@+1 {{result #0 must be 16-bit float or 32-bit float, but got 'f64'}}
  %c = spyreop.constant 0.0 : f64
  return
}

// -----

func.func @constant_bfloat() {
  // expected-error@+1 {{result #0 must be 16-bit float or 32-bit float, but got 'bf16'}}
  %c = spyreop.constant 0.0 : bf16
  return
}

// -----

func.func @constant_tensor() {
  // expected-error@+1 {{result #0 must be 16-bit float or 32-bit float, but got 'tensor<1xf16>'}}
  %c = spyreop.constant dense<0.0> : tensor<1xf16>
  return
}

// -----

func.func @constant_missing_value() {
  // expected-error@+1 {{requires attribute 'value'}}
  %c = "spyreop.constant"() : () -> f16
  return
}

// -----

func.func @constant_type_mismatch() {
  // expected-error@+1 {{failed to verify that all of {value, result} have same type}}
  %c = "spyreop.constant"() {value = 0.0 : f32} : () -> f16
  return
}

// -----

func.func @constant_unexpected_operand(%arg: f16) {
  // expected-error@+1 {{requires zero operands}}
  %c = "spyreop.constant"(%arg) {value = 0.0 : f16} : (f16) -> f16
  return
}

// -----

func.func @constant_no_result() {
  // expected-error@+1 {{requires one result}}
  "spyreop.constant"() {value = 0.0 : f16} : () -> ()
  return
}

// -----

func.func @constant_two_results() {
  // expected-error@+1 {{requires one result}}
  %c:2 = "spyreop.constant"() {value = 0.0 : f16} : () -> (f16, f16)
  return
}

// -----

func.func @constant_df16_result() {
  // expected-error@+1 {{result #0 must be 16-bit float or 32-bit float, but got '!spyreop.df16'}}
  %c = "spyreop.constant"() {value = 0.0 : f16} : () -> !spyreop.df16
  return
}
