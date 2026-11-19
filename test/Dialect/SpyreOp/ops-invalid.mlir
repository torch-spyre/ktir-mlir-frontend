// RUN: ktir-opt "%s" -split-input-file -verify-diagnostics

func.func @scalar_cast_wrong_widen_direction(%input: f32) -> f16 {
  // expected-error@+1 {{requires f16 -> f32 for dl16tofp32 or f32 -> f16 for fp32todl16}}
  %result = spyreop.numeric_cast %input {kind = "dl16tofp32"} : f32 -> f16
  return %result : f16
}

// -----

func.func @scalar_cast_wrong_narrow_direction(%input: f16) -> f32 {
  // expected-error@+1 {{requires f16 -> f32 for dl16tofp32 or f32 -> f16 for fp32todl16}}
  %result = spyreop.numeric_cast %input {kind = "fp32todl16"} : f16 -> f32
  return %result : f32
}

// -----

func.func @scalar_cast_same_type(%input: f16) -> f16 {
  // expected-error@+1 {{requires f16 -> f32 for dl16tofp32 or f32 -> f16 for fp32todl16}}
  %result = spyreop.numeric_cast %input {kind = "dl16tofp32"} : f16 -> f16
  return %result : f16
}

// -----

func.func @scalar_cast_bf16(%input: bf16) -> f32 {
  // expected-error@+1 {{operand #0 must be 16-bit float or 32-bit float, but got 'bf16'}}
  %result = spyreop.numeric_cast %input {kind = "dl16tofp32"} : bf16 -> f32
  return %result : f32
}

// -----

func.func @scalar_cast_df16(%input: f32) -> !spyreop.df16 {
  // expected-error@+1 {{result #0 must be 16-bit float or 32-bit float, but got '!spyreop.df16'}}
  %result = spyreop.numeric_cast %input {kind = "fp32todl16"} : f32 -> !spyreop.df16
  return %result : !spyreop.df16
}

// -----

func.func @scalar_cast_f64(%input: f16) -> f64 {
  // expected-error@+1 {{result #0 must be 16-bit float or 32-bit float, but got 'f64'}}
  %result = spyreop.numeric_cast %input {kind = "dl16tofp32"} : f16 -> f64
  return %result : f64
}

// -----

func.func @cast_scalar_to_tensor(%input: f16) -> tensor<2x32xf32> {
  // expected-error@+1 {{result #0 must be 16-bit float or 32-bit float, but got 'tensor<2x32xf32>'}}
  %result = spyreop.numeric_cast %input {kind = "dl16tofp32"} : f16 -> tensor<2x32xf32>
  return %result : tensor<2x32xf32>
}

// -----

func.func @cast_tensor_to_scalar(%input: tensor<2x32xf32>) -> f16 {
  // expected-error@+1 {{operand #0 must be 16-bit float or 32-bit float, but got 'tensor<2x32xf32>'}}
  %result = spyreop.numeric_cast %input {kind = "fp32todl16"} : tensor<2x32xf32> -> f16
  return %result : f16
}

// -----

func.func @numeric_cast_kind(%input: f16) -> f32 {
  // expected-error@+1 {{kind must be dl16tofp32 or fp32todl16}}
  %result = spyreop.numeric_cast %input {kind = "unknown"} : f16 -> f32
  return %result : f32
}

// -----

func.func @numeric_cast_tensor(%input: tensor<1x64xf16>) -> tensor<2x32xf32> {
  // expected-error@+1 {{operand #0 must be 16-bit float or 32-bit float, but got 'tensor<1x64xf16>'}}
  %result = spyreop.numeric_cast %input {kind = "dl16tofp32"} : tensor<1x64xf16> -> tensor<2x32xf32>
  return %result : tensor<2x32xf32>
}

// -----

func.func @numeric_cast_vector(%input: vector<64xf16>) -> vector<64xf32> {
  // expected-error@+1 {{operand #0 must be 16-bit float or 32-bit float, but got 'vector<64xf16>'}}
  %result = spyreop.numeric_cast %input {kind = "dl16tofp32"} : vector<64xf16> -> vector<64xf32>
  return %result : vector<64xf32>
}

// -----

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

// The identity's result is its operand, so the two cannot differ in type. The
// custom form names one type, so only the generic form can write the mismatch.
func.func @identity_type_mismatch(%arg0: i32) {
  // expected-error@+1 {{requires the same type for all operands and results}}
  %0 = "spyreop.identity"(%arg0) : (i32) -> f32
  return
}

// -----

func.func @identity_unsupported_type(%arg0: f64) {
  // expected-error@+1 {{operand #0 must be 16-bit float or 32-bit float or 32-bit signless integer, but got 'f64'}}
  %0 = spyreop.identity %arg0 : f64
  return
}
