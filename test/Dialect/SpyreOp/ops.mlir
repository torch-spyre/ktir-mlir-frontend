// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"
// RUN: ktir-opt "%s" --canonicalize --cse | FileCheck "%s" --check-prefix=CONSTANT

// CHECK-LABEL: func.func @addi32toi32(
// CHECK-SAME:    %[[A:.*]]: i32, %[[B:.*]]: i32) -> i32
func.func @addi32toi32(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK:         %[[R:.*]] = spyreop.addi32toi32 %[[A]], %[[B]]
  %0 = spyreop.addi32toi32 %arg0, %arg1
  // CHECK:         return %[[R]] : i32
  return %0 : i32
}

// CHECK-LABEL: func.func @addi64toi64(
// CHECK-SAME:    %[[A:.*]]: i64, %[[B:.*]]: i64) -> i64
func.func @addi64toi64(%arg0: i64, %arg1: i64) -> i64 {
  // CHECK:         %[[R:.*]] = spyreop.addi64toi64 %[[A]], %[[B]]
  %0 = spyreop.addi64toi64 %arg0, %arg1
  // CHECK:         return %[[R]] : i64
  return %0 : i64
}

// CHECK-LABEL: func.func @exp(
// CHECK-SAME:    %[[A:.*]]: f32) -> f32
func.func @exp(%arg0: f32) -> f32 {
  // CHECK:         %[[R:.*]] = spyreop.exp %[[A]] : f32
  %0 = spyreop.exp %arg0 : f32
  // CHECK:         return %[[R]] : f32
  return %0 : f32
}

// The mean and the mean of squares come out apart, one result each.
// CHECK-LABEL: func.func @exx2(
// CHECK-SAME:    %[[A:.*]]: !spyreop.df16) -> (!spyreop.df16, !spyreop.df16)
func.func @exx2(%arg0: !spyreop.df16) -> (!spyreop.df16, !spyreop.df16) {
  // CHECK:         %[[M:.*]], %[[SQ:.*]] = spyreop.exx2 %[[A]] : !spyreop.df16
  %0, %1 = spyreop.exx2 %arg0 : !spyreop.df16
  // CHECK:         return %[[M]], %[[SQ]] : !spyreop.df16, !spyreop.df16
  return %0, %1 : !spyreop.df16, !spyreop.df16
}

// Fused, the two come out as one value of a fused type.
// CHECK-LABEL: func.func @exx2_fused(
// CHECK-SAME:    %[[A:.*]]: f16) -> !spyreop.fp16_fused
func.func @exx2_fused(%arg0: f16) -> !spyreop.fp16_fused {
  // CHECK:         %[[R:.*]] = spyreop.exx2_fused %[[A]] : f16 -> !spyreop.fp16_fused
  %0 = spyreop.exx2_fused %arg0 : f16 -> !spyreop.fp16_fused
  // CHECK:         return %[[R]] : !spyreop.fp16_fused
  return %0 : !spyreop.fp16_fused
}

// CHECK-LABEL: func.func @exx2_fused_df16(
// CHECK-SAME:    %[[A:.*]]: !spyreop.df16) -> !spyreop.df16_fused
func.func @exx2_fused_df16(%arg0: !spyreop.df16) -> !spyreop.df16_fused {
  // CHECK:         %[[R:.*]] = spyreop.exx2_fused %[[A]] : !spyreop.df16 -> !spyreop.df16_fused
  %0 = spyreop.exx2_fused %arg0 : !spyreop.df16 -> !spyreop.df16_fused
  // CHECK:         return %[[R]] : !spyreop.df16_fused
  return %0 : !spyreop.df16_fused
}

// CHECK-LABEL: func.func @exx2_fused_f32(
// CHECK-SAME:    %[[A:.*]]: f32) -> !spyreop.fp32_fused
func.func @exx2_fused_f32(%arg0: f32) -> !spyreop.fp32_fused {
  // CHECK:         %[[R:.*]] = spyreop.exx2_fused %[[A]] : f32 -> !spyreop.fp32_fused
  %0 = spyreop.exx2_fused %arg0 : f32 -> !spyreop.fp32_fused
  // CHECK:         return %[[R]] : !spyreop.fp32_fused
  return %0 : !spyreop.fp32_fused
}

// The mean is taken to be zero, so only the mean of squares is accumulated.
// CHECK-LABEL: func.func @exx2_zeromean_fused(
// CHECK-SAME:    %[[A:.*]]: f16) -> !spyreop.fp16_fused
func.func @exx2_zeromean_fused(%arg0: f16) -> !spyreop.fp16_fused {
  // CHECK:         %[[R:.*]] = spyreop.exx2_zeromean_fused %[[A]] : f16 -> !spyreop.fp16_fused
  %0 = spyreop.exx2_zeromean_fused %arg0 : f16 -> !spyreop.fp16_fused
  // CHECK:         return %[[R]] : !spyreop.fp16_fused
  return %0 : !spyreop.fp16_fused
}

// CHECK-LABEL: func.func @exx2_zeromean_fused_f32(
// CHECK-SAME:    %[[A:.*]]: f32) -> !spyreop.fp32_fused
func.func @exx2_zeromean_fused_f32(%arg0: f32) -> !spyreop.fp32_fused {
  // CHECK:         %[[R:.*]] = spyreop.exx2_zeromean_fused %[[A]] : f32 -> !spyreop.fp32_fused
  %0 = spyreop.exx2_zeromean_fused %arg0 : f32 -> !spyreop.fp32_fused
  // CHECK:         return %[[R]] : !spyreop.fp32_fused
  return %0 : !spyreop.fp32_fused
}

// CHECK-LABEL: func.func @gelu(
// CHECK-SAME:    %[[A:.*]]: !spyreop.df16) -> !spyreop.df16
func.func @gelu(%arg0: !spyreop.df16) -> !spyreop.df16 {
  // CHECK:         %[[R:.*]] = spyreop.gelu %[[A]] : !spyreop.df16
  %0 = spyreop.gelu %arg0 : !spyreop.df16
  // CHECK:         return %[[R]] : !spyreop.df16
  return %0 : !spyreop.df16
}

// CHECK-LABEL: func.func @idx32toaddr(
// CHECK-SAME:    %[[I:.*]]: i32, %[[B:.*]]: i32, %[[S:.*]]: i32) -> i32
func.func @idx32toaddr(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
  // CHECK:         %[[R:.*]] = spyreop.idx32toaddr %[[I]] base %[[B]] stride %[[S]]
  %0 = spyreop.idx32toaddr %arg0 base %arg1 stride %arg2
  // CHECK:         return %[[R]] : i32
  return %0 : i32
}

// CHECK-LABEL: func.func @layernormnorm(
// CHECK-SAME:    %[[A:.*]]: !spyreop.df16, %[[SQ:.*]]: !spyreop.df16, %[[SC:.*]]: !spyreop.df16, %[[W:.*]]: !spyreop.df16, %[[B:.*]]: !spyreop.df16) -> !spyreop.df16
func.func @layernormnorm(%arg0: !spyreop.df16, %arg1: !spyreop.df16,
                         %arg2: !spyreop.df16, %arg3: !spyreop.df16,
                         %arg4: !spyreop.df16) -> !spyreop.df16 {
  // CHECK:         %[[R:.*]] = spyreop.layernormnorm %[[A]] squares %[[SQ]] scale %[[SC]] weight %[[W]] bias %[[B]] : !spyreop.df16
  %0 = spyreop.layernormnorm %arg0 squares %arg1 scale %arg2 weight %arg3 bias %arg4 : !spyreop.df16
  // CHECK:         return %[[R]] : !spyreop.df16
  return %0 : !spyreop.df16
}

// The mean and the mean of squares arrive apart, one operand each.
// CHECK-LABEL: func.func @layernormscale(
// CHECK-SAME:    %[[A:.*]]: !spyreop.df16, %[[SQ:.*]]: !spyreop.df16) -> !spyreop.df16
func.func @layernormscale(%arg0: !spyreop.df16, %arg1: !spyreop.df16) -> !spyreop.df16 {
  // CHECK:         %[[R:.*]] = spyreop.layernormscale %[[A]] squares %[[SQ]] : !spyreop.df16
  %0 = spyreop.layernormscale %arg0 squares %arg1 : !spyreop.df16
  // CHECK:         return %[[R]] : !spyreop.df16
  return %0 : !spyreop.df16
}

// Fused, the two arrive as one value of a fused type.
// CHECK-LABEL: func.func @layernormscale_fused(
// CHECK-SAME:    %[[A:.*]]: !spyreop.fp16_fused) -> f16
func.func @layernormscale_fused(%arg0: !spyreop.fp16_fused) -> f16 {
  // CHECK:         %[[R:.*]] = spyreop.layernormscale_fused %[[A]] : !spyreop.fp16_fused -> f16
  %0 = spyreop.layernormscale_fused %arg0 : !spyreop.fp16_fused -> f16
  // CHECK:         return %[[R]] : f16
  return %0 : f16
}

// CHECK-LABEL: func.func @layernormscale_fused_df16(
// CHECK-SAME:    %[[A:.*]]: !spyreop.df16_fused) -> !spyreop.df16
func.func @layernormscale_fused_df16(%arg0: !spyreop.df16_fused) -> !spyreop.df16 {
  // CHECK:         %[[R:.*]] = spyreop.layernormscale_fused %[[A]] : !spyreop.df16_fused -> !spyreop.df16
  %0 = spyreop.layernormscale_fused %arg0 : !spyreop.df16_fused -> !spyreop.df16
  // CHECK:         return %[[R]] : !spyreop.df16
  return %0 : !spyreop.df16
}

// CHECK-LABEL: func.func @layernormscale_fused_f32(
// CHECK-SAME:    %[[A:.*]]: !spyreop.fp32_fused) -> f32
func.func @layernormscale_fused_f32(%arg0: !spyreop.fp32_fused) -> f32 {
  // CHECK:         %[[R:.*]] = spyreop.layernormscale_fused %[[A]] : !spyreop.fp32_fused -> f32
  %0 = spyreop.layernormscale_fused %arg0 : !spyreop.fp32_fused -> f32
  // CHECK:         return %[[R]] : f32
  return %0 : f32
}

// CHECK-LABEL: func.func @muli32toi32(
// CHECK-SAME:    %[[A:.*]]: i32, %[[B:.*]]: i32) -> i32
func.func @muli32toi32(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK:         %[[R:.*]] = spyreop.muli32toi32 %[[A]], %[[B]]
  %0 = spyreop.muli32toi32 %arg0, %arg1
  // CHECK:         return %[[R]] : i32
  return %0 : i32
}

// CHECK-LABEL: func.func @realdiv(
// CHECK-SAME:    %[[A:.*]]: !spyreop.df16, %[[B:.*]]: !spyreop.df16) -> !spyreop.df16
func.func @realdiv(%arg0: !spyreop.df16, %arg1: !spyreop.df16) -> !spyreop.df16 {
  // CHECK:         %[[R:.*]] = spyreop.realdiv %[[A]], %[[B]] : !spyreop.df16
  %0 = spyreop.realdiv %arg0, %arg1 : !spyreop.df16
  // CHECK:         return %[[R]] : !spyreop.df16
  return %0 : !spyreop.df16
}

// CHECK-LABEL: func.func @reciprocal(
// CHECK-SAME:    %[[A:.*]]: !spyreop.df16) -> !spyreop.df16
func.func @reciprocal(%arg0: !spyreop.df16) -> !spyreop.df16 {
  // CHECK:         %[[R:.*]] = spyreop.reciprocal %[[A]] : !spyreop.df16
  %0 = spyreop.reciprocal %arg0 : !spyreop.df16
  // CHECK:         return %[[R]] : !spyreop.df16
  return %0 : !spyreop.df16
}

// CHECK-LABEL: func.func @rsqrt(
// CHECK-SAME:    %[[A:.*]]: !spyreop.df16) -> !spyreop.df16
func.func @rsqrt(%arg0: !spyreop.df16) -> !spyreop.df16 {
  // CHECK:         %[[R:.*]] = spyreop.rsqrt %[[A]] : !spyreop.df16
  %0 = spyreop.rsqrt %arg0 : !spyreop.df16
  // CHECK:         return %[[R]] : !spyreop.df16
  return %0 : !spyreop.df16
}

// CHECK-LABEL: func.func @sigmoid(
// CHECK-SAME:    %[[A:.*]]: !spyreop.df16) -> !spyreop.df16
func.func @sigmoid(%arg0: !spyreop.df16) -> !spyreop.df16 {
  // CHECK:         %[[R:.*]] = spyreop.sigmoid %[[A]] : !spyreop.df16
  %0 = spyreop.sigmoid %arg0 : !spyreop.df16
  // CHECK:         return %[[R]] : !spyreop.df16
  return %0 : !spyreop.df16
}

// CHECK-LABEL: func.func @silu(
// CHECK-SAME:    %[[A:.*]]: !spyreop.df16) -> !spyreop.df16
func.func @silu(%arg0: !spyreop.df16) -> !spyreop.df16 {
  // CHECK:         %[[R:.*]] = spyreop.silu %[[A]] : !spyreop.df16
  %0 = spyreop.silu %arg0 : !spyreop.df16
  // CHECK:         return %[[R]] : !spyreop.df16
  return %0 : !spyreop.df16
}

// CHECK-LABEL: func.func @softplus(
// CHECK-SAME:    %[[A:.*]]: !spyreop.df16) -> !spyreop.df16
func.func @softplus(%arg0: !spyreop.df16) -> !spyreop.df16 {
  // CHECK:         %[[R:.*]] = spyreop.softplus %[[A]] beta 2.000000e-01 threshold 1.000000e-01 : !spyreop.df16
  %0 = spyreop.softplus %arg0 beta 0.2 threshold 0.1 : !spyreop.df16
  // CHECK:         return %[[R]] : !spyreop.df16
  return %0 : !spyreop.df16
}

// CHECK-LABEL: func.func @sqrt(
// CHECK-SAME:    %[[A:.*]]: !spyreop.df16) -> !spyreop.df16
func.func @sqrt(%arg0: !spyreop.df16) -> !spyreop.df16 {
  // CHECK:         %[[R:.*]] = spyreop.sqrt %[[A]] : !spyreop.df16
  %0 = spyreop.sqrt %arg0 : !spyreop.df16
  // CHECK:         return %[[R]] : !spyreop.df16
  return %0 : !spyreop.df16
}

// CHECK-LABEL: func.func @slice_reduction_in_f16(
// CHECK-SAME:    %[[A:.*]]: tensor<64xf16>) -> tensor<64xf16>
func.func @slice_reduction_in_f16(%arg0: tensor<64xf16>) -> tensor<64xf16> {
  // CHECK:         %[[R:.*]] = spyreop.slice_reduction %[[A]] {reduction_kind = #spyreop.reduction_kind<add>, reduction_scope = #spyreop.reduction_scope<in_slice>} : tensor<64xf16> -> tensor<64xf16>
  %0 = spyreop.slice_reduction %arg0 {reduction_kind = #spyreop.reduction_kind<add>, reduction_scope = #spyreop.reduction_scope<in_slice>} : tensor<64xf16> -> tensor<64xf16>
  // CHECK:         return %[[R]] : tensor<64xf16>
  return %0 : tensor<64xf16>
}

// CHECK-LABEL: func.func @slice_reduction_across_f16(
// CHECK-SAME:    %[[A:.*]]: tensor<64xf16>) -> tensor<64xf16>
func.func @slice_reduction_across_f16(%arg0: tensor<64xf16>) -> tensor<64xf16> {
  // CHECK:         %[[R:.*]] = spyreop.slice_reduction %[[A]] {reduction_kind = #spyreop.reduction_kind<max>, reduction_scope = #spyreop.reduction_scope<across_slice>} : tensor<64xf16> -> tensor<64xf16>
  %0 = spyreop.slice_reduction %arg0 {reduction_kind = #spyreop.reduction_kind<max>, reduction_scope = #spyreop.reduction_scope<across_slice>} : tensor<64xf16> -> tensor<64xf16>
  // CHECK:         return %[[R]] : tensor<64xf16>
  return %0 : tensor<64xf16>
}

// CHECK-LABEL: func.func @select(
// CHECK-SAME:    %[[C:.*]]: !spyreop.df16, %[[T:.*]]: !spyreop.df16, %[[F:.*]]: !spyreop.df16) -> !spyreop.df16
func.func @select(%arg0: !spyreop.df16, %arg1: !spyreop.df16,
                  %arg2: !spyreop.df16) -> !spyreop.df16 {
  // CHECK:         %[[R:.*]] = spyreop.select %[[C]], %[[T]], %[[F]] : !spyreop.df16
  %0 = spyreop.select %arg0, %arg1, %arg2 : !spyreop.df16
  // CHECK:         return %[[R]] : !spyreop.df16
  return %0 : !spyreop.df16
}

// One op, the predicate an attribute. A case per predicate, at both widths.

// CHECK-LABEL: func.func @compare_equal(
// CHECK-SAME:    %[[L:.*]]: !spyreop.df16, %[[R0:.*]]: !spyreop.df16) -> !spyreop.df16
func.func @compare_equal(%arg0: !spyreop.df16, %arg1: !spyreop.df16) -> !spyreop.df16 {
  // CHECK:         %[[R:.*]] = spyreop.compare <equal> %[[L]], %[[R0]] : !spyreop.df16
  %0 = spyreop.compare <equal> %arg0, %arg1 : !spyreop.df16
  // CHECK:         return %[[R]] : !spyreop.df16
  return %0 : !spyreop.df16
}

// CHECK-LABEL: func.func @compare_notequal(
// CHECK-SAME:    %[[L:.*]]: f32, %[[R0:.*]]: f32) -> f32
func.func @compare_notequal(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK:         %[[R:.*]] = spyreop.compare <notequal> %[[L]], %[[R0]] : f32
  %0 = spyreop.compare <notequal> %arg0, %arg1 : f32
  // CHECK:         return %[[R]] : f32
  return %0 : f32
}

// CHECK-LABEL: func.func @compare_greaterthan(
// CHECK-SAME:    %[[L:.*]]: !spyreop.df16, %[[R0:.*]]: !spyreop.df16) -> !spyreop.df16
func.func @compare_greaterthan(%arg0: !spyreop.df16, %arg1: !spyreop.df16) -> !spyreop.df16 {
  // CHECK:         %[[R:.*]] = spyreop.compare <greaterthan> %[[L]], %[[R0]] : !spyreop.df16
  %0 = spyreop.compare <greaterthan> %arg0, %arg1 : !spyreop.df16
  // CHECK:         return %[[R]] : !spyreop.df16
  return %0 : !spyreop.df16
}

// CHECK-LABEL: func.func @compare_greaterequal(
// CHECK-SAME:    %[[L:.*]]: f32, %[[R0:.*]]: f32) -> f32
func.func @compare_greaterequal(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK:         %[[R:.*]] = spyreop.compare <greaterequal> %[[L]], %[[R0]] : f32
  %0 = spyreop.compare <greaterequal> %arg0, %arg1 : f32
  // CHECK:         return %[[R]] : f32
  return %0 : f32
}

// CHECK-LABEL: func.func @compare_lesserthan(
// CHECK-SAME:    %[[L:.*]]: !spyreop.df16, %[[R0:.*]]: !spyreop.df16) -> !spyreop.df16
func.func @compare_lesserthan(%arg0: !spyreop.df16, %arg1: !spyreop.df16) -> !spyreop.df16 {
  // CHECK:         %[[R:.*]] = spyreop.compare <lesserthan> %[[L]], %[[R0]] : !spyreop.df16
  %0 = spyreop.compare <lesserthan> %arg0, %arg1 : !spyreop.df16
  // CHECK:         return %[[R]] : !spyreop.df16
  return %0 : !spyreop.df16
}

// CHECK-LABEL: func.func @compare_lesserequal(
// CHECK-SAME:    %[[L:.*]]: f32, %[[R0:.*]]: f32) -> f32
func.func @compare_lesserequal(%arg0: f32, %arg1: f32) -> f32 {
  // CHECK:         %[[R:.*]] = spyreop.compare <lesserequal> %[[L]], %[[R0]] : f32
  %0 = spyreop.compare <lesserequal> %arg0, %arg1 : f32
  // CHECK:         return %[[R]] : f32
  return %0 : f32
}

// Equal ordinary and register-bound values must retain separate representations,
// while equal register-bound values may share one.
// CHECK-LABEL: func.func @constant_storage(
// CHECK: arith.constant 0.000000e+00 : f16
// CHECK: spyreop.constant 0.000000e+00 : f16
// CHECK: spyreop.constant 0.000000e+00 : f16
// CONSTANT-LABEL: func.func @constant_storage(
// CONSTANT: %[[PLAIN:.*]] = arith.constant 0.000000e+00 : f16
// CONSTANT: %[[REG:.*]] = spyreop.constant 0.000000e+00 : f16
// CONSTANT-NOT: spyreop.constant
// CONSTANT: return %[[PLAIN]], %[[REG]], %[[REG]]
func.func @constant_storage() -> (f16, f16, f16) {
  %plain = arith.constant 0.0 : f16
  %first = spyreop.constant 0.0 : f16
  %same = spyreop.constant 0.0 : f16
  return %plain, %first, %same : f16, f16, f16
}

// Signed zero and nonzero literals keep their exact value at both widths.
// CHECK-LABEL: func.func @constant_values(
// CHECK: spyreop.constant -0.000000e+00 : f16
// CHECK: spyreop.constant 1.500000e+00 : f16
// CHECK: spyreop.constant -0.000000e+00 : f32
// CHECK: spyreop.constant -1.500000e+00 : f32
// CONSTANT-LABEL: func.func @constant_values(
// CONSTANT: spyreop.constant -0.000000e+00 : f16
// CONSTANT: spyreop.constant 1.500000e+00 : f16
// CONSTANT: spyreop.constant -0.000000e+00 : f32
// CONSTANT: spyreop.constant -1.500000e+00 : f32
func.func @constant_values() -> (f16, f16, f32, f32) {
  %negative_zero = spyreop.constant -0.0 : f16
  %half = spyreop.constant 1.5 : f16
  %wide_zero = spyreop.constant -0.0 : f32
  %wide = spyreop.constant -1.5 : f32
  return %negative_zero, %half, %wide_zero, %wide : f16, f16, f32, f32
}

// A value with no consumers has no register requirement to preserve.
// CHECK-LABEL: func.func @unused_constant(
// CONSTANT-LABEL: func.func @unused_constant(
// CONSTANT-NEXT: return
func.func @unused_constant() {
  %unused = spyreop.constant 0.0 : f32
  return
}

// Infinity and NaN payloads are representation tests, not device math promises.
// CHECK-LABEL: func.func @constant_special_values(
// CHECK: spyreop.constant 0x7C00 : f16
// CHECK: spyreop.constant 0x7E01 : f16
// CHECK: spyreop.constant 0x7F800000 : f32
// CHECK: spyreop.constant 0x7FC00001 : f32
// CONSTANT-LABEL: func.func @constant_special_values(
// CONSTANT: spyreop.constant 0x7C00 : f16
// CONSTANT: spyreop.constant 0x7E01 : f16
// CONSTANT: spyreop.constant 0x7F800000 : f32
// CONSTANT: spyreop.constant 0x7FC00001 : f32
func.func @constant_special_values() -> (f16, f16, f32, f32) {
  %half_inf = spyreop.constant 0x7C00 : f16
  %half_nan = spyreop.constant 0x7E01 : f16
  %float_inf = spyreop.constant 0x7F800000 : f32
  %float_nan = spyreop.constant 0x7FC00001 : f32
  return %half_inf, %half_nan, %float_inf, %float_nan : f16, f16, f32, f32
}
