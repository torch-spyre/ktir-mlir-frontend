// RUN: ktir-opt "%s" | ktir-opt | FileCheck "%s"

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

// CHECK-LABEL: func.func @gelu(
// CHECK-SAME:    %[[A:.*]]: !spyreop.df16) -> !spyreop.df16
func.func @gelu(%arg0: !spyreop.df16) -> !spyreop.df16 {
  // CHECK:         %[[R:.*]] = spyreop.gelu %[[A]] : !spyreop.df16
  %0 = spyreop.gelu %arg0 : !spyreop.df16
  // CHECK:         return %[[R]] : !spyreop.df16
  return %0 : !spyreop.df16
}

// CHECK-LABEL: func.func @idx32toaddr(
// CHECK-SAME:    %[[A:.*]]: i32) -> i32
func.func @idx32toaddr(%arg0: i32) -> i32 {
  // CHECK:         %[[R:.*]] = spyreop.idx32toaddr %[[A]] base 0xDEADBEEF stride 4
  %0 = spyreop.idx32toaddr %arg0 base 0xDEADBEEF stride 4
  // CHECK:         return %[[R]] : i32
  return %0 : i32
}

// CHECK-LABEL: func.func @layernormscale(
// CHECK-SAME:    %[[A:.*]]: !spyreop.df16) -> !spyreop.df16
func.func @layernormscale(%arg0: !spyreop.df16) -> !spyreop.df16 {
  // CHECK:         %[[R:.*]] = spyreop.layernormscale %[[A]] : !spyreop.df16
  %0 = spyreop.layernormscale %arg0 : !spyreop.df16
  // CHECK:         return %[[R]] : !spyreop.df16
  return %0 : !spyreop.df16
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
