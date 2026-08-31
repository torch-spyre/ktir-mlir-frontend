//===-- SpyreOp.td -----------------------------------------*- tablegen -*-===//
//
// Copyright 2026 The KTIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations in the spyreop dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KTIR_DIALECT_SPYREOP_SPYREOP_TD
#define KTIR_DIALECT_SPYREOP_SPYREOP_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "ktir/Dialect/SpyreOp/SpyreOpTypes.td"

class SpyreOp_Op<string mnemonic, list<Trait> traits = []> :
    Op<SpyreOp_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Intrinsics
//===----------------------------------------------------------------------===//

// FIXME: Deprecate F16 IEEE-754 floating point type in favor of DF16.

class SpyreOp_IntrinsicOp<
  string mnemonic, 
  dag inputs, 
  dag outputs = (outs !getdagarg<Type>(inputs, 0):$result), 
  list<Trait> traits = [SameOperandsAndResultType]> 
  : SpyreOp_Op<mnemonic, traits # [Pure]> 
{
  let arguments = inputs;  
  let results = outputs;

  let assemblyFormat = "operands attr-dict `:` type($result)";
}

def Spyre_AddI32ToI32 : SpyreOp_IntrinsicOp<
  "addi32toi32",
  (ins I32:$lhs, I32:$rhs)
> {
  let summary = "Adds two 32-bit integers";

  let assemblyFormat = "operands attr-dict";
}

def Spyre_AddI64ToI64 : SpyreOp_IntrinsicOp<
  "addi64toi64", 
  (ins I64:$lhs, I64:$rhs)> {
  let summary = "Adds two 64-bit integers";

  let assemblyFormat = "operands attr-dict";
}

def Spyre_Exp : SpyreOp_IntrinsicOp<
  "exp", 
  (ins AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$operand)
> {
  let summary = "Evaluates the exponential function `exp(x)`";
}

def Spyre_Exx2 : SpyreOp_IntrinsicOp<
  "exx2", 
  (ins AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$operand),
  (outs 
    AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$result,
    AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$squares
  )
> {
  let summary = "Computes a mean and a mean of squares";

  let assemblyFormat = [{
    $operand attr-dict `:` type($result)
  }];
}

def Spyre_Exx2Fused : SpyreOp_IntrinsicOp<
  "exx2_fused", 
  (ins AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$operand),
  (outs AnyTypeOf<[SpyreOp_FP16Fused, SpyreOp_FP32Fused]>:$result),
  []
> {
  let summary = "Computes a mean and a mean of squares fused into one value";

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];
}

def Spyre_GeLU : SpyreOp_IntrinsicOp<
  "gelu", 
  (ins AnyTypeOf<[F16, SpyreOp_DF16]>:$operand)
> {
  let summary = "Evaluates the GeLU function `x * CDF(x)`";
}

def Spyre_Idx32ToAddr : SpyreOp_IntrinsicOp<
  "idx32toaddr", 
  (ins 
    I32:$index,
    I32:$base,
    I32:$stride
  )
> {
  let summary = "Generates an address via `b + (s*i)`";

  let assemblyFormat = [{
    $index `base` $base `stride` $stride attr-dict
  }];
}

def Spyre_LayerNormNorm : SpyreOp_IntrinsicOp<
  "layernormnorm", 
  (ins 
    AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$operand,
    AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$squares,
    AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$scale,
    AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$weight,
    AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$bias
  )
> {
  let summary = "Computes the normalising part of the `LayerNorm` function";

  let assemblyFormat = [{
    $operand `squares` $squares `scale` $scale
    `weight` $weight `bias` $bias attr-dict `:` type($result)
  }];
}

def Spyre_LayerNormScale : SpyreOp_IntrinsicOp<
  "layernormscale", 
  (ins 
    AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$operand,
    AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$squares
  )
> {
  let summary = "Computes the scaling part of the `LayerNorm` function";

  let assemblyFormat = [{
    $operand `squares` $squares attr-dict `:` type($result)
  }];
}

def Spyre_LayerNormScaleFused : SpyreOp_IntrinsicOp<
  "layernormscale_fused", 
  (ins AnyTypeOf<[SpyreOp_FP16Fused, SpyreOp_FP32Fused]>:$operand),
  (outs AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$result),
  []
> {
  let summary = "Computes the scaling part of the `LayerNorm` function from a fused mean and mean of squares";

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];
}

def Spyre_MulI32ToI32 : SpyreOp_IntrinsicOp<
  "muli32toi32",
  (ins I32:$lhs, I32:$rhs)
> {
  let summary = "Multiplies two 32-bit integers into a 32-bit result";

  let assemblyFormat = "operands attr-dict";
}

def Spyre_RealDiv : SpyreOp_IntrinsicOp<
  "realdiv", 
  (ins 
    AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$lhs,
    AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$rhs
  )
> {
  let summary = "Divides two real numbers";
}

def Spyre_Reciprocal : SpyreOp_IntrinsicOp<
  "reciprocal", 
  (ins AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$operand)
> {
  let summary = "Computes the multiplicative inverse";
}

def Spyre_RSqrt : SpyreOp_IntrinsicOp<
  "rsqrt",
  (ins AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$operand)
> {
  let summary = "Computes the reciprocal of the square root";
}

def Spyre_Sigmoid : SpyreOp_IntrinsicOp<
  "sigmoid", 
  (ins AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$operand)
> {
  let summary = "Evaluates the sigmoid function `1 / (1 + exp(-x))`";
}

def Spyre_SiLU : SpyreOp_IntrinsicOp<
  "silu",
  (ins AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$operand)
> {
  let summary = "Evaluates the SiLU function `x * sigmoid(x)`";
}

def Spyre_Softplus : SpyreOp_IntrinsicOp<
  "softplus", 
  (ins 
    AnyTypeOf<[F16, SpyreOp_DF16]>:$operand,
    // To store F16 and DF16 without a DF16Attr, we need an F32Attr.
    ConfinedAttr<F32Attr, [FloatNonZero]>:$beta,
    F32Attr:$threshold
  )
> {
  let summary = "Evaluates the softplus function `1/b * log(1 + b*exp(x))`";

  let assemblyFormat = [{
    $operand `beta` $beta `threshold` $threshold attr-dict `:` type($result)
  }];
}

def Spyre_Sqrt : SpyreOp_IntrinsicOp<
  "sqrt", 
  (ins AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$operand)
> {
  let summary = "Computes the square root";
}

// slice_reduction operates on a full stick at a time.
def Spyre_SliceReduction : SpyreOp_Op<"slice_reduction",
    [Pure, SameOperandsAndResultType]> {
  let summary = "Reduces elements within or across slices";
  let description = [{
    Collects a n-element row and performs a reduction controlled by two
    attributes:

    - `reduction_kind` selects the operator (e.g. `"add"`, `"max"`).
    - `reduction_scope` selects the direction:
      - `"in_slice"`:  reduces within every slice so the first element in every
        slice holds the accumulated result of its slice.
      - `"across_slice"`: reduces across every slice so the elements of the
        first slice hold the accumulated result of counterpart elements in all
        slices.

    The op takes and returns the same n-wide tensor type, matching
    the compute unit register width.  At the post_lowering stage this op is
    replaced by the FMA send/receive sequence defined in the device's
    `patterns ["post_lowering"]` block.
  }];

  let arguments = (ins AnyTypeOf<[AnyRankedTensor]>:$operand,
                       SpyreOp_ReductionKindAttr:$reduction_kind,
                       SpyreOp_ReductionScopeAttr:$reduction_scope);
  let results   = (outs AnyTypeOf<[AnyRankedTensor]>:$result);

  // Operand and result share the same type (enforced by SameOperandsAndResultType).
  let assemblyFormat = "$operand attr-dict `:` type($operand) `->` type($result)";
}

#endif  // KTIR_DIALECT_SPYREOP_SPYREOP_TD
