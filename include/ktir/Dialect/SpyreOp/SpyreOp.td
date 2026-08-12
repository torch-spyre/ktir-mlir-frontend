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
}

def Spyre_AddI32ToI32 : SpyreOp_IntrinsicOp<
  "addi32toi32",
  (ins I32:$lhs, I32:$rhs)
> {
  let summary = "";
  let description = [{
    TODO
  }];
}

def Spyre_AddI64ToI64 : SpyreOp_IntrinsicOp<
  "addi64toi64", 
  (ins I64:$lhs, I64:$rhs)> {
  let summary = "";
  let description = [{
    TODO
  }];
}

def Spyre_Exp : SpyreOp_IntrinsicOp<
  "exp", 
  (ins AnyTypeOf<[F32]>:$operand)
> {
  let summary = "";
  let description = [{
    TODO
  }];
}

def Spyre_GeLU : SpyreOp_IntrinsicOp<
  "gelu", 
  (ins AnyTypeOf<[F16, SpyreOp_DF16]>:$operand)
> {
  let summary = "";
  let description = [{
    TODO
  }];
}

def Spyre_Idx32ToAddr : SpyreOp_IntrinsicOp<
  "idx32toaddr", 
  (ins 
    I32:$indices,
    I32:$base_address,
    I32:$stride
  ),
  (outs I32:$result),
  [AllTypesMatch<["indices", "result"]>]
> {
  let summary = "";
  let description = [{
    TODO
  }];
}

def Spyre_LayerNormScale : SpyreOp_IntrinsicOp<
  "layernormscale", 
  (ins AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$operand)
> {
  let summary = "";
  let description = [{
    TODO
  }];
}

def Spyre_MulI32ToI32 : SpyreOp_IntrinsicOp<
  "muli32toi32",
  (ins I32:$operand)
> {
  let summary = "";
  let description = [{
    TODO
  }];
}

def Spyre_RealDiv : SpyreOp_IntrinsicOp<
  "realdiv", 
  (ins AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$operand)
> {
  let summary = "";
  let description = [{
    TODO
  }];
}

def Spyre_Reciprocal : SpyreOp_IntrinsicOp<
  "reciprocal", 
  (ins AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$operand)
> {
  let summary = "";
  let description = [{
    TODO
  }];
}

def Spyre_Sigmoid : SpyreOp_IntrinsicOp<
  "sigmoid", 
  (ins AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$operand)
> {
  let summary = "";
  let description = [{
    TODO
  }];
}

def Spyre_Softplus : SpyreOp_IntrinsicOp<
  "softplus", 
  (ins AnyTypeOf<[F16, SpyreOp_DF16]>:$operand)
> {
  let summary = "";
  let description = [{
    TODO
  }];
}

def Spyre_Sqrt : SpyreOp_IntrinsicOp<
  "sqrt", 
  (ins AnyTypeOf<[F16, SpyreOp_DF16, F32]>:$operand)
> {
  let summary = "";
  let description = [{
    TODO
  }];
}

#endif  // KTIR_DIALECT_SPYRE_SPYRE_TD
