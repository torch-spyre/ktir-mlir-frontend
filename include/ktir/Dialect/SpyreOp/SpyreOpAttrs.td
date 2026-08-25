//===-- SpyreOpAttrs.td ------------------------------------*- tablegen -*-===//
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
// This file defines the attributes in the spyreop dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KTIR_DIALECT_SPYREOP_SPYREOPATTRS_TD
#define KTIR_DIALECT_SPYREOP_SPYREOPATTRS_TD

include "ktir/Dialect/SpyreOp/SpyreOpDialect.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Attribute Constraints
//===----------------------------------------------------------------------===//

def FloatNonZero : AttrConstraint<
    CPred<"!::llvm::cast<::mlir::FloatAttr>($_self).getValue().isZero()">,
    "which is non-zero">;

//===----------------------------------------------------------------------===//
// slice_reduction enums
//===----------------------------------------------------------------------===//

def SpyreOp_ReductionKind_Add    : I32EnumAttrCase<"Add",    0, "add">;
def SpyreOp_ReductionKind_Sub    : I32EnumAttrCase<"Sub",    1, "sub">;
def SpyreOp_ReductionKind_Max    : I32EnumAttrCase<"Max",    2, "max">;
def SpyreOp_ReductionKind_Min    : I32EnumAttrCase<"Min",    3, "min">;
def SpyreOp_ReductionKind_Mul    : I32EnumAttrCase<"Mul",    4, "mul">;
def SpyreOp_ReductionKind_AbsMax : I32EnumAttrCase<"AbsMax", 5, "abs_max">;

def SpyreOp_ReductionKind : I32EnumAttr<
    "ReductionKind",
    "reduction operator",
    [SpyreOp_ReductionKind_Add,
     SpyreOp_ReductionKind_Sub,
     SpyreOp_ReductionKind_Max,
     SpyreOp_ReductionKind_Min,
     SpyreOp_ReductionKind_Mul,
     SpyreOp_ReductionKind_AbsMax]> {
  let cppNamespace = "::mlir::spyreop";
  let genSpecializedAttr = 0;
}

def SpyreOp_ReductionKindAttr : EnumAttr<SpyreOp_Dialect, SpyreOp_ReductionKind, "reduction_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def SpyreOp_ReductionScope_InSlice     : I32EnumAttrCase<"InSlice",     0, "in_slice">;
def SpyreOp_ReductionScope_AcrossSlice : I32EnumAttrCase<"AcrossSlice", 1, "across_slice">;

def SpyreOp_ReductionScope : I32EnumAttr<
    "ReductionScope",
    "reduction scope",
    [SpyreOp_ReductionScope_InSlice,
     SpyreOp_ReductionScope_AcrossSlice]> {
  let cppNamespace = "::mlir::spyreop";
  let genSpecializedAttr = 0;
}

def SpyreOp_ReductionScopeAttr : EnumAttr<SpyreOp_Dialect, SpyreOp_ReductionScope, "reduction_scope"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // KTIR_DIALECT_SPYREOP_SPYREOPATTRS_TD
