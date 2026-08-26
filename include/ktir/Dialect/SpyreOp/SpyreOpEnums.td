//===-- SpyreOpEnums.td -------------------------------------*- tablegen -*-==//
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
// This file defines the enumerations in the spyreop dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KTIR_DIALECT_SPYREOP_SPYREOPENUMS_TD
#define KTIR_DIALECT_SPYREOP_SPYREOPENUMS_TD

include "mlir/IR/EnumAttr.td"

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

// In-slice reduction: three FMA variants that differ in which operand is
// cross-lane shuffled before the multiply-and-accumulate.
def SpyreOp_ReductionScope_InSliceFmaSrc0Shuf : I32EnumAttrCase<"InSliceFmaSrc0Shuf", 0, "in_slice_fma_src0_shuf">;
def SpyreOp_ReductionScope_InSliceFmaSrc2Shuf : I32EnumAttrCase<"InSliceFmaSrc2Shuf", 1, "in_slice_fma_src2_shuf">;
def SpyreOp_ReductionScope_InSliceFmaBothShuf : I32EnumAttrCase<"InSliceFmaBothShuf", 2, "in_slice_fma_both_shuf">;

// Across-slice reduction: splat → scan_with_gap → fma16.
def SpyreOp_ReductionScope_AcrossSliceSplat   : I32EnumAttrCase<"AcrossSliceSplat",   3, "across_slice_splat">;
def SpyreOp_ReductionScope_AcrossSliceScanGap : I32EnumAttrCase<"AcrossSliceScanGap", 4, "across_slice_scan_gap">;
def SpyreOp_ReductionScope_AcrossSliceFma16   : I32EnumAttrCase<"AcrossSliceFma16",   5, "across_slice_fma16">;

def SpyreOp_ReductionScope : I32EnumAttr<
    "ReductionScope",
    "reduction step within a slice_reduction sequence",
    [SpyreOp_ReductionScope_InSliceFmaSrc0Shuf,
     SpyreOp_ReductionScope_InSliceFmaSrc2Shuf,
     SpyreOp_ReductionScope_InSliceFmaBothShuf,
     SpyreOp_ReductionScope_AcrossSliceSplat,
     SpyreOp_ReductionScope_AcrossSliceScanGap,
     SpyreOp_ReductionScope_AcrossSliceFma16]> {
  let cppNamespace = "::mlir::spyreop";
  let genSpecializedAttr = 0;
}

#endif  // KTIR_DIALECT_SPYREOP_SPYREOPENUMS_TD
