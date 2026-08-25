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
include "ktir/Dialect/SpyreOp/SpyreOpEnums.td"

//===----------------------------------------------------------------------===//
// Attribute Constraints
//===----------------------------------------------------------------------===//

def FloatNonZero : AttrConstraint<
    CPred<"!::llvm::cast<::mlir::FloatAttr>($_self).getValue().isZero()">,
    "which is non-zero">;

//===----------------------------------------------------------------------===//
// Custom Attributes
//===----------------------------------------------------------------------===//

def SpyreOp_ReductionKindAttr : EnumAttr<SpyreOp_Dialect, SpyreOp_ReductionKind, "reduction_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def SpyreOp_ReductionScopeAttr : EnumAttr<SpyreOp_Dialect, SpyreOp_ReductionScope, "reduction_scope"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // KTIR_DIALECT_SPYREOP_SPYREOPATTRS_TD
