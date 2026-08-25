//===-- SpyreOpDialect.td ----------------------------------*- tablegen -*-===//
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
// This file defines the spyreop dialect for IBM Spyre intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef KTIR_DIALECT_SPYREOP_SPYREOPDIALECT_TD
#define KTIR_DIALECT_SPYREOP_SPYREOPDIALECT_TD

include "mlir/IR/OpBase.td"

def SpyreOp_Dialect : Dialect {
  let name = "spyreop";
  let cppNamespace = "::mlir::spyreop";
  let summary = "IBM Spyre Intrinsics Dialect";
  let description = [{
    The `spyreop` dialect exposes a set of attributes, types and operations that
    map to intrinsics available to users of the IBM Spyre device.
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
   private:
    void registerAttributes();
    void registerOps();
    void registerTypes();
  }];
}

#endif // KTIR_DIALECT_SPYREOP_SPYREOPDIALECT_TD
