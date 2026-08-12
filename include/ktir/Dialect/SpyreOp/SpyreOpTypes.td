//===-- SpyreOpTypes.td ------------------------------------*- tablegen -*-===//
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
// This file defines the types in the spyreop dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KTIR_DIALECT_SPYREOP_SPYREOPTYPES_TD
#define KTIR_DIALECT_SPYREOP_SPYREOPTYPES_TD

include "ktir/Dialect/SpyreOp/SpyreOpAttrs.td"

include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class SpyreOp_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SpyreOp_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def SpyreOp_DF16 : SpyreOp_Type<"DF16", "df16", [
  VectorElementTypeInterface,
  MemRefElementTypeInterface
]> {
  let summary = "IBM df16 float";
  let description = [{
    The `df16` type represents floating-point numbers based on the IEEE-754
    binary format with the parameters 1-6-9, i.e.:

    ```
      |s||eeeeee||mmmmmmmmm|
       |       |          ^ 9-bit mantissa
       |       ^ 6-bit exponent
       ^ sign bit
    ```
  }];
}

#endif // KTIR_DIALECT_SPYRE_SPYRETYPES_TD
