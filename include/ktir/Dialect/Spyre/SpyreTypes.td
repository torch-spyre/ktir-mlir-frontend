//===-- SpyreTypes.td --------------------------------------*- tablegen -*-===//
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
// This file defines the types in the spyre dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KTIR_DIALECT_SPYRE_SPYRETYPES_TD
#define KTIR_DIALECT_SPYRE_SPYRETYPES_TD

include "ktir/Dialect/Spyre/SpyreAttrs.td"

include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class Spyre_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Spyre_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Spyre_DF16 : Spyre_Type<"DF16", "df16", [
  VectorElementTypeInterface, 
  MemRefElementTypeInterface
]> {
  let summary = "";
  let description = [{
    TODO
  }];
}

#endif // KTIR_DIALECT_SPYRE_SPYRETYPES_TD
