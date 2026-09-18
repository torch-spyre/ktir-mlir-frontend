//===-- SpyreOpEnums.td ------------------------------------*- tablegen -*-===//
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
// Wrapper that pulls in all the spyreop dialect enums for Python bindings.
//
// This must be the file where the enum ATTRIBUTES are defined, per the contract
// on GEN_ENUM_BINDINGS_TD_FILE: an op's enum operand takes the builder generated
// from the `EnumAttr`, not the one generated from the `I32EnumAttr`. Including
// only the enums leaves the caller hand-writing `#spyreop.…<keyword>`.
//
// It must not be the ops file either. Enums carry no dialect, so `-bind-dialect`
// cannot filter them, and a wider include reaches every enum in the closure. The
// attributes file keeps the scope to this dialect: it includes SpyreOpEnums.td
// and nothing else.
//
//===----------------------------------------------------------------------===//

include "ktir/Dialect/SpyreOp/SpyreOpAttrs.td"
