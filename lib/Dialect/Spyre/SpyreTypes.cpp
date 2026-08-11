//===-- SpyreTypes.cpp ------------------------------------------*- c++ -*-===//
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

#include "ktir/Dialect/Spyre/SpyreTypes.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace mlir::spyre;

//===----------------------------------------------------------------------===//
// SpyreDialect
//===----------------------------------------------------------------------===//

void SpyreDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ktir/Dialect/Spyre/SpyreTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "ktir/Dialect/Spyre/SpyreTypes.cpp.inc"
