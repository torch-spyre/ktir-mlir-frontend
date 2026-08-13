//===-- SpyreOpOps.cpp -------------------------------------------*- c++-*-===//
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

// clang-format off
#include "ktir/Dialect/SpyreOp/SpyreOp.h"
// clang-format on

#include <llvm/Support/NativeFormatting.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;
using namespace mlir::spyreop;

namespace {

auto parseHex32Literal(OpAsmParser& parser, IntegerAttr& attr) -> ParseResult {
  uint32_t result;
  if (parser.parseInteger(result)) {
    return failure();
  }

  attr = parser.getBuilder().getI32IntegerAttr(result);
  return success();
}

void printHex32Literal(OpAsmPrinter& printer, Operation* /*op*/,
                       IntegerAttr attr) {
  const auto bits = static_cast<uint32_t>(attr.getValue().getZExtValue());
  llvm::write_hex(printer.getStream(), bits, llvm::HexPrintStyle::PrefixUpper);
}

}  // namespace

//===----------------------------------------------------------------------===//
// SpyreOpDialect
//===----------------------------------------------------------------------===//

void SpyreOpDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "ktir/Dialect/SpyreOp/SpyreOp.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "ktir/Dialect/SpyreOp/SpyreOp.cpp.inc"
