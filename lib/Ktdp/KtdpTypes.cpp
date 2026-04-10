//===- KtdpTypes.cpp - KTDP dialect type implementations ------------------===//
//
// Implements:
//   AccessTileType::parse / ::print / ::verify
//   RuntimeArgType::parse / ::print / ::verify
//
//===----------------------------------------------------------------------===//

#include "Ktdp/KtdpTypes.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"

using namespace mlir;
using namespace mlir::ktdp;

//===----------------------------------------------------------------------===//
// AccessTileType
//===----------------------------------------------------------------------===//

LogicalResult AccessTileType::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape,
    Type elementType) {
  if (!elementType.isIndex())
    return emitError() << "tile element type must be 'index', but got: "
                       << elementType;

  for (int64_t dim : shape) {
    if (!ShapedType::isDynamic(dim) && dim < 0)
      return emitError() << "tile dimension size must be dynamic or >= 0";
  }

  return success();
}

void AccessTileType::print(AsmPrinter &printer) const {
  printer << "<";
  printer.printDimensionList(getShape());
  if (!getShape().empty()) printer << "x";
  printer << getElementType();
  printer << ">";
}

Type AccessTileType::parse(AsmParser &parser) {
  if (parser.parseLess()) return Type();

  SmallVector<int64_t, 4> shape;
  Type elementType;

  if (parser.parseDimensionList(shape, /*allowDynamic=*/true,
                                /*withTrailingX=*/true))
    return Type();

  if (parser.parseType(elementType)) return Type();

  if (!elementType.isIndex()) {
    parser.emitError(parser.getCurrentLocation(),
                     "tile element type must be 'index', but got: ")
        << elementType;
    return Type();
  }

  if (parser.parseGreater()) return Type();

  return AccessTileType::getChecked(
      [&] { return parser.emitError(parser.getCurrentLocation()); }, shape,
      elementType);
}

//===----------------------------------------------------------------------===//
// RuntimeArgType
//===----------------------------------------------------------------------===//

LogicalResult RuntimeArgType::verify(
    function_ref<InFlightDiagnostic()> emitError, Type underlyingType,
    std::optional<int64_t> granularity, std::optional<int64_t> upperbound) {
  if (!mlir::isa<IndexType, IntegerType, FloatType>(underlyingType)) {
    return emitError() << "runtime_arg underlying type must be a builtin type "
                          "(index, integer, or float), but got: "
                       << underlyingType;
  }

  if (granularity.has_value() && granularity.value() <= 0) {
    return emitError() << "runtime_arg granularity must be positive, but got: "
                       << granularity.value();
  }

  if (upperbound.has_value() && upperbound.value() <= 0) {
    return emitError() << "runtime_arg upperbound must be positive, but got: "
                       << upperbound.value();
  }

  return success();
}

void RuntimeArgType::print(AsmPrinter &printer) const {
  printer << "<" << getUnderlyingType();

  if (getGranularity().has_value()) {
    printer << ", granularity=" << getGranularity().value();
  }

  if (getUpperbound().has_value()) {
    printer << ", upperbound=" << getUpperbound().value();
  }

  printer << ">";
}

Type RuntimeArgType::parse(AsmParser &parser) {
  if (parser.parseLess()) return Type();

  Type underlyingType;
  std::optional<int64_t> granularity;
  std::optional<int64_t> upperbound;

  if (parser.parseType(underlyingType)) return Type();

  while (succeeded(parser.parseOptionalComma())) {
    if (succeeded(parser.parseOptionalKeyword("granularity"))) {
      if (granularity.has_value()) {
        parser.emitError(parser.getCurrentLocation(),
                         "duplicate 'granularity' keyword");
        return Type();
      }
      if (parser.parseEqual()) return Type();
      int64_t value;
      if (parser.parseInteger(value)) return Type();
      granularity = value;
    } else if (succeeded(parser.parseOptionalKeyword("upperbound"))) {
      if (upperbound.has_value()) {
        parser.emitError(parser.getCurrentLocation(),
                         "duplicate 'upperbound' keyword");
        return Type();
      }
      if (parser.parseEqual()) return Type();
      int64_t value;
      if (parser.parseInteger(value)) return Type();
      upperbound = value;
    } else {
      parser.emitError(parser.getCurrentLocation(),
                       "expected 'granularity' or 'upperbound' keyword");
      return Type();
    }
  }

  if (parser.parseGreater()) return Type();

  return RuntimeArgType::getChecked(
      [&] { return parser.emitError(parser.getCurrentLocation()); },
      underlyingType, granularity, upperbound);
}
