//===----------------------------------------------------------------------===//
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
// This file defines the operations in the KTDP operation set.
//
//===----------------------------------------------------------------------===//
#include "Ktdp/KtdpOps.hpp"

#include "Ktdp/KtdpTypes.hpp"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

using namespace mlir;
using namespace mlir::ktdp;

//===----------------------------------------------------------------------===//
// ConstructAccessTilesOp
//===----------------------------------------------------------------------===//

void ConstructAccessTilesOp::build(OpBuilder& builder, OperationState& result,
                                   AccessTileType result_type, Value base,
                                   AffineMap base_map, ValueRange indices,
                                   ValueRange symbol_operands,
                                   IntegerSet access_tile_set,
                                   AffineMap access_tile_order) {
  assert(access_tile_set.getNumDims() == base_map.getNumInputs() &&
         "inconsistent dimensions size in base_map and access_tile_set");
  assert(access_tile_set.getNumSymbols() == symbol_operands.size() &&
         "number of symbol operands must match symbols in access_tile_set");

  result.addOperands(base);
  result.addOperands(indices);
  result.addOperands(symbol_operands);

  result.addAttribute(getOperandSegmentSizesAttrName(result.name),
                      builder.getDenseI32ArrayAttr(
                          {1, static_cast<int32_t>(indices.size()),
                           static_cast<int32_t>(symbol_operands.size())}));

  result.addAttribute(getAccessTileSetAttrStrName(),
                      IntegerSetAttr::get(access_tile_set));
  result.addAttribute(getBaseMapAttrStrName(), AffineMapAttr::get(base_map));
  result.addAttribute(getAccessTileOrderAttrStrName(),
                      AffineMapAttr::get(access_tile_order));
  result.types.push_back(result_type);
}

// Overload without symbol_operands
void ConstructAccessTilesOp::build(OpBuilder& builder, OperationState& result,
                                   AccessTileType result_type, Value base,
                                   AffineMap base_map, ValueRange indices,
                                   IntegerSet access_tile_set,
                                   AffineMap access_tile_order) {
  build(builder, result, result_type, base, base_map, indices,
        /*symbol_operands=*/ValueRange{}, access_tile_set, access_tile_order);
}

::mlir::ParseResult ConstructAccessTilesOp::parse(OpAsmParser& parser,
                                                  OperationState& result) {
  auto& builder = parser.getBuilder();
  auto index_type = builder.getIndexType();

  Type base_type;
  AccessTileType result_type;
  OpAsmParser::UnresolvedOperand base;
  AffineMapAttr map_attr;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> map_operands;
  auto op_result =
      parser.parseOperand(base) ||
      parser.parseAffineMapOfSSAIds(
          map_operands, map_attr,
          ConstructAccessTilesOp::getBaseMapAttrStrName(), result.attributes);

  // Parse optional symbol operands with symbols(...) syntax
  SmallVector<OpAsmParser::UnresolvedOperand> symbol_operands;
  if (parser.parseOptionalKeyword("symbols").succeeded()) {
    if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, [&]() {
          symbol_operands.emplace_back();
          return parser.parseOperand(symbol_operands.back());
        })) {
      return parser.emitError(parser.getNameLoc())
             << "Failed to parse symbol operands";
    }
  }

  op_result =
      op_result || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(base_type) || parser.parseArrow() ||
      parser.parseType(result_type) ||
      parser.resolveOperand(base, base_type, result.operands) ||
      parser.resolveOperands(map_operands, index_type, result.operands) ||
      parser.resolveOperands(symbol_operands, index_type, result.operands) ||
      parser.addTypeToList(result_type, result.types);

  std::optional<NamedAttribute> access_tile_set_attr =
      result.attributes.getNamed(
          ConstructAccessTilesOp::getAccessTileSetAttrStrName());
  if (!access_tile_set_attr.has_value()) {
    return parser.emitError(parser.getNameLoc())
           << "Access tile set is missing in the operation";
  }

  auto access_tile_set =
      mlir::dyn_cast<IntegerSetAttr>(access_tile_set_attr->getValue())
          .getValue();

  if (access_tile_set.getNumDims() != map_attr.getValue().getNumResults()) {
    return parser.emitError(parser.getNameLoc())
           << "Access tile set and Base input dimensions should match";
  }

  // Verify symbol operands count matches access_tile_set symbols
  if (access_tile_set.getNumSymbols() != symbol_operands.size()) {
    return parser.emitError(parser.getNameLoc())
           << "access_tile_set has " << access_tile_set.getNumSymbols()
           << " symbols but got " << symbol_operands.size()
           << " symbol operands";
  }

  std::optional<NamedAttribute> access_tile_order_attr =
      result.attributes.getNamed(
          ConstructAccessTilesOp::getAccessTileOrderAttrStrName());
  if (!access_tile_order_attr.has_value()) {
    return parser.emitError(parser.getNameLoc())
           << "Access tile order is missing in the operation";
  }
  auto access_tile_order =
      mlir::dyn_cast<AffineMapAttr>(access_tile_order_attr->getValue())
          .getValue();
  if (access_tile_order.getNumInputs() != access_tile_order.getNumResults()) {
    return parser.emitError(parser.getNameLoc())
           << "Number of input dimensions and output dimensions in the "
              "access tile order should match";
  }
  if (access_tile_order.getNumResults() != access_tile_set.getNumDims()) {
    return parser.emitError(parser.getNameLoc())
           << "Number of dimensions in the access tile order and access tile "
              "set must match!";
  }

  // Add operand segment sizes attribute for AttrSizedOperandSegments
  result.addAttribute("operandSegmentSizes",
                      builder.getDenseI32ArrayAttr(
                          {1, static_cast<int32_t>(map_operands.size()),
                           static_cast<int32_t>(symbol_operands.size())}));

  return failure(op_result);
}

void ConstructAccessTilesOp::print(::mlir::OpAsmPrinter& p) {
  auto& op = *this;
  p << ' ' << op.getBase() << '[';
  if (AffineMapAttr map_attr =
          op->getAttrOfType<AffineMapAttr>(op.getBaseMapAttrStrName()))
    p.printAffineMapOfSSAIds(map_attr, op.getIndices());
  p << ']';

  // Print symbol operands if present
  if (!op.getSymbolOperands().empty()) {
    p << " symbols(";
    llvm::interleaveComma(op.getSymbolOperands(), p);
    p << ")";
  }

  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{op.getBaseMapAttrStrName(),
                       op.getOperandSegmentSizesAttrName()});

  p << " : " << op.getBase().getType() << " -> " << op.getType();
}

//===----------------------------------------------------------------------===//
// ConstructAccessTilesOp Verification
//===----------------------------------------------------------------------===//

LogicalResult ConstructAccessTilesOp::verify() {
  // Check that access_tile_set dimensions match base_map results
  auto access_tile_set = getAccessTileSet().getValue();
  auto base_map = getBaseMap();

  if (access_tile_set.getNumDims() != base_map.getNumResults()) {
    return emitOpError("access_tile_set dimensions (")
           << access_tile_set.getNumDims() << ") must match base_map dims ("
           << base_map.getNumResults() << ")";
  }

  // Check that base_map has correct number of operands
  if (base_map.getNumInputs() != getIndices().size()) {
    return emitOpError("base_map expects ")
           << base_map.getNumInputs() << " operands but got "
           << getIndices().size();
  }

  // Check that symbol_operands count matches access_tile_set symbols
  if (access_tile_set.getNumSymbols() != getSymbolOperands().size()) {
    return emitOpError("access_tile_set has ")
           << access_tile_set.getNumSymbols() << " symbols but got "
           << getSymbolOperands().size() << " symbol operands";
  }

  // Check that access_tile_order is a permutation (inputs == outputs)
  auto access_tile_order = getAccessTileOrder();
  if (access_tile_order.getNumInputs() != access_tile_order.getNumResults()) {
    return emitOpError("access_tile_order must have equal number of inputs (")
           << access_tile_order.getNumInputs() << ") and outputs ("
           << access_tile_order.getNumResults() << ")";
  }

  // Check that access_tile_order dimensions match access_tile_set dimensions
  if (access_tile_order.getNumResults() != access_tile_set.getNumDims()) {
    return emitOpError("access_tile_order dimensions (")
           << access_tile_order.getNumResults()
           << ") must match access_tile_set dimensions ("
           << access_tile_set.getNumDims() << ")";
  }

  // Verify result type matches expected shape from access_tile_order and
  // access_tile_set
  auto result_type = mlir::cast<AccessTileType>(getResult().getType());
  auto result_shape = result_type.getShape();

  // The result shape should be derivable from the access_tile_set bounds
  // TODO: This is a basic check - more sophisticated verification could be
  // added
  if (result_shape.size() != access_tile_set.getNumDims()) {
    return emitOpError("result type rank (")
           << result_shape.size() << ") must match access_tile_set dimensions ("
           << access_tile_set.getNumDims() << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConstructIndirectAccessTilesOp
//===----------------------------------------------------------------------===//

/// Collect exclusive captured variables from mapOperands.
///
/// intermediateVars: variables defined in the op's region
/// mapOperands: per-map operand lists (may contain both intermediate +
/// captured)
///
/// Returns:
///   ordered list of unique Values that are:
///     - present in mapOperands
///     - not present in intermediateVars
template <typename KeyT>
static void collectExclusiveCapturedVariables(
    const SmallVectorImpl<KeyT>& intermediate_vars,
    const SmallVectorImpl<SmallVector<KeyT>>& map_operands,
    SmallVectorImpl<KeyT>& captured_vars) {
  captured_vars.clear();

  // Fast membership check for intermediate variables
  llvm::DenseSet<KeyT> intermediate_set;
  intermediate_set.insert(intermediate_vars.begin(), intermediate_vars.end());

  // Track what we've already inserted (to avoid duplicates)
  llvm::DenseSet<KeyT> seen;

  for (const auto& operands : map_operands) {
    for (auto& v : operands) {
      // Skip intermediate variables
      if (intermediate_set.contains(v)) continue;

      // Insert only once
      if (seen.insert(v).second) captured_vars.push_back(v);
    }
  }
}

//===----------------------------------------------------------------------===//
// Canonicalize per-dimension affine maps to use a single consolidated operand
// list:  (captured_variables..., intermediate_variables...)
// by composing each map with a "remap" affine map from the consolidated domain
// to the map-local operand list.
//===----------------------------------------------------------------------===//

// Helper: build canonicalized maps + canonical operand list.
template <typename KeyT>
static LogicalResult canonicalizeAffineMapsToUnifiedOperands(
    MLIRContext* ctx,
    SmallVectorImpl<AffineMapAttr>& maps,              // per-dim subscript maps
    SmallVectorImpl<SmallVector<KeyT>>& map_operands,  // per-map operand lists
    SmallVectorImpl<KeyT>& captured_variables,  // SSA values defined above op
    SmallVectorImpl<KeyT>& intermediate_variables,  // region args
    SmallVectorImpl<Attribute>& canonical_maps,
    SmallVectorImpl<KeyT>& canonical_operands) {
  canonical_maps.clear();
  canonical_operands.clear();

  // Choose an order for the consolidated operand list.
  canonical_operands.append(captured_variables.begin(),
                            captured_variables.end());
  for (auto& iv : intermediate_variables) {
    canonical_operands.push_back(iv);
  }

  const unsigned unifiedNumDims = canonical_operands.size();
  const unsigned unifiedNumSyms =
      0;  // keep everything as dims in this approach

  // Map Value -> position in canonicalOperands.
  llvm::DenseMap<KeyT, unsigned> pos;
  pos.reserve(canonical_operands.size());
  for (unsigned i = 0; i < canonical_operands.size(); ++i)
    pos[canonical_operands[i]] = i;

  canonical_maps.reserve(maps.size());

  for (unsigned mi = 0; mi < maps.size(); ++mi) {
    AffineMap m = maps[mi].getAffineMap();
    SmallVector<KeyT> local_operands = map_operands[mi];

    if (m.getNumSymbols() != 0)
      return failure();  // or handle symbols explicitly (see notes).

    if (m.getNumDims() != local_operands.size())
      return failure();  // inconsistent (map dim count must match local operand
                         // count).

    // Build a "remap" map:
    //   remap : (U0..U_{k-1}) -> (Ui0, Ui1, ..., Ui_{n-1})
    // where each result picks the corresponding unified dim for the i-th local
    // operand.
    SmallVector<AffineExpr, 8> remap_results;
    remap_results.reserve(local_operands.size());

    for (auto& v : local_operands) {
      auto it = pos.find(v);
      if (it == pos.end())
        return failure();  // local operand wasn't in (captured + intermediate)
      remap_results.push_back(getAffineDimExpr(it->second, ctx));
    }

    AffineMap remap = AffineMap::get(/*dimCount=*/unifiedNumDims,
                                     /*symbolCount=*/unifiedNumSyms,
                                     /*results=*/remap_results, ctx);

    // Compose: oldMap(localDims...) . remap(unifiedDims...) =>
    // newMap(unifiedDims...) newMap : (U...) -> oldMap( remap(U...) )
    AffineMap newMap = m.compose(remap);

    canonical_maps.push_back(AffineMapAttr::get(newMap));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Build Method Contract: Subscript Map Canonicalization
//===----------------------------------------------------------------------===//
//
// The build method for `ktdp.construct_indirect_access_tile` assumes that all
// subscript affine maps have already been canonicalized with respect to a
// unified operand ordering.
//
// Canonical Operand Ordering
// --------------------------
// The canonical dimension ordering for all subscript maps must follow:
//
//   (captured_variables..., intermediate_variables...)
//
// That is, affine map dimensions must correspond first to the values supplied
// through the `captured_variables` operands, followed by the region arguments
// representing `intermediate_variables`.
//
// The builder does not perform this canonicalization automatically. Callers
// (e.g., the custom parser or transformation passes) are responsible for
// constructing affine maps whose dimension positions match this ordering.
//
// Example
// -------
// Consider the following operation:
//
//   %X_access_tile = ktdp.construct_indirect_access_tile
//                      intermediate_variables(%b, %h, %tkv, %dkv)
//       %X_mem_view[
//           ind(%Idx_mem_view[%b + %outside, (%tkv floordiv 64)]),
//           (%h),
//           (%tkv mod 64),
//           (%dkv)
//       ] {
//         variables_space_set = #X_var_space_set,
//         variables_space_order = #X_var_space_order
//       }
//       : memref<10000x8x64x128xf16>, memref<4x32xi32>
//         -> !ktdp.access_tile<4x8x2048x128xindex>
//
// Assume:
//
//   captured_variables       = [%outside]
//   intermediate_variables = (%b, %h, %tkv, %dkv)
//
// The canonical dimension ordering is:
//
//   d0 = outside
//   d1 = b
//   d2 = h
//   d3 = tkv
//   d4 = dkv
//
// Each per-dimension subscript map must therefore be constructed such that its
// affine dimensions are expressed over this complete ordered list.
//
// Canonicalized Maps
// ------------------
//
// To represent the following affine expression as a per-dim subscript
//   (b + outside, tkv floordiv 64)
//
// Specify the following map using the canonical dim order: (outside, b, h, tkv,
// dkv)
//
//   affine_map<(d0, d1, d2, d3, d4)
//              -> (d1 + d0, d3 floordiv 64)>
//
// To represent the expression (h):
//
//   affine_map<(d0, d1, d2, d3, d4)
//              -> (d2)>
//
// To represent the expression (tkv mod 64):
//
//   affine_map<(d0, d1, d2, d3, d4)
//              -> (d3 mod 64)>
//
// Summary
// -------
// * All subscript maps must share the same dimensional domain.
// * The domain ordering must be:
//       (captured_variables..., intermediate_variables...).
// * The build method assumes maps are already canonicalized to this form.
// * Verification may assert that map dimensionality matches the expected
//   number of variables.
//
void ConstructIndirectAccessTilesOp::build(
    OpBuilder& builder, OperationState& result, AccessTileType result_type,
    Value base, ArrayAttr per_dim_subscript_kinds,
    ArrayAttr per_dim_subscript_maps, ValueRange indirect_memrefs,
    ValueRange captured_variables, ValueRange symbol_operands,
    IntegerSet variables_space_set, AffineMap variables_space_order) {
  assert(variables_space_set.getNumDims() ==
             variables_space_order.getNumInputs() &&
         "Number of input dimensions in the variables_space_order should match "
         "with number of inputs in the variables_space_set");
  assert(variables_space_order.getNumInputs() ==
             variables_space_order.getNumResults() &&
         "Number of input dimensions and output dimensions in the variable "
         "space order should match");
  assert(variables_space_set.getNumSymbols() == symbol_operands.size() &&
         "number of symbol operands must match symbols in variables_space_set");

  result.addOperands(base);
  result.addOperands(indirect_memrefs);
  result.addOperands(captured_variables);
  result.addOperands(symbol_operands);

  // Add a body region with block arguments
  Region* bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block& bodyBlock = bodyRegion->front();
  ensureTerminator(*bodyRegion, builder, builder.getUnknownLoc());

  auto num_intermediate_variables = variables_space_set.getNumDims();
  for (unsigned i = 0; i < num_intermediate_variables; i++) {
    bodyBlock.addArgument(builder.getIndexType(), builder.getUnknownLoc());
  }

  result.addAttribute(
      getOperandSegmentSizesAttrName(result.name),
      builder.getDenseI32ArrayAttr({1, static_cast<int32_t>(indirect_memrefs.size()),
                                    static_cast<int32_t>(captured_variables.size()),
                                    static_cast<int32_t>(symbol_operands.size())}));

  result.addAttribute(getPerDimSubscriptKindsAttrStrName(),
                      per_dim_subscript_kinds);
  result.addAttribute(getPerDimSubscriptMapsAttrStrName(),
                      per_dim_subscript_maps);
  result.addAttribute(getVariablesSpaceSetAttrStrName(),
                      IntegerSetAttr::get(variables_space_set));
  result.addAttribute(getVariablesSpaceOrderAttrStrName(),
                      AffineMapAttr::get(variables_space_order));

  result.types.push_back(result_type);
}

// Overload without symbol_operands
void ConstructIndirectAccessTilesOp::build(
    OpBuilder& builder, OperationState& result, AccessTileType result_type,
    Value base, ArrayAttr per_dim_subscript_kinds,
    ArrayAttr per_dim_subscript_maps, ValueRange indirect_memrefs,
    ValueRange captured_variables, IntegerSet variables_space_set,
    AffineMap variables_space_order) {
  build(builder, result, result_type, base, per_dim_subscript_kinds,
        per_dim_subscript_maps, indirect_memrefs, captured_variables,
        /*symbol_operands=*/ValueRange{}, variables_space_set,
        variables_space_order);
}

/// Parse method for `ktdp.construct_indirect_access_tile`.
///
/// This method performs the following high-level steps:
///
/// 1. Parse the list of `intermediate_variables`.
/// 2. Parse the base memref followed by per-dimension subscripts.
///    Each subscript may be either:
///       - Direct:    (affine-map-of-SSA-ids)
///       - Indirect:  ind(memref[affine-map-of-SSA-ids])
/// 3. Collect subscript affine maps and their associated SSA operands.
/// 4. Construct an implicit region whose block arguments correspond to
///    the intermediate variables.
/// 5. Derive `captured_variables` (variables defined outside the operation)
///    from subscript operands.
/// 6. Canonicalize all subscript maps over a unified operand ordering:
///       (captured_variables..., intermediate_variables...)
/// 7. Attach canonicalized attributes to the operation state.
/// 8. Parse type signature and resolve all operands.
/// 9. Add result type.
///
/// The method ensures that:
///   * All subscript maps share a consistent canonical domain.
///   * Intermediate variables are represented as region arguments.
///   * captured variables are added as explicit operands.
///   * Operand segment sizes match ODS expectations.
///
///
::mlir::ParseResult ConstructIndirectAccessTilesOp::parse(
    OpAsmParser& parser, OperationState& result) {
  auto& builder = parser.getBuilder();

  SmallVector<OpAsmParser::UnresolvedOperand, 4> intermediate_variables;
  if (parser.parseKeyword("intermediate_variables") ||
      parser.parseOperandList(intermediate_variables,
                              AsmParser::Delimiter::Paren)) {
    return parser.emitError(parser.getNameLoc())
           << "error parsing intermediate_variables";
  }

  OpAsmParser::UnresolvedOperand base_memref;
  if (parser.parseOperand(base_memref) || parser.parseLSquare())
    return failure();

  int num_indirect_dims = 0;
  SmallVector<Attribute> per_dim_subscript_kinds;
  SmallVector<AffineMapAttr> base_dim_subscript_maps;
  SmallVector<OpAsmParser::UnresolvedOperand> indirect_memrefs;
  SmallVector<SmallVector<OpAsmParser::UnresolvedOperand>>
      base_dim_map_operands;

  while (parser.parseOptionalRSquare()) {
    if (!per_dim_subscript_kinds.empty()) {
      if (parser.parseComma()) {
        return parser.emitError(parser.getNameLoc())
               << "expecting a comma between subscripts";
      }
    }

    // Save current location to detect if we make progress
    auto startLoc = parser.getCurrentLocation();

    AffineMapAttr subscript;
    SmallVector<OpAsmParser::UnresolvedOperand> subscript_operands;

    // indirect subscript.
    if (parser.parseOptionalKeyword("ind").succeeded()) {
      per_dim_subscript_kinds.push_back(builder.getBoolAttr(1));
      OpAsmParser::UnresolvedOperand indirect_memref;
      if (parser.parseLParen() || parser.parseOperand(indirect_memref) ||
          parser.parseAffineMapOfSSAIds(
              subscript_operands, subscript,
              getPerDimSubscriptMapsAttrStrName(),  // Dummy - we will clear
              result.attributes) ||
          parser.parseRParen())
        return parser.emitError(parser.getNameLoc())
               << "error parsing the subscript ";

      indirect_memrefs.push_back(indirect_memref);
      num_indirect_dims++;
    } else {
      // direct subscript
      per_dim_subscript_kinds.push_back(builder.getBoolAttr(0));
      if (parser.parseAffineMapOfSSAIds(
              subscript_operands, subscript,
              getPerDimSubscriptMapsAttrStrName(),  // Dummy - we will clear
              result.attributes, AsmParser::Delimiter::Paren)) {
        return parser.emitError(parser.getNameLoc())
               << "error parsing the subscript ";
      }
    }

    // Safety check: ensure we consumed at least one token to prevent infinite
    // loop
    if (parser.getCurrentLocation() == startLoc) {
      return parser.emitError(startLoc) << "unexpected token in subscript "
                                           "list, expected 'ind', '(', or ']'";
    }

    base_dim_subscript_maps.push_back(subscript);
    base_dim_map_operands.push_back(subscript_operands);
  }

  // clear temporary attributes created so far
  result.attributes.clear();

  // Parse optional symbol operands with symbols(...) syntax
  SmallVector<OpAsmParser::UnresolvedOperand> symbol_operands;
  if (parser.parseOptionalKeyword("symbols").succeeded()) {
    if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren, [&]() {
          symbol_operands.emplace_back();
          return parser.parseOperand(symbol_operands.back());
        })) {
      return parser.emitError(parser.getNameLoc())
             << "Failed to parse symbol operands";
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return parser.emitError(parser.getNameLoc()) << "error parsing attributes";
  }

  {
    // Add region in the parse along with its block arguments
    result.regions.reserve(1);
    Region* region = result.addRegion();
    region->push_back(new Block);
    Block& bodyBlock = region->front();
    for (unsigned i = 0; i < intermediate_variables.size(); ++i) {
      bodyBlock.addArgument(builder.getIndexType(), builder.getUnknownLoc());
    }

    ensureTerminator(*region, parser.getBuilder(), result.location);
  }

  // canonicalize maps.
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> captured_variables;
  SmallVector<Attribute> base_dim_subscript_maps_canonicalized;
  {
    SmallVector<StringRef> intermediate_variables_str, captured_variables_str;
    SmallVector<SmallVector<StringRef>> base_dim_map_operands_str;
    for (auto& var : intermediate_variables) {
      intermediate_variables_str.push_back(var.name);
    }

    // construct captured variables defined outside.
    for (auto& outer : base_dim_map_operands) {
      SmallVector<StringRef> map_operands_str;
      for (auto var : outer) {
        map_operands_str.push_back(var.name);
      }

      base_dim_map_operands_str.push_back(map_operands_str);
    }

    collectExclusiveCapturedVariables<StringRef>(intermediate_variables_str,
                                                 base_dim_map_operands_str,
                                                 captured_variables_str);

    auto fillCapturedVariables =
        [&](SmallVectorImpl<SmallVector<OpAsmParser::UnresolvedOperand>>&
                map_operands,
            SmallVectorImpl<StringRef>& names) {
          llvm::DenseSet<StringRef> names_set(names.begin(), names.end());
          llvm::DenseSet<StringRef> seen;
          for (const auto& ops : map_operands) {
            for (const OpAsmParser::UnresolvedOperand& operand : ops) {
              if (names_set.contains(operand.name) &&
                  seen.insert(operand.name).second)
                captured_variables.push_back(operand);
            }
          }
        };

    fillCapturedVariables(base_dim_map_operands, captured_variables_str);

    SmallVector<StringRef> canonicalized_operands;
    if (canonicalizeAffineMapsToUnifiedOperands<StringRef>(
            builder.getContext(), base_dim_subscript_maps,
            base_dim_map_operands_str, captured_variables_str,
            intermediate_variables_str, base_dim_subscript_maps_canonicalized,
            canonicalized_operands)
            .failed()) {
      return parser.emitError(parser.getNameLoc())
             << "error canonicalizing subscripts";
    }
  }

  // Construct additional attributes
  result.addAttribute(getPerDimSubscriptKindsAttrStrName(),
                      builder.getArrayAttr(per_dim_subscript_kinds));
  result.addAttribute(
      getPerDimSubscriptMapsAttrStrName(),
      builder.getArrayAttr(base_dim_subscript_maps_canonicalized));

  // Parse types
  Type result_type;
  llvm::SmallVector<Type, 4> memref_types;
  if (parser.parseColonTypeList(memref_types) || parser.parseArrow() ||
      parser.parseType(result_type)) {
    return parser.emitError(parser.getNameLoc())
           << "error parsing result types";
  }

  if (memref_types.size() != static_cast<unsigned>(num_indirect_dims) + 1) {
    return parser.emitError(parser.getNameLoc())
           << "expected " << (num_indirect_dims + 1)
           << " types (base + " << num_indirect_dims
           << " indirect memrefs), but got " << memref_types.size();
  }

  // Resolve the base memref and associate with type.
  if (parser.resolveOperand(base_memref, memref_types[0], result.operands)) {
    return parser.emitError(parser.getNameLoc())
           << "error resolving base memref type";
  }

  // Resolve indirect memref and associate with their types.
  for (unsigned i = 0; i < indirect_memrefs.size(); ++i) {
    if (parser.resolveOperand(indirect_memrefs[i], memref_types[i + 1],
                              result.operands)) {
      return parser.emitError(parser.getNameLoc())
             << "error resolving indirect memref type";
    }
  }

  if (!captured_variables.empty()) {
    if (parser.resolveOperands(captured_variables, builder.getIndexType(),
                               result.operands)) {
      return parser.emitError(parser.getNameLoc())
             << "error resolving captured variables";
    }
  }

  // Resolve symbol operands
  if (!symbol_operands.empty()) {
    if (parser.resolveOperands(symbol_operands, builder.getIndexType(),
                               result.operands)) {
      return parser.emitError(parser.getNameLoc())
             << "error resolving symbol operands";
    }
  }

  // Verify symbol operands count matches variables_space_set symbols
  std::optional<NamedAttribute> variables_space_set_attr =
      result.attributes.getNamed(getVariablesSpaceSetAttrStrName());

  if (variables_space_set_attr.has_value()) {
    auto variables_space_set =
        mlir::dyn_cast<IntegerSetAttr>(variables_space_set_attr->getValue())
            .getValue();

    if (variables_space_set.getNumSymbols() != symbol_operands.size()) {
      return parser.emitError(parser.getNameLoc())
             << "variables_space_set has "
             << variables_space_set.getNumSymbols() << " symbols but got "
             << symbol_operands.size() << " symbol operands";
    }
  }

  // Update operand segment sizes to include symbol_operands
  result.addAttribute(getOperandSegmentSizesAttrName(result.name),
                      builder.getDenseI32ArrayAttr(
                          {1, num_indirect_dims, static_cast<int32_t>(captured_variables.size()),
                           static_cast<int32_t>(symbol_operands.size())}));

  // add result type
  if (parser.addTypeToList(result_type, result.types)) {
    return parser.emitError(parser.getNameLoc()) << "error adding result type";
  }

  return ParseResult::success();
}

void ConstructIndirectAccessTilesOp::print(::mlir::OpAsmPrinter& p) {
  auto& op = *this;

  // print intermediate variables
  auto arguments = getRegion().getArguments();
  p << " intermediate_variables(";
  p << arguments;
  p << ") ";

  // print base value
  p << op.getBase() << "[";

  // print subscript expressions
  auto ndims = op.getPerDimSubscriptKinds().size();
  auto per_dim_subscript_kinds = op.getPerDimSubscriptKinds();
  auto per_dim_subscript_maps = op.getPerDimSubscriptMaps();

  int indirect_memrefs_counter = 0;
  auto indirect_memrefs = op.getIndirectMemrefs();

  llvm::SmallVector<Value> all_variables(op.getCapturedVariables());
  for (auto var : op.getIntermediateVariables()) {
    all_variables.push_back(var);
  }

  for (unsigned i = 0; i < ndims; ++i) {
    AffineMapAttr map_attr =
        llvm::dyn_cast<AffineMapAttr>(per_dim_subscript_maps[i]);
    BoolAttr bool_attr = llvm::dyn_cast<BoolAttr>(per_dim_subscript_kinds[i]);
    if (bool_attr.getValue()) {
      if (i == 0)
        p << "ind(";
      else
        p << ", ind(";
      p << indirect_memrefs[indirect_memrefs_counter++] << "[";
      p.printAffineMapOfSSAIds(map_attr, all_variables);
      p << "])";
    } else {
      // direct subscript
      if (i == 0)
        p << "(";
      else
        p << ", (";
      p.printAffineMapOfSSAIds(map_attr, all_variables);
      p << ")";
    }
  }

  p << "]";

  // Print symbol operands if present
  if (!op.getSymbolOperands().empty()) {
    p << " symbols(";
    llvm::interleaveComma(op.getSymbolOperands(), p);
    p << ")";
  }

  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{getPerDimSubscriptKindsAttrStrName(),
                                           getPerDimSubscriptMapsAttrStrName(),
                                           getOperandSegmentSizesAttrName()});

  p << " : " << op.getBase().getType();

  indirect_memrefs_counter = 0;
  for (unsigned i = 0; i < ndims; ++i) {
    BoolAttr bool_attr = llvm::dyn_cast<BoolAttr>(per_dim_subscript_kinds[i]);
    if (bool_attr.getValue()) {
      p << ", " << indirect_memrefs[indirect_memrefs_counter++].getType();
    }
  }

  p << " -> " << op.getResult().getType();
}

//===----------------------------------------------------------------------===//
// ConstructIndirectAccessTilesOp Verification
//===----------------------------------------------------------------------===//

LogicalResult ConstructIndirectAccessTilesOp::verify() {
  // Check that variables_space_set dimensions match variables_space_order
  // inputs
  auto variables_space_set = getVariablesSpaceSet().getValue();
  auto variables_space_order = getVariablesSpaceOrder();

  if (variables_space_set.getNumDims() !=
      variables_space_order.getNumInputs()) {
    return emitOpError("variables_space_set dimensions (")
           << variables_space_set.getNumDims()
           << ") must match variables_space_order inputs ("
           << variables_space_order.getNumInputs() << ")";
  }

  // Check that variables_space_order is a permutation (inputs == outputs)
  if (variables_space_order.getNumInputs() !=
      variables_space_order.getNumResults()) {
    return emitOpError(
               "variables_space_order must have equal number of inputs (")
           << variables_space_order.getNumInputs() << ") and outputs ("
           << variables_space_order.getNumResults() << ")";
  }

  // Check that number of intermediate variables matches variables_space_set
  // dimensions
  auto intermediate_vars = getIntermediateVariables();
  if (intermediate_vars.size() != variables_space_set.getNumDims()) {
    return emitOpError("number of intermediate variables (")
           << intermediate_vars.size()
           << ") must match variables_space_set dimensions ("
           << variables_space_set.getNumDims() << ")";
  }

  // Check that symbol_operands count matches variables_space_set symbols
  if (variables_space_set.getNumSymbols() != getSymbolOperands().size()) {
    return emitOpError("variables_space_set has ")
           << variables_space_set.getNumSymbols() << " symbols but got "
           << getSymbolOperands().size() << " symbol operands";
  }

  // Check that per_dim_subscript_kinds and per_dim_subscript_maps have same
  // size
  auto per_dim_kinds = getPerDimSubscriptKinds();
  auto per_dim_maps = getPerDimSubscriptMaps();

  if (per_dim_kinds.size() != per_dim_maps.size()) {
    return emitOpError("per_dim_subscript_kinds size (")
           << per_dim_kinds.size()
           << ") must match per_dim_subscript_maps size ("
           << per_dim_maps.size() << ")";
  }

  // Check that number of subscripts matches base memref/tensor rank
  auto base_type = getBase().getType();
  unsigned base_rank = 0;
  if (auto memref_type = mlir::dyn_cast<MemRefType>(base_type)) {
    base_rank = memref_type.getRank();
  } else if (auto tensor_type = mlir::dyn_cast<RankedTensorType>(base_type)) {
    base_rank = tensor_type.getRank();
  } else {
    return emitOpError("base must be a ranked memref or tensor type");
  }

  if (per_dim_kinds.size() != base_rank) {
    return emitOpError("number of subscripts (")
           << per_dim_kinds.size() << ") must match base rank (" << base_rank
           << ")";
  }

  // Count indirect dimensions and verify against indirect_memrefs size
  unsigned num_indirect_dims = 0;
  for (auto kind_attr : per_dim_kinds) {
    if (auto bool_attr = mlir::dyn_cast<BoolAttr>(kind_attr)) {
      if (bool_attr.getValue()) {
        num_indirect_dims++;
      }
    } else {
      return emitOpError(
          "per_dim_subscript_kinds must contain BoolAttr elements");
    }
  }

  if (num_indirect_dims != getIndirectMemrefs().size()) {
    return emitOpError("number of indirect dimensions (")
           << num_indirect_dims << ") must match number of indirect memrefs ("
           << getIndirectMemrefs().size() << ")";
  }

  // Verify that all subscript maps have the correct number of dimensions
  // They should all use the canonical ordering: (captured_variables...,
  // intermediate_variables...)
  const unsigned expected_num_dims =
      getCapturedVariables().size() + intermediate_vars.size();

  for (size_t i = 0; i < per_dim_maps.size(); ++i) {
    auto map_attr = mlir::dyn_cast<AffineMapAttr>(per_dim_maps[i]);
    if (!map_attr) {
      return emitOpError("per_dim_subscript_maps[")
             << i << "] must be an AffineMapAttr";
    }

    auto map = map_attr.getValue();
    if (map.getNumDims() != expected_num_dims) {
      return emitOpError("subscript map for dimension ")
             << i << " has " << map.getNumDims() << " dimensions, but expected "
             << expected_num_dims
             << " (captured_variables + intermediate_variables)";
    }

    // For indirect subscripts, the map should produce coordinates for the
    // indirect memref
    auto is_indirect = mlir::cast<BoolAttr>(per_dim_kinds[i]).getValue();
    if (is_indirect) {
      // The map results should match the rank of the corresponding indirect
      // memref We need to find which indirect memref this corresponds to
      unsigned indirect_idx = 0;
      for (size_t j = 0; j < i; ++j) {
        if (mlir::cast<BoolAttr>(per_dim_kinds[j]).getValue()) {
          indirect_idx++;
        }
      }

      if (indirect_idx >= getIndirectMemrefs().size()) {
        return emitOpError(
            "indirect memref index computed goes beyond the bounds of the "
            "indirect memrefs available. This could happen if the "
            "PerDimSubscriptKinds aren't in sync with the number of indirect "
            "memrefs");
      }
      auto indirect_memref_type = getIndirectMemrefs()[indirect_idx].getType();
      unsigned indirect_rank = 0;
      if (auto memref_type = mlir::dyn_cast<MemRefType>(indirect_memref_type)) {
        indirect_rank = memref_type.getRank();
      } else if (auto tensor_type =
                     mlir::dyn_cast<RankedTensorType>(indirect_memref_type)) {
        indirect_rank = tensor_type.getRank();
      }

      if (map.getNumResults() != indirect_rank) {
        return emitOpError("indirect subscript map for dimension ")
               << i << " produces " << map.getNumResults()
               << " results, but indirect memref has rank " << indirect_rank;
      }
    } else {
      // For direct subscripts, the map should produce a single result
      if (map.getNumResults() != 1) {
        return emitOpError("direct subscript map for dimension ")
               << i << " must produce exactly 1 result, but produces "
               << map.getNumResults();
      }
    }
  }

  // Verify result type rank matches variables_space_set dimensions
  auto result_type = mlir::cast<AccessTileType>(getResult().getType());
  auto result_shape = result_type.getShape();
  if (result_shape.size() != variables_space_set.getNumDims()) {
    return emitOpError("result type rank (")
           << result_shape.size()
           << ") must match variables_space_set dimensions ("
           << variables_space_set.getNumDims() << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GetComputeTileId Verification
//===----------------------------------------------------------------------===//

LogicalResult GetComputeTileIdOp::verify() {
  if (getNumResults() < 1)
    return emitOpError("must return at least one index (num tile-grid dims >= 1)");
  for (auto res : getResults())
    if (!mlir::isa<IndexType>(res.getType()))
      return emitOpError("all results must be IndexType");
  return success();
}

//===----------------------------------------------------------------------===//
// ConstructMemoryViewOp Verification
//===----------------------------------------------------------------------===//

LogicalResult ConstructMemoryViewOp::verify() {
  unsigned nDims = getStaticSizes().size();
  if (getStaticStrides().size() != nDims)
    return emitOpError("static_sizes and static_strides must have equal length");
  unsigned dynSizes = llvm::count(getStaticSizes(), ShapedType::kDynamic);
  unsigned dynStrides = llvm::count(getStaticStrides(), ShapedType::kDynamic);
  if (getSizes().size() != dynSizes)
    return emitOpError("number of dynamic size operands does not match kDynamic entries in static_sizes");
  if (getStrides().size() != dynStrides)
    return emitOpError("number of dynamic stride operands does not match kDynamic entries in static_strides");
  auto memrefType = dyn_cast<MemRefType>(getResult().getType());
  if (!memrefType)
    return emitOpError("result must be a memref type");
  if (memrefType.getRank() != static_cast<int64_t>(nDims))
    return emitOpError("result memref rank does not match sizes/strides length");
  return success();
}

//===----------------------------------------------------------------------===//
// ConstructDistributedMemoryViewOp Verification
//===----------------------------------------------------------------------===//

LogicalResult ConstructDistributedMemoryViewOp::verify() {
  auto resultType = dyn_cast<MemRefType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be a memref type");
  for (auto memref : getMemrefs()) {
    auto mType = dyn_cast<MemRefType>(memref.getType());
    if (!mType)
      return emitOpError("all inputs must be memref types");
    if (mType.getElementType() != resultType.getElementType())
      return emitOpError("all input memrefs must have the same element type as the result");
    if (mType.getRank() != resultType.getRank())
      return emitOpError("all input memrefs must have the same rank as the result");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp Builders
//===----------------------------------------------------------------------===//

void LoadOp::build(OpBuilder& builder, OperationState& result, Value accessTile,
                   Type elementType) {
  auto accessTileType = mlir::cast<AccessTileType>(accessTile.getType());
  auto resultType =
      RankedTensorType::get(accessTileType.getShape(), elementType);
  build(builder, result, resultType, accessTile);
}

//===----------------------------------------------------------------------===//
// LoadOp Verification
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::verify() {
  auto tileType = mlir::cast<AccessTileType>(getAccessTile().getType());
  auto tensorType = dyn_cast<RankedTensorType>(getResult().getType());
  if (!tensorType)
    return emitOpError("result must be a ranked tensor");
  if (tileType.getShape() != tensorType.getShape())
    return emitOpError("access tile shape must match result tensor shape");
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp Verification
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::verify() {
  auto tileType = mlir::cast<AccessTileType>(getAccessTile().getType());
  auto tensorType = dyn_cast<RankedTensorType>(getDataTile().getType());
  if (!tensorType)
    return emitOpError("data_tile must be a ranked tensor");
  if (tileType.getShape() != tensorType.getShape())
    return emitOpError("data tile shape must match access tile shape");
  return success();
}

//===----------------------------------------------------------------------===//
// MemoryEffects
//===----------------------------------------------------------------------===//

void LoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

void StoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void ConstructIndirectAccessTilesOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Indirect subscripts implicitly read from their associated index views.
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// RuntimeArgExtractOp
//===----------------------------------------------------------------------===//

void RuntimeArgExtractOp::build(OpBuilder &builder, OperationState &result,
                                RuntimeArgExtractKind kind, Value input) {
  auto inputType = mlir::cast<RuntimeArgType>(input.getType());
  Type resultType;

  switch (kind) {
    case RuntimeArgExtractKind::value:
      resultType = inputType.getUnderlyingType();
      break;
    case RuntimeArgExtractKind::granularity:
    case RuntimeArgExtractKind::upperbound:
      resultType = builder.getIndexType();
      break;
  }

  build(builder, result, resultType, kind, input);
}

LogicalResult RuntimeArgExtractOp::verify() {
  auto inputType = mlir::cast<RuntimeArgType>(getInput().getType());
  auto kind = getKind();
  auto resultType = getResult().getType();

  switch (kind) {
    case RuntimeArgExtractKind::value:
      if (resultType != inputType.getUnderlyingType()) {
        return emitOpError("result type (")
               << resultType << ") must match runtime_arg underlying type ("
               << inputType.getUnderlyingType() << ") when extracting value";
      }
      break;

    case RuntimeArgExtractKind::granularity:
      if (!inputType.getGranularity().has_value()) {
        return emitOpError(
            "cannot extract granularity from runtime_arg that doesn't have one");
      }
      if (!resultType.isIndex()) {
        return emitOpError(
                   "result type must be index when extracting granularity, "
                   "but got ")
               << resultType;
      }
      break;

    case RuntimeArgExtractKind::upperbound:
      if (!inputType.getUpperbound().has_value()) {
        return emitOpError(
            "cannot extract upperbound from runtime_arg that doesn't have one");
      }
      if (!resultType.isIndex()) {
        return emitOpError(
                   "result type must be index when extracting upperbound, "
                   "but got ")
               << resultType;
      }
      break;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "Ktdp/KtdpOps.cpp.inc"
