//===- KtdpAttrs.hpp - KTDP dialect attrs public header ---*- C++ -*-===//
//
//===----------------------------------------------------------------------===//
#ifndef KTDP_KTDPATTRS_HPP_
#define KTDP_KTDPATTRS_HPP_

#include "mlir/IR/Attributes.h"
#include "Ktdp/KtdpDialect.hpp"
#include "Ktdp/KtdpAttrInterfaces.hpp.inc"
#include "Ktdp/KtdpEnums.hpp.inc"
#define GET_ATTRDEF_CLASSES
#include "Ktdp/KtdpAttrs.hpp.inc"
#undef GET_ATTRDEF_CLASSES

#endif // KTDP_KTDPATTRS_HPP_
