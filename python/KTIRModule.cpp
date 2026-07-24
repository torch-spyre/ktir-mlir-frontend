//===- KtdpExtensionNanobind.cpp - Extension module ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Ktdp-c/Dialects.h"
#include "mlir-c/Dialect/Arith.h"
#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/Linalg.h"
#include "mlir-c/Dialect/Math.h"
#include "mlir-c/Dialect/SCF.h"
#include "mlir-c/Dialect/Tensor.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/IRTypes.h"

namespace nb = nanobind;
namespace py = mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;

namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::ktdp {

struct PyAccessTileType : PyConcreteType<PyAccessTileType, PyShapedType> {
  static constexpr IsAFunctionTy isaFunction = mlirKtdpTypeIsAAccessTileType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirKtdpAccessTileTypeGetTypeID;
  static constexpr const char *pyClassName = "AccessTileType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](std::vector<int64_t> shape, MlirType elementType, 
              DefaultingPyMlirContext context) {
          return PyAccessTileType(
              context->getRef(),
              mlirKtdpAccessTileTypeGet(
                  shape.size(), shape.data(),
                  elementType));
        },
        nb::arg("shape"), nb::arg("element_type"), nb::arg("context") = nb::none());
  }

};

struct PyRuntimeArgType : PyConcreteType<PyRuntimeArgType, PyType> {
  static constexpr IsAFunctionTy isaFunction = mlirKtdpTypeIsARuntimeArgType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirKtdpRuntimeArgTypeGetTypeID;
  static constexpr const char *pyClassName = "RuntimeArgType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](MlirType underlyingType, std::optional<int64_t> granularity,
              std::optional<int64_t> upperbound, DefaultingPyMlirContext context) {
          return PyRuntimeArgType(
              context->getRef(),
              mlirKtdpRuntimeArgTypeGet(
                  context->get(), underlyingType,
                  granularity.value_or(-1),
                  upperbound.value_or(-1)));
        },
        nb::arg("underlying_type"),
        nb::arg("granularity") = nb::none(),
        nb::arg("upperbound") = nb::none(),
        nb::arg("context") = nb::none());
    c.def_prop_ro("underlying_type", [](PyRuntimeArgType &self) {
      return mlirKtdpRuntimeArgTypeGetUnderlyingType(self);
    });
    c.def_prop_ro("granularity", [](PyRuntimeArgType &self) -> std::optional<int64_t> {
      int64_t v = mlirKtdpRuntimeArgTypeGetGranularity(self);
      return v >= 0 ? std::optional<int64_t>(v) : std::nullopt;
    });
    c.def_prop_ro("upperbound", [](PyRuntimeArgType &self) -> std::optional<int64_t> {
      int64_t v = mlirKtdpRuntimeArgTypeGetUpperbound(self);
      return v >= 0 ? std::optional<int64_t>(v) : std::nullopt;
    });
  }
};

} // namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::ktdp

//===----------------------------------------------------------------------===//
// _ktir Module
//===----------------------------------------------------------------------===//

NB_MODULE(_ktir, m) {
  m.def(
      "register_dialects",
      [](py::DefaultingPyMlirContext context, bool load) {
        MlirContext context_ = context.get()->get();
        for (auto handle : {
            mlirGetDialectHandle__arith__(),
            mlirGetDialectHandle__func__(),
            mlirGetDialectHandle__ktdp__(),
            mlirGetDialectHandle__linalg__(),
            mlirGetDialectHandle__math__(),
            mlirGetDialectHandle__scf__(),
            mlirGetDialectHandle__tensor__(),
        }) {
          mlirDialectHandleRegisterDialect(handle, context_);
          if (load)
            mlirDialectHandleLoadDialect(handle, context_);
        }
      },
      nb::arg("context") = nb::none(), nb::arg("load") = true);

  //===--------------------------------------------------------------------===//
  // _ktir.ktdp Module
  //===--------------------------------------------------------------------===//

  auto ktdp = m.def_submodule("ktdp");
  py::ktdp::PyAccessTileType::bind(ktdp);
  py::ktdp::PyRuntimeArgType::bind(ktdp);
}
