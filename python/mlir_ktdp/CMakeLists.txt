################################################################################
# Generate packages and shared library
################################################################################

set(_source_components
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.affine
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.math
  MLIRPythonSources.Dialects.linalg
  MLIRPythonSources.Dialects.scf
  MLIRPythonSources.Dialects.tensor
  MLIRPythonSources.Dialects.memref
  MLIRPythonSources.Dialects.complex # linalg depends on

  KTIRPythonSources
  KTIRPythonExtensions
)

if(MLIR_PYTHON_STUBGEN_ENABLED)
  list(APPEND _source_components MLIRPythonExtension.Core.type_stub_gen)
endif()

add_mlir_python_common_capi_library(KTIRPythonCAPI
  INSTALL_COMPONENT KTIRPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${PYTHON_PACKAGE_DIR}/_mlir_libs"
  DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(KTIRPythonModules
  ROOT_PREFIX "${PYTHON_PACKAGE_DIR}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    KTIRPythonCAPI
)

