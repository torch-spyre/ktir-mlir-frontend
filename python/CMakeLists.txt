include(AddKTIRPython)

# NOTE: These are for tablegen only!
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)

# KTIR bindings contain their own copy of the MLIR bindings.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

# Where all the Python sources are.
set(PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_ktdp")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(KTIRPythonSources
  ROOT_DIR "${PYTHON_ROOT_DIR}"
  SOURCES
    __init__.py
)
if (MLIR_PYTHON_STUBGEN_ENABLED)
  declare_mlir_python_sources(KTIRPythonSources.type_stubs
    ADD_TO_PARENT KTIRPythonSources
  )
endif ()
declare_mlir_python_sources(KTIRPythonSources.Dialects
  ADD_TO_PARENT KTIRPythonSources
)
declare_mlir_python_sources(KTIRPythonExtensions
  ROOT_DIR "${PYTHON_ROOT_DIR}"
  SOURCES
    _mlir_libs/_ktir/py.typed
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT KTIRPythonSources.Dialects
  ROOT_DIR "${PYTHON_ROOT_DIR}"
  TD_FILE dialects/KtdpOps.td
  SOURCES 
    dialects/ktdp.py
  DIALECT_NAME ktdp
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(KTIRPythonExtensions.Main
  MODULE_NAME _ktir
  ADD_TO_PARENT KTIRPythonExtensions
  SOURCES
    KTIRModule.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIArith
    MLIRCAPIFunc
    MLIRCAPILinalg
    MLIRCAPIMath
    MLIRCAPISCF
    MLIRCAPITensor
    MLIRCAPITransforms
    KTIRKtdpCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)
target_link_libraries(KTIRPythonExtensions.Main INTERFACE KTIR::KtdpDialect)

if(MLIR_PYTHON_STUBGEN_ENABLED)
  add_ktir_python_type_stubs(KTIRPythonExtensions.Main
    ADD_TO_PARENT KTIRPythonSources.type_stubs
    DEPENDS_TARGETS
      KTIRPythonModules.extension._mlir.dso
      KTIRPythonModules.extension._ktir.dso
      KTIRPythonModules.sources.MLIRPythonSources.Core
      KTIRPythonModules.sources.MLIRPythonSources.Core.Python
      KTIRPythonModules.sources.MLIRPythonExtension.Core.type_stub_gen
    OUTPUTS
      _ktir/__init__.pyi
      _ktir/ktdp.pyi
  )
endif()

set(EXCLUDE_FROM_ALL)
if (NOT PROJECT_IS_TOP_LEVEL)
  set(EXCLUDE_FROM_ALL EXCLUDE_FROM_ALL)
endif ()
add_subdirectory(mlir_ktdp ${EXCLUDE_FROM_ALL})
