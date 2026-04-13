include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_ktdp`
# top level package (the API has been embedded in a relocatable way).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(KtdpPythonSources)

# SOURCES_GLOB picks up all .py files added to mlir_ktdp/ automatically.
# Note: cmake evaluates globs at configure time, so adding a new file requires
# re-running cmake configure (cmake -S . -B build ...) to pick it up.
# Non-.py files (e.g. .pyi stubs, data files) are NOT covered by this glob
# and must be added explicitly via a separate declare_mlir_python_sources call.
# This declaration controls the cmake build tree (bare cmake + PYTHONPATH) and
# the installed wheel (via cmake --install --component KtdpPythonModules).
# Pure-Python tools that are NOT part of the cmake build (e.g. ktdp_tools/)
# live outside mlir_ktdp/ and are packaged by scikit-build-core via
# wheel.packages in pyproject.toml — do NOT add them here.
declare_mlir_python_sources(KtdpPythonSources.Package
  ADD_TO_PARENT KtdpPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_ktdp"
  SOURCES_GLOB "*.py"
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT KtdpPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_ktdp"
  TD_FILE dialects/KtdpOps.td
  SOURCES
    dialects/ktdp_nanobind.py
    _mlir_libs/_ktdpDialectsNanobind/py.typed
  DIALECT_NAME ktdp)

declare_mlir_python_extension(KtdpPythonSources.NanobindExtension
  MODULE_NAME _ktdpDialectsNanobind
  ADD_TO_PARENT KtdpPythonSources
  SOURCES
        KtdpExtensionNanobind.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIArith
    MLIRCAPIFunc
    MLIRCAPILinalg
    MLIRCAPIMath
    MLIRCAPISCF
    MLIRCAPITensor
    MLIRCAPITransforms
    KtdpCAPI
)


################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(KtdpPythonCAPI
  INSTALL_COMPONENT KtdpPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../../.."
  DECLARED_SOURCES
    KtdpPythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
)

################################################################################
# Instantiation of all Python modules
################################################################################

set(KtdpPythonModules_ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")

if(MLIR_PYTHON_STUBGEN_ENABLED)
  # Everything here is very tightly coupled. See the ample descriptions at the bottom of
  # mlir/python/CMakeLists.txt.

  set(_core_type_stub_sources
    _mlir/__init__.pyi
    _mlir/ir.pyi
    _mlir/passmanager.pyi
    _mlir/rewrite.pyi
  )
  get_target_property(_core_extension_srcs MLIRPythonExtension.Core INTERFACE_SOURCES)
  mlir_generate_type_stubs(
    MODULE_NAME _mlir
    DEPENDS_TARGETS KtdpPythonModules.extension._mlir.dso
    OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_mlir_libs"
    OUTPUTS "${_core_type_stub_sources}"
    DEPENDS_TARGET_SRC_DEPS "${_core_extension_srcs}"
    IMPORT_PATHS "${KtdpPythonModules_ROOT_PREFIX}/_mlir_libs"
    VERBOSE
  )
  set(_mlir_typestub_gen_target "${NB_STUBGEN_CUSTOM_TARGET}")

  list(TRANSFORM _core_type_stub_sources PREPEND "_mlir_libs/")
  declare_mlir_python_sources(
    KtdpPythonExtension.Core.type_stub_gen
    ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
    ADD_TO_PARENT KtdpPythonSources
    SOURCES "${_core_type_stub_sources}"
  )

  get_target_property(_ktdpextension_srcs KtdpPythonSources.NanobindExtension INTERFACE_SOURCES)
  mlir_generate_type_stubs(
    MODULE_NAME mlir_ktdp._mlir_libs._ktdpDialectsNanobind
    DEPENDS_TARGETS
      KtdpPythonModules.extension._mlir.dso
      KtdpPythonModules.extension._ktdpDialectsNanobind.dso
    OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_mlir_libs"
    OUTPUTS
      _ktdpDialectsNanobind/__init__.pyi
      _ktdpDialectsNanobind/ktdp.pyi
    DEPENDS_TARGET_SRC_DEPS "${_ktdpextension_srcs}"
    IMPORT_PATHS "${KtdpPythonModules_ROOT_PREFIX}/.."
  )
  set(_ktdpDialectsNanobind_typestub_gen_target "${NB_STUBGEN_CUSTOM_TARGET}")

  declare_mlir_python_sources(
    KtdpPythonSources.type_stub_gen
    ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
    ADD_TO_PARENT KtdpPythonSources
    SOURCES
      _mlir_libs/_ktdpDialectsNanobind/__init__.pyi
      _mlir_libs/_ktdpDialectsNanobind/ktdp.pyi
  )
endif()

set(_declared_sources
  KtdpPythonSources
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.builtin
)
add_mlir_python_modules(KtdpPythonModules
  ROOT_PREFIX "${KtdpPythonModules_ROOT_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES "${_declared_sources}"
  COMMON_CAPI_LINK_LIBS
    KtdpPythonCAPI
)

if(MLIR_PYTHON_STUBGEN_ENABLED)
  add_dependencies(KtdpPythonModules "${_mlir_typestub_gen_target}")
  add_dependencies(KtdpPythonModules "${_ktdpDialectsNanobind_typestub_gen_target}")
endif()
