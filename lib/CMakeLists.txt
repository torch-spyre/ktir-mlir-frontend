add_subdirectory(CAPI/Dialect)
add_subdirectory(Dialect/KTDP)

#-------------------------------------------------------------------------------
# KTIRRegisterEverything
#-------------------------------------------------------------------------------

# IMPORTANT: Make sure to keep this in sync with the `RegisterEverything.cpp`
#            source file! If a new dialect, pass or extension is added, its
#            owning library must be added to one of the following lists.

# List all relevant exported MLIR targets.
set(mlir_dialect_libs
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRAnalysis
  MLIRPresburger
)
set(mlir_extension_libs)
set(mlir_conversion_libs)

# Get the names of all relevant exported KTIR targets.
get_property(_dialect_libs GLOBAL PROPERTY ${PROJECT_NAME}_DIALECT_LIBS)
get_property(_extension_libs GLOBAL PROPERTY ${PROJECT_NAME}_EXTENSION_LIBS)
get_property(_conversion_libs GLOBAL PROPERTY ${PROJECT_NAME}_CONVERSION_LIBS)

add_ktir_library(KTIRRegisterEverything
    RegisterEverything.cpp

  PARTIAL_SOURCES_INTENDED

  EXPORT_NAME RegisterEverything

  LINK_LIBS PUBLIC
    ${_dialect_libs}
    ${_extension_libs}
    ${_conversion_libs}

  LINK_MLIR_LIBS
    ${mlir_dialect_libs}
    ${mlir_extension_libs}
    ${mlir_conversion_libs}
)
