add_ktir_library(KTIRSpyreDialect
    SpyreAttrs.cpp
    SpyreDialect.cpp
    SpyreOps.cpp
    SpyreTypes.cpp

  GROUP DIALECT
  EXPORT_NAME SpyreDialect
  
  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ktir/Dialect/Spyre
    ${PROJECT_BINARY_DIR}/include/ktir/Dialect/Spyre

  DEPENDS
    KTIRSpyreIncGen

  LINK_MLIR_LIBS
    MLIRIR
    MLIRSideEffectInterfaces
)
