add_ktir_library(KTIRSpyreOpDialect
    SpyreOpAttrs.cpp
    SpyreOpDialect.cpp
    SpyreOpOps.cpp
    SpyreOpTypes.cpp

  GROUP DIALECT
  EXPORT_NAME SpyreOpDialect
  
  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ktir/Dialect/SpyreOp
    ${PROJECT_BINARY_DIR}/include/ktir/Dialect/SpyreOp

  DEPENDS
    KTIRSpyreOpIncGen

  LINK_MLIR_LIBS
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRSideEffectInterfaces
    MLIRBytecodeOpInterface
)
