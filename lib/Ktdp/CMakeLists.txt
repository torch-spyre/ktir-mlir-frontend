add_mlir_dialect_library(MLIRKtdp
        KtdpDialect.cpp
        KtdpAttrs.cpp
        KtdpOps.cpp
        KtdpTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Ktdp
  ${PROJECT_BINARY_DIR}/include/

  DEPENDS
  MLIRKtdpDialectIncGen
  MLIRKtdpAttrsIncGen
  MLIRKtdpTypesIncGen
  MLIRKtdpOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
)
