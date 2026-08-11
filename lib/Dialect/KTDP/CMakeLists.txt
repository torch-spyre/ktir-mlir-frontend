add_ktir_library(KTIRKTDPDialect
    KTDPAttrs.cpp
    KTDPDialect.cpp
    KTDPOps.cpp
    KTDPTypes.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ktir/Dialect/KTDP
    ${PROJECT_BINARY_DIR}/include/ktir/Dialect/KTDP

  GROUP DIALECT
  EXPORT_NAME KTDPDialect

  DEPENDS
    KTIRKTDPAttrInterfacesIncGen
    KTIRKTDPAttrsIncGen
    KTIRKTDPIncGen

  LINK_MLIR_LIBS
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
