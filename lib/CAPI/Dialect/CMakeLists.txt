add_ktir_public_c_api_library(KTIRKTDPCAPI
    KTDP.cpp

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ktir-c/Dialect

  EXPORT_NAME KTDPCAPI

  LINK_LIBS PUBLIC
    KTIRKtdpDialect
    KTIRConvertToKTIR
)

add_ktir_public_c_api_library(KTIRSpyreOpCAPI
    SpyreOp.cpp
    
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ktir-c/Dialect

  EXPORT_NAME SpyreOPCAPI

  LINK_LIBS PUBLIC
    KTIRSpyreOpDialect
)
