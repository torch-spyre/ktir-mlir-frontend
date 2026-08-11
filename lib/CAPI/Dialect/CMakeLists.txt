add_ktir_public_c_api_library(KTIRKTDPCAPI
    KTDP.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ktir-c/Dialect

  EXPORT_NAME KTDPCAPI

  LINK_LIBS PUBLIC
    KTIRKTDPDialect
)
