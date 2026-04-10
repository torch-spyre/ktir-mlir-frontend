add_mlir_public_c_api_library(KtdpCAPI
        Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRKtdp
)
