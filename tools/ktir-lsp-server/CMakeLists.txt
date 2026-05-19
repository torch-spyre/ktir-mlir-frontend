# Get the names of all relevant exported KTIR targets.
get_property(_dialect_libs GLOBAL PROPERTY ${PROJECT_NAME}_DIALECT_LIBS)
get_property(_extension_libs GLOBAL PROPERTY ${PROJECT_NAME}_EXTENSION_LIBS)

add_ktir_tool(ktir-lsp-server 
    ktir-lsp-server.cpp

  EXPORT_NAME lsp-server
)
target_link_libraries(ktir-lsp-server PRIVATE
  ${_dialect_libs}
  ${_extension_libs}
)

llvm_update_compile_flags(ktir-lsp-server)
mlir_target_link_libraries(ktir-lsp-server PRIVATE
  MLIRLspServerLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
)
