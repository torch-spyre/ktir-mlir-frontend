get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  MLIRKtdp
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
)
add_llvm_executable(ktir-opt ktir-opt.cpp)

llvm_update_compile_flags(ktir-opt)
target_link_libraries(ktir-opt PRIVATE ${LIBS})
