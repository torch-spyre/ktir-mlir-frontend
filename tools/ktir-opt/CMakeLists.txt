get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  MLIRKtdp
)
add_llvm_executable(ktir-opt ktir-opt.cpp)

llvm_update_compile_flags(ktir-opt)

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  set(DEPS PRIVATE -Wl,--whole-archive ${LIBS} -Wl,--no-whole-archive)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
  set(DEPS PRIVATE -Wl,-all_load ${LIBS})
endif()

target_link_libraries(ktir-opt PRIVATE ${DEPS})
