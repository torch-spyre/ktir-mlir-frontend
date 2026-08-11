# Get the names of all relevant exported KTIR targets.
get_property(_dialect_libs GLOBAL PROPERTY ${PROJECT_NAME}_DIALECT_LIBS)
get_property(_extension_libs GLOBAL PROPERTY ${PROJECT_NAME}_EXTENSION_LIBS)
get_property(_conversion_libs GLOBAL PROPERTY ${PROJECT_NAME}_CONVERSION_LIBS)

add_ktir_tool(ktir-opt 
    ktir-opt.cpp

  EXPORT_NAME opt

  SUPPORT_PLUGINS
)
target_link_libraries(ktir-opt PRIVATE
  KTIR::RegisterEverything
  ${_dialect_libs}
  ${_extension_libs}
  ${_conversion_libs}
)

mlir_target_link_libraries(ktir-opt PRIVATE
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRRegisterAllExtensions
)
llvm_update_compile_flags(ktir-opt)
export_executable_symbols_for_plugins(ktir-opt)
