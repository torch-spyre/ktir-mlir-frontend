set(PROJECT_INCLUDE_DIRS
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

#
# Tablegen helpers
#

# Creates a TableGen target and adds it to ktir-headers.
function(add_ktir_tablegen_target name)
  add_public_tablegen_target(${name})
  set(TABLEGEN_OUTPUT "")
  add_dependencies(ktir-headers ${name})
endfunction()

# Creates a canonical MLIR dialect TableGen target and adds it to ktir-headers.
function(add_ktir_dialect dialect dialect_namespace)
  # TODO: Examine if we want to stick to the .hpp convention or use .h
  set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
  mlir_tablegen(${dialect}.hpp.inc -gen-op-decls)
  mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
  mlir_tablegen(${dialect}Types.hpp.inc -gen-typedef-decls -typedefs-dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Dialect.hpp.inc -gen-dialect-decls -dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
  add_ktir_tablegen_target(KTIR${dialect}IncGen)
endfunction()

# Creates an MLIR TableGen documentation target and adds it to ktir-doc.
function(add_ktir_doc doc_filename output_file output_directory command)
  set(LLVM_TARGET_DEFINITIONS ${doc_filename}.td)
  set(TABLEGEN_OUTPUT "")
  tablegen(MLIR ${output_file}.md ${command} -allow-hugo-specific-features ${ARGN})
  set(_output "${KTIR_BINARY_DIR}/doc/${output_directory}${output_file}.md")
  add_custom_command(
    OUTPUT "${_output}"
    COMMAND ${CMAKE_COMMAND} -E copy "${TABLEGEN_OUTPUT}" "${_output}"
    DEPENDS "${TABLEGEN_OUTPUT}"
  )
  set_source_files_properties(${_output} PROPERTIES GENERATED TRUE)
  add_custom_target("KTIR${output_file}DocGen" DEPENDS "${_output}")
  add_dependencies(ktir-doc "KTIR${output_file}DocGen")
endfunction()

#
# Target creation helpers
#

# Creates a tool executable target.
macro(add_ktir_tool name)
  cmake_parse_arguments(ARG "" "EXPORT_NAME" "" ${ARGN})

  if(NOT KTIR_BUILD_TOOLS)
    # Tools are always included, but not necessarily built.
    set(EXCLUDE_FROM_ALL ON)
  endif()

  add_llvm_executable(${name} ${ARG_UNPARSED_ARGUMENTS})

  if(KTIR_BUILD_TOOLS AND ARG_EXPORT_NAME)
    # To unify the interface for all consumers, create an alias target.
    set_target_properties(${name} PROPERTIES ARG_EXPORT_NAME ${ARG_EXPORT_NAME})
    add_executable("${PROJECT_NAME}::${ARG_EXPORT_NAME}" ALIAS ${name})

    # Tools each install as their own exported component.
    install(TARGETS ${name}
      COMPONENT ${name}
      EXPORT ${PROJECT_NAME}
    )
  endif()
endmacro()

# Creates a library target.
function(add_ktir_library name)
  cmake_parse_arguments(ARG "" "GROUP;EXPORT_NAME" "" ${ARGN})

  add_mlir_library(${name} 
    ${ARG_UNPARSED_ARGUMENTS} 
    DISABLE_INSTALL 
    EXCLUDE_FROM_LIBMLIR
  )
  target_include_directories(${name} PUBLIC ${PROJECT_INCLUDE_DIRS})

  if(ARG_GROUP)
    set(_groups DIALECT CONVERSION EXTENSION TRANSLATION)
    if(NOT ARG_GROUP IN_LIST _groups)
      message(FATAL_ERROR "GROUP must be one of ${_groups}")
    endif()
    if(NOT ARG_EXPORT_NAME)
      message(FATAL_ERROR "${ARG_GROUP} library requires EXPORT_NAME")
    endif()

    set(_qualified_name ${name})
    if(ARG_EXPORT_NAME)
      set(_qualified_name "${PROJECT_NAME}::${ARG_EXPORT_NAME}")
    endif()
    set_property(GLOBAL APPEND PROPERTY "KTIR_${ARG_GROUP}_LIBS" ${_qualified_name})
  endif()

  if(ARG_EXPORT_NAME)
    _install_ktir_library(${ARGV})
  endif()
endfunction()

# Creates a CAPI library target.
function(add_ktir_public_c_api_library name)
  cmake_parse_arguments(ARG "" "EXPORT_NAME" "" ${ARGN})

  add_mlir_public_c_api_library(${name}
    ${ARG_UNPARSED_ARGUMENTS} 
    DISABLE_INSTALL
  )
  target_include_directories(${name} PUBLIC ${PROJECT_INCLUDE_DIRS})

  if(ARG_EXPORT_NAME)
    _install_ktir_library(${ARGV})
  endif()

  add_dependencies(ktir-capi ${name})
endfunction()

function(_install_ktir_library name)
  cmake_parse_arguments(ARG "" "EXPORT_NAME" "" ${ARGN})

  if (DEFINED ARG_EXPORT_NAME)
    # To unify the interface for all consumers, create an alias target.
    set_target_properties(${name} PROPERTIES EXPORT_NAME ${ARG_EXPORT_NAME})
    add_library("${PROJECT_NAME}::${ARG_EXPORT_NAME}" ALIAS ${name})
  endif()

  # Libraries each install as their own exported component.
  install(TARGETS ${name}
    EXPORT ${PROJECT_NAME}
    COMPONENT ${name}
  )
endfunction()
