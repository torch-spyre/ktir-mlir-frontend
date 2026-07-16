include(AddMLIRPython)

set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)
set(PYTHON_PACKAGE_DIR 
  "${PROJECT_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
)

file(MAKE_DIRECTORY "${PYTHON_PACKAGE_DIR}")

# Patch the MLIR function of the same name to fix install targets.
function(_mlir_python_install_sources name source_root_dir destination)
  foreach(source_relative_path ${ARGN})
    get_filename_component(
      dest_relative_dir "${source_relative_path}" DIRECTORY
      BASE_DIR "${source_root_dir}"
    )
    install(
      FILES "${source_root_dir}/${source_relative_path}"
      DESTINATION "${destination}/${dest_relative_dir}"
      COMPONENT ${PROJECT_NAME}-python-sources
    )
  endforeach()
  install(TARGETS ${name}
    EXPORT ${PROJECT_NAME}
    COMPONENT ${PROJECT_NAME}-python-sources
  )
endfunction()

if(MLIR_PYTHON_STUBGEN_ENABLED)
  # Convenience function to declare a stubgen target.
  function(add_ktir_python_type_stubs extension)
    cmake_parse_arguments(ARG "" "ADD_TO_PARENT" "DEPENDS_TARGETS;OUTPUTS;IMPORT_PATHS" ${ARGN})

    get_target_property(_extension_sources ${extension} INTERFACE_SOURCES)
    get_target_property(_extension_module_name ${extension} mlir_python_EXTENSION_MODULE_NAME)

    mlir_generate_type_stubs(
      MODULE_NAME "${MLIR_PYTHON_PACKAGE_PREFIX}._mlir_libs.${_extension_module_name}"
      DEPENDS_TARGETS "${NB_LIBRARY_TARGET_NAME};${ARG_DEPENDS_TARGETS}"
      OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_mlir_libs"
      OUTPUTS "${ARG_OUTPUTS}"
      DEPENDS_TARGET_SRC_DEPS "${_extension_sources}"
      IMPORT_PATHS "${PYTHON_PACKAGE_DIR}/..;${ARG_IMPORT_PATHS}"
    )

    set(_generated_outputs "${ARG_OUTPUTS}")
    list(TRANSFORM _generated_outputs PREPEND "_mlir_libs/")

    declare_mlir_python_sources(
      ${extension}.type_stub_gen
      ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
      ADD_TO_PARENT ${ARG_ADD_TO_PARENT}
      SOURCES "${_generated_outputs}"
    )
  endfunction()
endif()
