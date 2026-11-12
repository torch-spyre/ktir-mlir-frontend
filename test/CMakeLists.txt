llvm_canonicalize_cmake_booleans(KTIR_ENABLE_PYTHON_BINDINGS)

# Ask CMake where ktir-opt is actually built so lit doesn't have to guess.
get_target_property(KTIR_TOOLS_DIR ktir-opt RUNTIME_OUTPUT_DIRECTORY)
if(NOT KTIR_TOOLS_DIR)
  # No explicit output dir: the exe lands in the dir where the target was defined.
  get_target_property(KTIR_TOOLS_DIR ktir-opt BINARY_DIR)
endif()
message(STATUS "ktir-opt tools dir: ${KTIR_TOOLS_DIR}")

configure_lit_site_cfg(
        lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(KTIR_TEST_DEPENDS
  ktir-opt
  FileCheck
)
if(KTIR_ENABLE_PYTHON_BINDINGS)
  list(APPEND KTIR_TEST_DEPENDS KTIRPythonModules)
endif()

add_lit_testsuite(check-ktir "Running the KTIR regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
    ${KTIR_TEST_DEPENDS}
)
set_target_properties(check-ktir PROPERTIES FOLDER "Tests")

add_lit_testsuites(KTIR
    ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${KTIR_TEST_DEPENDS}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
)
