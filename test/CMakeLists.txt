configure_lit_site_cfg(
        lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(KTIR_TEST_DEPENDS
  ktir-opt
)

add_lit_testsuite(check-ktir "Running the KTIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${KTIR_TEST_DEPENDS}
)
set_target_properties(check-ktir PROPERTIES FOLDER "Tests")

add_lit_testsuites(KTIR . DEPENDS ${KTIR_TEST_DEPENDS})
