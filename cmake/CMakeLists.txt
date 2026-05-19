include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# The build-tree directory where we'll generate a build-time CMake package.
set(PROJECT_CMAKE_DIR "${PROJECT_BINARY_DIR}/lib/cmake/${PROJECT_NAME}")

get_property(${PROJECT_NAME}_DIALECT_LIBS GLOBAL PROPERTY ${PROJECT_NAME}_DIALECT_LIBS)
get_property(${PROJECT_NAME}_CONVERSION_LIBS GLOBAL PROPERTY ${PROJECT_NAME}_CONVERSION_LIBS)
get_property(${PROJECT_NAME}_EXTENSION_LIBS GLOBAL PROPERTY ${PROJECT_NAME}_EXTENSION_LIBS)
get_property(${PROJECT_NAME}_TRANSLATION_LIBS GLOBAL PROPERTY ${PROJECT_NAME}_TRANSLATION_LIBS)

#
# Install tree
#

install(DIRECTORY 
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_BINARY_DIR}/include/
  TYPE INCLUDE
  COMPONENT ktir-headers
  FILES_MATCHING
    PATTERN "*.td"
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.h.inc"
    PATTERN "*.hpp.inc"
    REGEX "CMakeFiles" EXCLUDE 
)

install(DIRECTORY
    ${PROJECT_BINARY_DIR}/doc/
  DESTINATION doc
  COMPONENT ktir-doc
  FILES_MATCHING
    PATTERN "*.md"
    REGEX "CMakeFiles" EXCLUDE
)

install(EXPORT ${PROJECT_NAME}
  FILE "${PROJECT_NAME}Targets.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
  NAMESPACE ${PROJECT_NAME}::
  COMPONENT ktir-cmake
)

install(FILES
    "${PROJECT_CMAKE_DIR}/${PROJECT_NAME}Config.cmake"
    "${PROJECT_CMAKE_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
)

#
# Build tree
#

export(EXPORT ${PROJECT_NAME}
  NAMESPACE ${PROJECT_NAME}::
  FILE "${PROJECT_CMAKE_DIR}/${PROJECT_NAME}Targets.cmake"
)

#
# Build & install tree
#

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in"
    "${PROJECT_CMAKE_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
)

write_basic_package_version_file(
    "${PROJECT_CMAKE_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)
