cmake_minimum_required(VERSION 3.20)
project(oai-api)

set(oai-api_MAJOR_VERSION 0)
set(oai-api_MINOR_VERSION 1)
set(oai-api_PATCH_VERSION 0)
set(oai-api_VERSION
  ${oai-api_MAJOR_VERSION}.${oai-api_MINOR_VERSION}.${oai-api_PATCH_VERSION})


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(FetchContent)
set(CPR_USE_SYSTEM_CURL ON CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
FetchContent_Declare(cpr GIT_REPOSITORY https://github.com/libcpr/cpr.git
                         GIT_TAG  1.14.2)
FetchContent_MakeAvailable(cpr)

# find dependencies
find_package(cpr REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

# Library target
add_library(oai-api SHARED
  src/client.cpp
  src/agent.cpp
  src/builder.cpp)
target_compile_features(oai-api PUBLIC cxx_std_23)

target_include_directories(oai-api PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(oai-api PUBLIC
  cpr::cpr
  fmt::fmt
  nlohmann_json::nlohmann_json
  ${OpenCV_LIBS})

# Example executable
add_executable(echo_agent examples/echo_agent.cpp)
target_link_libraries(echo_agent PRIVATE oai-api)

# Install
include(GNUInstallDirs)
install(TARGETS oai-api
  EXPORT oai-apiTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT oai-apiTargets
  FILE oai-apiTargets.cmake
  NAMESPACE oai-api::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/oai-api)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/oai-apiConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/oai-apiConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/oai-api)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/oai-apiConfigVersion.cmake
  VERSION ${oai-api_VERSION}
  COMPATIBILITY SameMajorVersion)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/oai-apiConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/oai-apiConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/oai-api)
