# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.20)
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/package.xml" oai_api_VERSION REGEX "<version>")
string(REGEX REPLACE ".*<version>([^<]+)</version>.*" "\\1"
  oai_api_VERSION "${oai_api_VERSION}")
project(oai_api VERSION "${oai_api_VERSION}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/package.xml")

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
find_package(fmt REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

# Library target
add_library(oai_api STATIC
  src/client.cpp
  src/agent.cpp
  src/builder.cpp
  src/serialize.cpp)
target_compile_features(oai_api PUBLIC cxx_std_23)

target_include_directories(oai_api PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(oai_api PUBLIC
  cpr::cpr
  fmt::fmt
  ZLIB::ZLIB
  nlohmann_json::nlohmann_json
  ${OpenCV_LIBS})

# Example executable
add_executable(echo_agent examples/echo_agent.cpp)
target_link_libraries(echo_agent PRIVATE oai_api)

add_executable(decisions_test examples/decisions_test.cpp)
target_link_libraries(decisions_test PRIVATE oai_api)

# Tests (no network): serialization, extra_body merging, SSE parsing
include(CTest)
if(BUILD_TESTING)
  add_executable(oai_api_test test/oai_api_test.cpp)
  target_include_directories(oai_api_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(oai_api_test PRIVATE oai_api)
  add_test(NAME oai_api_test COMMAND oai_api_test)
endif()

# Install
include(GNUInstallDirs)
install(TARGETS oai_api
  EXPORT oai_apiTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT oai_apiTargets
  FILE oai_apiTargets.cmake
  NAMESPACE oai_api::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/oai_api)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/oai_apiConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/oai_apiConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/oai_api)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/oai_apiConfigVersion.cmake
  VERSION ${oai_api_VERSION}
  COMPATIBILITY SameMajorVersion)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/oai_apiConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/oai_apiConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/oai_api)
