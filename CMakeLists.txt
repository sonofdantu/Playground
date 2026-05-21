cmake_minimum_required(VERSION 3.24)

project(edge_scene_describer
  VERSION 0.1.0
  DESCRIPTION "Edge C++ scene description runtime using ONNX Runtime GenAI-ready backends"
  LANGUAGES CXX)

option(SCENE_DESC_BUILD_TESTS "Build lightweight C++ tests" ON)
option(SCENE_DESC_ENABLE_ORT_GENAI "Enable ONNX Runtime GenAI backend wiring" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

function(scene_desc_configure_target target_name)
  target_compile_features(${target_name} PUBLIC cxx_std_20)
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4 /permissive- /EHsc)
  else()
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

add_library(scene_describer_core
  src/analyzer/analyzer.cpp
  src/backends/backend_factory.cpp
  src/backends/mock_backend.cpp
  src/backends/ort_genai_backend.cpp
  src/core/config.cpp
  src/core/image.cpp
)

target_include_directories(scene_describer_core
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)

scene_desc_configure_target(scene_describer_core)

if(SCENE_DESC_ENABLE_ORT_GENAI)
  find_package(OnnxRuntimeGenAI REQUIRED)
  target_compile_definitions(scene_describer_core PRIVATE SCENE_DESC_HAS_ORT_GENAI=1)
  target_link_libraries(scene_describer_core PRIVATE OnnxRuntimeGenAI::OnnxRuntimeGenAI)
endif()

add_executable(scene_describer src/cli/main.cpp)
scene_desc_configure_target(scene_describer)
target_link_libraries(scene_describer PRIVATE scene_describer_core)

add_executable(scene_analyzer src/analyzer_cli/main.cpp)
scene_desc_configure_target(scene_analyzer)
target_link_libraries(scene_analyzer PRIVATE scene_describer_core)

add_executable(scene_describer_benchmark src/benchmarks/benchmark_main.cpp)
scene_desc_configure_target(scene_describer_benchmark)
target_link_libraries(scene_describer_benchmark PRIVATE scene_describer_core)

if(SCENE_DESC_ENABLE_ORT_GENAI)
  add_executable(scene_model_probe src/tools/model_probe.cpp)
  scene_desc_configure_target(scene_model_probe)
  target_link_libraries(scene_model_probe PRIVATE OnnxRuntimeGenAI::OnnxRuntimeGenAI)
endif()

if(SCENE_DESC_BUILD_TESTS)
  enable_testing()
  add_executable(scene_describer_config_test tests/config_test.cpp)
  scene_desc_configure_target(scene_describer_config_test)
  target_link_libraries(scene_describer_config_test PRIVATE scene_describer_core)
  add_test(NAME config_test COMMAND scene_describer_config_test)

  add_executable(scene_describer_analyzer_test tests/analyzer_test.cpp)
  scene_desc_configure_target(scene_describer_analyzer_test)
  target_link_libraries(scene_describer_analyzer_test PRIVATE scene_describer_core)
  add_test(NAME analyzer_test COMMAND scene_describer_analyzer_test)
endif()
