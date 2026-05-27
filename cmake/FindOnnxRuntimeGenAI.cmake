# Minimal package finder for ONNX Runtime GenAI.
#
# Expected layout:
#   <root>/include/onnxruntime_genai_cxx.h
#   <root>/lib/<onnxruntime-genai library>
#
# Configure with:
#   cmake -DSCENE_DESC_ENABLE_ORT_GENAI=ON \
#     -DOnnxRuntimeGenAI_ROOT=/path/to/ort-genai \
#     -DOnnxRuntime_ROOT=/path/to/onnxruntime ...

find_path(OnnxRuntimeGenAI_INCLUDE_DIR
  NAMES ort_genai.h onnxruntime_genai_cxx.h onnxruntime_genai.h
  HINTS
    "${OnnxRuntimeGenAI_ROOT}/include"
    "$ENV{OnnxRuntimeGenAI_ROOT}/include"
)

find_library(OnnxRuntimeGenAI_LIBRARY
  NAMES onnxruntime-genai onnxruntime_genai
  HINTS
    "${OnnxRuntimeGenAI_ROOT}/lib"
    "$ENV{OnnxRuntimeGenAI_ROOT}/lib"
)

find_path(OnnxRuntime_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h onnxruntime_c_api.h
  HINTS
    "${OnnxRuntime_ROOT}/include"
    "$ENV{OnnxRuntime_ROOT}/include"
)

find_library(OnnxRuntime_LIBRARY
  NAMES onnxruntime
  HINTS
    "${OnnxRuntime_ROOT}/lib"
    "$ENV{OnnxRuntime_ROOT}/lib"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OnnxRuntimeGenAI
  REQUIRED_VARS
    OnnxRuntimeGenAI_INCLUDE_DIR
    OnnxRuntimeGenAI_LIBRARY
    OnnxRuntime_INCLUDE_DIR
    OnnxRuntime_LIBRARY)

if(OnnxRuntimeGenAI_FOUND AND NOT TARGET OnnxRuntimeGenAI::OnnxRuntimeGenAI)
  add_library(OnnxRuntimeGenAI::OnnxRuntimeGenAI UNKNOWN IMPORTED)
  set_target_properties(OnnxRuntimeGenAI::OnnxRuntimeGenAI PROPERTIES
    IMPORTED_LOCATION "${OnnxRuntimeGenAI_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${OnnxRuntimeGenAI_INCLUDE_DIR}")
endif()

if(OnnxRuntimeGenAI_FOUND AND NOT TARGET OnnxRuntime::OnnxRuntime)
  add_library(OnnxRuntime::OnnxRuntime UNKNOWN IMPORTED)
  set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
    IMPORTED_LOCATION "${OnnxRuntime_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${OnnxRuntime_INCLUDE_DIR}")
endif()
