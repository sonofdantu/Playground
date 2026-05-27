#pragma once

#include "scene_describer/scene_describer.hpp"

#if SCENE_DESC_HAS_ORT_GENAI
#include <ort_genai.h>
#endif

#include <cstddef>
#include <string>

namespace scene_describer {

#if SCENE_DESC_HAS_ORT_GENAI

bool ShouldUseQwen35RawOrtCuda(const RuntimeConfig& config, const std::string& model_type);

Result<SceneDescription> DescribeQwen35RawOrtCuda(const RuntimeConfig& config,
                                                  OgaMultiModalProcessor& processor,
                                                  OgaNamedTensors& inputs,
                                                  const std::string& model_type,
                                                  size_t input_token_count,
                                                  size_t image_count);

#endif

}  // namespace scene_describer
