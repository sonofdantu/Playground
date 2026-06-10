#pragma once

#include "scene_describer/scene_describer.hpp"

#if SCENE_DESC_HAS_ORT_GENAI
#include <ort_genai.h>
#endif

#include <cstddef>
#include <memory>
#include <string>

namespace scene_describer {

#if SCENE_DESC_HAS_ORT_GENAI

bool ShouldUseQwen35RawOrtCuda(const RuntimeConfig& config, const std::string& model_type);

class Qwen35RawOrtCudaRunner {
 public:
  explicit Qwen35RawOrtCudaRunner(RuntimeConfig config);
  ~Qwen35RawOrtCudaRunner();

  Qwen35RawOrtCudaRunner(const Qwen35RawOrtCudaRunner&) = delete;
  Qwen35RawOrtCudaRunner& operator=(const Qwen35RawOrtCudaRunner&) = delete;
  Qwen35RawOrtCudaRunner(Qwen35RawOrtCudaRunner&&) noexcept;
  Qwen35RawOrtCudaRunner& operator=(Qwen35RawOrtCudaRunner&&) noexcept;

  Result<SceneDescription> Describe(OgaMultiModalProcessor& processor,
                                    OgaNamedTensors& inputs,
                                    const std::string& model_type,
                                    size_t input_token_count,
                                    size_t image_count);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

#endif

}  // namespace scene_describer
