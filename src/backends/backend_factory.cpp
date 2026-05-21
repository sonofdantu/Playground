#include "scene_describer/scene_describer.hpp"

#include <memory>

namespace scene_describer {

std::unique_ptr<ISceneDescriber> CreateMockSceneDescriber();
std::unique_ptr<ISceneDescriber> CreateOrtGenAiSceneDescriber(RuntimeConfig config);

Result<std::unique_ptr<ISceneDescriber>> CreateSceneDescriber(const RuntimeConfig& config) {
  if (config.backend == "mock") {
    return CreateMockSceneDescriber();
  }
  if (config.backend == "ort-genai") {
    return CreateOrtGenAiSceneDescriber(config);
  }
  return Status(ErrorCode::kInvalidArgument, "unknown backend: " + config.backend);
}

}  // namespace scene_describer

