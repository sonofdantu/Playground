#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "scene_describer/config.hpp"
#include "scene_describer/image.hpp"
#include "scene_describer/status.hpp"

namespace scene_describer {

struct SceneDescriptionRequest {
  std::string image_path;
  std::vector<std::string> image_paths;
  std::optional<Image> decoded_image;
  std::string prompt;
  GenerationOptions generation;
};

struct SceneDescription {
  std::string text;
  std::map<std::string, std::string> metadata;
};

class ISceneDescriber {
 public:
  virtual ~ISceneDescriber() = default;
  virtual Result<SceneDescription> Describe(const SceneDescriptionRequest& request) = 0;
};

Result<std::unique_ptr<ISceneDescriber>> CreateSceneDescriber(const RuntimeConfig& config);

}  // namespace scene_describer
