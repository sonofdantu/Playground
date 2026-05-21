#include "scene_describer/scene_describer.hpp"

#include <memory>
#include <sstream>

namespace scene_describer {
namespace {

class MockSceneDescriber final : public ISceneDescriber {
 public:
  Result<SceneDescription> Describe(const SceneDescriptionRequest& request) override {
    if (!request.decoded_image.has_value()) {
      return Status(ErrorCode::kInvalidArgument, "mock backend requires decoded image data");
    }

    const auto& image = *request.decoded_image;
    SceneDescription description;
    std::ostringstream text;
    text << "Mock scene description for " << image.width << "x" << image.height << " image with " << image.channels
         << " channel";
    if (image.channels != 1) {
      text << "s";
    }
    text << ". Prompt: " << request.prompt;

    description.text = text.str();
    description.metadata["backend"] = "mock";
    description.metadata["image_width"] = std::to_string(image.width);
    description.metadata["image_height"] = std::to_string(image.height);
    description.metadata["max_new_tokens"] = std::to_string(request.generation.max_new_tokens);
    return description;
  }
};

}  // namespace

std::unique_ptr<ISceneDescriber> CreateMockSceneDescriber() {
  return std::make_unique<MockSceneDescriber>();
}

}  // namespace scene_describer
