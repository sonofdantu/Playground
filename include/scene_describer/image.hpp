#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "scene_describer/status.hpp"

namespace scene_describer {

struct Image {
  int width{0};
  int height{0};
  int channels{0};
  std::vector<std::uint8_t> pixels;
  std::string source_path;
};

Result<Image> LoadImage(const std::filesystem::path& path);

}  // namespace scene_describer

