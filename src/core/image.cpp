#include "scene_describer/image.hpp"

#include <cctype>
#include <exception>
#include <fstream>
#include <limits>
#include <string>

namespace scene_describer {
namespace {

Result<std::string> ReadToken(std::istream& input) {
  std::string token;
  char ch = '\0';

  while (input.get(ch)) {
    if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
      continue;
    }
    if (ch == '#') {
      std::string ignored;
      std::getline(input, ignored);
      continue;
    }
    token.push_back(ch);
    break;
  }

  while (input.get(ch)) {
    if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
      break;
    }
    if (ch == '#') {
      input.unget();
      break;
    }
    token.push_back(ch);
  }

  if (token.empty()) {
    return Status(ErrorCode::kParseError, "unexpected end of image header");
  }
  return token;
}

}  // namespace

Result<Image> LoadImage(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return Status(ErrorCode::kIoError, "unable to open image: " + path.string());
  }

  auto magic = ReadToken(input);
  if (!magic.ok()) {
    return magic.status();
  }

  int channels = 0;
  if (magic.value() == "P6") {
    channels = 3;
  } else if (magic.value() == "P5") {
    channels = 1;
  } else {
    return Status(ErrorCode::kUnsupported,
                  "only binary PPM/PGM images are supported in the bootstrap loader; got " + magic.value());
  }

  auto width_token = ReadToken(input);
  auto height_token = ReadToken(input);
  auto max_value_token = ReadToken(input);
  if (!width_token.ok()) {
    return width_token.status();
  }
  if (!height_token.ok()) {
    return height_token.status();
  }
  if (!max_value_token.ok()) {
    return max_value_token.status();
  }

  int width = 0;
  int height = 0;
  int max_value = 0;
  try {
    width = std::stoi(width_token.value());
    height = std::stoi(height_token.value());
    max_value = std::stoi(max_value_token.value());
  } catch (const std::exception& ex) {
    return Status(ErrorCode::kParseError, std::string("invalid image header: ") + ex.what());
  }

  if (width <= 0 || height <= 0 || max_value != 255) {
    return Status(ErrorCode::kUnsupported, "image must have positive dimensions and max value 255");
  }

  const auto pixel_count = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) *
                           static_cast<std::uint64_t>(channels);
  if (pixel_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status(ErrorCode::kUnsupported, "image is too large for this process");
  }

  Image image;
  image.width = width;
  image.height = height;
  image.channels = channels;
  image.source_path = path.string();
  image.pixels.resize(static_cast<std::size_t>(pixel_count));

  input.read(reinterpret_cast<char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
  if (input.gcount() != static_cast<std::streamsize>(image.pixels.size())) {
    return Status(ErrorCode::kParseError, "image ended before all pixel data was read");
  }

  return image;
}

}  // namespace scene_describer
