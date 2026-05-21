#pragma once

#include <filesystem>
#include <optional>
#include <string>

#include "scene_describer/status.hpp"

namespace scene_describer {

struct GenerationOptions {
  int max_new_tokens{128};
  float temperature{0.2F};
  float top_p{0.9F};
  bool deterministic{true};
};

struct RuntimeConfig {
  std::string backend{"mock"};
  std::string model_dir;
  std::string execution_provider{"cpu"};
  std::string prompt{"Describe this image in one concise paragraph."};
  GenerationOptions generation;
  bool emit_json{false};
};

struct RuntimeConfigOverrides {
  std::optional<std::string> backend;
  std::optional<std::string> model_dir;
  std::optional<std::string> execution_provider;
  std::optional<std::string> prompt;
  std::optional<int> max_new_tokens;
  std::optional<bool> emit_json;
};

Result<RuntimeConfig> LoadRuntimeConfig(const std::filesystem::path& path);
Status ApplyConfigOverrides(RuntimeConfig& config, const RuntimeConfigOverrides& overrides);
Status ValidateRuntimeConfig(const RuntimeConfig& config);

}  // namespace scene_describer

