#include "scene_describer/config.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
  scene_describer::RuntimeConfig config;
  scene_describer::RuntimeConfigOverrides overrides;
  overrides.backend = "ort-genai";
  overrides.model_dir = "models/example";
  overrides.max_new_tokens = 32;

  auto status = scene_describer::ApplyConfigOverrides(config, overrides);
  if (!status.ok()) {
    std::cerr << status.message() << "\n";
    return EXIT_FAILURE;
  }

  if (config.backend != "ort-genai" || config.model_dir != "models/example" ||
      config.generation.max_new_tokens != 32) {
    std::cerr << "config override did not apply expected values\n";
    return EXIT_FAILURE;
  }

  const auto bom_config_path = std::filesystem::temp_directory_path() / "scene_describer_config_bom.ini";
  {
    std::ofstream output(bom_config_path, std::ios::binary);
    output << "\xEF\xBB\xBF"
           << "backend=mock\n"
           << "max_new_tokens=17\n";
  }

  auto loaded = scene_describer::LoadRuntimeConfig(bom_config_path);
  std::filesystem::remove(bom_config_path);
  if (!loaded.ok()) {
    std::cerr << loaded.status().message() << "\n";
    return EXIT_FAILURE;
  }

  if (loaded.value().backend != "mock" || loaded.value().generation.max_new_tokens != 17) {
    std::cerr << "BOM config did not load expected values\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
