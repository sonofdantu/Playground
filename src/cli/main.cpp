#include "scene_describer/config.hpp"
#include "scene_describer/image.hpp"
#include "scene_describer/scene_describer.hpp"

#include <filesystem>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

namespace {

using scene_describer::RuntimeConfig;
using scene_describer::RuntimeConfigOverrides;

void PrintUsage(std::ostream& output) {
  output << "scene_describer --image <path> [options]\n\n"
         << "Options:\n"
         << "  --config <path>             Load key=value runtime config\n"
         << "  --backend <mock|ort-genai>  Select backend (default: mock)\n"
         << "  --model-dir <path>          ONNX Runtime GenAI model package directory\n"
         << "  --execution-provider <name> Execution provider hint: cpu, cuda, dml, qnn\n"
         << "  --image <path>              Image path. Bootstrap loader supports PPM/PGM\n"
         << "  --prompt <text>             Prompt sent to the model\n"
         << "  --max-new-tokens <n>        Max generated tokens\n"
         << "  --json                      Emit machine-readable JSON\n"
         << "  --help                      Show this help\n";
}

std::string JsonEscape(const std::string& input) {
  std::ostringstream escaped;
  for (const char ch : input) {
    switch (ch) {
      case '\\':
        escaped << "\\\\";
        break;
      case '"':
        escaped << "\\\"";
        break;
      case '\n':
        escaped << "\\n";
        break;
      case '\r':
        escaped << "\\r";
        break;
      case '\t':
        escaped << "\\t";
        break;
      default:
        escaped << ch;
        break;
    }
  }
  return escaped.str();
}

struct CliParseResult {
  bool help{false};
  std::filesystem::path config_path;
  std::filesystem::path image_path;
  RuntimeConfigOverrides overrides;
  std::string error;
};

CliParseResult ParseArgs(int argc, char** argv) {
  CliParseResult result;

  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    auto require_value = [&](const std::string& option) -> std::string {
      if (index + 1 >= argc) {
        result.error = "missing value for " + option;
        return {};
      }
      ++index;
      return argv[index];
    };

    if (arg == "--help" || arg == "-h") {
      result.help = true;
    } else if (arg == "--config") {
      result.config_path = require_value(arg);
    } else if (arg == "--backend") {
      result.overrides.backend = require_value(arg);
    } else if (arg == "--model-dir") {
      result.overrides.model_dir = require_value(arg);
    } else if (arg == "--execution-provider") {
      result.overrides.execution_provider = require_value(arg);
    } else if (arg == "--image") {
      result.image_path = require_value(arg);
    } else if (arg == "--prompt") {
      result.overrides.prompt = require_value(arg);
    } else if (arg == "--max-new-tokens") {
      const auto value = require_value(arg);
      try {
        result.overrides.max_new_tokens = std::stoi(value);
      } catch (const std::exception& ex) {
        result.error = std::string("invalid --max-new-tokens value: ") + ex.what();
      }
    } else if (arg == "--json") {
      result.overrides.emit_json = true;
    } else {
      result.error = "unknown option: " + arg;
    }

    if (!result.error.empty()) {
      break;
    }
  }

  return result;
}

}  // namespace

int main(int argc, char** argv) {
  const auto cli = ParseArgs(argc, argv);
  if (!cli.error.empty()) {
    std::cerr << "error: " << cli.error << "\n\n";
    PrintUsage(std::cerr);
    return 2;
  }
  if (cli.help) {
    PrintUsage(std::cout);
    return 0;
  }
  if (cli.image_path.empty()) {
    std::cerr << "error: --image is required\n\n";
    PrintUsage(std::cerr);
    return 2;
  }

  RuntimeConfig config;
  if (!cli.config_path.empty()) {
    auto loaded = scene_describer::LoadRuntimeConfig(cli.config_path);
    if (!loaded.ok()) {
      std::cerr << "error: " << loaded.status().message() << "\n";
      return 2;
    }
    config = loaded.value();
  }

  auto override_status = scene_describer::ApplyConfigOverrides(config, cli.overrides);
  if (!override_status.ok()) {
    std::cerr << "error: " << override_status.message() << "\n";
    return 2;
  }

  auto backend = scene_describer::CreateSceneDescriber(config);
  if (!backend.ok()) {
    std::cerr << "error: " << backend.status().message() << "\n";
    return 2;
  }

  scene_describer::SceneDescriptionRequest request;
  request.image_path = cli.image_path.string();
  request.prompt = config.prompt;
  request.generation = config.generation;

  if (config.backend == "mock") {
    auto image = scene_describer::LoadImage(cli.image_path);
    if (!image.ok()) {
      std::cerr << "error: " << image.status().message() << "\n";
      return 2;
    }
    request.decoded_image = std::move(image.value());
  }

  auto description = backend.value()->Describe(request);
  if (!description.ok()) {
    std::cerr << "error: " << description.status().message() << "\n";
    return 3;
  }

  if (config.emit_json) {
    const auto& value = description.value();
    std::cout << "{\n"
              << "  \"text\": \"" << JsonEscape(value.text) << "\",\n"
              << "  \"backend\": \"" << JsonEscape(config.backend) << "\",\n"
              << "  \"image\": {\n"
              << "    \"path\": \"" << JsonEscape(request.image_path) << "\"";
    if (request.decoded_image.has_value()) {
      std::cout << ",\n"
                << "    \"width\": " << request.decoded_image->width << ",\n"
                << "    \"height\": " << request.decoded_image->height << ",\n"
                << "    \"channels\": " << request.decoded_image->channels << "\n"
                << "  }";
    } else {
      std::cout << "\n"
                << "  }";
    }
    if (!value.metadata.empty()) {
      std::cout << ",\n"
                << "  \"metadata\": {\n";
      size_t emitted = 0;
      for (const auto& [key, metadata_value] : value.metadata) {
        std::cout << "    \"" << JsonEscape(key) << "\": \"" << JsonEscape(metadata_value) << "\"";
        ++emitted;
        std::cout << (emitted == value.metadata.size() ? "\n" : ",\n");
      }
      std::cout << "  }\n";
    } else {
      std::cout << "\n";
    }
    std::cout << "}\n";
  } else {
    std::cout << description.value().text << "\n";
  }

  return 0;
}
