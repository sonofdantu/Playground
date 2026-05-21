#include "scene_describer/config.hpp"

#include <algorithm>
#include <cctype>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>

namespace scene_describer {
namespace {

std::string Trim(std::string value) {
  auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(),
                                          [&](char c) { return !is_space(static_cast<unsigned char>(c)); }));
  value.erase(std::find_if(value.rbegin(), value.rend(),
                           [&](char c) { return !is_space(static_cast<unsigned char>(c)); }).base(),
              value.end());
  return value;
}

std::string Lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::string StripUtf8Bom(std::string value) {
  if (value.size() >= 3 && static_cast<unsigned char>(value[0]) == 0xEF &&
      static_cast<unsigned char>(value[1]) == 0xBB && static_cast<unsigned char>(value[2]) == 0xBF) {
    value.erase(0, 3);
  }
  return value;
}

Result<bool> ParseBool(const std::string& value) {
  const auto lower = Lower(Trim(value));
  if (lower == "true" || lower == "1" || lower == "yes" || lower == "on") {
    return true;
  }
  if (lower == "false" || lower == "0" || lower == "no" || lower == "off") {
    return false;
  }
  return Status(ErrorCode::kParseError, "invalid boolean value: " + value);
}

Status ApplyKeyValue(RuntimeConfig& config, const std::string& raw_key, const std::string& raw_value) {
  const auto key = Lower(Trim(raw_key));
  const auto value = Trim(raw_value);

  try {
    if (key == "backend") {
      config.backend = Lower(value);
    } else if (key == "model_dir") {
      config.model_dir = value;
    } else if (key == "execution_provider") {
      config.execution_provider = Lower(value);
    } else if (key == "prompt") {
      config.prompt = value;
    } else if (key == "max_new_tokens") {
      config.generation.max_new_tokens = std::stoi(value);
    } else if (key == "temperature") {
      config.generation.temperature = std::stof(value);
    } else if (key == "top_p") {
      config.generation.top_p = std::stof(value);
    } else if (key == "deterministic") {
      auto parsed = ParseBool(value);
      if (!parsed.ok()) {
        return parsed.status();
      }
      config.generation.deterministic = parsed.value();
    } else if (key == "emit_json") {
      auto parsed = ParseBool(value);
      if (!parsed.ok()) {
        return parsed.status();
      }
      config.emit_json = parsed.value();
    } else {
      return Status(ErrorCode::kParseError, "unknown config key: " + raw_key);
    }
  } catch (const std::exception& ex) {
    return Status(ErrorCode::kParseError, "invalid value for key '" + raw_key + "': " + ex.what());
  }

  return Status::Ok();
}

}  // namespace

Result<RuntimeConfig> LoadRuntimeConfig(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    return Status(ErrorCode::kIoError, "unable to open config file: " + path.string());
  }

  RuntimeConfig config;
  std::string line;
  int line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line_number == 1) {
      line = StripUtf8Bom(std::move(line));
    }
    const auto trimmed = Trim(line);
    if (trimmed.empty() || trimmed[0] == '#') {
      continue;
    }

    const auto delimiter = trimmed.find('=');
    if (delimiter == std::string::npos) {
      return Status(ErrorCode::kParseError,
                    "expected key=value at " + path.string() + ":" + std::to_string(line_number));
    }

    auto status = ApplyKeyValue(config, trimmed.substr(0, delimiter), trimmed.substr(delimiter + 1));
    if (!status.ok()) {
      return status;
    }
  }

  auto status = ValidateRuntimeConfig(config);
  if (!status.ok()) {
    return status;
  }
  return config;
}

Status ApplyConfigOverrides(RuntimeConfig& config, const RuntimeConfigOverrides& overrides) {
  if (overrides.backend.has_value()) {
    config.backend = Lower(*overrides.backend);
  }
  if (overrides.model_dir.has_value()) {
    config.model_dir = *overrides.model_dir;
  }
  if (overrides.execution_provider.has_value()) {
    config.execution_provider = Lower(*overrides.execution_provider);
  }
  if (overrides.prompt.has_value()) {
    config.prompt = *overrides.prompt;
  }
  if (overrides.max_new_tokens.has_value()) {
    config.generation.max_new_tokens = *overrides.max_new_tokens;
  }
  if (overrides.emit_json.has_value()) {
    config.emit_json = *overrides.emit_json;
  }
  return ValidateRuntimeConfig(config);
}

Status ValidateRuntimeConfig(const RuntimeConfig& config) {
  if (config.backend != "mock" && config.backend != "ort-genai") {
    return Status(ErrorCode::kInvalidArgument, "backend must be one of: mock, ort-genai");
  }
  if (config.backend == "ort-genai" && config.model_dir.empty()) {
    return Status(ErrorCode::kInvalidArgument, "model_dir is required for backend=ort-genai");
  }
  if (config.generation.max_new_tokens <= 0) {
    return Status(ErrorCode::kInvalidArgument, "max_new_tokens must be positive");
  }
  if (config.generation.temperature < 0.0F) {
    return Status(ErrorCode::kInvalidArgument, "temperature must be non-negative");
  }
  if (config.generation.top_p <= 0.0F || config.generation.top_p > 1.0F) {
    return Status(ErrorCode::kInvalidArgument, "top_p must be in (0, 1]");
  }
  return Status::Ok();
}

}  // namespace scene_describer
