#include "scene_describer/analyzer.hpp"
#include "scene_describer/config.hpp"
#include "scene_describer/image.hpp"
#include "scene_describer/scene_describer.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using scene_describer::RuntimeConfig;
using scene_describer::RuntimeConfigOverrides;

struct AnalyzerCli {
  bool help{false};
  bool print_prompt{false};
  std::filesystem::path config_path;
  std::filesystem::path template_dir{"prompts/analyzer"};
  RuntimeConfigOverrides overrides;
  scene_describer::AnalyzerRequest request;
  std::vector<std::string> track_specs;
  std::string error;
};

void PrintUsage(std::ostream& output) {
  output << "scene_analyzer --image <path> [options]\n\n"
         << "Options:\n"
         << "  --config <path>             Load key=value runtime config\n"
         << "  --templates <dir>           Prompt template directory (default: prompts/analyzer)\n"
         << "  --backend <mock|ort-genai>  Select backend\n"
         << "  --model-dir <path>          ONNX Runtime GenAI model package directory\n"
         << "  --execution-provider <name> Execution provider hint: cpu, cuda, dml, qnn\n"
         << "  --image <path>              Add an image frame. May be repeated\n"
         << "  --detail-image <path>       Add a high-resolution detail crop. May be repeated\n"
         << "  --timestamp-ms <n>          Timestamp for the most recent frame\n"
         << "  --frame-note <text>         Note for the most recent image or detail crop\n"
         << "  --request-id <id>           Request identifier\n"
         << "  --history <text>            Prior summary. May be repeated\n"
         << "  --track <spec>              Track metadata: frame,id,label,x,y,w,h,confidence\n"
         << "  --prompt <text>             Extra prompt appended to analyzer prompt\n"
         << "  --max-new-tokens <n>        Max generated tokens\n"
         << "  --json                      Emit machine-readable JSON\n"
         << "  --print-prompt              Print the built prompt without inference\n"
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

std::vector<std::string> Split(const std::string& value, char delimiter) {
  std::vector<std::string> parts;
  std::string part;
  std::istringstream input(value);
  while (std::getline(input, part, delimiter)) {
    parts.push_back(part);
  }
  return parts;
}

bool ParseInt64(const std::string& value, std::int64_t& output, std::string& error, const std::string& option) {
  try {
    size_t parsed_count = 0;
    output = std::stoll(value, &parsed_count);
    if (parsed_count != value.size()) {
      error = "invalid " + option + " value: trailing characters";
      return false;
    }
    return true;
  } catch (const std::exception& ex) {
    error = "invalid " + option + " value: " + ex.what();
    return false;
  }
}

bool ParseTrackSpec(const std::string& spec, scene_describer::AnalyzerRequest& request, std::string& error) {
  const auto parts = Split(spec, ',');
  if (parts.size() != 8) {
    error = "track spec must be frame,id,label,x,y,w,h,confidence";
    return false;
  }

  try {
    const auto frame_index = static_cast<std::size_t>(std::stoull(parts[0]));
    if (frame_index >= request.frames.size()) {
      error = "track frame index is out of range: " + parts[0];
      return false;
    }

    scene_describer::TrackMetadata track;
    track.track_id = parts[1];
    track.label = parts[2];
    track.bbox.x = std::stod(parts[3]);
    track.bbox.y = std::stod(parts[4]);
    track.bbox.width = std::stod(parts[5]);
    track.bbox.height = std::stod(parts[6]);
    track.confidence = std::stod(parts[7]);
    request.frames[frame_index].tracks.push_back(track);
    return true;
  } catch (const std::exception& ex) {
    error = std::string("invalid track spec: ") + ex.what();
    return false;
  }
}

AnalyzerCli ParseArgs(int argc, char** argv) {
  AnalyzerCli result;

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
    } else if (arg == "--templates") {
      result.template_dir = require_value(arg);
    } else if (arg == "--backend") {
      result.overrides.backend = require_value(arg);
    } else if (arg == "--model-dir") {
      result.overrides.model_dir = require_value(arg);
    } else if (arg == "--execution-provider") {
      result.overrides.execution_provider = require_value(arg);
    } else if (arg == "--request-id") {
      result.request.request_id = require_value(arg);
    } else if (arg == "--image") {
      scene_describer::AnalyzerFrame frame;
      frame.image_path = require_value(arg);
      frame.frame_id = "frame-" + std::to_string(result.request.frames.size());
      frame.timestamp_ms = static_cast<std::int64_t>(result.request.frames.size());
      result.request.frames.push_back(std::move(frame));
    } else if (arg == "--detail-image") {
      scene_describer::AnalyzerFrame frame;
      frame.image_path = require_value(arg);
      frame.frame_id = "detail-" + std::to_string(result.request.frames.size());
      frame.detail_view = true;
      frame.timestamp_ms = result.request.frames.empty() ? 0 : result.request.frames.back().timestamp_ms;
      result.request.frames.push_back(std::move(frame));
    } else if (arg == "--timestamp-ms") {
      const auto value = require_value(arg);
      if (result.request.frames.empty()) {
        result.error = "--timestamp-ms requires a preceding --image or --detail-image";
      } else {
        std::int64_t timestamp = 0;
        if (ParseInt64(value, timestamp, result.error, arg)) {
          result.request.frames.back().timestamp_ms = timestamp;
        }
      }
    } else if (arg == "--frame-note") {
      const auto value = require_value(arg);
      if (result.request.frames.empty()) {
        result.error = "--frame-note requires a preceding --image or --detail-image";
      } else {
        result.request.frames.back().note = value;
      }
    } else if (arg == "--history") {
      result.request.prior_summaries.push_back(require_value(arg));
    } else if (arg == "--track") {
      result.track_specs.push_back(require_value(arg));
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
    } else if (arg == "--print-prompt") {
      result.print_prompt = true;
    } else {
      result.error = "unknown option: " + arg;
    }

    if (!result.error.empty()) {
      break;
    }
  }

  if (result.error.empty()) {
    for (const auto& track_spec : result.track_specs) {
      if (!ParseTrackSpec(track_spec, result.request, result.error)) {
        break;
      }
    }
  }

  return result;
}

}  // namespace

int main(int argc, char** argv) {
  auto cli = ParseArgs(argc, argv);
  if (!cli.error.empty()) {
    std::cerr << "error: " << cli.error << "\n\n";
    PrintUsage(std::cerr);
    return 2;
  }
  if (cli.help) {
    PrintUsage(std::cout);
    return 0;
  }
  if (cli.request.frames.empty()) {
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

  auto templates = scene_describer::LoadPromptTemplates(cli.template_dir);
  if (!templates.ok()) {
    std::cerr << "error: " << templates.status().message() << "\n";
    return 2;
  }

  auto analyzer_prompt = scene_describer::BuildAnalyzerPrompt(cli.request, templates.value());
  if (!analyzer_prompt.ok()) {
    std::cerr << "error: " << analyzer_prompt.status().message() << "\n";
    return 2;
  }

  if (cli.overrides.prompt.has_value()) {
    analyzer_prompt.value() += "\n\nAdditional request instruction:\n" + *cli.overrides.prompt;
  }

  if (cli.print_prompt) {
    std::cout << analyzer_prompt.value() << "\n";
    return 0;
  }

  auto backend = scene_describer::CreateSceneDescriber(config);
  if (!backend.ok()) {
    std::cerr << "error: " << backend.status().message() << "\n";
    return 2;
  }

  scene_describer::SceneDescriptionRequest request;
  request.prompt = analyzer_prompt.value();
  request.generation = config.generation;
  request.image_path = cli.request.frames.front().image_path;
  for (const auto& frame : cli.request.frames) {
    request.image_paths.push_back(frame.image_path);
  }

  if (config.backend == "mock") {
    auto image = scene_describer::LoadImage(request.image_path);
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

  auto latest_timestamp = scene_describer::LatestFrameTimestampMs(cli.request);
  if (!latest_timestamp.ok()) {
    std::cerr << "error: " << latest_timestamp.status().message() << "\n";
    return 2;
  }

  scene_describer::AnalyzerResult result;
  result.request_id = cli.request.request_id;
  result.summary = description.value().text;
  result.latest_timestamp_ms = latest_timestamp.value();
  result.metadata = description.value().metadata;
  std::size_t source_frame_count = 0;
  std::size_t detail_image_count = 0;
  for (const auto& frame : cli.request.frames) {
    if (frame.detail_view) {
      ++detail_image_count;
    } else {
      ++source_frame_count;
    }
  }
  result.metadata["frame_count"] = std::to_string(source_frame_count);
  result.metadata["detail_image_count"] = std::to_string(detail_image_count);
  result.metadata["input_image_count"] = std::to_string(cli.request.frames.size());
  result.metadata["track_count"] = "0";
  std::size_t track_count = 0;
  for (const auto& frame : cli.request.frames) {
    track_count += frame.tracks.size();
  }
  result.metadata["track_count"] = std::to_string(track_count);

  if (config.emit_json) {
    std::cout << "{\n"
              << "  \"request_id\": \"" << JsonEscape(result.request_id) << "\",\n"
              << "  \"summary\": \"" << JsonEscape(result.summary) << "\",\n"
              << "  \"latest_timestamp_ms\": " << result.latest_timestamp_ms << ",\n"
              << "  \"metadata\": {\n";
    std::size_t emitted = 0;
    for (const auto& [key, value] : result.metadata) {
      std::cout << "    \"" << JsonEscape(key) << "\": \"" << JsonEscape(value) << "\"";
      ++emitted;
      std::cout << (emitted == result.metadata.size() ? "\n" : ",\n");
    }
    std::cout << "  }\n"
              << "}\n";
  } else {
    std::cout << result.summary << "\n";
  }

  return 0;
}
