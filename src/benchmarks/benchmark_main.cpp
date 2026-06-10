#include "scene_describer/config.hpp"
#include "scene_describer/image.hpp"
#include "scene_describer/scene_describer.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using scene_describer::RuntimeConfig;
using scene_describer::RuntimeConfigOverrides;
using Clock = std::chrono::steady_clock;

struct BenchmarkArgs {
  bool help{false};
  bool json{false};
  std::filesystem::path config_path;
  std::vector<std::filesystem::path> image_paths;
  RuntimeConfigOverrides overrides;
  int warmups{1};
  int repeats{5};
  std::string error;
};

struct Stats {
  double minimum{0.0};
  double median{0.0};
  double mean{0.0};
  double maximum{0.0};
};

struct TimedDescription {
  double elapsed_ms{0.0};
  scene_describer::SceneDescription description;
};

void PrintUsage(std::ostream& output) {
  output << "scene_describer_benchmark --image <path> [options]\n\n"
         << "Options:\n"
         << "  --config <path>             Load key=value runtime config\n"
         << "  --backend <mock|ort-genai>  Select backend (default: mock)\n"
         << "  --model-dir <path>          ONNX Runtime GenAI model package directory\n"
         << "  --execution-provider <name> Execution provider hint: cpu, cuda, dml, qnn\n"
         << "  --image <path>              Image path; repeat for frame batches\n"
         << "  --prompt <text>             Prompt sent to the model\n"
         << "  --max-new-tokens <n>        Max generated tokens\n"
         << "  --warmups <n>               Warmup requests before measurement (default: 1)\n"
         << "  --repeats <n>               Measured requests (default: 5)\n"
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

bool ParseInteger(const std::string& option, const std::string& value, int minimum, int& output,
                  std::string& error) {
  try {
    size_t parsed_count = 0;
    const int parsed = std::stoi(value, &parsed_count);
    if (parsed_count != value.size()) {
      error = "invalid " + option + " value: trailing characters";
      return false;
    }
    if (parsed < minimum) {
      error = option + " must be at least " + std::to_string(minimum);
      return false;
    }
    output = parsed;
    return true;
  } catch (const std::exception& ex) {
    error = "invalid " + option + " value: " + ex.what();
    return false;
  }
}

std::optional<int> ParseIntegerMetadata(const std::map<std::string, std::string>& metadata,
                                        const std::string& key) {
  const auto found = metadata.find(key);
  if (found == metadata.end()) {
    return std::nullopt;
  }
  try {
    size_t parsed_count = 0;
    const int parsed = std::stoi(found->second, &parsed_count);
    if (parsed_count != found->second.size()) {
      return std::nullopt;
    }
    return parsed;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

BenchmarkArgs ParseArgs(int argc, char** argv) {
  BenchmarkArgs result;

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
      const auto value = require_value(arg);
      if (result.error.empty()) {
        result.image_paths.push_back(value);
      }
    } else if (arg == "--prompt") {
      result.overrides.prompt = require_value(arg);
    } else if (arg == "--max-new-tokens") {
      const auto value = require_value(arg);
      int parsed = 0;
      if (result.error.empty() && ParseInteger(arg, value, 1, parsed, result.error)) {
        result.overrides.max_new_tokens = parsed;
      }
    } else if (arg == "--warmups") {
      const auto value = require_value(arg);
      if (result.error.empty()) {
        ParseInteger(arg, value, 0, result.warmups, result.error);
      }
    } else if (arg == "--repeats") {
      const auto value = require_value(arg);
      if (result.error.empty()) {
        ParseInteger(arg, value, 1, result.repeats, result.error);
      }
    } else if (arg == "--json") {
      result.json = true;
    } else {
      result.error = "unknown option: " + arg;
    }

    if (!result.error.empty()) {
      break;
    }
  }

  return result;
}

double ElapsedMilliseconds(Clock::time_point start, Clock::time_point finish) {
  return std::chrono::duration<double, std::milli>(finish - start).count();
}

Stats ComputeStats(const std::vector<double>& values) {
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());

  Stats stats;
  stats.minimum = sorted.front();
  stats.maximum = sorted.back();
  const auto count = sorted.size();
  if (count % 2 == 0) {
    stats.median = (sorted[(count / 2) - 1] + sorted[count / 2]) / 2.0;
  } else {
    stats.median = sorted[count / 2];
  }
  stats.mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(count);
  return stats;
}

scene_describer::Result<TimedDescription> RunOnce(scene_describer::ISceneDescriber& backend,
                                                  const scene_describer::SceneDescriptionRequest& request) {
  const auto start = Clock::now();
  auto description = backend.Describe(request);
  const auto finish = Clock::now();
  if (!description.ok()) {
    return description.status();
  }

  TimedDescription timed;
  timed.elapsed_ms = ElapsedMilliseconds(start, finish);
  timed.description = std::move(description.value());
  return timed;
}

scene_describer::Result<scene_describer::SceneDescriptionRequest> PrepareRequest(
    const BenchmarkArgs& args, const RuntimeConfig& config) {
  scene_describer::SceneDescriptionRequest request;
  request.image_path = args.image_paths.front().string();
  request.image_paths.reserve(args.image_paths.size());
  for (const auto& image_path : args.image_paths) {
    request.image_paths.push_back(image_path.string());
  }
  request.prompt = config.prompt;
  request.generation = config.generation;

  if (config.backend == "mock") {
    auto image = scene_describer::LoadImage(args.image_paths.front());
    if (!image.ok()) {
      return image.status();
    }
    request.decoded_image = std::move(image.value());
  }

  return request;
}

void WriteStatsJson(std::ostream& output, const Stats& stats) {
  output << "{\n"
         << "    \"min\": " << stats.minimum << ",\n"
         << "    \"median\": " << stats.median << ",\n"
         << "    \"mean\": " << stats.mean << ",\n"
         << "    \"max\": " << stats.maximum << "\n"
         << "  }";
}

void WriteMetadataJson(std::ostream& output, const std::map<std::string, std::string>& metadata) {
  output << "{\n";
  size_t emitted = 0;
  for (const auto& [key, value] : metadata) {
    output << "    \"" << JsonEscape(key) << "\": \"" << JsonEscape(value) << "\"";
    ++emitted;
    output << (emitted == metadata.size() ? "\n" : ",\n");
  }
  output << "  }";
}

void WritePathArrayJson(std::ostream& output, const std::vector<std::filesystem::path>& paths) {
  output << "[";
  for (size_t index = 0; index < paths.size(); ++index) {
    output << (index == 0 ? "\n" : ",\n")
           << "    \"" << JsonEscape(paths[index].string()) << "\"";
  }
  if (!paths.empty()) {
    output << "\n  ";
  }
  output << "]";
}

}  // namespace

int main(int argc, char** argv) {
  const auto args = ParseArgs(argc, argv);
  if (!args.error.empty()) {
    std::cerr << "error: " << args.error << "\n\n";
    PrintUsage(std::cerr);
    return 2;
  }
  if (args.help) {
    PrintUsage(std::cout);
    return 0;
  }
  if (args.image_paths.empty()) {
    std::cerr << "error: --image is required\n\n";
    PrintUsage(std::cerr);
    return 2;
  }

  RuntimeConfig config;
  if (!args.config_path.empty()) {
    auto loaded = scene_describer::LoadRuntimeConfig(args.config_path);
    if (!loaded.ok()) {
      std::cerr << "error: " << loaded.status().message() << "\n";
      return 2;
    }
    config = loaded.value();
  }

  auto override_status = scene_describer::ApplyConfigOverrides(config, args.overrides);
  if (!override_status.ok()) {
    std::cerr << "error: " << override_status.message() << "\n";
    return 2;
  }

  auto request = PrepareRequest(args, config);
  if (!request.ok()) {
    std::cerr << "error: " << request.status().message() << "\n";
    return 2;
  }

  const auto load_start = Clock::now();
  auto backend = scene_describer::CreateSceneDescriber(config);
  const auto load_finish = Clock::now();
  if (!backend.ok()) {
    std::cerr << "error: " << backend.status().message() << "\n";
    return 2;
  }
  const double model_load_ms = ElapsedMilliseconds(load_start, load_finish);

  for (int index = 0; index < args.warmups; ++index) {
    auto warmup = RunOnce(*backend.value(), request.value());
    if (!warmup.ok()) {
      std::cerr << "error: warmup " << (index + 1) << " failed: " << warmup.status().message() << "\n";
      return 3;
    }
  }

  std::vector<double> latencies_ms;
  latencies_ms.reserve(static_cast<size_t>(args.repeats));
  std::vector<double> token_rates;
  token_rates.reserve(static_cast<size_t>(args.repeats));
  scene_describer::SceneDescription last_description;

  for (int index = 0; index < args.repeats; ++index) {
    auto measured = RunOnce(*backend.value(), request.value());
    if (!measured.ok()) {
      std::cerr << "error: repeat " << (index + 1) << " failed: " << measured.status().message() << "\n";
      return 3;
    }

    latencies_ms.push_back(measured.value().elapsed_ms);
    const auto generated_tokens = ParseIntegerMetadata(measured.value().description.metadata, "generated_tokens");
    if (generated_tokens.has_value() && *generated_tokens > 0 && measured.value().elapsed_ms > 0.0) {
      token_rates.push_back(static_cast<double>(*generated_tokens) * 1000.0 / measured.value().elapsed_ms);
    }
    last_description = std::move(measured.value().description);
  }

  const auto latency_stats = ComputeStats(latencies_ms);
  const auto throughput_stats = token_rates.empty() ? std::optional<Stats>{} : std::optional<Stats>{ComputeStats(token_rates)};
  const auto last_generated_tokens = ParseIntegerMetadata(last_description.metadata, "generated_tokens");

  std::cout << std::fixed << std::setprecision(3);
  if (args.json) {
    std::cout << "{\n"
              << "  \"backend\": \"" << JsonEscape(config.backend) << "\",\n"
              << "  \"model_dir\": \"" << JsonEscape(config.model_dir) << "\",\n"
              << "  \"execution_provider\": \"" << JsonEscape(config.execution_provider) << "\",\n"
              << "  \"image\": \"" << JsonEscape(request.value().image_path) << "\",\n"
              << "  \"image_count\": " << request.value().image_paths.size() << ",\n"
              << "  \"images\": ";
    WritePathArrayJson(std::cout, args.image_paths);
    std::cout << ",\n"
              << "  \"warmups\": " << args.warmups << ",\n"
              << "  \"repeats\": " << args.repeats << ",\n"
              << "  \"model_load_ms\": " << model_load_ms << ",\n"
              << "  \"latency_ms\": ";
    WriteStatsJson(std::cout, latency_stats);
    if (last_generated_tokens.has_value()) {
      std::cout << ",\n"
                << "  \"generated_tokens\": " << *last_generated_tokens;
    }
    if (throughput_stats.has_value()) {
      std::cout << ",\n"
                << "  \"generated_tokens_per_second\": ";
      WriteStatsJson(std::cout, *throughput_stats);
    }
    std::cout << ",\n"
              << "  \"last_text\": \"" << JsonEscape(last_description.text) << "\"";
    if (!last_description.metadata.empty()) {
      std::cout << ",\n"
                << "  \"metadata\": ";
      WriteMetadataJson(std::cout, last_description.metadata);
    }
    std::cout << "\n"
              << "}\n";
  } else {
    std::cout << "backend: " << config.backend << "\n"
              << "image_count: " << request.value().image_paths.size() << "\n"
              << "model_load_ms: " << model_load_ms << "\n"
              << "latency_ms: min=" << latency_stats.minimum << " median=" << latency_stats.median
              << " mean=" << latency_stats.mean << " max=" << latency_stats.maximum << "\n";
    if (last_generated_tokens.has_value()) {
      std::cout << "generated_tokens: " << *last_generated_tokens << "\n";
    }
    if (throughput_stats.has_value()) {
      std::cout << "generated_tokens_per_second: min=" << throughput_stats->minimum
                << " median=" << throughput_stats->median << " mean=" << throughput_stats->mean
                << " max=" << throughput_stats->maximum << "\n";
    }
    std::cout << "last_text: " << last_description.text << "\n";
  }

  return 0;
}
