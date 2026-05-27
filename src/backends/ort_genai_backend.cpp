#include "scene_describer/scene_describer.hpp"

#include "qwen35_raw_ort_cuda.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if SCENE_DESC_HAS_ORT_GENAI
#include <ort_genai.h>
#endif

namespace scene_describer {
namespace {

#if SCENE_DESC_HAS_ORT_GENAI

std::string NormalizeProvider(std::string provider) {
  std::transform(provider.begin(), provider.end(), provider.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return provider;
}

std::string BuildPrompt(const std::string& model_type, const std::string& user_prompt, int image_count) {
  const auto normalized = NormalizeProvider(model_type);

  if (normalized.find("qwen") != std::string::npos || normalized.find("fara") != std::string::npos) {
    std::ostringstream prompt;
    prompt << "<|im_start|>user\n";
    for (int index = 0; index < image_count; ++index) {
      prompt << "<|vision_start|><|image_pad|><|vision_end|>";
    }
    prompt << user_prompt << "<|im_end|>\n<|im_start|>assistant\n";
    return prompt.str();
  }

  if (normalized == "phi3v") {
    std::ostringstream prompt;
    for (int index = 0; index < image_count; ++index) {
      prompt << "<|image_" << (index + 1) << "|>\n";
    }
    prompt << user_prompt;
    return prompt.str();
  }

  return user_prompt;
}

void ConfigureProvider(OgaConfig& config, const RuntimeConfig& runtime_config) {
  const auto provider = NormalizeProvider(runtime_config.execution_provider);
  if (provider.empty() || provider == "follow_config") {
    return;
  }

  config.ClearProviders();
  if (provider != "cpu") {
    config.AppendProvider(runtime_config.execution_provider.c_str());
  }
}

bool ShouldBootstrapOgaOnCpu(const RuntimeConfig& runtime_config) {
  const auto provider = NormalizeProvider(runtime_config.execution_provider);
  if (provider != "cuda") {
    return false;
  }
  auto model_dir = runtime_config.model_dir;
  std::transform(model_dir.begin(), model_dir.end(), model_dir.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return model_dir.find("qwen3.5") != std::string::npos || model_dir.find("qwen35") != std::string::npos;
}

size_t GetInputTokenCount(OgaNamedTensors& inputs) {
  auto input_ids = inputs.Get("input_ids");
  if (!input_ids) {
    return 0;
  }

  const auto shape = input_ids->Shape();
  if (shape.empty()) {
    return 0;
  }

  const auto token_count = shape.back();
  return token_count > 0 ? static_cast<size_t>(token_count) : 0;
}

std::string TrimLeadingWhitespace(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  value.erase(0, first);
  return value;
}

std::string StripLeadingThinkBlock(std::string value) {
  value = TrimLeadingWhitespace(std::move(value));
  constexpr std::string_view think_start = "<think>";
  constexpr std::string_view think_end = "</think>";
  if (value.rfind(think_start, 0) != 0) {
    return value;
  }

  const auto end_position = value.find(think_end);
  if (end_position == std::string::npos) {
    return value;
  }

  value.erase(0, end_position + think_end.size());
  return TrimLeadingWhitespace(std::move(value));
}

#endif

class OrtGenAiSceneDescriber final : public ISceneDescriber {
 public:
  explicit OrtGenAiSceneDescriber(RuntimeConfig config) : config_(std::move(config)) {
#if SCENE_DESC_HAS_ORT_GENAI
    auto oga_config = OgaConfig::Create(config_.model_dir.c_str());
    if (!ShouldBootstrapOgaOnCpu(config_)) {
      ConfigureProvider(*oga_config, config_);
    }
    model_ = OgaModel::Create(*oga_config);
    processor_ = OgaMultiModalProcessor::Create(*model_);
#endif
  }

  Result<SceneDescription> Describe(const SceneDescriptionRequest& request) override {
#if SCENE_DESC_HAS_ORT_GENAI
    try {
      std::vector<std::string> requested_image_paths;
      if (!request.image_paths.empty()) {
        requested_image_paths = request.image_paths;
      } else if (!request.image_path.empty()) {
        requested_image_paths.push_back(request.image_path);
      }

      if (requested_image_paths.empty()) {
        return Status(ErrorCode::kInvalidArgument, "image_path is required for backend=ort-genai");
      }

      std::vector<const char*> image_paths;
      image_paths.reserve(requested_image_paths.size());
      for (const auto& image_path : requested_image_paths) {
        if (!std::filesystem::exists(image_path)) {
          return Status(ErrorCode::kIoError, "image file does not exist: " + image_path);
        }
        image_paths.push_back(image_path.c_str());
      }

      auto images = OgaImages::Load(image_paths);
      const std::string model_type = static_cast<const char*>(model_->GetType());
      const auto prompt = BuildPrompt(model_type, request.prompt, static_cast<int>(image_paths.size()));
      auto inputs = processor_->ProcessImages(prompt.c_str(), images.get());
      const auto input_token_count = GetInputTokenCount(*inputs);

      if (ShouldUseQwen35RawOrtCuda(config_, model_type)) {
        return DescribeQwen35RawOrtCuda(config_,
                                        *processor_,
                                        *inputs,
                                        model_type,
                                        input_token_count,
                                        requested_image_paths.size());
      }

      auto params = OgaGeneratorParams::Create(*model_);
      const auto requested_new_tokens = std::max(1, request.generation.max_new_tokens);
      const auto max_length = input_token_count > 0
                                  ? static_cast<int>(input_token_count) + requested_new_tokens
                                  : std::max(512, requested_new_tokens + 512);
      params->SetSearchOption("max_length", static_cast<double>(max_length));
      params->SetSearchOption("batch_size", 1.0);
      params->SetSearchOptionBool("do_sample", !request.generation.deterministic);
      if (!request.generation.deterministic) {
        params->SetSearchOption("temperature", static_cast<double>(request.generation.temperature));
        params->SetSearchOption("top_p", static_cast<double>(request.generation.top_p));
      }

      auto generator = OgaGenerator::Create(*model_, *params);
      generator->SetInputs(*inputs);
      while (!generator->IsDone()) {
        generator->GenerateNextToken();
      }

      const auto* sequence = generator->GetSequenceData(0);
      const auto sequence_count = generator->GetSequenceCount(0);
      const auto decode_offset = std::min(input_token_count, sequence_count);
      auto output = processor_->Decode(sequence + decode_offset, sequence_count - decode_offset);

      SceneDescription description;
      description.text = StripLeadingThinkBlock(static_cast<const char*>(output));
      description.metadata["backend"] = "ort-genai";
      description.metadata["model_type"] = model_type;
      description.metadata["model_dir"] = config_.model_dir;
      description.metadata["image_count"] = std::to_string(image_paths.size());
      description.metadata["input_tokens"] = std::to_string(input_token_count);
      description.metadata["generated_tokens"] = std::to_string(sequence_count - decode_offset);
      return description;
    } catch (const std::exception& ex) {
      return Status(ErrorCode::kRuntimeError, std::string("ONNX Runtime GenAI inference failed: ") + ex.what());
    }
#else
    (void)request;
    return Status(ErrorCode::kBackendUnavailable,
                  "backend=ort-genai requires configuring and linking ONNX Runtime GenAI");
#endif
  }

 private:
  RuntimeConfig config_;
#if SCENE_DESC_HAS_ORT_GENAI
  std::unique_ptr<OgaModel> model_;
  std::unique_ptr<OgaMultiModalProcessor> processor_;
#endif
};

}  // namespace

std::unique_ptr<ISceneDescriber> CreateOrtGenAiSceneDescriber(RuntimeConfig config) {
  return std::make_unique<OrtGenAiSceneDescriber>(std::move(config));
}

}  // namespace scene_describer
