#include "qwen35_raw_ort_cuda.hpp"

#if SCENE_DESC_HAS_ORT_GENAI

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace scene_describer {
namespace {

constexpr int64_t kBatchSize = 1;
constexpr int64_t kHiddenSize = 2048;
constexpr int32_t kImStartTokenId = 248044;
constexpr int32_t kImEndTokenId = 248046;

std::string NormalizeProvider(std::string provider) {
  std::transform(provider.begin(), provider.end(), provider.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return provider;
}

size_t ElementCount(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return 1;
  }
  size_t count = 1;
  for (const auto dim : shape) {
    if (dim < 0) {
      throw std::runtime_error("cannot count tensor elements with dynamic dimension");
    }
    count *= static_cast<size_t>(dim);
  }
  return count;
}

std::filesystem::path FindModelFile(const std::filesystem::path& model_dir, const std::string& prefix) {
  const auto onnx_dir = model_dir / "onnx";
  std::vector<std::filesystem::path> matches;
  for (const auto& entry : std::filesystem::directory_iterator(onnx_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto filename = entry.path().filename().string();
    if (filename.rfind(prefix, 0) == 0 && entry.path().extension() == ".onnx") {
      matches.push_back(entry.path());
    }
  }
  if (matches.empty()) {
    throw std::runtime_error("missing ONNX graph with prefix " + prefix + " under " + onnx_dir.string());
  }
  std::sort(matches.begin(), matches.end());
  return matches.front();
}

std::vector<std::string> GetSessionInputNames(const Ort::Session& session) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> names;
  names.reserve(session.GetInputCount());
  for (size_t index = 0; index < session.GetInputCount(); ++index) {
    auto name = session.GetInputNameAllocated(index, allocator);
    names.emplace_back(name.get());
  }
  return names;
}

std::vector<std::string> GetSessionOutputNames(const Ort::Session& session) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> names;
  names.reserve(session.GetOutputCount());
  for (size_t index = 0; index < session.GetOutputCount(); ++index) {
    auto name = session.GetOutputNameAllocated(index, allocator);
    names.emplace_back(name.get());
  }
  return names;
}

std::vector<const char*> ToCStrs(const std::vector<std::string>& names) {
  std::vector<const char*> cstrs;
  cstrs.reserve(names.size());
  for (const auto& name : names) {
    cstrs.push_back(name.c_str());
  }
  return cstrs;
}

Ort::SessionOptions MakeCudaSessionOptions() {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  options.DisableMemPattern();
  OrtCUDAProviderOptions cuda_options{};
  cuda_options.device_id = 0;
  options.AppendExecutionProvider_CUDA(cuda_options);
  return options;
}

template <typename T>
std::vector<T> CopyOgaTensorData(OgaNamedTensors& inputs, const char* name, std::vector<int64_t>& shape) {
  auto tensor = inputs.Get(name);
  shape = tensor->Shape();
  const auto count = ElementCount(shape);
  const auto* data = static_cast<const T*>(tensor->Data());
  return std::vector<T>(data, data + count);
}

std::vector<int64_t> CopyInputIdsAsInt64(OgaNamedTensors& inputs, std::vector<int64_t>& shape) {
  auto tensor = inputs.Get("input_ids");
  shape = tensor->Shape();
  const auto count = ElementCount(shape);
  const auto* data = static_cast<const int32_t*>(tensor->Data());
  std::vector<int64_t> output;
  output.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    output.push_back(static_cast<int64_t>(data[index]));
  }
  return output;
}

std::vector<int64_t> MakeAttentionMask(int64_t total_length) {
  return std::vector<int64_t>(static_cast<size_t>(total_length), 1);
}

std::vector<int64_t> MakePositionIds(int64_t start_position, int64_t length) {
  std::vector<int64_t> position_ids(static_cast<size_t>(3 * length));
  for (int64_t axis = 0; axis < 3; ++axis) {
    for (int64_t index = 0; index < length; ++index) {
      position_ids[static_cast<size_t>(axis * length + index)] = start_position + index;
    }
  }
  return position_ids;
}

struct StateValue {
  std::string input_name;
  Ort::Value value{nullptr};

  StateValue(std::string name, Ort::Value state_value) : input_name(std::move(name)), value(std::move(state_value)) {}
};

std::vector<int64_t> ConcreteStateShape(const std::string& name, std::vector<int64_t> shape) {
  if (name.find(".key") != std::string::npos || name.find(".value") != std::string::npos) {
    for (size_t index = 0; index < shape.size(); ++index) {
      if (index == 0) {
        shape[index] = kBatchSize;
      } else if (index == 2) {
        shape[index] = 0;
      } else if (shape[index] < 0) {
        throw std::runtime_error("unexpected dynamic key/value state dimension for " + name);
      }
    }
    return shape;
  }

  for (auto& dim : shape) {
    if (dim < 0) {
      dim = kBatchSize;
    }
  }
  return shape;
}

std::vector<StateValue> MakeInitialDecoderStates(const Ort::Session& decoder_session, Ort::AllocatorWithDefaultOptions& allocator) {
  std::vector<StateValue> states;
  const auto input_names = GetSessionInputNames(decoder_session);

  for (size_t index = 0; index < input_names.size(); ++index) {
    const auto& name = input_names[index];
    if (name == "inputs_embeds" || name == "attention_mask" || name == "position_ids") {
      continue;
    }

    auto input_type_info = decoder_session.GetInputTypeInfo(index);
    const auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    const auto element_type = tensor_info.GetElementType();
    auto shape = ConcreteStateShape(name, tensor_info.GetShape());
    const auto element_count = ElementCount(shape);
    size_t element_size = 0;
    if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      element_size = sizeof(uint16_t);
    } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      element_size = sizeof(float);
    } else {
      throw std::runtime_error("unsupported decoder state tensor type " + std::to_string(static_cast<int>(element_type)) +
                               " for " + name);
    }

    auto value = Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), element_type);
    if (element_count > 0) {
      std::memset(value.GetTensorMutableRawData(), 0, element_count * element_size);
    }
    states.emplace_back(name, std::move(value));
  }

  return states;
}

std::unordered_map<std::string, std::string> MakeOutputToInputStateMap(const std::vector<std::string>& output_names) {
  std::unordered_map<std::string, std::string> mapping;
  for (const auto& output_name : output_names) {
    constexpr std::string_view prefix = "present.";
    if (output_name.rfind(prefix, 0) == 0) {
      mapping.emplace(output_name, "past_key_values." + output_name.substr(prefix.size()));
    }
  }
  return mapping;
}

int32_t ArgmaxLastLogit(const Ort::Value& logits) {
  const auto info = logits.GetTensorTypeAndShapeInfo();
  const auto shape = info.GetShape();
  if (shape.size() != 3 || shape[0] != 1 || shape[1] < 1 || shape[2] < 1) {
    throw std::runtime_error("unexpected logits shape");
  }
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    throw std::runtime_error("expected float16 decoder logits");
  }

  const auto vocab_size = static_cast<size_t>(shape[2]);
  const auto last_row = static_cast<size_t>(shape[1] - 1) * vocab_size;
  const auto* data = logits.GetTensorData<Ort::Float16_t>() + last_row;
  int32_t best_index = 0;
  float best_value = -std::numeric_limits<float>::infinity();
  for (size_t index = 0; index < vocab_size; ++index) {
    const auto value = data[index].ToFloat();
    if (value > best_value) {
      best_value = value;
      best_index = static_cast<int32_t>(index);
    }
  }
  return best_index;
}

std::string StripLeadingThinkBlock(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  value.erase(0, first);

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
  const auto second_first = value.find_first_not_of(" \t\r\n");
  if (second_first == std::string::npos) {
    return {};
  }
  value.erase(0, second_first);
  return value;
}

}  // namespace

bool ShouldUseQwen35RawOrtCuda(const RuntimeConfig& config, const std::string& model_type) {
  return NormalizeProvider(config.execution_provider) == "cuda" && model_type == "qwen3_5";
}

Result<SceneDescription> DescribeQwen35RawOrtCuda(const RuntimeConfig& config,
                                                  OgaMultiModalProcessor& processor,
                                                  OgaNamedTensors& inputs,
                                                  const std::string& model_type,
                                                  size_t input_token_count,
                                                  size_t image_count) {
  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "scene-describer-qwen35-raw-ort-cuda");
    auto session_options = MakeCudaSessionOptions();
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;

    const std::filesystem::path model_dir(config.model_dir);
    Ort::Session vision_session(env, FindModelFile(model_dir, "vision_encoder_").c_str(), session_options);
    Ort::Session embedding_session(env, FindModelFile(model_dir, "embed_tokens_").c_str(), session_options);
    Ort::Session decoder_session(env, FindModelFile(model_dir, "decoder_model_merged_").c_str(), session_options);

    std::vector<int64_t> pixel_shape;
    auto pixel_values = CopyOgaTensorData<float>(inputs, "pixel_values", pixel_shape);
    std::vector<int64_t> image_grid_shape;
    auto image_grid_thw = CopyOgaTensorData<int64_t>(inputs, "image_grid_thw", image_grid_shape);
    std::vector<int64_t> input_ids_shape;
    auto input_ids = CopyInputIdsAsInt64(inputs, input_ids_shape);

    std::vector<const char*> vision_input_names{"pixel_values", "image_grid_thw"};
    std::vector<const char*> vision_output_names{"image_features"};
    std::vector<Ort::Value> vision_inputs;
    vision_inputs.push_back(Ort::Value::CreateTensor<float>(
        memory_info, pixel_values.data(), pixel_values.size(), pixel_shape.data(), pixel_shape.size()));
    vision_inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        memory_info, image_grid_thw.data(), image_grid_thw.size(), image_grid_shape.data(), image_grid_shape.size()));
    auto vision_outputs = vision_session.Run(Ort::RunOptions{nullptr},
                                             vision_input_names.data(),
                                             vision_inputs.data(),
                                             vision_inputs.size(),
                                             vision_output_names.data(),
                                             vision_output_names.size());
    if (vision_outputs.empty()) {
      throw std::runtime_error("vision encoder returned no image_features");
    }
    const auto image_features_shape = vision_outputs.front().GetTensorTypeAndShapeInfo().GetShape();
    if (image_features_shape.size() != 2 || image_features_shape[1] != kHiddenSize) {
      throw std::runtime_error("unexpected image_features shape");
    }

    const auto decoder_output_names = GetSessionOutputNames(decoder_session);
    const auto decoder_output_name_ptrs = ToCStrs(decoder_output_names);
    const auto output_to_input_state = MakeOutputToInputStateMap(decoder_output_names);

    std::vector<StateValue> states = MakeInitialDecoderStates(decoder_session, allocator);
    std::vector<int32_t> generated_tokens;
    generated_tokens.reserve(static_cast<size_t>(std::max(1, config.generation.max_new_tokens)));

    int64_t total_length = static_cast<int64_t>(input_token_count);
    std::vector<int64_t> next_input_ids = std::move(input_ids);
    std::vector<int64_t> next_input_ids_shape = std::move(input_ids_shape);
    std::vector<float> empty_image_features_storage(1, 0.0F);
    std::vector<int64_t> empty_image_features_shape{0, kHiddenSize};

    for (int step = 0; step < std::max(1, config.generation.max_new_tokens); ++step) {
      std::vector<const char*> embedding_input_names{"input_ids", "image_features"};
      std::vector<const char*> embedding_output_names{"inputs_embeds"};
      std::vector<Ort::Value> embedding_inputs;
      embedding_inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory_info,
                                                                    next_input_ids.data(),
                                                                    next_input_ids.size(),
                                                                    next_input_ids_shape.data(),
                                                                    next_input_ids_shape.size()));
      if (step == 0) {
        embedding_inputs.push_back(std::move(vision_outputs.front()));
      } else {
        embedding_inputs.push_back(Ort::Value::CreateTensor<float>(memory_info,
                                                                   empty_image_features_storage.data(),
                                                                   0,
                                                                   empty_image_features_shape.data(),
                                                                   empty_image_features_shape.size()));
      }
      auto embedding_outputs = embedding_session.Run(Ort::RunOptions{nullptr},
                                                     embedding_input_names.data(),
                                                     embedding_inputs.data(),
                                                     embedding_inputs.size(),
                                                     embedding_output_names.data(),
                                                     embedding_output_names.size());
      if (embedding_outputs.empty()) {
        throw std::runtime_error("embedding graph returned no inputs_embeds");
      }

      const auto inputs_embeds_shape = embedding_outputs.front().GetTensorTypeAndShapeInfo().GetShape();
      if (inputs_embeds_shape.size() != 3 || inputs_embeds_shape[0] != kBatchSize || inputs_embeds_shape[2] != kHiddenSize) {
        throw std::runtime_error("unexpected inputs_embeds shape");
      }
      const auto current_length = inputs_embeds_shape[1];
      auto attention_mask = MakeAttentionMask(total_length);
      auto position_ids = MakePositionIds(total_length - current_length, current_length);
      std::vector<int64_t> attention_mask_shape{kBatchSize, total_length};
      std::vector<int64_t> position_ids_shape{3, kBatchSize, current_length};

      std::vector<std::string> decoder_input_names{"inputs_embeds", "attention_mask", "position_ids"};
      decoder_input_names.reserve(3 + states.size());
      std::vector<Ort::Value> decoder_inputs;
      decoder_inputs.reserve(3 + states.size());
      decoder_inputs.push_back(std::move(embedding_outputs.front()));
      decoder_inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory_info,
                                                                 attention_mask.data(),
                                                                 attention_mask.size(),
                                                                 attention_mask_shape.data(),
                                                                 attention_mask_shape.size()));
      decoder_inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory_info,
                                                                 position_ids.data(),
                                                                 position_ids.size(),
                                                                 position_ids_shape.data(),
                                                                 position_ids_shape.size()));
      for (auto& state : states) {
        decoder_input_names.push_back(state.input_name);
        decoder_inputs.push_back(std::move(state.value));
      }
      const auto decoder_input_name_ptrs = ToCStrs(decoder_input_names);
      auto decoder_outputs = decoder_session.Run(Ort::RunOptions{nullptr},
                                                 decoder_input_name_ptrs.data(),
                                                 decoder_inputs.data(),
                                                 decoder_inputs.size(),
                                                 decoder_output_name_ptrs.data(),
                                                 decoder_output_name_ptrs.size());
      if (decoder_outputs.empty()) {
        throw std::runtime_error("decoder returned no logits");
      }

      const int32_t next_token = ArgmaxLastLogit(decoder_outputs.front());
      generated_tokens.push_back(next_token);

      std::vector<StateValue> next_states;
      next_states.reserve(decoder_outputs.size() - 1);
      for (size_t output_index = 1; output_index < decoder_outputs.size(); ++output_index) {
        const auto mapping = output_to_input_state.find(decoder_output_names[output_index]);
        if (mapping != output_to_input_state.end()) {
          next_states.emplace_back(mapping->second, std::move(decoder_outputs[output_index]));
        }
      }
      states = std::move(next_states);

      total_length += 1;
      next_input_ids = {next_token};
      next_input_ids_shape = {kBatchSize, 1};
      if (next_token == kImEndTokenId || next_token == kImStartTokenId) {
        break;
      }
    }

    auto decoded = processor.Decode(generated_tokens.data(), generated_tokens.size());

    SceneDescription description;
    description.text = StripLeadingThinkBlock(static_cast<const char*>(decoded));
    description.metadata["backend"] = "ort-genai";
    description.metadata["model_type"] = model_type;
    description.metadata["model_dir"] = config.model_dir;
    description.metadata["image_count"] = std::to_string(image_count);
    description.metadata["input_tokens"] = std::to_string(input_token_count);
    description.metadata["generated_tokens"] = std::to_string(generated_tokens.size());
    description.metadata["execution_provider"] = "raw-ort-cuda";
    description.metadata["generation"] = "greedy";
    return description;
  } catch (const std::exception& ex) {
    return Status(ErrorCode::kRuntimeError, std::string("Qwen3.5 raw ONNX Runtime CUDA inference failed: ") + ex.what());
  }
}

}  // namespace scene_describer

#endif
