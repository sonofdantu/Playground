#include <ort_genai.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ProbeOptions {
  std::filesystem::path model_dir;
  std::filesystem::path image_path;
  std::string execution_provider{"cpu"};
  std::string prompt{"Describe the visible scene in one concise sentence."};
  std::string stage{"generate"};
  int max_new_tokens{8};
  bool help{false};
};

std::string Normalize(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

void PrintUsage(std::ostream& output) {
  output << "scene_model_probe --model-dir <path> [options]\n\n"
         << "Options:\n"
         << "  --model-dir <path>          ORT GenAI model package directory\n"
         << "  --image <path>              Optional image for processor/generation stages\n"
         << "  --execution-provider <name> Execution provider hint: cpu, cuda, dml, qnn\n"
         << "  --prompt <text>             Prompt text\n"
         << "  --max-new-tokens <n>        Tokens to generate in the generate stage\n"
         << "  --stage <name>              config, model, processor, inputs, generator, token, generate\n"
         << "  --help                      Show this help\n";
}

ProbeOptions ParseArgs(int argc, char** argv) {
  ProbeOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    auto require_value = [&](const std::string& option) -> std::string {
      if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for " + option);
      }
      ++index;
      return argv[index];
    };

    if (arg == "--help" || arg == "-h") {
      options.help = true;
    } else if (arg == "--model-dir") {
      options.model_dir = require_value(arg);
    } else if (arg == "--image") {
      options.image_path = require_value(arg);
    } else if (arg == "--execution-provider") {
      options.execution_provider = require_value(arg);
    } else if (arg == "--prompt") {
      options.prompt = require_value(arg);
    } else if (arg == "--max-new-tokens") {
      options.max_new_tokens = std::stoi(require_value(arg));
    } else if (arg == "--stage") {
      options.stage = Normalize(require_value(arg));
    } else {
      throw std::invalid_argument("unknown option: " + arg);
    }
  }
  return options;
}

void ConfigureProvider(OgaConfig& config, const std::string& requested_provider) {
  const auto provider = Normalize(requested_provider);
  if (provider.empty() || provider == "follow_config") {
    return;
  }

  config.ClearProviders();
  if (provider != "cpu") {
    config.AppendProvider(requested_provider.c_str());
  }
}

std::string BuildPrompt(const std::string& model_type, const std::string& user_prompt, int image_count) {
  const auto normalized = Normalize(model_type);
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

std::string ShapeToString(const std::vector<int64_t>& shape) {
  std::ostringstream output;
  output << "[";
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) {
      output << ", ";
    }
    output << shape[index];
  }
  output << "]";
  return output.str();
}

void PrintNamedTensors(OgaNamedTensors& tensors) {
  const auto names = tensors.GetNames();
  std::cout << "named_tensors=" << names->Count() << "\n";
  for (size_t index = 0; index < names->Count(); ++index) {
    const char* name = names->Get(index);
    auto tensor = tensors.Get(name);
    std::cout << "  " << name << " type=" << static_cast<int>(tensor->Type())
              << " shape=" << ShapeToString(tensor->Shape()) << "\n";
  }
}

bool ReachedStage(const ProbeOptions& options, const std::string& stage) {
  return options.stage == stage;
}

size_t GetInputTokenCount(OgaNamedTensors& inputs) {
  auto input_ids = inputs.Get("input_ids");
  if (!input_ids) {
    return 0;
  }
  const auto shape = input_ids->Shape();
  return shape.empty() || shape.back() < 0 ? 0 : static_cast<size_t>(shape.back());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = ParseArgs(argc, argv);
    if (options.help) {
      PrintUsage(std::cout);
      return 0;
    }
    if (options.model_dir.empty()) {
      std::cerr << "error: --model-dir is required\n\n";
      PrintUsage(std::cerr);
      return 2;
    }
    if (!std::filesystem::exists(options.model_dir)) {
      std::cerr << "error: model directory does not exist: " << options.model_dir.string() << "\n";
      return 2;
    }
    if (options.max_new_tokens < 1) {
      std::cerr << "error: --max-new-tokens must be at least 1\n";
      return 2;
    }

    std::cout << "stage=config\n";
    std::cout.flush();
    auto config = OgaConfig::Create(options.model_dir.string().c_str());
    ConfigureProvider(*config, options.execution_provider);
    std::cout << "ok=config\n";
    if (ReachedStage(options, "config")) {
      return 0;
    }

    std::cout << "stage=model\n";
    std::cout.flush();
    auto model = OgaModel::Create(*config);
    const std::string model_type = static_cast<const char*>(model->GetType());
    const std::string device_type = static_cast<const char*>(model->GetDeviceType());
    std::cout << "ok=model type=" << model_type << " device=" << device_type << "\n";
    if (ReachedStage(options, "model")) {
      return 0;
    }

    std::cout << "stage=processor\n";
    std::cout.flush();
    auto processor = OgaMultiModalProcessor::Create(*model);
    std::cout << "ok=processor\n";
    if (ReachedStage(options, "processor")) {
      return 0;
    }

    if (options.image_path.empty()) {
      std::cerr << "error: --image is required for stage=" << options.stage << "\n";
      return 2;
    }
    if (!std::filesystem::exists(options.image_path)) {
      std::cerr << "error: image file does not exist: " << options.image_path.string() << "\n";
      return 2;
    }

    std::cout << "stage=inputs\n";
    std::cout.flush();
    const std::string image_path_string = options.image_path.string();
    std::vector<const char*> image_paths{image_path_string.c_str()};
    auto images = OgaImages::Load(image_paths);
    const auto prompt = BuildPrompt(model_type, options.prompt, static_cast<int>(image_paths.size()));
    auto inputs = processor->ProcessImages(prompt.c_str(), images.get());
    PrintNamedTensors(*inputs);
    const auto input_token_count = GetInputTokenCount(*inputs);
    std::cout << "ok=inputs input_tokens=" << input_token_count << "\n";
    if (ReachedStage(options, "inputs")) {
      return 0;
    }

    std::cout << "stage=generator\n";
    std::cout.flush();
    auto params = OgaGeneratorParams::Create(*model);
    const auto max_length = static_cast<double>(input_token_count + static_cast<size_t>(options.max_new_tokens));
    params->SetSearchOption("max_length", max_length);
    params->SetSearchOption("batch_size", 1.0);
    params->SetSearchOptionBool("do_sample", false);
    auto generator = OgaGenerator::Create(*model, *params);
    generator->SetInputs(*inputs);
    std::cout << "ok=generator\n";
    if (ReachedStage(options, "generator")) {
      return 0;
    }

    std::cout << "stage=token\n";
    std::cout.flush();
    generator->GenerateNextToken();
    std::cout << "ok=token sequence_count=" << generator->GetSequenceCount(0) << "\n";
    if (ReachedStage(options, "token")) {
      return 0;
    }

    std::cout << "stage=generate\n";
    std::cout.flush();
    while (!generator->IsDone()) {
      generator->GenerateNextToken();
    }
    const auto* sequence = generator->GetSequenceData(0);
    const auto sequence_count = generator->GetSequenceCount(0);
    const auto decode_offset = std::min(input_token_count, sequence_count);
    auto output = processor->Decode(sequence + decode_offset, sequence_count - decode_offset);
    std::cout << "ok=generate generated_tokens=" << (sequence_count - decode_offset) << "\n";
    std::cout << static_cast<const char*>(output) << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 3;
  }
}
