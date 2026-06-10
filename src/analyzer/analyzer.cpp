#include "scene_describer/analyzer.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace scene_describer {
namespace {

Result<std::string> ReadTextFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    return Status(ErrorCode::kIoError, "unable to open prompt template: " + path.string());
  }

  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::string ReplaceAll(std::string value, const std::string& needle, const std::string& replacement) {
  if (needle.empty()) {
    return value;
  }

  std::size_t offset = 0;
  while ((offset = value.find(needle, offset)) != std::string::npos) {
    value.replace(offset, needle.size(), replacement);
    offset += replacement.size();
  }
  return value;
}

std::string FormatNumber(double value) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(3) << value;
  return output.str();
}

std::string RenderTrack(const TrackMetadata& track) {
  std::ostringstream output;
  output << "- track_id=" << (track.track_id.empty() ? "<unknown>" : track.track_id);
  if (!track.label.empty()) {
    output << ", label=" << track.label;
  }
  output << ", bbox=[x=" << FormatNumber(track.bbox.x) << ", y=" << FormatNumber(track.bbox.y)
         << ", w=" << FormatNumber(track.bbox.width) << ", h=" << FormatNumber(track.bbox.height) << "]";
  if (track.confidence > 0.0) {
    output << ", confidence=" << FormatNumber(track.confidence);
  }
  for (const auto& [key, value] : track.attributes) {
    output << ", " << key << "=" << value;
  }
  return output.str();
}

std::string RenderFrames(const AnalyzerRequest& request) {
  std::ostringstream output;
  for (std::size_t frame_index = 0; frame_index < request.frames.size(); ++frame_index) {
    const auto& frame = request.frames[frame_index];
    output << (frame.detail_view ? "Detail view " : "Frame ") << frame_index;
    if (!frame.frame_id.empty()) {
      output << " (" << frame.frame_id << ")";
    }
    output << ":\n";
    output << "- image_path: " << frame.image_path << "\n";
    output << "- timestamp_ms: " << frame.timestamp_ms << "\n";
    if (frame.detail_view) {
      output << "- view_type: high-resolution detail crop\n";
    }
    if (!frame.note.empty()) {
      output << "- note: " << frame.note << "\n";
    }
    if (frame.tracks.empty()) {
      output << "- tracks: none provided\n";
    } else {
      output << "- tracks:\n";
      for (const auto& track : frame.tracks) {
        output << "  " << RenderTrack(track) << "\n";
      }
    }
    if (frame_index + 1 < request.frames.size()) {
      output << "\n";
    }
  }
  return output.str();
}

std::string RenderHistory(const AnalyzerRequest& request) {
  if (request.prior_summaries.empty()) {
    return "No prior summaries were provided.";
  }

  std::ostringstream output;
  for (std::size_t index = 0; index < request.prior_summaries.size(); ++index) {
    output << "- prior_summary_" << index << ": " << request.prior_summaries[index];
    if (index + 1 < request.prior_summaries.size()) {
      output << "\n";
    }
  }
  return output.str();
}

Status ValidateRequest(const AnalyzerRequest& request) {
  if (request.frames.empty()) {
    return Status(ErrorCode::kInvalidArgument, "analyzer request requires at least one frame");
  }

  for (std::size_t index = 0; index < request.frames.size(); ++index) {
    const auto& frame = request.frames[index];
    if (frame.image_path.empty()) {
      return Status(ErrorCode::kInvalidArgument, "analyzer frame " + std::to_string(index) + " is missing image_path");
    }
    if (index > 0 && frame.timestamp_ms < request.frames[index - 1].timestamp_ms) {
      return Status(ErrorCode::kInvalidArgument, "analyzer frame timestamps must be monotonic");
    }
  }

  return Status::Ok();
}

}  // namespace

HistoryStore::HistoryStore(std::size_t max_entries) : max_entries_(max_entries) {}

std::vector<HistoryEntry> HistoryStore::Snapshot() const {
  return {entries_.begin(), entries_.end()};
}

Status HistoryStore::Add(HistoryEntry entry) {
  if (!entries_.empty() && entry.timestamp_ms < entries_.back().timestamp_ms) {
    return Status(ErrorCode::kInvalidArgument, "history timestamps must be monotonic");
  }

  if (max_entries_ == 0) {
    return Status::Ok();
  }

  entries_.push_back(std::move(entry));
  while (entries_.size() > max_entries_) {
    entries_.pop_front();
  }
  return Status::Ok();
}

std::optional<std::int64_t> HistoryStore::LatestTimestampMs() const {
  if (entries_.empty()) {
    return std::nullopt;
  }
  return entries_.back().timestamp_ms;
}

PromptTemplates DefaultPromptTemplates() {
  PromptTemplates templates;
  templates.base_guardrails =
      "You are an edge surveillance scene analyzer. Describe only visible evidence. "
      "Do not identify people, infer intent, or add details not supported by the frames.";
  templates.task_rules =
      "Return a concise operational summary. Mention visible objects, scene changes, track continuity, and notable "
      "actions. Inspect high-resolution detail crops for small, held, or airborne objects. If track metadata conflicts "
      "with image evidence, say that the track metadata is uncertain.";
  templates.local_batch = "Current batch:\n{frames}";
  templates.history_context = "Recent context:\n{history}";
  return templates;
}

Result<PromptTemplates> LoadPromptTemplates(const std::filesystem::path& template_dir) {
  PromptTemplates templates;

  auto base = ReadTextFile(template_dir / "base_guardrails.txt");
  if (!base.ok()) {
    return base.status();
  }
  templates.base_guardrails = base.value();

  auto task_rules = ReadTextFile(template_dir / "task_rules.txt");
  if (!task_rules.ok()) {
    return task_rules.status();
  }
  templates.task_rules = task_rules.value();

  auto local_batch = ReadTextFile(template_dir / "local_batch.txt");
  if (!local_batch.ok()) {
    return local_batch.status();
  }
  templates.local_batch = local_batch.value();

  auto history_context = ReadTextFile(template_dir / "history_context.txt");
  if (!history_context.ok()) {
    return history_context.status();
  }
  templates.history_context = history_context.value();

  return templates;
}

Result<std::string> BuildAnalyzerPrompt(const AnalyzerRequest& request, const PromptTemplates& templates) {
  auto status = ValidateRequest(request);
  if (!status.ok()) {
    return status;
  }

  const auto frames = RenderFrames(request);
  const auto history = RenderHistory(request);
  auto local_batch = ReplaceAll(templates.local_batch, "{frames}", frames);
  auto history_context = ReplaceAll(templates.history_context, "{history}", history);
  auto request_id = request.request_id.empty() ? "<unspecified>" : request.request_id;
  local_batch = ReplaceAll(local_batch, "{request_id}", request_id);
  history_context = ReplaceAll(history_context, "{request_id}", request_id);

  std::ostringstream prompt;
  prompt << templates.base_guardrails << "\n\n"
         << templates.task_rules << "\n\n"
         << history_context << "\n\n"
         << local_batch << "\n\n"
         << "Write the analyzer result as one concise paragraph.";
  return prompt.str();
}

Result<std::int64_t> LatestFrameTimestampMs(const AnalyzerRequest& request) {
  auto status = ValidateRequest(request);
  if (!status.ok()) {
    return status;
  }

  auto latest = request.frames.front().timestamp_ms;
  for (const auto& frame : request.frames) {
    latest = std::max(latest, frame.timestamp_ms);
  }
  return latest;
}

}  // namespace scene_describer
