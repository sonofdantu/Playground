#pragma once

#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "scene_describer/status.hpp"

namespace scene_describer {

struct BoundingBox {
  double x{0.0};
  double y{0.0};
  double width{0.0};
  double height{0.0};
};

struct TrackMetadata {
  std::string track_id;
  std::string label;
  BoundingBox bbox;
  double confidence{0.0};
  std::map<std::string, std::string> attributes;
};

struct AnalyzerFrame {
  std::string frame_id;
  std::string image_path;
  std::int64_t timestamp_ms{0};
  std::vector<TrackMetadata> tracks;
};

struct AnalyzerRequest {
  std::string request_id;
  std::vector<AnalyzerFrame> frames;
  std::vector<std::string> prior_summaries;
};

struct AnalyzerResult {
  std::string request_id;
  std::string summary;
  std::int64_t latest_timestamp_ms{0};
  std::map<std::string, std::string> metadata;
};

struct PromptTemplates {
  std::string base_guardrails;
  std::string task_rules;
  std::string local_batch;
  std::string history_context;
};

struct HistoryEntry {
  std::int64_t timestamp_ms{0};
  std::string summary;
};

class HistoryStore {
 public:
  explicit HistoryStore(std::size_t max_entries);

  [[nodiscard]] std::vector<HistoryEntry> Snapshot() const;
  Status Add(HistoryEntry entry);
  [[nodiscard]] std::optional<std::int64_t> LatestTimestampMs() const;

 private:
  std::size_t max_entries_{0};
  std::deque<HistoryEntry> entries_;
};

Result<PromptTemplates> LoadPromptTemplates(const std::filesystem::path& template_dir);
PromptTemplates DefaultPromptTemplates();
Result<std::string> BuildAnalyzerPrompt(const AnalyzerRequest& request, const PromptTemplates& templates);
Result<std::int64_t> LatestFrameTimestampMs(const AnalyzerRequest& request);

}  // namespace scene_describer
