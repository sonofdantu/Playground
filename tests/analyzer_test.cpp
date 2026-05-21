#include "scene_describer/analyzer.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

int main() {
  scene_describer::AnalyzerRequest request;
  request.request_id = "test-request";

  scene_describer::AnalyzerFrame frame;
  frame.frame_id = "frame-0";
  frame.image_path = "tmp/smoke.png";
  frame.timestamp_ms = 1000;

  scene_describer::TrackMetadata track;
  track.track_id = "t-1";
  track.label = "vehicle";
  track.bbox = {1.0, 2.0, 30.0, 40.0};
  track.confidence = 0.87;
  frame.tracks.push_back(track);
  request.frames.push_back(frame);
  request.prior_summaries.push_back("A vehicle was previously visible near the entry lane.");

  auto prompt = scene_describer::BuildAnalyzerPrompt(request, scene_describer::DefaultPromptTemplates());
  if (!prompt.ok()) {
    std::cerr << prompt.status().message() << "\n";
    return EXIT_FAILURE;
  }

  const auto& prompt_text = prompt.value();
  if (prompt_text.find("track_id=t-1") == std::string::npos ||
      prompt_text.find("A vehicle was previously visible") == std::string::npos ||
      prompt_text.find("tmp/smoke.png") == std::string::npos) {
    std::cerr << "prompt did not contain expected frame, track, and history context\n";
    return EXIT_FAILURE;
  }

  scene_describer::HistoryStore history(2);
  auto status = history.Add({1000, "first"});
  if (!status.ok()) {
    std::cerr << status.message() << "\n";
    return EXIT_FAILURE;
  }
  status = history.Add({900, "older"});
  if (status.ok()) {
    std::cerr << "history accepted non-monotonic timestamp\n";
    return EXIT_FAILURE;
  }
  status = history.Add({1100, "second"});
  if (!status.ok()) {
    std::cerr << status.message() << "\n";
    return EXIT_FAILURE;
  }
  status = history.Add({1200, "third"});
  if (!status.ok()) {
    std::cerr << status.message() << "\n";
    return EXIT_FAILURE;
  }
  if (history.Snapshot().size() != 2 || history.LatestTimestampMs().value_or(0) != 1200) {
    std::cerr << "history retention did not keep the expected latest entries\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
