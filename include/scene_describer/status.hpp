#pragma once

#include <string>
#include <utility>

namespace scene_describer {

enum class ErrorCode {
  kOk = 0,
  kInvalidArgument,
  kIoError,
  kParseError,
  kUnsupported,
  kBackendUnavailable,
  kRuntimeError,
};

class Status {
 public:
  Status() = default;
  Status(ErrorCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  static Status Ok() { return {}; }

  [[nodiscard]] bool ok() const { return code_ == ErrorCode::kOk; }
  [[nodiscard]] ErrorCode code() const { return code_; }
  [[nodiscard]] const std::string& message() const { return message_; }

 private:
  ErrorCode code_{ErrorCode::kOk};
  std::string message_;
};

template <typename T>
class Result {
 public:
  Result(T value) : value_(std::move(value)), status_(Status::Ok()), has_value_(true) {}
  Result(Status status) : status_(std::move(status)), has_value_(false) {}

  [[nodiscard]] bool ok() const { return status_.ok(); }
  [[nodiscard]] const Status& status() const { return status_; }

  [[nodiscard]] const T& value() const { return value_; }
  [[nodiscard]] T& value() { return value_; }

 private:
  T value_{};
  Status status_;
  bool has_value_{false};
};

}  // namespace scene_describer

