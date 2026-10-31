#pragma once

// Responses API: POST /responses, GET|DELETE /responses/{id}.

#include <functional>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "oai_api/types.hpp"

namespace oai {

// A plain prompt string or a conversation. Messages are converted to Responses input items:
// TOOL messages -> function_call_output, assistant tool_calls -> function_call.
using ResponseInput = std::variant<std::string, std::vector<Message>>;

struct ResponseRequest
{
  std::string model{};
  ResponseInput input{};
  std::optional<std::string> instructions{};
  std::vector<ToolDefinition> tools{};
  nlohmann::json tool_choice{};
  std::optional<float> temperature{};
  std::optional<float> top_p{};
  std::optional<int> max_output_tokens{};
  nlohmann::json reasoning{}; // e.g. {"effort":"low","summary":"auto"}
  nlohmann::json text{};      // e.g. {"format":{"type":"json_schema","name":...,"schema":...}}
  std::optional<std::string> previous_response_id{};
  std::optional<bool> store{};
  nlohmann::json metadata{};
  nlohmann::json extra_body{};
};

struct Response
{
  std::string id{};
  std::string model{};
  std::string status{};               // "completed" | "in_progress" | "incomplete" | "failed" | ...
  std::string output_text{};          // all output_text parts concatenated
  std::vector<ToolCall> tool_calls{}; // function_call items; ToolCall::id holds call_id
  Usage usage{};
  nlohmann::json raw{};
};

// One server-sent event, e.g. type "response.output_text.delta". data is the full event object.
struct ResponseEvent
{
  std::string type{};
  nlohmann::json data{};

  [[nodiscard]] std::string delta() const { return data.value("delta", ""); }
};

// Return false to cancel the stream.
using EventCallback = std::function<bool(const ResponseEvent&)>;

} // namespace oai
