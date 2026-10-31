#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace oai {

// ---------------------------------------------------------------------------
// Roles
// ---------------------------------------------------------------------------

enum class Role
{
  SYSTEM,
  DEVELOPER,
  USER,
  ASSISTANT,
  TOOL
};

// ---------------------------------------------------------------------------
// Content blocks
// ---------------------------------------------------------------------------

struct TextBlock
{
  std::string text{};
};

struct ImageBlock
{
  // Exactly one of (base64_data+mime_type) or url must be set.
  std::string base64_data; // base64-encoded image bytes
  std::string mime_type{}; // "image/png" | "image/jpeg"
  std::string url{};       // HTTPS URL (alternative to base64)
};

// Structured JSON evidence (Responses/Decisions `input_json`). Chat completions has no JSON part,
// so it is sent there as dumped text.
struct JsonBlock
{
  nlohmann::json data{};
};

using ContentBlock = std::variant<TextBlock, ImageBlock, JsonBlock>;

// ---------------------------------------------------------------------------
// Tool call / definition  (defined before Message so Message can use them)
// ---------------------------------------------------------------------------

struct ToolCall
{
  std::string id{};
  std::string function_name{};
  std::string arguments_json{}; // raw JSON string from the API
};

struct ToolDefinition
{
  std::string name{};
  std::string description{};
  // JSON Schema object describing parameters, e.g.:
  // {"type":"object","properties":{"x":{"type":"number"}},"required":["x"]}
  std::string parameters_schema_json{};
};

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

struct Message
{
  Role role{};
  std::vector<ContentBlock> content{};
  std::optional<std::string> tool_call_id{}; // set when role == Tool
  std::optional<std::string> name{};         // tool name, when role == Tool
  std::vector<ToolCall> tool_calls{};        // set on assistant messages that request tools
};

// ---------------------------------------------------------------------------
// Usage / models / errors
// ---------------------------------------------------------------------------

// Token accounting. Chat-style (prompt/completion) and Responses-style (input/output) names are
// both mapped onto input/output.
struct Usage
{
  int input_tokens{};
  int output_tokens{};
  int total_tokens{};
  int cached_tokens{};
  std::optional<double> cost{}; // OpenRouter: credits charged
};

struct Model
{
  std::string id{};
  std::string owned_by{};
  std::int64_t created{};
};

struct ApiError
{
  int http_status{}; // 0 for transport / parse errors
  std::string message{};
  std::string type{}; // error.type from the API body, if any
  std::string code{}; // error.code from the API body, if any
};

// ---------------------------------------------------------------------------
// Image helpers
// ---------------------------------------------------------------------------

// Encode a cv::Mat to a PNG ImageBlock (base64).
// Returns an error string on failure.
inline std::expected<ImageBlock, std::string> encodeImage(const cv::Mat& image)
{
  std::vector<uchar> buf{};
  if (!cv::imencode(".png", image, buf))
  {
    return std::unexpected{"cv::imencode failed"};
  }

  static constexpr std::string_view K_ALPHABET =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  std::string encoded{};
  encoded.reserve(((buf.size() + 2) / 3) * 4);

  for (std::size_t i = 0; i < buf.size(); i += 3)
  {
    const unsigned B0 = buf[i];
    const unsigned B1 = (i + 1 < buf.size()) ? buf[i + 1] : 0u;
    const unsigned B2 = (i + 2 < buf.size()) ? buf[i + 2] : 0u;

    encoded += K_ALPHABET[(B0 >> 2) & 0x3f];
    encoded += K_ALPHABET[((B0 << 4) | (B1 >> 4)) & 0x3f];
    encoded += (i + 1 < buf.size()) ? K_ALPHABET[((B1 << 2) | (B2 >> 6)) & 0x3f] : '=';
    encoded += (i + 2 < buf.size()) ? K_ALPHABET[B2 & 0x3f] : '=';
  }

  return ImageBlock{std::move(encoded), "image/png", ""};
}

// Wrap an HTTPS URL into an ImageBlock without downloading or encoding.
inline ImageBlock imageFromUrl(std::string_view url)
{
  return ImageBlock{"", "", std::string{url}};
}

} // namespace oai
