#pragma once

#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace oai {

// ---------------------------------------------------------------------------
// Roles
// ---------------------------------------------------------------------------

enum class Role
{
  SYSTEM,
  USER,
  ASSISTANT,
  TOOL
};

// ---------------------------------------------------------------------------
// Content blocks
// ---------------------------------------------------------------------------

struct TextBlock
{
  std::string text;
};

struct ImageBlock
{
  // Exactly one of (base64_data+mime_type) or url must be set.
  std::string base64_data; // base64-encoded image bytes
  std::string mime_type;   // "image/png" | "image/jpeg"
  std::string url;         // HTTPS URL (alternative to base64)
};

using ContentBlock = std::variant<TextBlock, ImageBlock>;

// ---------------------------------------------------------------------------
// Tool call / definition  (defined before Message so Message can use them)
// ---------------------------------------------------------------------------

struct ToolCall
{
  std::string id;
  std::string function_name;
  std::string arguments_json; // raw JSON string from the API
};

struct ToolDefinition
{
  std::string name;
  std::string description;
  // JSON Schema object describing parameters, e.g.:
  // {"type":"object","properties":{"x":{"type":"number"}},"required":["x"]}
  std::string parameters_schema_json;
};

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

struct Message
{
  Role role;
  std::vector<ContentBlock> content;
  std::optional<std::string> tool_call_id; // set when role == Tool
  std::optional<std::string> name;         // tool name, when role == Tool
  std::vector<ToolCall> tool_calls;        // set on assistant messages that request tools
};

// Convenience constructors
inline Message systemMessage(std::string text)
{
  return {Role::SYSTEM, {TextBlock{std::move(text)}}, {}, {}, {}};
}

inline Message userMessage(std::string text)
{
  return {Role::USER, {TextBlock{std::move(text)}}, {}, {}, {}};
}

inline Message userMessage(std::vector<ContentBlock> blocks)
{
  return {Role::USER, std::move(blocks), {}, {}, {}};
}

// ---------------------------------------------------------------------------
// Completion response
// ---------------------------------------------------------------------------

struct CompletionChoice
{
  std::string finish_reason;          // "stop" | "tool_calls"
  std::optional<std::string> content; // present when finish_reason == "stop"
  std::vector<ToolCall> tool_calls;   // present when finish_reason == "tool_calls"
};

struct CompletionResponse
{
  std::string id;
  CompletionChoice choice;
  int prompt_tokens{};
  int completion_tokens{};
};

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

struct ApiError
{
  int http_status{};
  std::string message;
};

// ---------------------------------------------------------------------------
// Image helpers
// ---------------------------------------------------------------------------

// Encode a cv::Mat to a PNG ImageBlock (base64).
// Returns an error string on failure.
inline std::expected<ImageBlock, std::string> encodeImage(const cv::Mat& image)
{
  std::vector<uchar> buf;
  if (!cv::imencode(".png", image, buf))
  {
    return std::unexpected{"cv::imencode failed"};
  }

  static constexpr std::string_view K_ALPHABET =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  std::string encoded;
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
