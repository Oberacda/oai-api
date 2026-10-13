#pragma once

#include <chrono>
#include <expected>
#include <string>
#include <vector>

#include "oai_api/types.hpp"

namespace oai::detail {

struct ClientConfig
{
  std::string base_url; // e.g. "http://localhost:11434/v1"
  std::string api_key;  // Bearer token; empty string if not required
  std::string model;    // e.g. "gpt-4o", "llama3.2-vision"
  float temperature{0.7F};
  int max_tokens{4096};
  std::chrono::milliseconds connect_timeout{5000};
  std::chrono::milliseconds request_timeout{60000};
};

} // namespace oai::detail

namespace oai {

// ---------------------------------------------------------------------------
// OpenAIClient
//
// Thin synchronous HTTP client for the /chat/completions endpoint.
// Uses cpr for HTTP and nlohmann/json for serialization.
// Construct via ClientBuilder.
// ---------------------------------------------------------------------------

class OpenAIClient
{
public:
  explicit OpenAIClient(detail::ClientConfig config);

  // Send a chat completion request and block until the response arrives.
  // Returns the parsed CompletionResponse or an ApiError.
  [[nodiscard]] std::expected<CompletionResponse, ApiError>
  complete(const std::vector<Message>& messages, const std::vector<ToolDefinition>& tools = {});

  // Check that the server is reachable and the configured model is listed.
  // Returns an error describing the first problem found.
  [[nodiscard]] std::expected<void, ApiError> isHealthy();

  [[nodiscard]] const detail::ClientConfig& config() const { return m_config; }

private:
  detail::ClientConfig m_config;
};

} // namespace oai
