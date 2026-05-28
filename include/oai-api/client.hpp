#pragma once

#include <expected>
#include <string>
#include <vector>

#include "oai-api/types.hpp"

namespace oai {

// ---------------------------------------------------------------------------
// ClientConfig
// ---------------------------------------------------------------------------

struct ClientConfig
{
  std::string base_url; // e.g. "http://localhost:11434/v1"
  std::string api_key;  // Bearer token; empty string if not required
  std::string model;    // e.g. "gpt-4o", "llama3.2-vision"
  float temperature{0.7F};
  int max_tokens{4096};
};

// ---------------------------------------------------------------------------
// OpenAIClient
//
// Thin synchronous HTTP client for the /chat/completions endpoint.
// Uses cpr for HTTP and nlohmann/json for serialization.
// ---------------------------------------------------------------------------

class OpenAIClient
{
public:
  explicit OpenAIClient(ClientConfig config);

  // Send a chat completion request and block until the response arrives.
  // Returns the parsed CompletionResponse or an ApiError.
  [[nodiscard]] std::expected<CompletionResponse, ApiError>
  complete(const std::vector<Message>& messages, const std::vector<ToolDefinition>& tools = {});

  [[nodiscard]] const ClientConfig& config() const { return m_config; }

private:
  ClientConfig m_config;
};

} // namespace oai
