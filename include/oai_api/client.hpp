#pragma once

#include <chrono>
#include <cstdint>
#include <expected>
#include <functional>
#include <future>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "oai_api/chat.hpp"
#include "oai_api/decisions.hpp"
#include "oai_api/media.hpp"
#include "oai_api/responses.hpp"
#include "oai_api/types.hpp"

namespace oai::detail {

struct ClientConfig
{
  std::string base_url; // e.g. "http://localhost:11434/v1"
  std::string api_key;  // Bearer token; empty string if not required
  // Defaults applied to requests that leave these unset.
  std::string model; // e.g. "gpt-4o", "llama3.2-vision"
  std::optional<float> temperature;
  std::optional<int> max_tokens;
  // Sent with every request (e.g. OpenAI-Organization, OpenRouter HTTP-Referer / X-Title).
  std::vector<std::pair<std::string, std::string>> headers;
  // Merge-patched into every JSON request body, before the request's own extra_body.
  nlohmann::json extra_body;
  std::chrono::milliseconds connect_timeout{5000};
  std::chrono::milliseconds request_timeout{60000};
};

} // namespace oai::detail

namespace oai {

enum class Method : std::uint8_t
{
  GET,
  POST,
  DELETE
};

// ---------------------------------------------------------------------------
// OpenAIClient
//
// Synchronous HTTP client for the OpenAI REST API and compatible servers (Ollama, vLLM,
// OpenRouter, ...). Every call blocks; use async() for a std::future. The client is cheap to copy
// and safe to use from several threads at once. Construct via ClientBuilder.
// ---------------------------------------------------------------------------

class OpenAIClient
{
public:
  explicit OpenAIClient(detail::ClientConfig config);

  // --- Chat / completions / embeddings ------------------------------------
  [[nodiscard]] std::expected<ChatResponse, ApiError> chat(const ChatRequest& request) const;

  // Streams deltas to on_chunk (on the calling thread). Returns when the stream ends, fails, or
  // on_chunk returns false (cancel; not an error).
  [[nodiscard]] std::expected<void, ApiError> chatStream(const ChatRequest& request,
                                                         ChunkCallback on_chunk) const;

  [[nodiscard]] std::expected<CompletionResponse, ApiError>
  completion(const CompletionRequest& request) const;

  [[nodiscard]] std::expected<EmbeddingResponse, ApiError>
  embeddings(const EmbeddingRequest& request) const;

  // --- Models -------------------------------------------------------------
  [[nodiscard]] std::expected<std::vector<Model>, ApiError> listModels() const;
  [[nodiscard]] std::expected<Model, ApiError> retrieveModel(std::string_view id) const;

  // Check that the server is reachable and the default model is listed.
  [[nodiscard]] std::expected<void, ApiError> isHealthy() const;

  // --- Responses ----------------------------------------------------------
  [[nodiscard]] std::expected<Response, ApiError>
  createResponse(const ResponseRequest& request) const;
  [[nodiscard]] std::expected<void, ApiError> createResponseStream(const ResponseRequest& request,
                                                                   EventCallback on_event) const;
  [[nodiscard]] std::expected<Response, ApiError> retrieveResponse(std::string_view id) const;
  [[nodiscard]] std::expected<void, ApiError> deleteResponse(std::string_view id) const;

  // --- Audio / images -----------------------------------------------------
  [[nodiscard]] std::expected<Transcription, ApiError>
  transcribe(const TranscriptionRequest& request) const;
  [[nodiscard]] std::expected<Transcription, ApiError>
  translate(const TranscriptionRequest& request) const;
  // Returns the encoded audio bytes.
  [[nodiscard]] std::expected<std::string, ApiError> speech(const SpeechRequest& request) const;
  [[nodiscard]] std::expected<std::vector<ImageResult>, ApiError>
  generateImage(const ImageRequest& request) const;

  // --- Decisions ----------------------------------------------------------
  [[nodiscard]] std::expected<DecisionResponse, ApiError>
  decide(const DecisionRequest& request) const;

  // --- Raw escape hatch ---------------------------------------------------
  // Any JSON endpoint, e.g. request(Method::GET, "/files"). The client-level extra_body is not
  // applied here; body is sent as given (omitted when null).
  [[nodiscard]] std::expected<nlohmann::json, ApiError>
  request(Method method, std::string_view path, const nlohmann::json& body = nullptr) const;

  // Run any call on a worker thread:
  //   auto fut = client.async(&OpenAIClient::chat, request);
  //   auto fut = client.async([](const OpenAIClient& c) { return c.listModels(); });
  // The client and arguments are copied into the task, so pass owning types (std::string, not
  // std::string_view / const char*). Stream callbacks then run on the worker thread.
  template <typename TFn, typename... TArgs>
  [[nodiscard]] auto async(TFn fn, TArgs... args) const
  {
    return std::async(std::launch::async,
                      [self = *this, fn = std::move(fn), ... args = std::move(args)]() {
                        return std::invoke(fn, self, args...);
                      });
  }

  [[nodiscard]] const detail::ClientConfig& config() const { return m_config; }

private:
  detail::ClientConfig m_config;
};

} // namespace oai
