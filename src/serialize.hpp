#pragma once

// Internal: JSON (de)serialization for every endpoint, plus a server-sent-events parser.
// Parse functions throw nlohmann::json::exception on malformed input; the client catches it.

#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "oai_api/client.hpp"

namespace oai::detail {

using json = nlohmann::json;

// Merge-patch client then request extra_body into body (non-object values are ignored).
void applyExtraBody(json& body, const json& client_extra, const json& request_extra);

ApiError parseApiError(int http_status, const std::string& body);
Usage parseUsage(const json& j);
Model parseModel(const json& j);

// Chat-completions wire format.
json messageToJson(const Message& msg);
json toolDefinitionToJson(const ToolDefinition& def);
json chatRequestToJson(const ChatRequest& req, const ClientConfig& cfg);
ChatResponse parseChatResponse(const json& j);
ChatChunk parseChatChunk(const json& j);

json completionRequestToJson(const CompletionRequest& req, const ClientConfig& cfg);
CompletionResponse parseCompletionResponse(const json& j);

json embeddingRequestToJson(const EmbeddingRequest& req, const ClientConfig& cfg);
EmbeddingResponse parseEmbeddingResponse(const json& j);

// Responses-API wire format (also used by Decisions).
json messagesToResponsesInput(const std::vector<Message>& messages);
json responseRequestToJson(const ResponseRequest& req, const ClientConfig& cfg);
Response parseResponse(const json& j);

json speechRequestToJson(const SpeechRequest& req, const ClientConfig& cfg);
json imageRequestToJson(const ImageRequest& req, const ClientConfig& cfg);
std::vector<ImageResult> parseImages(const json& j);

json decisionRequestToJson(const DecisionRequest& req, const ClientConfig& cfg);
DecisionResponse parseDecisionResponse(const json& j);

// Incremental text/event-stream parser. feed() may be called with arbitrary byte chunks; each
// complete event's data payload (multi-line data joined by '\n') is passed to on_data.
class SseParser
{
public:
  using DataCallback = std::function<bool(std::string_view)>;

  // Returns false as soon as on_data returns false.
  bool feed(std::string_view chunk, const DataCallback& on_data);
  // Dispatch a trailing event that was not terminated by a blank line.
  bool finish(const DataCallback& on_data);

private:
  std::string m_buffer;
  std::string m_data;
};

} // namespace oai::detail
