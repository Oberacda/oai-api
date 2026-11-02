// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the copyright holder nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

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
// order: question names, used to restore request order for keyed (OpenRouter) answers.
DecisionResponse parseDecisionResponse(const json& j, const std::vector<std::string>& order = {});

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
