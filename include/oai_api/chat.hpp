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

// Chat completions, legacy completions and embeddings.

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "oai_api/types.hpp"

namespace oai {

// ---------------------------------------------------------------------------
// POST /chat/completions
//
// Unset optionals / null json fields are omitted from the request. An empty model and unset
// temperature / max_tokens fall back to the client defaults. extra_body is merge-patched into the
// final body (after the client-level extra_body), e.g. OpenRouter routing:
//   .extra_body = {{"provider", {{"order", {"anthropic"}}, {"allow_fallbacks", false}}}}
// ---------------------------------------------------------------------------

struct ChatRequest
{
  std::string model{};
  std::vector<Message> messages{};
  std::vector<ToolDefinition> tools{};
  nlohmann::json tool_choice{}; // "auto" | "none" | "required" | {"type":"function",...}
  std::optional<float> temperature{};
  std::optional<float> top_p{};
  std::optional<int> max_tokens{};
  std::optional<int> max_completion_tokens{};
  std::optional<int> n{};
  std::vector<std::string> stop{};
  std::optional<int> seed{};
  std::optional<float> presence_penalty{};
  std::optional<float> frequency_penalty{};
  nlohmann::json response_format{}; // see jsonSchemaFormat()
  std::optional<std::string> reasoning_effort{};
  std::optional<bool> logprobs{};
  std::optional<int> top_logprobs{};
  std::optional<bool> parallel_tool_calls{};
  std::optional<std::string> user{};
  nlohmann::json extra_body{};
};

struct ChatChoice
{
  int index{};
  std::string finish_reason{}; // "stop" | "length" | "tool_calls" | "content_filter"
  std::optional<std::string> content{};
  std::optional<std::string> refusal{};
  std::vector<ToolCall> tool_calls{};
};

struct ChatResponse
{
  std::string id{};
  std::string model{};
  std::vector<ChatChoice> choices{};
  Usage usage{};
  nlohmann::json raw{};

  // Content of the first choice, or "" if absent.
  [[nodiscard]] std::string text() const
  {
    return choices.empty() ? std::string{} : choices.front().content.value_or("");
  }
};

// Streaming tool-call fragment. Fragments with the same index belong to one call; id and name
// arrive on the first fragment, arguments are split across fragments and must be concatenated.
struct ToolCallDelta
{
  int index{};
  std::string id{};
  std::string function_name{};
  std::string arguments{};
};

struct ChatChunk
{
  std::string id{};
  std::string model{};
  std::string content{}; // delta text of choice 0 (may be empty)
  std::vector<ToolCallDelta> tool_calls{};
  std::optional<std::string> finish_reason{};
  std::optional<Usage> usage{}; // only on the final chunk
  nlohmann::json raw{};
};

// Return false to cancel the stream.
using ChunkCallback = std::function<bool(const ChatChunk&)>;

// response_format for structured outputs.
inline nlohmann::json
jsonSchemaFormat(const std::string& name, nlohmann::json schema, bool strict = true)
{
  return {{"type", "json_schema"},
          {"json_schema", {{"name", name}, {"schema", std::move(schema)}, {"strict", strict}}}};
}

// ---------------------------------------------------------------------------
// POST /completions (legacy text completion)
// ---------------------------------------------------------------------------

struct CompletionRequest
{
  std::string model{};
  std::string prompt{};
  std::optional<std::string> suffix{};
  std::optional<int> max_tokens{};
  std::optional<float> temperature{};
  std::optional<float> top_p{};
  std::optional<int> n{};
  std::vector<std::string> stop{};
  std::optional<int> seed{};
  std::optional<bool> echo{};
  nlohmann::json extra_body{};
};

struct CompletionChoice
{
  int index{};
  std::string text{};
  std::string finish_reason{};
};

struct CompletionResponse
{
  std::string id{};
  std::string model{};
  std::vector<CompletionChoice> choices{};
  Usage usage{};
  nlohmann::json raw{};

  [[nodiscard]] std::string text() const
  {
    return choices.empty() ? std::string{} : choices.front().text;
  }
};

// ---------------------------------------------------------------------------
// POST /embeddings
// ---------------------------------------------------------------------------

struct EmbeddingRequest
{
  std::string model{};
  std::vector<std::string> input{};
  std::optional<int> dimensions{};
  std::optional<std::string> user{};
  nlohmann::json extra_body{};
};

struct EmbeddingResponse
{
  std::string model{};
  std::vector<std::vector<float>> data{}; // one vector per input, in input order
  Usage usage{};
  nlohmann::json raw{};
};

} // namespace oai
