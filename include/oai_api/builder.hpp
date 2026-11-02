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

// oai_api/builder.hpp
// Fluent builder API for Message, OpenAIClient, and Agent.

#include <chrono>
#include <expected>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "oai_api/agent.hpp"
#include "oai_api/client.hpp"
#include "oai_api/tool.hpp"
#include "oai_api/types.hpp"

namespace oai {

// ---------------------------------------------------------------------------
// MessageBuilder
//
// Constructs a Message for any role by chaining text() and image() calls.
// If an image(cv::Mat) call fails to encode, build() propagates the error.
//
// Example:
//   auto msg = oai::MessageBuilder::user()
//                .text("What colour is this?")
//                .image(mat)
//                .build();
//   if (!msg) { /* handle msg.error() */ }
// ---------------------------------------------------------------------------

class MessageBuilder
{
public:
  [[nodiscard]] static MessageBuilder user();
  [[nodiscard]] static MessageBuilder system();
  [[nodiscard]] static MessageBuilder assistant();
  [[nodiscard]] static MessageBuilder developer();

  // Append a text block.
  MessageBuilder& text(std::string value);

  // Encode a cv::Mat as a base64 PNG block. Stores the first encode error;
  // subsequent image() calls are still accepted but build() will fail.
  MessageBuilder& image(const cv::Mat& mat);

  // Append a pre-built ImageBlock directly (URL or already-encoded).
  MessageBuilder& image(ImageBlock block);

  // Append structured JSON evidence (input_json in Responses/Decisions; dumped text in chat).
  MessageBuilder& json(nlohmann::json data);

  // Consume the builder. Returns unexpected if any image(cv::Mat) call failed.
  [[nodiscard]] std::expected<Message, std::string> build();

private:
  explicit MessageBuilder(Role role);

  Role m_role;
  std::vector<ContentBlock> m_content;
  std::optional<std::string> m_error;
};

// ---------------------------------------------------------------------------
// ClientBuilder
//
// Constructs an OpenAIClient. build() is infallible. model / temperature / maxTokens are defaults
// for requests that leave them unset.
//
// Example:
//   auto client = oai::ClientBuilder{}
//                   .baseUrl("http://localhost:11434/v1")
//                   .model("llama3.2")
//                   .temperature(0.2f)
//                   .maxTokens(512)
//                   .header("X-Title", "my-robot")                       // e.g. OpenRouter
//                   .extraBody({{"provider", {{"order", {"anthropic"}}}}})  // every request
//                   .build();
// ---------------------------------------------------------------------------

class ClientBuilder
{
public:
  ClientBuilder& baseUrl(std::string url);
  ClientBuilder& apiKey(std::string key);
  ClientBuilder& model(std::string name);
  ClientBuilder& temperature(float t);
  ClientBuilder& maxTokens(int n);
  ClientBuilder& connectTimeout(std::chrono::milliseconds timeout);
  ClientBuilder& requestTimeout(std::chrono::milliseconds timeout);
  // Extra HTTP header sent with every request.
  ClientBuilder& header(std::string name, std::string value);
  // Merge-patched into every JSON request body (a request's own extra_body wins).
  ClientBuilder& extraBody(nlohmann::json body);

  [[nodiscard]] OpenAIClient build();

private:
  detail::ClientConfig m_cfg;
};

// ---------------------------------------------------------------------------
// AgentBuilder
//
// Constructs an Agent with inline tool registration.
// client() must be called before build().
//
// Example:
//   auto agent = oai::AgentBuilder{}
//                  .systemPrompt("You are a helpful assistant.")
//                  .maxSteps(5)
//                  .client(std::move(client))
//                  .tool("get_time", "Returns UTC time.",
//                        R"({"type":"object","properties":{},"required":[]})",
//                        [](std::string_view) -> std::expected<std::string, std::string> {
//                          return utc_now();
//                        })
//                  .build();
// ---------------------------------------------------------------------------

class AgentBuilder
{
public:
  AgentBuilder& systemPrompt(std::string prompt);
  AgentBuilder& maxSteps(int steps);

  // Provide the underlying HTTP client.
  AgentBuilder& client(OpenAIClient c);

  // Register a tool inline.
  template <ToolCallable THandler>
  AgentBuilder&
  tool(std::string name, std::string description, std::string schema_json, THandler&& handler)
  {
    m_tools.add(FunctionTool::make(
      ToolDefinition{std::move(name), std::move(description), std::move(schema_json)},
      std::forward<THandler>(handler)));
    return *this;
  }

  // Replace all tools registered via tool() with a pre-built ToolRegistry.
  AgentBuilder& tools(ToolRegistry registry);

  // Consume the builder. Requires that client() was called.
  [[nodiscard]] Agent build();

private:
  detail::AgentConfig m_cfg;
  std::optional<OpenAIClient> m_client;
  ToolRegistry m_tools;
};

} // namespace oai
