#pragma once

// oai_api/builder.hpp
// Fluent builder API for Message, OpenAIClient, and Agent.

#include <expected>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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

  // Append a text block.
  MessageBuilder& text(std::string value);

  // Encode a cv::Mat as a base64 PNG block. Stores the first encode error;
  // subsequent image() calls are still accepted but build() will fail.
  MessageBuilder& image(const cv::Mat& mat);

  // Append a pre-built ImageBlock directly (URL or already-encoded).
  MessageBuilder& image(ImageBlock block);

  // Consume the builder. Returns unexpected if any image(cv::Mat) call failed.
  [[nodiscard]] std::expected<Message, std::string> build();

private:
  explicit MessageBuilder(Role role);

  Role                       m_role;
  std::vector<ContentBlock>  m_content;
  std::optional<std::string> m_error;
};

// ---------------------------------------------------------------------------
// ClientBuilder
//
// Constructs an OpenAIClient. build() is infallible.
//
// Example:
//   auto client = oai::ClientBuilder{}
//                   .baseUrl("http://localhost:11434/v1")
//                   .model("llama3.2")
//                   .temperature(0.2f)
//                   .maxTokens(512)
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
  AgentBuilder& tool(std::string name, std::string description, std::string schema_json,
                     THandler&& handler)
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
  detail::AgentConfig         m_cfg;
  std::optional<OpenAIClient> m_client;
  ToolRegistry                m_tools;
};

} // namespace oai
