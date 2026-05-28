#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "oai-api/client.hpp"
#include "oai-api/tool.hpp"
#include "oai-api/types.hpp"

namespace oai::detail {

struct AgentConfig
{
  std::string system_prompt;
  int max_steps{10}; // max tool-call rounds before giving up
};

} // namespace oai::detail

namespace oai {

// ---------------------------------------------------------------------------
// Agent
//
// Wraps an OpenAIClient and ToolRegistry into a ReAct-style execution loop:
//   1. Build prompt from system_prompt + conversation history + new task
//   2. Call the LLM
//   3. If the model requests tool calls, dispatch them and feed results back
//   4. Repeat until finish_reason == "stop" or max_steps exceeded
//
// Construct via AgentBuilder.
// ---------------------------------------------------------------------------

class Agent
{
public:
  Agent(detail::AgentConfig config, OpenAIClient client, ToolRegistry tools = {});

  // Run a task to completion and return the final answer string.
  // The conversation history is preserved across calls; use reset() to clear it.
  [[nodiscard]] std::expected<std::string, ApiError> run(std::string_view task);

  // Access full conversation history (useful for debugging / logging).
  [[nodiscard]] const std::vector<Message>& history() const { return m_history; }

  // Clear the conversation history so the next run() starts fresh.
  void reset() { m_history.clear(); }

  // Replace the tool registry after construction.
  void setTools(ToolRegistry tools) { m_tools = std::move(tools); }

private:
  detail::AgentConfig m_config;
  OpenAIClient m_client;
  ToolRegistry m_tools;
  std::vector<Message> m_history;

  // Append an assistant message that carries tool_calls back into history.
  void pushAssistantToolCalls(const CompletionResponse& resp);
};

} // namespace oai
