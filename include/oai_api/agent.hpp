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

#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "oai_api/client.hpp"
#include "oai_api/tool.hpp"
#include "oai_api/types.hpp"

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

  // Check that the backing server is reachable and the configured model is available.
  [[nodiscard]] std::expected<void, ApiError> isHealthy() const { return m_client.isHealthy(); }

private:
  detail::AgentConfig m_config;
  OpenAIClient m_client;
  ToolRegistry m_tools;
  std::vector<Message> m_history;
};

} // namespace oai
