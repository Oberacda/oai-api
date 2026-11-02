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

#include "oai_api/agent.hpp"

#include <fmt/core.h>

namespace oai {

Agent::Agent(detail::AgentConfig config, OpenAIClient client, ToolRegistry tools)
  : m_config{std::move(config)}
  , m_client{std::move(client)}
  , m_tools{std::move(tools)}
{
}

std::expected<std::string, ApiError> Agent::run(std::string_view task)
{
  // Seed history with the system prompt on first call.
  if (m_history.empty() && !m_config.system_prompt.empty())
  {
    m_history.push_back(Message{Role::SYSTEM, {TextBlock{m_config.system_prompt}}, {}, {}, {}});
  }

  m_history.push_back(Message{Role::USER, {TextBlock{std::string{task}}}, {}, {}, {}});

  const auto DEFS = m_tools.definitions();

  for (int step = 0; step < m_config.max_steps; ++step)
  {
    auto result = m_client.chat(ChatRequest{.messages = m_history, .tools = DEFS});
    if (!result)
    {
      return std::unexpected{result.error()};
    }
    if (result->choices.empty())
    {
      return std::unexpected{ApiError{
        .http_status = 0, .message = "response contained no choices", .type = {}, .code = {}}};
    }
    const ChatChoice& choice = result->choices.front();

    if (choice.finish_reason == "tool_calls" && !choice.tool_calls.empty())
    {
      // Record the assistant's tool-call request in history.
      Message assistant_msg;
      assistant_msg.role       = Role::ASSISTANT;
      assistant_msg.tool_calls = choice.tool_calls;
      m_history.push_back(std::move(assistant_msg));

      // Dispatch each tool and push tool-result messages.
      for (const auto& tc : choice.tool_calls)
      {
        auto tool_result = m_tools.invoke(tc.function_name, tc.arguments_json);

        std::string result_text =
          tool_result ? *tool_result : fmt::format("[tool error: {}]", tool_result.error());

        Message tool_msg;
        tool_msg.role         = Role::TOOL;
        tool_msg.tool_call_id = tc.id;
        tool_msg.name         = tc.function_name;
        tool_msg.content.push_back(TextBlock{std::move(result_text)});
        m_history.push_back(std::move(tool_msg));
      }

      continue; // send updated history back to the model
    }

    // finish_reason == "stop": we have a final answer.
    const std::string ANSWER = choice.content.value_or("");

    Message assistant_msg;
    assistant_msg.role = Role::ASSISTANT;
    assistant_msg.content.push_back(TextBlock{ANSWER});
    m_history.push_back(std::move(assistant_msg));

    return ANSWER;
  }

  return std::unexpected{
    ApiError{.http_status = 0,
             .message     = fmt::format("agent exceeded max_steps ({})", m_config.max_steps),
             .type        = {},
             .code        = {}}};
}

} // namespace oai
