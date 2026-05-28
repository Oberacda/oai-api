#include "oai-api/agent.hpp"

#include <fmt/core.h>

namespace oai {

Agent::Agent(AgentConfig config, OpenAIClient client, ToolRegistry tools)
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
    m_history.push_back(systemMessage(m_config.system_prompt));
  }

  m_history.push_back(userMessage(std::string{task}));

  const auto DEFS = m_tools.definitions();

  for (int step = 0; step < m_config.max_steps; ++step)
  {
    auto result = m_client.complete(m_history, DEFS);
    if (!result)
    {
      return std::unexpected{result.error()};
    }
    const CompletionResponse& resp = *result;

    if (resp.choice.finish_reason == "tool_calls" && !resp.choice.tool_calls.empty())
    {
      // Record the assistant's tool-call request in history.
      Message assistant_msg;
      assistant_msg.role       = Role::ASSISTANT;
      assistant_msg.tool_calls = resp.choice.tool_calls;
      m_history.push_back(std::move(assistant_msg));

      // Dispatch each tool and push tool-result messages.
      for (const auto& tc : resp.choice.tool_calls)
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
    const std::string ANSWER = resp.choice.content.value_or("");

    Message assistant_msg;
    assistant_msg.role = Role::ASSISTANT;
    assistant_msg.content.push_back(TextBlock{ANSWER});
    m_history.push_back(std::move(assistant_msg));

    return ANSWER;
  }

  return std::unexpected{
    ApiError{0, fmt::format("agent exceeded max_steps ({})", m_config.max_steps)}};
}

} // namespace oai
