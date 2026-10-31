#include "oai_api/builder.hpp"

#include <algorithm>

namespace oai {

// ---------------------------------------------------------------------------
// MessageBuilder
// ---------------------------------------------------------------------------

MessageBuilder::MessageBuilder(Role role)
  : m_role{role}
{
}

MessageBuilder MessageBuilder::user()
{
  return MessageBuilder{Role::USER};
}

MessageBuilder MessageBuilder::system()
{
  return MessageBuilder{Role::SYSTEM};
}

MessageBuilder MessageBuilder::assistant()
{
  return MessageBuilder{Role::ASSISTANT};
}

MessageBuilder MessageBuilder::developer()
{
  return MessageBuilder{Role::DEVELOPER};
}

MessageBuilder& MessageBuilder::text(std::string value)
{
  m_content.push_back(TextBlock{std::move(value)});
  return *this;
}

MessageBuilder& MessageBuilder::image(const cv::Mat& mat)
{
  auto result = encodeImage(mat);
  if (result)
  {
    m_content.push_back(std::move(*result));
  }
  else if (!m_error)
  {
    m_error = std::move(result.error());
  }
  return *this;
}

MessageBuilder& MessageBuilder::image(ImageBlock block)
{
  m_content.push_back(std::move(block));
  return *this;
}

MessageBuilder& MessageBuilder::json(nlohmann::json data)
{
  m_content.push_back(JsonBlock{std::move(data)});
  return *this;
}

std::expected<Message, std::string> MessageBuilder::build()
{
  if (m_error)
  {
    return std::unexpected{std::move(*m_error)};
  }
  return Message{m_role, std::move(m_content), {}, {}, {}};
}

// ---------------------------------------------------------------------------
// ClientBuilder
// ---------------------------------------------------------------------------

ClientBuilder& ClientBuilder::baseUrl(std::string url)
{
  m_cfg.base_url = std::move(url);
  return *this;
}

ClientBuilder& ClientBuilder::apiKey(std::string key)
{
  m_cfg.api_key = std::move(key);
  return *this;
}

ClientBuilder& ClientBuilder::model(std::string name)
{
  m_cfg.model = std::move(name);
  return *this;
}

ClientBuilder& ClientBuilder::temperature(float t)
{
  m_cfg.temperature = t;
  return *this;
}

ClientBuilder& ClientBuilder::maxTokens(int n)
{
  m_cfg.max_tokens = n;
  return *this;
}

ClientBuilder& ClientBuilder::connectTimeout(std::chrono::milliseconds timeout)
{
  m_cfg.connect_timeout = std::max(timeout, std::chrono::milliseconds{1});
  return *this;
}

ClientBuilder& ClientBuilder::requestTimeout(std::chrono::milliseconds timeout)
{
  m_cfg.request_timeout = std::max(timeout, std::chrono::milliseconds{1});
  return *this;
}

ClientBuilder& ClientBuilder::header(std::string name, std::string value)
{
  m_cfg.headers.emplace_back(std::move(name), std::move(value));
  return *this;
}

ClientBuilder& ClientBuilder::extraBody(nlohmann::json body)
{
  m_cfg.extra_body = std::move(body);
  return *this;
}

OpenAIClient ClientBuilder::build()
{
  return OpenAIClient{std::move(m_cfg)};
}

// ---------------------------------------------------------------------------
// AgentBuilder
// ---------------------------------------------------------------------------

AgentBuilder& AgentBuilder::systemPrompt(std::string prompt)
{
  m_cfg.system_prompt = std::move(prompt);
  return *this;
}

AgentBuilder& AgentBuilder::maxSteps(int steps)
{
  m_cfg.max_steps = steps;
  return *this;
}

AgentBuilder& AgentBuilder::client(OpenAIClient c)
{
  m_client = std::move(c);
  return *this;
}

AgentBuilder& AgentBuilder::tools(ToolRegistry registry)
{
  m_tools = std::move(registry);
  return *this;
}

Agent AgentBuilder::build()
{
  return Agent{std::move(m_cfg), std::move(*m_client), std::move(m_tools)};
}

} // namespace oai
