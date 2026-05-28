#include "oai-api/client.hpp"

#include <cpr/cpr.h>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

namespace oai {

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// JSON serialisation helpers
// ---------------------------------------------------------------------------

static json contentBlockToJson(const ContentBlock& block)
{
  return std::visit(
    [](const auto& b) -> json {
      if constexpr (std::is_same_v<std::decay_t<decltype(b)>, TextBlock>)
      {
        return {{"type", "text"}, {"text", b.text}};
      }
      else
      {
        // ImageBlock
        if (!b.url.empty())
        {
          return {{"type", "image_url"}, {"image_url", {{"url", b.url}}}};
        }
        const std::string data_url = fmt::format("data:{};base64,{}", b.mime_type, b.base64_data);
        return {{"type", "image_url"}, {"image_url", {{"url", data_url}}}};
      }
    },
    block);
}

static std::string roleToString(Role role)
{
  switch (role)
  {
    case Role::SYSTEM:
      return "system";
    case Role::USER:
      return "user";
    case Role::ASSISTANT:
      return "assistant";
    case Role::TOOL:
      return "tool";
  }
  return "user";
}

static json messageToJson(const Message& msg)
{
  json j;
  j["role"] = roleToString(msg.role);

  // Tool result messages use a plain string content.
  if (msg.role == Role::TOOL)
  {
    if (!msg.content.empty())
    {
      if (const auto* t = std::get_if<TextBlock>(&msg.content[0]))
      {
        j["content"] = t->text;
      }
    }
    else
    {
      j["content"] = "";
    }
    if (msg.tool_call_id)
    {
      j["tool_call_id"] = *msg.tool_call_id;
    }
    if (msg.name)
    {
      j["name"] = *msg.name;
    }
    return j;
  }

  // Assistant messages that requested tool calls must carry tool_calls back.
  if (msg.role == Role::ASSISTANT && !msg.tool_calls.empty())
  {
    j["content"] = nullptr; // null content is valid when tool_calls present
    json tc_arr  = json::array();
    for (const auto& tc : msg.tool_calls)
    {
      tc_arr.push_back(
        {{"id", tc.id},
         {"type", "function"},
         {"function", {{"name", tc.function_name}, {"arguments", tc.arguments_json}}}});
    }
    j["tool_calls"] = std::move(tc_arr);
    return j;
  }

  // Normal text/multimodal content.
  if (msg.content.size() == 1 && std::holds_alternative<TextBlock>(msg.content[0]))
  {
    j["content"] = std::get<TextBlock>(msg.content[0]).text;
  }
  else
  {
    json arr = json::array();
    for (const auto& block : msg.content)
    {
      arr.push_back(contentBlockToJson(block));
    }
    j["content"] = std::move(arr);
  }

  return j;
}

static json toolDefinitionToJson(const ToolDefinition& def)
{
  return {{"type", "function"},
          {"function",
           {{"name", def.name},
            {"description", def.description},
            {"parameters", json::parse(def.parameters_schema_json)}}}};
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

static std::expected<CompletionResponse, ApiError> parseResponse(int http_status,
                                                                 const std::string& body)
{
  if (http_status < 200 || http_status >= 300)
  {
    std::string msg;
    try
    {
      auto j = json::parse(body);
      msg    = j.value("/error/message"_json_pointer, body);
    }
    catch (...)
    {
      msg = body;
    }
    return std::unexpected{ApiError{http_status, std::move(msg)}};
  }

  try
  {
    const auto J = json::parse(body);

    CompletionResponse resp;
    resp.id                = J.value("id", "");
    resp.prompt_tokens     = J.value("/usage/prompt_tokens"_json_pointer, 0);
    resp.completion_tokens = J.value("/usage/completion_tokens"_json_pointer, 0);

    const auto& choice        = J.at("choices").at(0);
    resp.choice.finish_reason = choice.value("finish_reason", "stop");

    const auto& api_msg = choice.at("message");

    if (api_msg.contains("content") && !api_msg["content"].is_null())
    {
      resp.choice.content = api_msg["content"].get<std::string>();
    }

    if (api_msg.contains("tool_calls") && api_msg["tool_calls"].is_array())
    {
      for (const auto& tc : api_msg["tool_calls"])
      {
        ToolCall call;
        call.id             = tc.value("id", "");
        call.function_name  = tc.value("/function/name"_json_pointer, "");
        call.arguments_json = tc.value("/function/arguments"_json_pointer, "{}");
        resp.choice.tool_calls.push_back(std::move(call));
      }
    }

    return resp;
  }
  catch (const json::exception& e)
  {
    return std::unexpected{ApiError{http_status, fmt::format("JSON parse error: {}", e.what())}};
  }
}

// ---------------------------------------------------------------------------
// OpenAIClient implementation
// ---------------------------------------------------------------------------

OpenAIClient::OpenAIClient(detail::ClientConfig config)
  : m_config{std::move(config)}
{
}

std::expected<CompletionResponse, ApiError>
OpenAIClient::complete(const std::vector<Message>& messages,
                       const std::vector<ToolDefinition>& tools)
{
  json body;
  body["model"]       = m_config.model;
  body["temperature"] = m_config.temperature;
  body["max_tokens"]  = m_config.max_tokens;

  json msgs = json::array();
  for (const auto& m : messages)
  {
    msgs.push_back(messageToJson(m));
  }
  body["messages"] = std::move(msgs);

  if (!tools.empty())
  {
    json tool_arr = json::array();
    for (const auto& t : tools)
    {
      tool_arr.push_back(toolDefinitionToJson(t));
    }
    body["tools"]       = std::move(tool_arr);
    body["tool_choice"] = "auto";
  }

  cpr::Header headers{
    {"Content-Type", "application/json"},
    {"Accept", "application/json"},
  };
  if (!m_config.api_key.empty())
  {
    headers["Authorization"] = fmt::format("Bearer {}", m_config.api_key);
  }

  const std::string URL   = m_config.base_url + "/chat/completions";
  const auto CPR_RESP     = cpr::Post(cpr::Url{URL}, headers, cpr::Body{body.dump()});

  if (CPR_RESP.error.code != cpr::ErrorCode::OK)
  {
    return std::unexpected{ApiError{0, fmt::format("HTTP error: {}", CPR_RESP.error.message)}};
  }

  return parseResponse(static_cast<int>(CPR_RESP.status_code), CPR_RESP.text);
}

} // namespace oai
