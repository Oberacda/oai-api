#include "serialize.hpp"

#include <algorithm>
#include <charconv>
#include <type_traits>
#include <utility>

#include <fmt/core.h>

namespace oai::detail {

namespace {

template <typename T>
void setIf(json& j, const char* key, const std::optional<T>& value)
{
  if (value)
  {
    j[key] = *value;
  }
}

void setIf(json& j, const char* key, const json& value)
{
  if (!value.is_null())
  {
    j[key] = value;
  }
}

void setIf(json& j, const char* key, const std::vector<std::string>& value)
{
  if (!value.empty())
  {
    j[key] = value;
  }
}

std::string modelOr(const std::string& model, const ClientConfig& cfg)
{
  return model.empty() ? cfg.model : model;
}

std::string roleToString(Role role)
{
  switch (role)
  {
    case Role::SYSTEM:
      return "system";
    case Role::DEVELOPER:
      return "developer";
    case Role::USER:
      return "user";
    case Role::ASSISTANT:
      return "assistant";
    case Role::TOOL:
      return "tool";
  }
  return "user";
}

std::string imageUrl(const ImageBlock& b)
{
  return b.url.empty() ? fmt::format("data:{};base64,{}", b.mime_type, b.base64_data) : b.url;
}

std::string firstText(const Message& msg)
{
  for (const auto& block : msg.content)
  {
    if (const auto* t = std::get_if<TextBlock>(&block))
    {
      return t->text;
    }
  }
  return {};
}

// Chat-completions content part.
json chatContentBlockToJson(const ContentBlock& block)
{
  return std::visit(
    [](const auto& b) -> json {
      using T = std::decay_t<decltype(b)>;
      if constexpr (std::is_same_v<T, TextBlock>)
      {
        return {{"type", "text"}, {"text", b.text}};
      }
      else if constexpr (std::is_same_v<T, ImageBlock>)
      {
        return {{"type", "image_url"}, {"image_url", {{"url", imageUrl(b)}}}};
      }
      else
      {
        return {{"type", "text"}, {"text", b.data.dump()}};
      }
    },
    block);
}

// Responses / Decisions content part. Assistant text is output_text, everything else input_*.
json responsesContentBlockToJson(const ContentBlock& block, bool assistant)
{
  return std::visit(
    [assistant](const auto& b) -> json {
      using T = std::decay_t<decltype(b)>;
      if constexpr (std::is_same_v<T, TextBlock>)
      {
        return {{"type", assistant ? "output_text" : "input_text"}, {"text", b.text}};
      }
      else if constexpr (std::is_same_v<T, ImageBlock>)
      {
        return {{"type", "input_image"}, {"image_url", imageUrl(b)}};
      }
      else
      {
        return {{"type", "input_json"}, {"data", b.data}};
      }
    },
    block);
}

json responsesToolToJson(const ToolDefinition& def)
{
  return {{"type", "function"},
          {"name", def.name},
          {"description", def.description},
          {"parameters", json::parse(def.parameters_schema_json)}};
}

json inputToJson(const std::variant<std::string, std::vector<Message>>& input)
{
  if (const auto* s = std::get_if<std::string>(&input))
  {
    return *s;
  }
  return messagesToResponsesInput(std::get<std::vector<Message>>(input));
}

json instructionsToJson(const DecisionInstructions& instructions)
{
  if (const auto* s = std::get_if<std::string>(&instructions))
  {
    return *s;
  }
  json arr = json::array();
  for (const auto& block : std::get<std::vector<ContentBlock>>(instructions))
  {
    arr.push_back(responsesContentBlockToJson(block, false));
  }
  return arr;
}

std::string stringOrDump(const json& j)
{
  if (j.is_null())
  {
    return {};
  }
  return j.is_string() ? j.get<std::string>() : j.dump();
}

} // namespace

// ---------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------

void applyExtraBody(json& body, const json& client_extra, const json& request_extra)
{
  if (client_extra.is_object())
  {
    body.merge_patch(client_extra);
  }
  if (request_extra.is_object())
  {
    body.merge_patch(request_extra);
  }
}

ApiError parseApiError(int http_status, const std::string& body)
{
  ApiError err{.http_status = http_status, .message = body, .type = {}, .code = {}};
  try
  {
    const auto J = json::parse(body);
    if (J.is_object() && J.contains("error"))
    {
      const auto& e = J["error"];
      if (e.is_object())
      {
        err.message = e.value("message", body);
        err.type    = stringOrDump(e.value("type", json{}));
        err.code    = stringOrDump(e.value("code", json{}));
      }
      else
      {
        err.message = stringOrDump(e);
      }
    }
  }
  catch (const json::exception&) // NOLINT(bugprone-empty-catch): non-JSON body stays the message
  {
  }
  return err;
}

Usage parseUsage(const json& j)
{
  Usage u;
  if (!j.is_object())
  {
    return u;
  }
  u.input_tokens  = j.value("input_tokens", j.value("prompt_tokens", 0));
  u.output_tokens = j.value("output_tokens", j.value("completion_tokens", 0));
  u.total_tokens  = j.value("total_tokens", u.input_tokens + u.output_tokens);
  u.cached_tokens = j.value("/input_tokens_details/cached_tokens"_json_pointer,
                            j.value("/prompt_tokens_details/cached_tokens"_json_pointer, 0));
  if (j.contains("cost") && j["cost"].is_number())
  {
    u.cost = j["cost"].get<double>();
  }
  return u;
}

Model parseModel(const json& j)
{
  return Model{.id       = j.value("id", ""),
               .owned_by = j.value("owned_by", ""),
               .created  = j.value("created", std::int64_t{0})};
}

// ---------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------

json messageToJson(const Message& msg)
{
  json j;
  j["role"] = roleToString(msg.role);

  // Tool result messages use a plain string content.
  if (msg.role == Role::TOOL)
  {
    j["content"] = firstText(msg);
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
    const std::string TEXT = firstText(msg);
    j["content"]           = TEXT.empty() ? json(nullptr) : json(TEXT);
    json tc_arr            = json::array();
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

  if (msg.content.size() == 1 && std::holds_alternative<TextBlock>(msg.content[0]))
  {
    j["content"] = std::get<TextBlock>(msg.content[0]).text;
  }
  else
  {
    json arr = json::array();
    for (const auto& block : msg.content)
    {
      arr.push_back(chatContentBlockToJson(block));
    }
    j["content"] = std::move(arr);
  }
  return j;
}

json toolDefinitionToJson(const ToolDefinition& def)
{
  return {{"type", "function"},
          {"function",
           {{"name", def.name},
            {"description", def.description},
            {"parameters", json::parse(def.parameters_schema_json)}}}};
}

json chatRequestToJson(const ChatRequest& req, const ClientConfig& cfg)
{
  json body;
  body["model"] = modelOr(req.model, cfg);

  json msgs = json::array();
  for (const auto& m : req.messages)
  {
    msgs.push_back(messageToJson(m));
  }
  body["messages"] = std::move(msgs);

  if (!req.tools.empty())
  {
    json tools = json::array();
    for (const auto& t : req.tools)
    {
      tools.push_back(toolDefinitionToJson(t));
    }
    body["tools"] = std::move(tools);
  }
  setIf(body, "tool_choice", req.tool_choice);
  setIf(body, "temperature", req.temperature ? req.temperature : cfg.temperature);
  setIf(body, "top_p", req.top_p);
  if (req.max_tokens || req.max_completion_tokens)
  {
    setIf(body, "max_tokens", req.max_tokens);
    setIf(body, "max_completion_tokens", req.max_completion_tokens);
  }
  else
  {
    setIf(body, "max_tokens", cfg.max_tokens);
  }
  setIf(body, "n", req.n);
  setIf(body, "stop", req.stop);
  setIf(body, "seed", req.seed);
  setIf(body, "presence_penalty", req.presence_penalty);
  setIf(body, "frequency_penalty", req.frequency_penalty);
  setIf(body, "response_format", req.response_format);
  setIf(body, "reasoning_effort", req.reasoning_effort);
  setIf(body, "logprobs", req.logprobs);
  setIf(body, "top_logprobs", req.top_logprobs);
  setIf(body, "parallel_tool_calls", req.parallel_tool_calls);
  setIf(body, "user", req.user);

  applyExtraBody(body, cfg.extra_body, req.extra_body);
  return body;
}

ChatResponse parseChatResponse(const json& j)
{
  ChatResponse resp;
  resp.id    = j.value("id", "");
  resp.model = j.value("model", "");
  resp.usage = parseUsage(j.value("usage", json{}));

  for (const auto& c : j.at("choices"))
  {
    ChatChoice choice;
    choice.index         = c.value("index", 0);
    choice.finish_reason = stringOrDump(c.value("finish_reason", json{}));

    const auto& msg = c.at("message");
    if (msg.contains("content") && msg["content"].is_string())
    {
      choice.content = msg["content"].get<std::string>();
    }
    if (msg.contains("refusal") && msg["refusal"].is_string())
    {
      choice.refusal = msg["refusal"].get<std::string>();
    }
    if (msg.contains("tool_calls") && msg["tool_calls"].is_array())
    {
      for (const auto& tc : msg["tool_calls"])
      {
        choice.tool_calls.push_back(
          ToolCall{.id             = tc.value("id", ""),
                   .function_name  = tc.value("/function/name"_json_pointer, ""),
                   .arguments_json = tc.value("/function/arguments"_json_pointer, "{}")});
      }
    }
    resp.choices.push_back(std::move(choice));
  }
  resp.raw = j;
  return resp;
}

ChatChunk parseChatChunk(const json& j)
{
  ChatChunk chunk;
  chunk.id    = j.value("id", "");
  chunk.model = j.value("model", "");
  if (j.contains("usage") && j["usage"].is_object())
  {
    chunk.usage = parseUsage(j["usage"]);
  }

  const auto& choices = j.value("choices", json::array());
  if (!choices.empty())
  {
    const auto& c = choices.at(0);
    if (c.contains("finish_reason") && c["finish_reason"].is_string())
    {
      chunk.finish_reason = c["finish_reason"].get<std::string>();
    }
    const auto& delta = c.value("delta", json::object());
    if (delta.contains("content") && delta["content"].is_string())
    {
      chunk.content = delta["content"].get<std::string>();
    }
    if (delta.contains("tool_calls") && delta["tool_calls"].is_array())
    {
      for (const auto& tc : delta["tool_calls"])
      {
        chunk.tool_calls.push_back(ToolCallDelta{
          .index         = tc.value("index", 0),
          .id            = stringOrDump(tc.value("id", json{})),
          .function_name = stringOrDump(tc.value("/function/name"_json_pointer, json{})),
          .arguments     = stringOrDump(tc.value("/function/arguments"_json_pointer, json{}))});
      }
    }
  }
  chunk.raw = j;
  return chunk;
}

// ---------------------------------------------------------------------------
// Legacy completions / embeddings
// ---------------------------------------------------------------------------

json completionRequestToJson(const CompletionRequest& req, const ClientConfig& cfg)
{
  json body;
  body["model"]  = modelOr(req.model, cfg);
  body["prompt"] = req.prompt;
  setIf(body, "suffix", req.suffix);
  setIf(body, "max_tokens", req.max_tokens ? req.max_tokens : cfg.max_tokens);
  setIf(body, "temperature", req.temperature ? req.temperature : cfg.temperature);
  setIf(body, "top_p", req.top_p);
  setIf(body, "n", req.n);
  setIf(body, "stop", req.stop);
  setIf(body, "seed", req.seed);
  setIf(body, "echo", req.echo);
  applyExtraBody(body, cfg.extra_body, req.extra_body);
  return body;
}

CompletionResponse parseCompletionResponse(const json& j)
{
  CompletionResponse resp;
  resp.id    = j.value("id", "");
  resp.model = j.value("model", "");
  resp.usage = parseUsage(j.value("usage", json{}));
  for (const auto& c : j.at("choices"))
  {
    resp.choices.push_back(
      CompletionChoice{.index         = c.value("index", 0),
                       .text          = c.value("text", ""),
                       .finish_reason = stringOrDump(c.value("finish_reason", json{}))});
  }
  resp.raw = j;
  return resp;
}

json embeddingRequestToJson(const EmbeddingRequest& req, const ClientConfig& cfg)
{
  json body;
  body["model"]           = modelOr(req.model, cfg);
  body["input"]           = req.input;
  body["encoding_format"] = "float";
  setIf(body, "dimensions", req.dimensions);
  setIf(body, "user", req.user);
  applyExtraBody(body, cfg.extra_body, req.extra_body);
  return body;
}

EmbeddingResponse parseEmbeddingResponse(const json& j)
{
  EmbeddingResponse resp;
  resp.model = j.value("model", "");
  resp.usage = parseUsage(j.value("usage", json{}));

  std::vector<std::pair<int, std::vector<float>>> indexed;
  for (const auto& d : j.at("data"))
  {
    indexed.emplace_back(d.value("index", static_cast<int>(indexed.size())),
                         d.at("embedding").get<std::vector<float>>());
  }
  std::ranges::sort(indexed, {}, &std::pair<int, std::vector<float>>::first);
  for (auto& [_, vec] : indexed)
  {
    resp.data.push_back(std::move(vec));
  }
  resp.raw = j;
  return resp;
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

json messagesToResponsesInput(const std::vector<Message>& messages)
{
  json items = json::array();
  for (const auto& msg : messages)
  {
    if (msg.role == Role::TOOL)
    {
      items.push_back({{"type", "function_call_output"},
                       {"call_id", msg.tool_call_id.value_or("")},
                       {"output", firstText(msg)}});
      continue;
    }

    const bool ASSISTANT = msg.role == Role::ASSISTANT;
    if (!msg.content.empty())
    {
      json parts = json::array();
      for (const auto& block : msg.content)
      {
        parts.push_back(responsesContentBlockToJson(block, ASSISTANT));
      }
      items.push_back({{"role", roleToString(msg.role)}, {"content", std::move(parts)}});
    }
    for (const auto& tc : msg.tool_calls)
    {
      items.push_back({{"type", "function_call"},
                       {"call_id", tc.id},
                       {"name", tc.function_name},
                       {"arguments", tc.arguments_json}});
    }
  }
  return items;
}

json responseRequestToJson(const ResponseRequest& req, const ClientConfig& cfg)
{
  json body;
  body["model"] = modelOr(req.model, cfg);
  body["input"] = inputToJson(req.input);
  setIf(body, "instructions", req.instructions);
  if (!req.tools.empty())
  {
    json tools = json::array();
    for (const auto& t : req.tools)
    {
      tools.push_back(responsesToolToJson(t));
    }
    body["tools"] = std::move(tools);
  }
  setIf(body, "tool_choice", req.tool_choice);
  setIf(body, "temperature", req.temperature ? req.temperature : cfg.temperature);
  setIf(body, "top_p", req.top_p);
  setIf(body, "max_output_tokens", req.max_output_tokens ? req.max_output_tokens : cfg.max_tokens);
  setIf(body, "reasoning", req.reasoning);
  setIf(body, "text", req.text);
  setIf(body, "previous_response_id", req.previous_response_id);
  setIf(body, "store", req.store);
  setIf(body, "metadata", req.metadata);
  applyExtraBody(body, cfg.extra_body, req.extra_body);
  return body;
}

Response parseResponse(const json& j)
{
  Response resp;
  resp.id     = j.value("id", "");
  resp.model  = j.value("model", "");
  resp.status = j.value("status", "");
  resp.usage  = parseUsage(j.value("usage", json{}));

  for (const auto& item : j.value("output", json::array()))
  {
    const std::string TYPE = item.value("type", "");
    if (TYPE == "message")
    {
      for (const auto& part : item.value("content", json::array()))
      {
        if (part.value("type", "") == "output_text")
        {
          resp.output_text += part.value("text", "");
        }
      }
    }
    else if (TYPE == "function_call")
    {
      resp.tool_calls.push_back(ToolCall{.id             = item.value("call_id", ""),
                                         .function_name  = item.value("name", ""),
                                         .arguments_json = item.value("arguments", "{}")});
    }
  }
  resp.raw = j;
  return resp;
}

// ---------------------------------------------------------------------------
// Audio / images
// ---------------------------------------------------------------------------

json speechRequestToJson(const SpeechRequest& req, const ClientConfig& cfg)
{
  json body;
  body["model"] = modelOr(req.model, cfg);
  body["input"] = req.input;
  body["voice"] = req.voice;
  setIf(body, "response_format", req.response_format);
  setIf(body, "speed", req.speed);
  setIf(body, "instructions", req.instructions);
  applyExtraBody(body, cfg.extra_body, req.extra_body);
  return body;
}

json imageRequestToJson(const ImageRequest& req, const ClientConfig& cfg)
{
  json body;
  body["model"]  = modelOr(req.model, cfg);
  body["prompt"] = req.prompt;
  setIf(body, "n", req.n);
  setIf(body, "size", req.size);
  setIf(body, "quality", req.quality);
  setIf(body, "response_format", req.response_format);
  setIf(body, "output_format", req.output_format);
  setIf(body, "background", req.background);
  applyExtraBody(body, cfg.extra_body, req.extra_body);
  return body;
}

std::vector<ImageResult> parseImages(const json& j)
{
  std::vector<ImageResult> out;
  for (const auto& d : j.at("data"))
  {
    out.push_back(ImageResult{.b64_json       = d.value("b64_json", ""),
                              .url            = d.value("url", ""),
                              .revised_prompt = d.value("revised_prompt", "")});
  }
  return out;
}

// ---------------------------------------------------------------------------
// Decisions
// ---------------------------------------------------------------------------

namespace {

// OpenRouter state / instructions item: text as plain string, images as image_url parts, JSON
// as-is.
json openRouterPart(const ContentBlock& block)
{
  return std::visit(
    [](const auto& b) -> json {
      using T = std::decay_t<decltype(b)>;
      if constexpr (std::is_same_v<T, TextBlock>)
      {
        return b.text;
      }
      else if constexpr (std::is_same_v<T, ImageBlock>)
      {
        return {{"type", "image_url"}, {"image_url", {{"url", imageUrl(b)}}}};
      }
      else
      {
        return b.data;
      }
    },
    block);
}

json openRouterInstructions(const DecisionInstructions& instructions)
{
  if (const auto* s = std::get_if<std::string>(&instructions))
  {
    return *s;
  }
  json arr = json::array();
  for (const auto& block : std::get<std::vector<ContentBlock>>(instructions))
  {
    arr.push_back(openRouterPart(block));
  }
  return arr;
}

json openRouterState(const std::variant<std::string, std::vector<Message>>& input)
{
  if (const auto* s = std::get_if<std::string>(&input))
  {
    return *s;
  }
  json arr = json::array();
  for (const auto& msg : std::get<std::vector<Message>>(input))
  {
    for (const auto& block : msg.content)
    {
      arr.push_back(openRouterPart(block));
    }
  }
  return arr;
}

std::string levelGuidance(const DecisionLevel& level)
{
  if (level.description.empty())
  {
    return level.label;
  }
  return level.label.empty() ? level.description : level.label + ": " + level.description;
}

json openAiQuestion(const DecisionQuestion& q)
{
  return std::visit(
    [](const auto& question) -> json {
      using T = std::decay_t<decltype(question)>;
      json out{{"name", question.name},
               {"instructions", instructionsToJson(question.instructions)}};
      if constexpr (std::is_same_v<T, PredicateQuestion>)
      {
        out["type"] = "predicate";
      }
      else if constexpr (std::is_same_v<T, ChoiceQuestion>)
      {
        out["type"]    = "choice";
        out["choices"] = json::array();
        for (const auto& c : question.choices)
        {
          out["choices"].push_back({{"value", c.value}, {"description", c.description}});
        }
      }
      else
      {
        out["type"]   = "score";
        out["levels"] = json::array();
        for (const auto& l : question.levels)
        {
          out["levels"].push_back({{"label", l.label}, {"description", l.description}});
        }
      }
      return out;
    },
    q);
}

json openRouterQuestion(const DecisionQuestion& q)
{
  return std::visit(
    [](const auto& question) -> json {
      using T = std::decay_t<decltype(question)>;
      json out{{"instructions", openRouterInstructions(question.instructions)}};
      if constexpr (std::is_same_v<T, PredicateQuestion>)
      {
        out["type"]     = "noul";
        out["criteria"] = {{"true", question.true_criteria}, {"false", question.false_criteria}};
      }
      else if constexpr (std::is_same_v<T, ChoiceQuestion>)
      {
        out["type"]     = "choice";
        out["criteria"] = json::object();
        for (const auto& c : question.choices)
        {
          out["criteria"][c.value] = c.description.empty() ? json(nullptr) : json(c.description);
        }
      }
      else
      {
        out["type"]     = "score";
        out["criteria"] = json::array();
        for (const auto& l : question.levels)
        {
          out["criteria"].push_back(levelGuidance(l));
        }
      }
      return out;
    },
    q);
}

std::optional<double> optDouble(const json& obj, const char* key)
{
  if (obj.contains(key) && obj[key].is_number())
  {
    return obj[key].get<double>();
  }
  return std::nullopt;
}

DecisionAnswer parseAnswer(const json& a, std::string name)
{
  DecisionAnswer ans;
  ans.type        = a.value("type", "");
  ans.name        = std::move(name);
  ans.probability = optDouble(a, "probability");
  ans.score       = optDouble(a, "score");
  ans.confidence  = optDouble(a, "confidence");
  if (a.contains("choice") && a["choice"].is_string())
  {
    ans.choice = a["choice"].get<std::string>();
  }
  if (ans.type == "noul") // OpenRouter yes/no -> predicate
  {
    ans.type        = "predicate";
    ans.probability = optDouble(a, "noul");
  }

  const auto& probs = a.value("probabilities", json{});
  if (probs.is_array()) // OpenAI: [{value, label?, probability}]
  {
    for (const auto& p : probs)
    {
      DecisionProbability prob;
      const auto& value = p.value("value", json{});
      if (value.is_number_integer())
      {
        prob.level = value.get<int>();
      }
      else
      {
        prob.value = stringOrDump(value);
      }
      prob.label       = p.value("label", "");
      prob.probability = p.value("probability", 0.0);
      ans.probabilities.push_back(std::move(prob));
    }
  }
  else if (probs.is_object()) // OpenRouter: {label: p} or {"<index>": p} + legend
  {
    const bool SCORE  = ans.type == "score";
    const auto LEGEND = a.value("legend", json::object());
    for (const auto& [key, p] : probs.items())
    {
      DecisionProbability prob;
      prob.probability = p.is_number() ? p.get<double>() : 0.0;
      if (SCORE)
      {
        std::from_chars(
          key.data(), key.data() + key.size(), prob.level); // stays -1 if not a number
        prob.label = stringOrDump(LEGEND.value(key, json{}));
      }
      else
      {
        prob.value = key;
      }
      ans.probabilities.push_back(std::move(prob));
    }
    if (SCORE)
    {
      std::ranges::sort(ans.probabilities, {}, &DecisionProbability::level);
    }
  }
  return ans;
}

} // namespace

json decisionRequestToJson(const DecisionRequest& req, const ClientConfig& cfg)
{
  json body;
  body["model"] = modelOr(req.model, cfg);
  if (req.format == DecisionFormat::OPENROUTER)
  {
    body["state"]     = openRouterState(req.input);
    body["questions"] = json::object();
    for (const auto& q : req.questions)
    {
      const auto& name =
        std::visit([](const auto& question) -> const std::string& { return question.name; }, q);
      body["questions"][name] = openRouterQuestion(q);
    }
  }
  else
  {
    body["input"]     = inputToJson(req.input);
    body["questions"] = json::array();
    for (const auto& q : req.questions)
    {
      body["questions"].push_back(openAiQuestion(q));
    }
    setIf(body, "prompt_cache_key", req.prompt_cache_key);
  }
  applyExtraBody(body, cfg.extra_body, req.extra_body);
  return body;
}

DecisionResponse parseDecisionResponse(const json& j, const std::vector<std::string>& order)
{
  DecisionResponse resp;
  resp.id       = j.value("id", "");
  resp.model    = j.value("model", "");
  resp.provider = j.value("provider", "");
  resp.usage    = parseUsage(j.value("usage", json{}));

  const auto& answers = j.at("answers");
  if (answers.is_array())
  {
    for (const auto& a : answers)
    {
      resp.answers.push_back(parseAnswer(a, a.value("name", "")));
    }
  }
  else
  {
    // Keyed by question name; restore request order (json objects iterate alphabetically).
    for (const auto& name : order)
    {
      if (answers.contains(name))
      {
        resp.answers.push_back(parseAnswer(answers[name], name));
      }
    }
    for (const auto& [name, a] : answers.items())
    {
      if (std::ranges::find(order, name) == order.end())
      {
        resp.answers.push_back(parseAnswer(a, name));
      }
    }
  }
  resp.raw = j;
  return resp;
}

// ---------------------------------------------------------------------------
// SSE
// ---------------------------------------------------------------------------

bool SseParser::feed(std::string_view chunk, const DataCallback& on_data)
{
  m_buffer.append(chunk);
  std::size_t start = 0;
  std::size_t end   = 0;
  while ((end = m_buffer.find('\n', start)) != std::string::npos)
  {
    std::string_view line{m_buffer.data() + start, end - start};
    start = end + 1;
    if (!line.empty() && line.back() == '\r')
    {
      line.remove_suffix(1);
    }

    if (line.empty())
    {
      if (!m_data.empty())
      {
        const std::string DATA = std::exchange(m_data, {});
        if (!on_data(DATA))
        {
          m_buffer.erase(0, start);
          return false;
        }
      }
    }
    else if (line.starts_with("data:"))
    {
      line.remove_prefix(5);
      if (line.starts_with(' '))
      {
        line.remove_prefix(1);
      }
      if (!m_data.empty())
      {
        m_data += '\n';
      }
      m_data.append(line);
    }
    // event:, id:, retry: and ":" comments are ignored; payloads carry their own type.
  }
  m_buffer.erase(0, start);
  return true;
}

bool SseParser::finish(const DataCallback& on_data)
{
  if (!m_buffer.empty())
  {
    feed("\n", on_data); // terminate a trailing line without newline
  }
  if (m_data.empty())
  {
    return true;
  }
  return on_data(std::exchange(m_data, {}));
}

} // namespace oai::detail
