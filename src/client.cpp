#include "oai_api/client.hpp"

#include <filesystem>

#include <cpr/cpr.h>
#include <fmt/core.h>

#include "serialize.hpp"

namespace oai {

using json = nlohmann::json;

namespace {

// Bytes of a streamed body kept for error reporting when the server answers non-2xx.
constexpr std::size_t MAX_ERROR_BODY = std::size_t{64} * 1024;

struct Endpoint
{
  std::string_view path;
  std::string_view content_type; // empty: let cpr set it (multipart) or no body
  std::string_view accept;
};

void configure(cpr::Session& session, const detail::ClientConfig& cfg, const Endpoint& ep)
{
  const auto& [path, content_type, accept] = ep;
  cpr::Header headers{{"Accept", std::string{accept}}};
  if (!content_type.empty())
  {
    headers["Content-Type"] = std::string{content_type};
  }
  if (!cfg.api_key.empty())
  {
    headers["Authorization"] = fmt::format("Bearer {}", cfg.api_key);
  }
  for (const auto& [key, value] : cfg.headers)
  {
    headers[key] = value;
  }

  session.SetUrl(cpr::Url{cfg.base_url + std::string{path}});
  session.SetHeader(headers);
  session.SetConnectTimeout(cpr::ConnectTimeout{cfg.connect_timeout});
  session.SetTimeout(cpr::Timeout{cfg.request_timeout});
}

cpr::Response perform(cpr::Session& session, Method method)
{
  switch (method)
  {
    case Method::GET:
      return session.Get();
    case Method::DELETE:
      return session.Delete();
    case Method::POST:
      break;
  }
  return session.Post();
}

// Transport failure or non-2xx -> ApiError; otherwise the body.
std::expected<std::string, ApiError> checked(cpr::Response resp)
{
  if (resp.error.code != cpr::ErrorCode::OK)
  {
    return std::unexpected{ApiError{.http_status = 0,
                                    .message = fmt::format("HTTP error: {}", resp.error.message),
                                    .type    = {},
                                    .code    = {}}};
  }
  const auto STATUS = static_cast<int>(resp.status_code);
  if (STATUS < 200 || STATUS >= 300)
  {
    return std::unexpected{detail::parseApiError(STATUS, resp.text)};
  }
  return std::move(resp.text);
}

ApiError parseError(const std::exception& e)
{
  return ApiError{
    .http_status = 0, .message = fmt::format("JSON error: {}", e.what()), .type = {}, .code = {}};
}

std::expected<json, ApiError>
sendJson(const detail::ClientConfig& cfg, Method method, std::string_view path, const json& body)
{
  cpr::Session session;
  configure(session,
            cfg,
            {.path         = path,
             .content_type = body.is_null() ? "" : "application/json",
             .accept       = "application/json"});
  if (!body.is_null())
  {
    session.SetBody(cpr::Body{body.dump()});
  }
  auto text = checked(perform(session, method));
  if (!text)
  {
    return std::unexpected{std::move(text.error())};
  }
  if (text->empty())
  {
    return json{};
  }
  try
  {
    return json::parse(*text);
  }
  catch (const json::exception& e)
  {
    return std::unexpected{parseError(e)};
  }
}

// Build a request body and parse the response JSON, mapping JSON exceptions to ApiError.
template <typename TBuild, typename TParse>
auto postAndParse(const detail::ClientConfig& cfg,
                  std::string_view path,
                  TBuild&& build,
                  TParse&& parse) -> std::expected<decltype(parse(json{})), ApiError>
{
  json body;
  try
  {
    body = build();
  }
  catch (const json::exception& e)
  {
    return std::unexpected{parseError(e)};
  }
  auto resp = sendJson(cfg, Method::POST, path, body);
  if (!resp)
  {
    return std::unexpected{std::move(resp.error())};
  }
  try
  {
    return parse(*resp);
  }
  catch (const json::exception& e)
  {
    return std::unexpected{parseError(e)};
  }
}

// POST body with stream=true and dispatch each JSON event. on_event returns false to cancel.
std::expected<void, ApiError> streamJson(const detail::ClientConfig& cfg,
                                         std::string_view path,
                                         json body,
                                         const std::function<bool(const json&)>& on_event)
{
  body["stream"] = true;
  cpr::Session session;
  configure(session,
            cfg,
            {.path = path, .content_type = "application/json", .accept = "text/event-stream"});
  session.SetBody(cpr::Body{body.dump()});

  detail::SseParser parser;
  std::string error_body;
  std::optional<ApiError> stream_error;
  bool cancelled = false;
  bool done      = false;

  const detail::SseParser::DataCallback ON_DATA = [&](std::string_view data) {
    if (done || data == "[DONE]")
    {
      done = true;
      return true;
    }
    try
    {
      const auto J = json::parse(data);
      if ((J.contains("error") && !J["error"].is_null()) || J.value("type", "") == "error")
      {
        stream_error = detail::parseApiError(0, std::string{data});
        if (stream_error->message == data && J.contains("message"))
        {
          stream_error->message = J.value("message", std::string{data});
        }
        return false;
      }
      if (!on_event(J))
      {
        cancelled = true;
        return false;
      }
      return true;
    }
    catch (const json::exception& e)
    {
      stream_error = parseError(e);
      return false;
    }
  };

  session.SetWriteCallback(cpr::WriteCallback{[&](std::string_view data, intptr_t) {
    if (error_body.size() < MAX_ERROR_BODY)
    {
      error_body.append(data.substr(0, MAX_ERROR_BODY - error_body.size()));
    }
    return parser.feed(data, ON_DATA);
  }});

  auto resp = session.Post();
  if (stream_error)
  {
    return std::unexpected{std::move(*stream_error)};
  }
  if (cancelled)
  {
    return {};
  }
  resp.text = std::move(error_body); // so checked() can parse an HTTP error body
  if (auto ok = checked(std::move(resp)); !ok)
  {
    return std::unexpected{std::move(ok.error())};
  }
  parser.finish(ON_DATA);
  if (stream_error)
  {
    return std::unexpected{std::move(*stream_error)};
  }
  return {};
}

std::expected<Transcription, ApiError> postAudio(const detail::ClientConfig& cfg,
                                                 std::string_view path,
                                                 const TranscriptionRequest& req,
                                                 bool with_language)
{
  if (req.file_path.empty() && req.audio.empty())
  {
    return std::unexpected{ApiError{.http_status = 0,
                                    .message     = "TranscriptionRequest needs file_path or audio",
                                    .type        = {},
                                    .code        = {}}};
  }
  cpr::Multipart form{{"model", req.model.empty() ? cfg.model : req.model}};
  if (!req.file_path.empty())
  {
    form.parts.emplace_back("file", cpr::Files{cpr::File{req.file_path}});
  }
  else
  {
    form.parts.emplace_back(
      "file", cpr::Buffer{req.audio.begin(), req.audio.end(), std::filesystem::path{req.filename}});
  }
  if (with_language && req.language)
  {
    form.parts.emplace_back("language", *req.language);
  }
  if (req.prompt)
  {
    form.parts.emplace_back("prompt", *req.prompt);
  }
  if (req.response_format)
  {
    form.parts.emplace_back("response_format", *req.response_format);
  }
  if (req.temperature)
  {
    form.parts.emplace_back("temperature", fmt::format("{}", *req.temperature));
  }
  for (const auto& g : req.timestamp_granularities)
  {
    form.parts.emplace_back("timestamp_granularities[]", g);
  }
  json extra = json::object();
  detail::applyExtraBody(extra, cfg.extra_body, req.extra_body);
  for (const auto& [key, value] : extra.items())
  {
    if (value.is_primitive() && !value.is_null())
    {
      form.parts.emplace_back(key, value.is_string() ? value.get<std::string>() : value.dump());
    }
  }

  cpr::Session session;
  configure(session, cfg, {.path = path, .content_type = "", .accept = "application/json"});
  session.SetMultipart(std::move(form));
  auto text = checked(session.Post());
  if (!text)
  {
    return std::unexpected{std::move(text.error())};
  }
  // text / srt / vtt formats return a plain body.
  try
  {
    auto j = json::parse(*text);
    return Transcription{.text = j.value("text", ""), .raw = std::move(j)};
  }
  catch (const json::exception&)
  {
    return Transcription{.text = std::move(*text), .raw = {}};
  }
}

} // namespace

// ---------------------------------------------------------------------------
// OpenAIClient
// ---------------------------------------------------------------------------

OpenAIClient::OpenAIClient(detail::ClientConfig config)
  : m_config{std::move(config)}
{
}

std::expected<ChatResponse, ApiError> OpenAIClient::chat(const ChatRequest& request) const
{
  return postAndParse(
    m_config,
    "/chat/completions",
    [&] { return detail::chatRequestToJson(request, m_config); },
    detail::parseChatResponse);
}

std::expected<void, ApiError> OpenAIClient::chatStream(const ChatRequest& request,
                                                       ChunkCallback on_chunk) const
{
  json body;
  try
  {
    body                   = detail::chatRequestToJson(request, m_config);
    body["stream_options"] = {{"include_usage", true}};
  }
  catch (const json::exception& e)
  {
    return std::unexpected{parseError(e)};
  }
  return streamJson(m_config, "/chat/completions", std::move(body), [&](const json& j) {
    return on_chunk(detail::parseChatChunk(j));
  });
}

std::expected<CompletionResponse, ApiError>
OpenAIClient::completion(const CompletionRequest& request) const
{
  return postAndParse(
    m_config,
    "/completions",
    [&] { return detail::completionRequestToJson(request, m_config); },
    detail::parseCompletionResponse);
}

std::expected<EmbeddingResponse, ApiError>
OpenAIClient::embeddings(const EmbeddingRequest& request) const
{
  return postAndParse(
    m_config,
    "/embeddings",
    [&] { return detail::embeddingRequestToJson(request, m_config); },
    detail::parseEmbeddingResponse);
}

std::expected<std::vector<Model>, ApiError> OpenAIClient::listModels() const
{
  auto resp = sendJson(m_config, Method::GET, "/models", nullptr);
  if (!resp)
  {
    return std::unexpected{std::move(resp.error())};
  }
  try
  {
    std::vector<Model> models;
    for (const auto& m : resp->at("data"))
    {
      models.push_back(detail::parseModel(m));
    }
    return models;
  }
  catch (const json::exception& e)
  {
    return std::unexpected{parseError(e)};
  }
}

std::expected<Model, ApiError> OpenAIClient::retrieveModel(std::string_view id) const
{
  auto resp = sendJson(m_config, Method::GET, fmt::format("/models/{}", id), nullptr);
  if (!resp)
  {
    return std::unexpected{std::move(resp.error())};
  }
  try
  {
    return detail::parseModel(*resp);
  }
  catch (const json::exception& e)
  {
    return std::unexpected{parseError(e)};
  }
}

std::expected<void, ApiError> OpenAIClient::isHealthy() const
{
  auto models = listModels();
  if (!models)
  {
    return std::unexpected{std::move(models.error())};
  }
  for (const auto& m : *models)
  {
    if (m.id == m_config.model)
    {
      return {};
    }
  }
  return std::unexpected{
    ApiError{.http_status = 0,
             .message     = fmt::format("model '{}' not found on server", m_config.model),
             .type        = {},
             .code        = {}}};
}

std::expected<Response, ApiError> OpenAIClient::createResponse(const ResponseRequest& request) const
{
  return postAndParse(
    m_config,
    "/responses",
    [&] { return detail::responseRequestToJson(request, m_config); },
    detail::parseResponse);
}

std::expected<void, ApiError> OpenAIClient::createResponseStream(const ResponseRequest& request,
                                                                 EventCallback on_event) const
{
  json body;
  try
  {
    body = detail::responseRequestToJson(request, m_config);
  }
  catch (const json::exception& e)
  {
    return std::unexpected{parseError(e)};
  }
  return streamJson(m_config, "/responses", std::move(body), [&](const json& j) {
    return on_event(ResponseEvent{.type = j.value("type", ""), .data = j});
  });
}

std::expected<Response, ApiError> OpenAIClient::retrieveResponse(std::string_view id) const
{
  auto resp = sendJson(m_config, Method::GET, fmt::format("/responses/{}", id), nullptr);
  if (!resp)
  {
    return std::unexpected{std::move(resp.error())};
  }
  try
  {
    return detail::parseResponse(*resp);
  }
  catch (const json::exception& e)
  {
    return std::unexpected{parseError(e)};
  }
}

std::expected<void, ApiError> OpenAIClient::deleteResponse(std::string_view id) const
{
  auto resp = sendJson(m_config, Method::DELETE, fmt::format("/responses/{}", id), nullptr);
  if (!resp)
  {
    return std::unexpected{std::move(resp.error())};
  }
  return {};
}

std::expected<Transcription, ApiError>
OpenAIClient::transcribe(const TranscriptionRequest& request) const
{
  return postAudio(m_config, "/audio/transcriptions", request, true);
}

std::expected<Transcription, ApiError>
OpenAIClient::translate(const TranscriptionRequest& request) const
{
  return postAudio(m_config, "/audio/translations", request, false);
}

std::expected<std::string, ApiError> OpenAIClient::speech(const SpeechRequest& request) const
{
  cpr::Session session;
  configure(session,
            m_config,
            {.path = "/audio/speech", .content_type = "application/json", .accept = "*/*"});
  session.SetBody(cpr::Body{detail::speechRequestToJson(request, m_config).dump()});
  return checked(session.Post());
}

std::expected<std::vector<ImageResult>, ApiError>
OpenAIClient::generateImage(const ImageRequest& request) const
{
  return postAndParse(
    m_config,
    "/images/generations",
    [&] { return detail::imageRequestToJson(request, m_config); },
    detail::parseImages);
}

std::expected<DecisionResponse, ApiError> OpenAIClient::decide(const DecisionRequest& request) const
{
  return postAndParse(
    m_config,
    "/decisions",
    [&] { return detail::decisionRequestToJson(request, m_config); },
    [&](const json& j) {
      std::vector<std::string> order;
      for (const auto& q : request.questions)
      {
        order.push_back(std::visit([](const auto& question) { return question.name; }, q));
      }
      return detail::parseDecisionResponse(j, order);
    });
}

std::expected<json, ApiError>
OpenAIClient::request(Method method, std::string_view path, const json& body) const
{
  return sendJson(m_config, method, path, body);
}

} // namespace oai
