// Offline checks for request serialization, extra_body merging, response parsing and SSE parsing.

#include <cstdlib>
#include <string>
#include <vector>

#include <fmt/core.h>

#include <oai_api/oai_api.hpp>

#include "serialize.hpp"

using json = nlohmann::json;
using namespace oai;

// assert() is compiled out under NDEBUG (RelWithDebInfo); this is not.
#define CHECK(cond)                                                                                \
  do                                                                                               \
  {                                                                                                \
    if (!(cond))                                                                                   \
    {                                                                                              \
      fmt::print(stderr, "FAIL {}:{}: {}\n", __FILE__, __LINE__, #cond);                           \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (false)

static detail::ClientConfig makeConfig()
{
  detail::ClientConfig cfg;
  cfg.model       = "default-model";
  cfg.temperature = 0.5F;
  cfg.max_tokens  = 100;
  cfg.extra_body  = {{"provider", {{"order", {"anthropic", "openai"}}, {"allow_fallbacks", false}}},
                     {"route", "fallback"}};
  return cfg;
}

static void testChatRequest()
{
  const auto CFG = makeConfig();

  ChatRequest req{.messages = {Message{Role::USER, {TextBlock{"hi"}}, {}, {}, {}}},
                  .tools    = {ToolDefinition{"f", "does f", R"({"type":"object"})"}}};
  req.extra_body = {{"provider", {{"allow_fallbacks", true}}}, {"route", nullptr}};

  const auto J = detail::chatRequestToJson(req, CFG);
  CHECK(J["model"] == "default-model");
  CHECK(J["temperature"] == 0.5F);
  CHECK(J["max_tokens"] == 100);
  CHECK(!J.contains("max_completion_tokens"));
  CHECK(!J.contains("tool_choice"));
  CHECK(J["messages"][0]["content"] == "hi");
  CHECK(J["tools"][0]["function"]["name"] == "f");
  // Client extra_body applied, request extra_body wins, null removes.
  CHECK(J["provider"]["order"][0] == "anthropic");
  CHECK(J["provider"]["allow_fallbacks"] == true);
  CHECK(!J.contains("route"));

  // Explicit request values beat client defaults; max_completion_tokens suppresses default
  // max_tokens.
  ChatRequest req2{.model = "m2", .temperature = 0.0F, .max_completion_tokens = 7};
  req2.response_format = jsonSchemaFormat("out", {{"type", "object"}});
  const auto J2        = detail::chatRequestToJson(req2, CFG);
  CHECK(J2["model"] == "m2");
  CHECK(J2["temperature"] == 0.0F);
  CHECK(!J2.contains("max_tokens"));
  CHECK(J2["max_completion_tokens"] == 7);
  CHECK(J2["response_format"]["json_schema"]["strict"] == true);
}

static void testMessages()
{
  Message assistant{Role::ASSISTANT, {}, {}, {}, {ToolCall{"c1", "f", R"({"x":1})"}}};
  Message tool{Role::TOOL, {TextBlock{"42"}}, "c1", "f", {}};
  Message user{Role::USER,
               {TextBlock{"look"}, imageFromUrl("https://x/y.png"), JsonBlock{{{"a", 1}}}},
               {},
               {},
               {}};

  const auto A = detail::messageToJson(assistant);
  CHECK(A["content"].is_null());
  CHECK(A["tool_calls"][0]["function"]["arguments"] == R"({"x":1})");
  CHECK(detail::messageToJson(tool)["tool_call_id"] == "c1");
  const auto U = detail::messageToJson(user);
  CHECK(U["content"][1]["image_url"]["url"] == "https://x/y.png");
  CHECK(U["content"][2]["text"] == R"({"a":1})");

  const auto ITEMS = detail::messagesToResponsesInput({user, assistant, tool});
  CHECK(ITEMS[0]["content"][0]["type"] == "input_text");
  CHECK(ITEMS[0]["content"][1]["image_url"] == "https://x/y.png");
  CHECK(ITEMS[0]["content"][2]["type"] == "input_json");
  CHECK(ITEMS[0]["content"][2]["data"]["a"] == 1);
  CHECK(ITEMS[1]["type"] == "function_call");
  CHECK(ITEMS[1]["call_id"] == "c1");
  CHECK(ITEMS[2]["type"] == "function_call_output");
  CHECK(ITEMS[2]["output"] == "42");
}

static void testChatParsing()
{
  const auto RESP = detail::parseChatResponse(json::parse(R"({
    "id":"x","model":"m",
    "choices":[{"index":0,"finish_reason":"tool_calls","message":{"content":null,
      "tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}]}}],
    "usage":{"prompt_tokens":3,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":1}}})"));
  CHECK(RESP.choices.size() == 1);
  CHECK(!RESP.choices[0].content);
  CHECK(RESP.text().empty());
  CHECK(RESP.choices[0].tool_calls[0].function_name == "f");
  CHECK(RESP.usage.input_tokens == 3);
  CHECK(RESP.usage.output_tokens == 4);
  CHECK(RESP.usage.total_tokens == 7);
  CHECK(RESP.usage.cached_tokens == 1);

  const auto CHUNK = detail::parseChatChunk(json::parse(
    R"({"id":"x","choices":[{"index":0,"delta":{"content":"He",
        "tool_calls":[{"index":1,"function":{"arguments":"{\"a"}}]},"finish_reason":null}]})"));
  CHECK(CHUNK.content == "He");
  CHECK(!CHUNK.finish_reason);
  CHECK(CHUNK.tool_calls[0].index == 1);
  CHECK(CHUNK.tool_calls[0].arguments == R"({"a)");
  CHECK(CHUNK.tool_calls[0].id.empty());

  const auto USAGE_CHUNK =
    detail::parseChatChunk(json::parse(R"({"choices":[],"usage":{"prompt_tokens":1}})"));
  CHECK(USAGE_CHUNK.usage && USAGE_CHUNK.usage->input_tokens == 1);
}

static void testResponses()
{
  const auto CFG = makeConfig();
  ResponseRequest req{.input = "hello", .tools = {ToolDefinition{"f", "d", "{}"}}};
  const auto J = detail::responseRequestToJson(req, CFG);
  CHECK(J["input"] == "hello");
  CHECK(J["tools"][0]["name"] == "f"); // flat, unlike chat
  CHECK(J["max_output_tokens"] == 100);

  const auto RESP = detail::parseResponse(json::parse(R"({"id":"r","status":"completed",
    "output":[{"type":"reasoning"},
      {"type":"message","content":[{"type":"output_text","text":"Hel"},{"type":"output_text","text":"lo"}]},
      {"type":"function_call","call_id":"c9","name":"f","arguments":"{}"}],
    "usage":{"input_tokens":9,"output_tokens":12,"total_tokens":21}})"));
  CHECK(RESP.output_text == "Hello");
  CHECK(RESP.tool_calls[0].id == "c9");
  CHECK(RESP.usage.total_tokens == 21);
}

static void testDecisions()
{
  DecisionRequest req;
  req.input = std::vector<Message>{
    Message{Role::USER,
            {TextBlock{"Evaluate the ticket."}, JsonBlock{{{"order", {{"charges", {49, 49}}}}}}},
            {},
            {},
            {}}};
  req.questions = {
    PredicateQuestion{.name = "duplicate_charge", .instructions = "Duplicate charge?"},
    ChoiceQuestion{.name         = "department",
                   .instructions = "Route it.",
                   .choices      = {{"billing", "Payments"}, {"technical", "Bugs"}}},
    ScoreQuestion{.name         = "severity",
                  .instructions = std::vector<ContentBlock>{TextBlock{"How bad?"}},
                  .levels       = {{"Cosmetic", "minor"}, {"Blocked", "major"}}},
  };
  const auto J = detail::decisionRequestToJson(req, makeConfig());
  CHECK(J["model"] == "gpt-6-luna");
  CHECK(J["input"][0]["content"][1]["type"] == "input_json");
  CHECK(J["questions"][0]["type"] == "predicate");
  CHECK(J["questions"][1]["choices"][1]["value"] == "technical");
  CHECK(J["questions"][2]["type"] == "score");
  CHECK(J["questions"][2]["instructions"][0]["type"] == "input_text");
  CHECK(J["questions"][2]["levels"][0]["label"] == "Cosmetic");

  const auto RESP = detail::parseDecisionResponse(json::parse(R"({"answers":[
    {"type":"predicate","name":"duplicate_charge","probability":0.92},
    {"type":"choice","name":"department","choice":"billing",
     "probabilities":[{"value":"billing","probability":0.95},{"value":"technical","probability":0.05}],
     "confidence":0.93},
    {"type":"score","name":"severity","score":1.1,
     "probabilities":[{"value":0,"label":"Cosmetic","probability":0.1},{"value":1,"label":"Blocked","probability":0.9}],
     "confidence":0.55},
    {"type":"refusal","name":"other"}],
    "usage":{"input_tokens":50,"input_tokens_details":{"cached_tokens":10,"cache_write_tokens":0}}})"));
  CHECK(RESP.answers.size() == 4);
  CHECK(*RESP.answers[0].probability == 0.92);
  CHECK(*RESP.answers[1].choice == "billing");
  CHECK(RESP.answers[1].probabilities[1].value == "technical");
  CHECK(*RESP.answers[2].score == 1.1);
  CHECK(RESP.answers[2].probabilities[1].level == 1);
  CHECK(RESP.answers[2].probabilities[1].label == "Blocked");
  CHECK(RESP.answers[3].type == "refusal");
  CHECK(!RESP.answers[3].probability);
  CHECK(RESP.usage.cached_tokens == 10);
}

static void testDecisionsOpenRouter()
{
  DecisionRequest req{.format = DecisionFormat::OPENROUTER, .model = "openai/gpt-6-luna-decisions"};
  req.input     = std::vector<Message>{Message{Role::USER,
                                               {TextBlock{"Checkout shows a blank screen."},
                                                imageFromUrl("https://x/y.png"),
                                                JsonBlock{{{"customer_tier", "enterprise"}}}},
                                               {},
                                               {},
                                               {}}};
  req.questions = {
    PredicateQuestion{.name           = "is_bug",
                      .instructions   = "Is it a defect?",
                      .true_criteria  = "Broken behavior.",
                      .false_criteria = "Feature request."},
    ChoiceQuestion{.name         = "team",
                   .instructions = "Which team?",
                   .choices      = {{"payments", "Billing"}, {"frontend", ""}}},
    ScoreQuestion{.name         = "urgency",
                  .instructions = "How urgent?",
                  .levels       = {{"Low", "Can wait"}, {"High", ""}}},
  };
  req.prompt_cache_key = "ignored";
  req.extra_body       = {{"provider", {{"order", {"openai"}}}}};

  const auto J = detail::decisionRequestToJson(req, makeConfig());
  CHECK(J["model"] == "openai/gpt-6-luna-decisions");
  CHECK(!J.contains("input") && !J.contains("prompt_cache_key"));
  CHECK(J["state"][0] == "Checkout shows a blank screen.");
  CHECK(J["state"][1]["image_url"]["url"] == "https://x/y.png");
  CHECK(J["state"][2]["customer_tier"] == "enterprise");
  CHECK(J["questions"].is_object());
  CHECK(J["questions"]["is_bug"]["type"] == "noul");
  CHECK(J["questions"]["is_bug"]["criteria"]["true"] == "Broken behavior.");
  CHECK(J["questions"]["team"]["criteria"]["payments"] == "Billing");
  CHECK(J["questions"]["team"]["criteria"]["frontend"].is_null());
  CHECK(J["questions"]["urgency"]["criteria"][0] == "Low: Can wait");
  CHECK(J["questions"]["urgency"]["criteria"][1] == "High");
  CHECK(J["provider"]["order"][0] == "openai");

  // Example 200 response from the OpenRouter API reference.
  const auto RESP = detail::parseDecisionResponse(json::parse(R"({
    "id":"gen-dec-1","model":"m","provider":"OpenAI",
    "answers":{
      "is_bug":{"type":"noul","noul":0.96},
      "team":{"type":"choice","choice":"payments","confidence":0.75,
              "probabilities":{"account":0,"frontend":0.16,"payments":0.84}},
      "urgency":{"type":"score","score":1.99,"confidence":0.99,
                 "probabilities":{"0":0,"1":0.01,"2":0.99},
                 "legend":{"0":"Next release","1":"This week","2":"Now"}}},
    "usage":{"input_tokens":476,"output_tokens":70,"cost":0.000019992}})"),
                                                  {"urgency", "is_bug", "team"});
  CHECK(RESP.provider == "OpenAI");
  CHECK(RESP.answers.size() == 3);
  CHECK(RESP.answers[0].name == "urgency"); // request order, not alphabetical
  CHECK(RESP.answers[1].type == "predicate" && RESP.answers[1].probability.value_or(0) == 0.96);
  CHECK(RESP.answers[2].choice.value_or("") == "payments");
  CHECK(RESP.answers[2].probabilities.size() == 3);
  CHECK(RESP.answers[0].score.value_or(0) == 1.99);
  CHECK(RESP.answers[0].probabilities[2].level == 2);
  CHECK(RESP.answers[0].probabilities[2].label == "Now");
  CHECK(RESP.answers[0].probabilities[2].probability == 0.99);
  CHECK(RESP.usage.input_tokens == 476 && RESP.usage.cost.value_or(0) == 0.000019992);
}

static void testErrors()
{
  auto e = detail::parseApiError(
    400, R"({"error":{"message":"bad","type":"invalid_request_error","code":"x"}})");
  CHECK(e.http_status == 400 && e.message == "bad" && e.type == "invalid_request_error" &&
        e.code == "x");
  e = detail::parseApiError(404, R"({"error":"model not found"})");
  CHECK(e.message == "model not found");
  e = detail::parseApiError(502, "Bad Gateway");
  CHECK(e.message == "Bad Gateway");
}

static void testSse()
{
  detail::SseParser parser;
  std::vector<std::string> events;
  const detail::SseParser::DataCallback ON = [&](std::string_view d) {
    events.emplace_back(d);
    return true;
  };

  // Split across chunk boundaries, CRLF, comments, event: lines and multi-line data.
  CHECK(parser.feed(": keepalive\n\nevent: x\ndata: {\"a\"", ON));
  CHECK(parser.feed(":1}\r\n\r\ndata: line1\ndata: line2\n\nda", ON));
  CHECK(parser.feed("ta: [DONE]\n\n", ON));
  CHECK(events.size() == 3);
  CHECK(events[0] == R"({"a":1})");
  CHECK(events[1] == "line1\nline2");
  CHECK(events[2] == "[DONE]");

  // Trailing event without blank line is flushed by finish().
  CHECK(parser.feed("data: tail", ON));
  CHECK(events.size() == 3);
  CHECK(parser.finish(ON));
  CHECK(events.size() == 4 && events[3] == "tail");

  // Returning false stops dispatch.
  detail::SseParser stopper;
  int seen = 0;
  CHECK(!stopper.feed("data: 1\n\ndata: 2\n\n", [&](std::string_view) { return ++seen < 1; }));
  CHECK(seen == 1);
}

int main()
{
  testChatRequest();
  testMessages();
  testChatParsing();
  testResponses();
  testDecisions();
  testDecisionsOpenRouter();
  testErrors();
  testSse();
  fmt::print("oai_api_test: all checks passed\n");
  return 0;
}
