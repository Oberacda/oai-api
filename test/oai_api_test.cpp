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

// Offline checks for request serialization, extra_body merging, response parsing and SSE parsing.

#include <cstdlib>
#include <stdexcept>
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

static void testDecisionsVllm()
{
  DecisionRequest req{.format = DecisionFormat::VLLM, .model = "vllm-sr/d3-nano"};
  req.input =
    std::vector<Message>{Message{Role::USER,
                                 {TextBlock{"What does the image show?"},
                                  ImageBlock{.base64_data = "PNG-A", .mime_type = "image/png"},
                                  imageFromUrl("https://example.com/b.jpg")},
                                 {},
                                 {},
                                 {}}};
  req.questions = {
    ChoiceQuestion{.name = "kind",
                   .instructions =
                     std::vector<ContentBlock>{TextBlock{"What kind?"}, JsonBlock{{{"k", 1}}}},
                   .choices = {{"chart", "A chart"}, {"photo", ""}}},
  };
  const auto J = detail::decisionRequestToJson(req, makeConfig());
  CHECK(J["state"] == json::array({"What does the image show?"}));
  CHECK(J["images"] == json::array({"data:image/png;base64,PNG-A", "https://example.com/b.jpg"}));
  CHECK(J["questions"]["kind"]["instructions"] == "What kind?\n{\"k\":1}");
  CHECK(J["questions"]["kind"]["criteria"]["photo"].is_null());

  // Response shape from vLLM docs (docs/models/structured_decisions_models.md).
  const auto RESP = detail::parseDecisionResponse(
    json::parse(R"({"id":"decision-1","object":"structured_decision","model":"vllm-sr/d3-nano",
      "answers":{"urgent":{"type":"noul","noul":0.02,"probabilities":{"yes":0.02,"no":0.98},
                           "confidence":0.95},
                 "kind":{"type":"choice","choice":"chart","probabilities":{"chart":0.9,"photo":0.1},
                         "confidence":0.85}},
      "usage":{"input_tokens":142,"output_tokens":3}})"),
    {"urgent", "kind"});
  CHECK(RESP.answers.size() == 2);
  CHECK(RESP.answers[0].type == "predicate" && RESP.answers[0].probability == 0.02);
  CHECK(RESP.answers[1].choice == "chart" && RESP.answers[1].confidence == 0.85);
}

static void testDecisionsOllama()
{
  DecisionRequest req{.format = DecisionFormat::OLLAMA, .model = "nimble"};
  req.input =
    std::vector<Message>{Message{Role::USER,
                                 {TextBlock{"Ticket text."},
                                  JsonBlock{{{"priority", "high"}}},
                                  ImageBlock{.base64_data = "PNG-A", .mime_type = "image/png"},
                                  ImageBlock{.base64_data = "JPEG-B", .mime_type = "image/jpeg"}},
                                 {},
                                 {},
                                 {}}};
  req.questions = {
    PredicateQuestion{.name           = "urgent",
                      .instructions   = "Needs attention today?",
                      .true_criteria  = "The ticket is urgent.",
                      .false_criteria = "The ticket can wait."},
    ChoiceQuestion{.name         = "team",
                   .instructions = std::vector<ContentBlock>{TextBlock{"Choose a team."},
                                                             TextBlock{"Use the ticket context."}},
                   .choices      = {{"billing", "Payments"}, {"support", ""}}},
    ScoreQuestion{.name         = "severity",
                  .instructions = std::vector<ContentBlock>{TextBlock{"How severe is it?"},
                                                            JsonBlock{{{"field", "severity"}}}},
                  .levels       = {{"Low", "Minor impact"}, {"High", "Major impact"}}},
  };
  req.prompt_cache_key = "must-be-omitted";
  req.extra_body       = {{"keep_alive", "5m"}, {"route", nullptr}};

  const auto J = detail::decisionRequestToJson(req, makeConfig());
  CHECK(J["model"] == "nimble");
  CHECK(J["state"].is_array());
  CHECK(J["state"].size() == 2);
  CHECK(J["state"][0] == "Ticket text.");
  CHECK(J["state"][1]["priority"] == "high");
  CHECK(J["images"] == json::array({"PNG-A", "JPEG-B"}));
  CHECK(!J.contains("input") && !J.contains("prompt_cache_key"));
  CHECK(J["questions"].is_object());
  CHECK(J["questions"]["urgent"]["type"] == "noul");
  CHECK(J["questions"]["urgent"]["criteria"]["true"] == "The ticket is urgent.");
  CHECK(J["questions"]["team"]["instructions"].is_array());
  CHECK(J["questions"]["team"]["instructions"][0] == "Choose a team.");
  CHECK(J["questions"]["team"]["instructions"][1] == "Use the ticket context.");
  CHECK(J["questions"]["team"]["criteria"]["billing"] == "Payments");
  CHECK(J["questions"]["team"]["criteria"]["support"].is_null());
  CHECK(J["questions"]["severity"]["instructions"][0] == "How severe is it?");
  CHECK(J["questions"]["severity"]["instructions"][1]["field"] == "severity");
  CHECK(J["questions"]["severity"]["criteria"][0] == "Low: Minor impact");
  CHECK(J["provider"]["order"][0] == "anthropic");
  CHECK(J["keep_alive"] == "5m");
  CHECK(!J.contains("route"));

  DecisionRequest string_state_req{.format = DecisionFormat::OLLAMA, .model = "nimble"};
  string_state_req.input     = "A plain text state";
  string_state_req.questions = {PredicateQuestion{.name = "urgent", .instructions = "Urgent?"}};
  const auto STRING_STATE    = detail::decisionRequestToJson(string_state_req, makeConfig());
  CHECK(STRING_STATE["state"] == "A plain text state");
  CHECK(!STRING_STATE.contains("images"));

  bool threw = false;
  try
  {
    DecisionRequest image_url_req{.format = DecisionFormat::OLLAMA};
    image_url_req.input = std::vector<Message>{
      Message{Role::USER, {imageFromUrl("https://example.com/a.png")}, {}, {}, {}}};
    (void)detail::decisionRequestToJson(image_url_req, makeConfig());
  }
  catch (const std::invalid_argument&)
  {
    threw = true;
  }
  CHECK(threw);

  DecisionRequest public_url_req{.format = DecisionFormat::OLLAMA};
  public_url_req.input = std::vector<Message>{
    Message{Role::USER, {imageFromUrl("https://example.com/a.png")}, {}, {}, {}}};
  const auto PUBLIC_RESULT =
    ClientBuilder{}.baseUrl("http://127.0.0.1:1/v1").build().decide(public_url_req);
  CHECK(!PUBLIC_RESULT);
  CHECK(PUBLIC_RESULT.error().http_status == 0);
  CHECK(PUBLIC_RESULT.error().type == "invalid_request_error");
  CHECK(PUBLIC_RESULT.error().message ==
        "Ollama decision images require raw base64 data, not URLs");

  threw = false;
  try
  {
    DecisionRequest data_url_req{.format = DecisionFormat::OLLAMA};
    data_url_req.input = std::vector<Message>{
      Message{Role::USER, {imageFromUrl("data:image/png;base64,AAAA")}, {}, {}, {}}};
    (void)detail::decisionRequestToJson(data_url_req, makeConfig());
  }
  catch (const std::invalid_argument&)
  {
    threw = true;
  }
  CHECK(threw);

  threw = false;
  try
  {
    DecisionRequest bad_instructions{.format = DecisionFormat::OLLAMA};
    bad_instructions.questions = {PredicateQuestion{
      .name = "bad",
      .instructions =
        std::vector<ContentBlock>{ImageBlock{.base64_data = "PNG", .mime_type = "image/png"}}}};
    (void)detail::decisionRequestToJson(bad_instructions, makeConfig());
  }
  catch (const std::invalid_argument&)
  {
    threw = true;
  }
  CHECK(threw);

  const auto RESP = detail::parseDecisionResponse(json::parse(R"({
    "model":"nimble",
    "answers":{
      "severity":{"type":"score","score":0.75,"confidence":0.8,
                   "probabilities":{"0":0.25,"1":0.75},
                   "legend":{"0":"Low","1":"High"}},
      "urgent":{"type":"noul","noul":0.91},
      "team":{"type":"choice","choice":"billing",
               "probabilities":{"billing":0.8,"support":0.2},"confidence":0.7}},
    "usage":{"input_tokens":17,"output_tokens":3}})"),
                                                  {"team", "urgent", "severity"});
  CHECK(RESP.model == "nimble");
  CHECK(RESP.answers.size() == 3);
  CHECK(RESP.answers[0].name == "team");
  CHECK(RESP.answers[1].type == "predicate" && RESP.answers[1].probability.value_or(0) == 0.91);
  CHECK(RESP.answers[2].score.value_or(0) == 0.75);
  CHECK(RESP.answers[2].probabilities[1].label == "High");
  CHECK(RESP.usage.input_tokens == 17 && RESP.usage.output_tokens == 3);
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
  testDecisionsOllama();
  testDecisionsVllm();
  testErrors();
  testSse();
  fmt::print("oai_api_test: all checks passed\n");
  return 0;
}
