// Manual test for the Decisions API.
//
//   OAI_API_KEY=sk-or-... ./decisions_test                       # OpenRouter (default)
//   OAI_API_KEY=sk-... OAI_BASE_URL=https://api.openai.com/v1 OAI_MODEL=gpt-6-luna \
//     ./decisions_test                                           # OpenAI directly
//
// Env vars: OAI_API_KEY (required), OAI_BASE_URL (default https://openrouter.ai/api/alpha),
//           OAI_MODEL (default openai/gpt-6-luna-decisions).
// The OpenRouter wire format is used when the base URL contains "openrouter", else OpenAI's.

#include <cstdlib>
#include <string>

#include <fmt/core.h>

#include <opencv2/imgproc.hpp>

#include <oai_api/oai_api.hpp>

static std::string envOr(const char* name, std::string fallback)
{
  const char* value = std::getenv(name);
  return (value != nullptr && *value != '\0') ? std::string{value} : std::move(fallback);
}

int main()
{
  const std::string API_KEY  = envOr("OAI_API_KEY", "");
  const std::string BASE_URL = envOr("OAI_BASE_URL", "https://openrouter.ai/api/alpha");
  const std::string MODEL    = envOr("OAI_MODEL", "openai/gpt-6-luna-decisions");
  const bool OPENROUTER      = BASE_URL.find("openrouter") != std::string::npos;
  if (API_KEY.empty())
  {
    fmt::print(stderr, "Set OAI_API_KEY (see the comment at the top of decisions_test.cpp).\n");
    return 2;
  }

  auto client = oai::ClientBuilder{}
                  .baseUrl(BASE_URL)
                  .apiKey(API_KEY)
                  .model(MODEL)
                  .header("X-Title", "oai_api decisions_test") // shown in OpenRouter logs
                  .requestTimeout(std::chrono::seconds{60})
                  .build();

  // Shared evidence: text, structured JSON, and a generated image (a red circle on white). The
  // "shape_color" question can only be answered correctly if the model actually sees the image.
  // 512x512: Jev needs >= 256 px per side and >= 78,400 px total; Luna accepts smaller.
  cv::Mat photo(512, 512, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::circle(photo, {256, 256}, 160, cv::Scalar(0, 0, 255), cv::FILLED); // BGR: red

  auto evidence = oai::MessageBuilder::user()
                    .text("Support ticket: 'I was charged twice for my order and the box arrived "
                          "crushed. I need this fixed before Friday.'")
                    .json({{"order", {{"id", 1042}, {"charges", {49.0, 49.0}}}}})
                    .text("Photo attached by the customer:")
                    .image(photo)
                    .build();
  if (!evidence)
  {
    fmt::print(stderr, "image encoding failed: {}\n", evidence.error());
    return 1;
  }

  oai::DecisionRequest req{
    .format = OPENROUTER ? oai::DecisionFormat::OPENROUTER : oai::DecisionFormat::OPENAI,
    .model  = MODEL,
    .input  = std::vector<oai::Message>{std::move(*evidence)},
    .questions =
      {
        oai::PredicateQuestion{.name           = "duplicate_charge",
                               .instructions   = "Does the order show a duplicate charge?",
                               .true_criteria  = "The same amount was charged more than once.",
                               .false_criteria = "Each amount was charged once."},
        oai::ChoiceQuestion{.name         = "department",
                            .instructions = "Which team should handle the ticket first?",
                            .choices      = {{"billing", "Payments, refunds, charges"},
                                             {"shipping", "Delivery and damaged goods"},
                                             {"technical", "Bugs and app problems"}}},
        oai::ScoreQuestion{.name         = "urgency",
                           .instructions = "How urgent is this ticket?",
                           .levels       = {{"Low", "Can wait a week"},
                                            {"Medium", "Should be handled in a few days"},
                                            {"High", "Needs action within a day or two"}}},
        oai::ChoiceQuestion{.name         = "shape_color",
                            .instructions = "What color is the shape in the attached photo?",
                            .choices      = {{"red", ""}, {"green", ""}, {"blue", ""}}},
      },
  };

  fmt::print("POST {}/decisions  model={}  format={}\n\n",
             BASE_URL,
             MODEL,
             OPENROUTER ? "openrouter" : "openai");
  auto resp = client.decide(req);
  if (!resp)
  {
    fmt::print(
      stderr, "decide() failed: HTTP {} {}\n", resp.error().http_status, resp.error().message);
    return 1;
  }

  for (const auto& a : resp->answers)
  {
    fmt::print("[{}] {}\n", a.type, a.name);
    if (a.probability)
    {
      fmt::print("  probability: {:.3f}\n", *a.probability);
    }
    if (a.choice)
    {
      fmt::print("  choice: {}\n", *a.choice);
    }
    if (a.score)
    {
      fmt::print("  score: {:.2f}\n", *a.score);
    }
    if (a.confidence)
    {
      fmt::print("  confidence: {:.3f}\n", *a.confidence);
    }
    for (const auto& p : a.probabilities)
    {
      fmt::print("    {:<40} {:.3f}\n", p.level >= 0 ? p.label : p.value, p.probability);
    }
  }
  fmt::print("\nprovider={} input_tokens={} output_tokens={}",
             resp->provider.empty() ? "-" : resp->provider,
             resp->usage.input_tokens,
             resp->usage.output_tokens);
  if (resp->usage.cost)
  {
    fmt::print(" cost={:.8f}", *resp->usage.cost);
  }
  fmt::print("\n");
  return 0;
}
