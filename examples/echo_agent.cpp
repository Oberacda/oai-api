#include <chrono>
#include <ctime>
#include <print>
#include <string>
#include <string_view>

#include <opencv2/core.hpp>

#include <oai-api/oai_api.hpp>

static std::string utc_now()
{
  const auto        now = std::chrono::system_clock::now();
  const std::time_t t   = std::chrono::system_clock::to_time_t(now);
  char              buf[64]{};
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  return buf;
}

int main()
{
  // ------------------------------------------------------------------
  // 1. Build the agent
  // ------------------------------------------------------------------
  auto agent = oai::AgentBuilder{}
                 .systemPrompt("You are a helpful assistant.")
                 .maxSteps(5)
                 .client(oai::ClientBuilder{}
                           .baseUrl("http://localhost:11434/v1")
                           .model("llama3.2")
                           .temperature(0.2f)
                           .maxTokens(512)
                           .build())
                 .tool("get_current_time",
                       "Returns the current UTC date and time.",
                       R"({"type":"object","properties":{},"required":[]})",
                       [](std::string_view /*args*/) -> std::expected<std::string, std::string> {
                         return utc_now();
                       })
                 .build();

  // ------------------------------------------------------------------
  // 2. Run a task
  // ------------------------------------------------------------------
  std::println("--- Task: time query ---");
  auto answer = agent.run("What is the current UTC time?");
  if (answer)
  {
    std::println("Answer: {}", *answer);
  }
  else
  {
    std::println("Error ({}): {}", answer.error().http_status, answer.error().message);
    return 1;
  }

  // ------------------------------------------------------------------
  // 3. Vision example — multimodal message via MessageBuilder
  //    (requires a vision-capable model such as llama3.2-vision)
  // ------------------------------------------------------------------
  std::println("\n--- Task: vision query ---");

  cv::Mat test_img(100, 100, CV_8UC3, cv::Scalar(0, 200, 0));

  auto vision_msg = oai::MessageBuilder::user()
                      .text("What colour is the square in this image?")
                      .image(test_img)
                      .build();
  if (!vision_msg)
  {
    std::println("Image encode failed: {}", vision_msg.error());
    return 1;
  }

  auto vis_client = oai::ClientBuilder{}
                      .baseUrl("http://localhost:11434/v1")
                      .model("llama3.2-vision")
                      .build();

  auto vis_resp = vis_client.complete({
    oai::MessageBuilder::system().text("You are a vision assistant.").build().value(),
    std::move(*vision_msg),
  });

  if (vis_resp)
  {
    std::println("Vision answer: {}", vis_resp->choice.content.value_or("(no content)"));
  }
  else
  {
    std::println("Vision error ({}): {}", vis_resp.error().http_status, vis_resp.error().message);
  }

  return 0;
}
