#include <chrono>
#include <cstdio>
#include <ctime>
#include <string>
#include <string_view>

#include <fmt/core.h>
#include <opencv2/core.hpp>

#include <oai_api/oai_api.hpp>

static std::string utc_now()
{
  const auto now      = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  char buf[64]{};
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
  fmt::print("--- Task: time query ---\n");
  auto answer = agent.run("What is the current UTC time?");
  if (answer)
  {
    fmt::print("Answer: {}\n", *answer);
  }
  else
  {
    fmt::print("Error ({}): {}\n", answer.error().http_status, answer.error().message);
    return 1;
  }

  // ------------------------------------------------------------------
  // 3. Vision example — multimodal message via MessageBuilder
  //    (requires a vision-capable model such as llama3.2-vision)
  // ------------------------------------------------------------------
  fmt::print("\n--- Task: vision query ---\n");

  cv::Mat test_img(100, 100, CV_8UC3, cv::Scalar(0, 200, 0));

  auto vision_msg = oai::MessageBuilder::user()
                      .text("What colour is the square in this image?")
                      .image(test_img)
                      .build();
  if (!vision_msg)
  {
    fmt::print("Image encode failed: {}\n", vision_msg.error());
    return 1;
  }

  auto vis_client =
    oai::ClientBuilder{}.baseUrl("http://localhost:11434/v1").model("llama3.2-vision").build();

  auto vis_resp = vis_client.chat(
    {.messages = {
       oai::MessageBuilder::system().text("You are a vision assistant.").build().value(),
       std::move(*vision_msg),
     }});

  if (vis_resp)
  {
    fmt::print("Vision answer: {}\n", vis_resp->text());
  }
  else
  {
    fmt::print("Vision error ({}): {}\n", vis_resp.error().http_status, vis_resp.error().message);
  }

  // ------------------------------------------------------------------
  // 4. Streaming chat — tokens are printed as they arrive
  // ------------------------------------------------------------------
  fmt::print("\n--- Task: streaming ---\n");
  auto client = oai::ClientBuilder{}.baseUrl("http://localhost:11434/v1").model("llama3.2").build();

  const oai::ChatRequest HAIKU{
    .messages = {oai::MessageBuilder::user().text("Write a haiku about robots.").build().value()}};
  auto streamed = client.chatStream(HAIKU, [](const oai::ChatChunk& chunk) {
    fmt::print("{}", chunk.content);
    std::fflush(stdout);
    return true; // false cancels
  });
  fmt::print("\n");
  if (!streamed)
  {
    fmt::print("Stream error: {}\n", streamed.error().message);
  }

  // ------------------------------------------------------------------
  // 5. Async — run requests concurrently and wait on std::future
  // ------------------------------------------------------------------
  fmt::print("\n--- Task: async ---\n");
  auto models_future = client.async(&oai::OpenAIClient::listModels);
  auto embed_future  = client.async(&oai::OpenAIClient::embeddings,
                                   oai::EmbeddingRequest{.input = {"robot", "forest"}});

  if (auto models = models_future.get())
  {
    fmt::print("Server lists {} model(s)\n", models->size());
  }
  if (auto embed = embed_future.get())
  {
    fmt::print("Got {} embedding(s) of dim {}\n",
               embed->data.size(),
               embed->data.empty() ? 0 : embed->data.front().size());
  }
  else
  {
    fmt::print("Embeddings error: {}\n", embed.error().message);
  }

  return 0;
}
