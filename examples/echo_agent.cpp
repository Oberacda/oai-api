#include <chrono>
#include <ctime>
#include <print>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include <oai-api/oai_api.hpp>

// ---------------------------------------------------------------------------
// Helper: current UTC time as a string
// ---------------------------------------------------------------------------

static std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[64]{};
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return buf;
}

int main() {
    // ------------------------------------------------------------------
    // 1. Build the client
    // ------------------------------------------------------------------
    oai::ClientConfig cfg;
    cfg.base_url    = "http://localhost:11434/v1"; // Ollama default endpoint
    cfg.api_key     = "";                          // not required for local Ollama
    cfg.model       = "llama3.2";                  // change to any installed model
    cfg.temperature = 0.2f;
    cfg.max_tokens  = 512;

    oai::OpenAIClient client{cfg};

    // ------------------------------------------------------------------
    // 2. Define a tool: get_current_time
    // ------------------------------------------------------------------
    oai::ToolRegistry tools;
    tools.add(oai::FunctionTool{
        .definition = {
            .name        = "get_current_time",
            .description = "Returns the current UTC date and time.",
            .parameters_schema_json = R"({"type":"object","properties":{},"required":[]})"
        },
        .handler = [](std::string_view /*args*/) -> std::expected<std::string, std::string> {
            return utc_now();
        }
    });

    // ------------------------------------------------------------------
    // 3. Run the agent on a simple task
    // ------------------------------------------------------------------
    oai::Agent agent{
        oai::AgentConfig{.system_prompt = "You are a helpful assistant.", .max_steps = 5},
        std::move(client),
        std::move(tools)
    };

    std::println("--- Task: time query ---");
    auto answer = agent.run("What is the current UTC time?");
    if (answer) {
        std::println("Answer: {}", *answer);
    } else {
        std::println("Error ({}): {}", answer.error().http_status, answer.error().message);
        return 1;
    }

    // ------------------------------------------------------------------
    // 4. Vision example: load an image and ask about it
    //    (requires a vision-capable model such as llava or llama3.2-vision)
    // ------------------------------------------------------------------
    std::println("\n--- Task: vision query ---");

    // Create a small synthetic test image (100x100 green square).
    cv::Mat test_img(100, 100, CV_8UC3, cv::Scalar(0, 200, 0));

    auto img_block = oai::encode_image(test_img);
    if (!img_block) {
        std::println("encode_image failed: {}", img_block.error());
        return 1;
    }

    oai::Agent vision_agent{
        oai::AgentConfig{.system_prompt = "You are a vision assistant.", .max_steps = 3},
        oai::OpenAIClient{oai::ClientConfig{
            .base_url = "http://localhost:11434/v1",
            .model    = "llama3.2-vision", // change to a vision model
        }}
    };

    std::vector<oai::ContentBlock> vision_content;
    vision_content.push_back(oai::TextBlock{"What colour is the square in this image?"});
    vision_content.push_back(*img_block);

    auto vision_answer = vision_agent.run("Describe the image.");

    // The vision message needs to be sent directly via a user message with image.
    // We reset and push the multimodal message manually for demonstration.
    vision_agent.reset();
    oai::OpenAIClient vis_client{oai::ClientConfig{
        .base_url = "http://localhost:11434/v1",
        .model    = "llama3.2-vision",
    }};
    auto vis_resp = vis_client.complete({
        oai::system_message("You are a vision assistant."),
        oai::user_message(std::move(vision_content))
    });

    if (vis_resp) {
        std::println("Vision answer: {}", vis_resp->choice.content.value_or("(no content)"));
    } else {
        std::println("Vision error ({}): {}", vis_resp.error().http_status, vis_resp.error().message);
    }

    return 0;
}
