# oai-api

A C++23 library for talking to OpenAI-compatible LLM APIs — including local models via [Ollama](https://ollama.com). Supports text, vision, tool/function calling, and a built-in ReAct agent loop.

## Dependencies

| Library | Purpose |
|---|---|
| [cpr](https://github.com/libcpr/cpr) | HTTP client |
| [fmt](https://github.com/fmtlib/fmt) | String formatting |
| [nlohmann/json](https://github.com/nlohmann/json) | JSON serialization |
| [OpenCV](https://opencv.org) | Image encoding for vision requests |

## Building

**Inside the ROS 2 / colcon workspace (recommended):**

```bash
colcon build --packages-select oai-api
```

**Standalone CMake:**

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j$(nproc)
```

## Quick start

Include the single umbrella header:

```cpp
#include <oai-api/oai_api.hpp>
```

### Plain chat completion

```cpp
oai::OpenAIClient client{oai::ClientConfig{
    .base_url    = "http://localhost:11434/v1", // Ollama; use "https://api.openai.com/v1" for OpenAI
    .api_key     = "",                          // leave empty for local Ollama
    .model       = "llama3.2",
    .temperature = 0.7f,
    .max_tokens  = 1024,
}};

auto result = client.complete({oai::userMessage("What is 2 + 2?")});
if (result) {
    std::println("{}", result->choice.content.value_or(""));
} else {
    std::println("Error {}: {}", result.error().http_status, result.error().message);
}
```

### Tool / function calling

```cpp
oai::ToolRegistry tools;
tools.add(oai::FunctionTool{
    .definition = {
        .name                   = "get_weather",
        .description            = "Returns the current temperature for a city.",
        .parameters_schema_json = R"({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        })",
    },
    .handler = [](std::string_view args) -> std::expected<std::string, std::string> {
        // parse args, call your weather API, return result string
        return "22°C, sunny";
    },
});

oai::Agent agent{
    oai::AgentConfig{.system_prompt = "You are a helpful assistant.", .max_steps = 5},
    std::move(client),
    std::move(tools),
};

auto answer = agent.run("What's the weather like in Berlin?");
if (answer) std::println("{}", *answer);
```

The agent automatically runs the ReAct loop: it calls the model, dispatches any requested tools, feeds results back, and repeats until the model produces a final answer or `max_steps` is reached.

### Vision (multimodal)

Pass a `cv::Mat` image alongside text using `encodeImage`:

```cpp
cv::Mat frame = cv::imread("scene.png");

auto img = oai::encodeImage(frame);
if (!img) { /* handle encoding error */ }

auto response = client.complete({
    oai::systemMessage("You are a vision assistant."),
    oai::userMessage({
        oai::TextBlock{"Describe what you see."},
        *img,
    }),
});
```

Or use a URL directly, without downloading the image:

```cpp
oai::ImageBlock img = oai::imageFromUrl("https://example.com/photo.jpg");
```

## API reference

### `ClientConfig`

| Field | Default | Description |
|---|---|---|
| `base_url` | — | API base URL, e.g. `"http://localhost:11434/v1"` |
| `api_key` | `""` | Bearer token; omit for local endpoints |
| `model` | — | Model name, e.g. `"gpt-4o"`, `"llama3.2-vision"` |
| `temperature` | `0.7` | Sampling temperature |
| `max_tokens` | `4096` | Maximum completion tokens |

### `AgentConfig`

| Field | Default | Description |
|---|---|---|
| `system_prompt` | `""` | Injected as the first system message |
| `max_steps` | `10` | Maximum tool-call rounds before returning an error |

### Error handling

All fallible functions return `std::expected<T, E>` — no exceptions cross the API boundary. Check with `if (result)` and access errors via `.error()`.

## License

BSD-3-Clause
