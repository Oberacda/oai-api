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
colcon build --packages-select oai_api
```

**Standalone CMake:**

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j$(nproc)
```

## Quick start

Include the single umbrella header:

```cpp
#include <oai_api/oai_api.hpp>
```

### Plain chat completion

```cpp
auto client = oai::ClientBuilder{}
    .baseUrl("http://localhost:11434/v1")  // Ollama; use "https://api.openai.com/v1" for OpenAI
    .model("llama3.2")
    .temperature(0.7f)
    .maxTokens(1024)
    .build();

auto result = client.complete({oai::MessageBuilder::user().text("What is 2 + 2?").build().value()});
if (result) {
    std::println("{}", result->choice.content.value_or(""));
} else {
    std::println("Error {}: {}", result.error().http_status, result.error().message);
}
```

### Tool / function calling

```cpp
auto agent = oai::AgentBuilder{}
    .systemPrompt("You are a helpful assistant.")
    .maxSteps(5)
    .client(oai::ClientBuilder{}
        .baseUrl("http://localhost:11434/v1")
        .model("llama3.2")
        .build())
    .tool("get_weather",
          "Returns the current temperature for a city.",
          R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})",
          [](std::string_view args) -> std::expected<std::string, std::string> {
              // parse args, call your weather API, return result string
              return "22°C, sunny";
          })
    .build();

auto answer = agent.run("What's the weather like in Berlin?");
if (answer) std::println("{}", *answer);
```

The agent automatically runs the ReAct loop: it calls the model, dispatches any requested tools, feeds results back, and repeats until the model produces a final answer or `max_steps` is reached.

### Vision (multimodal)

Use `MessageBuilder` to construct multimodal messages with `cv::Mat` images:

```cpp
cv::Mat frame = cv::imread("scene.png");

auto msg = oai::MessageBuilder::user()
    .text("Describe what you see.")
    .image(frame)
    .build();
if (!msg) { /* handle encoding error */ }

auto response = client.complete({
    oai::MessageBuilder::system().text("You are a vision assistant.").build().value(),
    std::move(*msg),
});
```

Or use a URL directly, without downloading the image:

```cpp
auto msg = oai::MessageBuilder::user()
    .text("What's in this image?")
    .image(oai::imageFromUrl("https://example.com/photo.jpg"))
    .build();
```

## API reference

### `ClientBuilder`

| Method | Default | Description |
|---|---|---|
| `baseUrl(url)` | — | API base URL, e.g. `"http://localhost:11434/v1"` |
| `apiKey(key)` | `""` | Bearer token; omit for local endpoints |
| `model(name)` | — | Model name, e.g. `"gpt-4o"`, `"llama3.2-vision"` |
| `temperature(t)` | `0.7` | Sampling temperature |
| `maxTokens(n)` | `4096` | Maximum completion tokens |

### `AgentBuilder`

| Method | Default | Description |
|---|---|---|
| `systemPrompt(prompt)` | `""` | Injected as the first system message |
| `maxSteps(steps)` | `10` | Maximum tool-call rounds before returning an error |
| `client(c)` | — | Provide the `OpenAIClient` (required) |
| `tool(name, desc, schema, handler)` | — | Register a tool with inline handler |
| `tools(registry)` | — | Replace all tools with a pre-built `ToolRegistry` |

### `MessageBuilder`

| Method | Description |
|---|---|
| `user()`, `system()`, `assistant()` | Static factory for the role |
| `text(value)` | Append a text content block |
| `image(mat)` | Encode a `cv::Mat` as base64 PNG |
| `image(block)` | Append a pre-built `ImageBlock` (URL or encoded) |
| `build()` | Returns `std::expected<Message, std::string>` |

### Error handling

All fallible functions return `std::expected<T, E>` — no exceptions cross the API boundary. Check with `if (result)` and access errors via `.error()`.

## License

BSD-3-Clause
