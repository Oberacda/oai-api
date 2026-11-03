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

### vLLM decisions (Decision 3.0)

`vllm serve vllm-sr/d3-nano` (or any `vllm-sr/d3*` model) serves decisions only on
`/v1/systemone`. Use `DecisionFormat::VLLM` with base URL `http://localhost:8000/v1`. The
body matches Ollama's, but images are sent as http(s) or data URLs (URLs work too), and
question instructions are flattened to one string. Try it with `OAI_FORMAT=vllm ./build/decisions_test`.

### Ollama decisions

Use [Ollama's Decision API](https://docs.ollama.com/capabilities/decision) with a System One model:

```cpp
auto client = oai::ClientBuilder{}
    .baseUrl("http://localhost:11434/v1")
    .build();

oai::DecisionRequest request{
    .format = oai::DecisionFormat::OLLAMA,
    .model = "nimble",
    .input = "I was charged twice. Please refund the extra payment.",
    .questions = {
        oai::PredicateQuestion{
            .name = "refund",
            .instructions = "Is the customer requesting a refund?"},
    },
};
auto result = client.decide(request); // POST /v1/systemone
if (result) {
    std::println("Refund probability: {}", result->answers[0].probability.value_or(0));
}
```

Local requests need no API key. Choice and score questions use the existing `ChoiceQuestion`
and `ScoreQuestion` types; yes/no answers are normalized to `predicate`. Use `extra_body`
for Ollama options such as `keep_alive`.

For structured state or images, supply `std::vector<oai::Message>` as `input`. Text and JSON
blocks form the state array; encoded image blocks are sent separately in `images`, in order.
Images are shared by all questions and require a vision model such as `clef-flash`.
Ollama rejects image URLs and images in question instructions. The library reports these as
`ApiError` before sending the request. A base URL of `https://ollama.com/v1` and an API key
can be used for direct cloud requests.

The bundled example exercises text, JSON, images, and all three question types:

```bash
ollama pull clef-flash
OAI_FORMAT=ollama ./build/decisions_test
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
