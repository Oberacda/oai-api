# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`oai-api` is a C++23 header+source library that wraps the OpenAI `/chat/completions` API (and any OpenAI-compatible endpoint such as Ollama). It provides synchronous HTTP calls, multimodal message construction (text + base64 images), tool/function calling, and a ReAct-style agent loop.

## Building

This package lives inside a ROS 2 / colcon workspace at `../../..` relative to this directory. The preferred build path is colcon from the workspace root:

```bash
cd ~/repos/semantic_perception_ws
colcon build --packages-select oai_api
```

Plain CMake (standalone, no ROS tooling):

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j$(nproc)
```

Dependencies required on the system: `cpr`, `fmt`, `nlohmann_json`, `OpenCV`.

## Running the example

After a colcon build, source the workspace and run the bundled echo agent:

```bash
source install/setup.bash
./build/oai_api/echo_agent
```

The example targets a local Ollama instance at `http://localhost:11434/v1` with model `llama3.2`. Edit `examples/echo_agent.cpp` to point at a different endpoint or model.

## Code style

- **Formatter**: `clang-format` with the project `.clang-format` (2-space indent, 100-column limit, Allman-style braces). Run `clang-format -i` before committing.
- **Linter**: `clang-tidy` with the project `.clang-tidy`. Run it against the compile-commands database produced by CMake.
- **Naming conventions** (enforced by clang-tidy):
  - Private/protected members: `m_` prefix, `lower_case`
  - Global variables: `g_` prefix
  - Classes/structs/enums: `CamelCase`
  - Methods and free functions: `camelBack`
  - Constants and `constexpr` variables: `UPPER_CASE`
  - Template type parameters: `T` prefix, `CamelCase`

## Architecture

All public API lives under `include/oai_api/`; implementations are in `src/`. Include everything via the umbrella header `<oai_api/oai_api.hpp>`.

### Layer overview

```
Agent          (agent.hpp / agent.cpp)
  └─ runs the ReAct loop: call LLM → dispatch tools → repeat until "stop"
  └─ owns conversation history (vector<Message>); history persists across run() calls

OpenAIClient   (client.hpp / client.cpp)
  └─ thin synchronous HTTP wrapper around /chat/completions
  └─ uses cpr for HTTP, nlohmann/json for serialization
  └─ handles both text and multimodal (image_url) content blocks

ToolRegistry   (tool.hpp)
  └─ owns FunctionTool objects keyed by name
  └─ builds the tools[] array for the API request
  └─ dispatches invocations by function name

Types          (types.hpp)
  └─ Message, ContentBlock (TextBlock | ImageBlock), ToolCall, CompletionResponse, ApiError
  └─ encodeImage(): cv::Mat → base64 PNG ImageBlock (inline, uses OpenCV imencode)
  └─ imageFromUrl(): wraps an HTTPS URL into an ImageBlock without downloading
```

### Key design points

- `OpenAIClient::complete()` is **synchronous**; it blocks until the HTTP response arrives. There is no async interface.
- `Agent::run()` preserves multi-turn history; call `reset()` to start a fresh conversation.
- Tool handlers satisfy the `ToolCallable` concept: `(std::string_view args_json) -> std::expected<std::string, std::string>`. The agent surfaces tool errors as `[tool error: ...]` text rather than aborting.
- `ImageBlock` holds either `(base64_data + mime_type)` or `url` — exactly one must be populated. `client.cpp` emits a `data:<mime>;base64,<data>` URL for the former, or a plain `image_url` object for the latter.
- The library uses `std::expected` throughout for error propagation; no exceptions are thrown across the public API boundary.
