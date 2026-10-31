# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`oai-api` is a C++23 library for the OpenAI REST API and OpenAI-compatible servers (Ollama, vLLM, OpenRouter). Typed endpoints: chat completions (with SSE streaming), legacy completions, embeddings, models, Responses (with streaming), images, audio (transcriptions/translations/speech) and Decisions; everything else via a raw `request()` escape hatch. It also provides `std::future` async calls, multimodal message construction, tool calling, and a ReAct-style agent loop.

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

Dependencies required on the system: `fmt`, `nlohmann_json`, `OpenCV`, `curl`, `zlib` (`cpr` is fetched by CMake).

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
  └─ ReAct loop: client.chat() → dispatch tools → repeat until "stop"
  └─ owns conversation history (vector<Message>); history persists across run() calls

OpenAIClient   (client.hpp / client.cpp)
  └─ one method per endpoint, all const + synchronous; async(&OpenAIClient::x, args...) → std::future
  └─ chatStream / createResponseStream: SSE via cpr WriteCallback; callback returns false to cancel
  └─ request(Method, path, json): untyped access to any other endpoint
  └─ private transport helpers (auth, headers, timeouts, ApiError mapping) live in client.cpp

Request/response types (chat.hpp, responses.hpp, media.hpp, decisions.hpp)
  └─ aggregates: unset optionals / null json are omitted from the request body
  └─ every request has extra_body; every response keeps the parsed `raw` json

Serialization  (src/serialize.hpp / serialize.cpp — internal)
  └─ request→json and json→response per endpoint, Message → chat or Responses wire format
  └─ SseParser (pure, unit-tested)

ToolRegistry   (tool.hpp)      — FunctionTool objects keyed by name; builds tools[]; dispatches
Types          (types.hpp)     — Message, ContentBlock (Text | Image | Json), ToolCall, Usage, Model, ApiError, image helpers
```

### Key design points

- All calls are **synchronous**; `async()` copies the client and arguments into a `std::async` task (pass owning types, not `string_view`).
- Client-level `model`, `temperature`, `max_tokens` are *defaults* applied only when the request leaves them unset.
- `extra_body` is merge-patched into the JSON body: typed fields → client `extraBody()` → request `extra_body` (later wins, `null` deletes). Used for provider extensions such as OpenRouter `provider` routing or Ollama `options`. Extra headers go through `ClientBuilder::header()`.
- `Agent::run()` preserves multi-turn history; call `reset()` to start a fresh conversation.
- Tool handlers satisfy the `ToolCallable` concept: `(std::string_view args_json) -> std::expected<std::string, std::string>`. The agent surfaces tool errors as `[tool error: ...]` text rather than aborting.
- `ImageBlock` holds either `(base64_data + mime_type)` or `url` — exactly one must be populated.
- The library uses `std::expected` throughout for error propagation; no exceptions are thrown across the public API boundary.

## Tests

`test/oai_api_test.cpp` covers serialization, extra_body merging, response parsing, and SSE parsing offline (no network). Run it with `ctest --test-dir build` (plain CMake) or `colcon test --packages-select oai_api`.
