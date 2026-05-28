#pragma once

#include <expected>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "oai-api/types.hpp"

namespace oai {

// ---------------------------------------------------------------------------
// Concept: ToolCallable
//
// Any callable that accepts a JSON arguments string and returns either a
// result string or an error string.
// ---------------------------------------------------------------------------

template <typename T>
concept ToolCallable = requires(T t, std::string_view args) {
  { t(args) } -> std::same_as<std::expected<std::string, std::string>>;
};

// ---------------------------------------------------------------------------
// FunctionTool
//
// Type-erased wrapper around a ToolCallable with its OpenAI tool definition.
// ---------------------------------------------------------------------------

struct FunctionTool
{
  ToolDefinition definition;
  std::function<std::expected<std::string, std::string>(std::string_view)> handler;

  // Convenience constructor from any ToolCallable lambda/function.
  template <ToolCallable TF>
  static FunctionTool make(ToolDefinition def, TF&& fn)
  {
    return FunctionTool{std::move(def), std::forward<TF>(fn)};
  }
};

// ---------------------------------------------------------------------------
// ToolRegistry
//
// Owns a set of FunctionTools; dispatches invocations by name.
// ---------------------------------------------------------------------------

class ToolRegistry
{
public:
  // Register a tool. Overwrites any existing tool with the same name.
  void add(FunctionTool tool)
  {
    std::string name = tool.definition.name;
    m_tools.emplace(std::move(name), std::move(tool));
  }

  // Build the list of ToolDefinitions to pass to the API.
  [[nodiscard]] std::vector<ToolDefinition> definitions() const
  {
    std::vector<ToolDefinition> defs;
    defs.reserve(m_tools.size());
    for (const auto& [_, tool] : m_tools)
    {
      defs.push_back(tool.definition);
    }
    return defs;
  }

  // Invoke a tool by name with the raw JSON arguments string the API returned.
  [[nodiscard]] std::expected<std::string, std::string> invoke(std::string_view name,
                                                               std::string_view args_json) const
  {
    if (auto it = m_tools.find(std::string{name}); it != m_tools.end())
    {
      return it->second.handler(args_json);
    }
    return std::unexpected{std::string{"unknown tool: "} + std::string{name}};
  }

  [[nodiscard]] bool empty() const { return m_tools.empty(); }
  [[nodiscard]] std::size_t size() const { return m_tools.size(); }

private:
  std::unordered_map<std::string, FunctionTool> m_tools;
};

} // namespace oai
