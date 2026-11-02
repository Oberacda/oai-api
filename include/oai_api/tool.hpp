// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the copyright holder nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <expected>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "oai_api/types.hpp"

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
  template <ToolCallable TF> static FunctionTool make(ToolDefinition def, TF&& fn)
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
