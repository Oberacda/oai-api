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

// Decisions API: POST /decisions (Ollama: /systemone). Evaluates shared input and returns typed
// answers (probability, choice from a fixed set, or score on ordered levels).
//
// Three wire formats share these types:
//   OPENAI      base_url https://api.openai.com/v1, model "gpt-6-luna"
//               https://developers.openai.com/api/docs/guides/decisions
//   OPENROUTER  base_url https://openrouter.ai/api/alpha, model e.g. "openai/gpt-6-luna-decisions"
//               https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request
//   OLLAMA      base_url http://localhost:11434/v1, model e.g. "clef-flash"
//               POST /systemone; https://docs.ollama.com/api/systemone
// Answers are normalized: OpenRouter/Ollama "noul" answers come back as type "predicate".

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "oai_api/types.hpp"

namespace oai {

// A string, or text/image parts specific to the question.
using DecisionInstructions = std::variant<std::string, std::vector<ContentBlock>>;

enum class DecisionFormat : std::uint8_t
{
  OPENAI,
  OPENROUTER,
  OLLAMA
};

// Probability that the condition is true.
struct PredicateQuestion
{
  std::string name{};
  DecisionInstructions instructions{};
  // OpenRouter/Ollama noul criteria: what counts as true / false.
  std::string true_criteria{"Yes"};
  std::string false_criteria{"No"};
};

struct DecisionChoiceOption
{
  std::string value{};
  std::string description{};
};

// One of the supplied values.
struct ChoiceQuestion
{
  std::string name{};
  DecisionInstructions instructions{};
  std::vector<DecisionChoiceOption> choices{};
};

struct DecisionLevel
{
  std::string label{};
  std::string description{};
};

// Probability-weighted level index; levels ordered lowest to highest (2..10 levels).
struct ScoreQuestion
{
  std::string name{};
  DecisionInstructions instructions{};
  std::vector<DecisionLevel> levels{};
};

using DecisionQuestion = std::variant<PredicateQuestion, ChoiceQuestion, ScoreQuestion>;

struct DecisionRequest
{
  DecisionFormat format{DecisionFormat::OPENAI};
  std::string model{"gpt-6-luna"};
  // Ollama: text/JSON blocks become state items; base64 images are shared by all questions.
  // Image URLs and images in question instructions are unsupported by Ollama.
  std::variant<std::string, std::vector<Message>> input{}; // JsonBlock -> input_json (OpenAI)
  std::vector<DecisionQuestion> questions{};               // provider-specific limits
  std::optional<std::string> prompt_cache_key{};           // OpenAI only
  nlohmann::json extra_body{};                             // e.g. OpenRouter {"provider": {...}}
};

struct DecisionProbability
{
  std::string value{}; // choice answers
  int level{-1};       // score answers: level index
  std::string label{}; // score answers
  double probability{};
};

struct DecisionAnswer
{
  std::string type{}; // "predicate" | "choice" | "score" | "refusal"
  std::string name{};
  std::optional<double> probability{}; // predicate
  std::optional<std::string> choice{}; // choice
  std::optional<double> score{};       // score
  std::optional<double> confidence{};  // choice, score
  std::vector<DecisionProbability> probabilities{};
};

struct DecisionResponse
{
  std::string id{};
  std::string model{};
  std::string provider{};                // OpenRouter only
  std::vector<DecisionAnswer> answers{}; // same order as the questions
  Usage usage{};
  nlohmann::json raw{};
};

} // namespace oai
