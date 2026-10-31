#pragma once

// Decisions API: POST /decisions. Evaluates questions against shared input and returns typed
// answers (probability, choice from a fixed set, or score on ordered levels).
//
// Two wire formats share these types:
//   OPENAI      base_url https://api.openai.com/v1, model "gpt-6-luna"
//               https://developers.openai.com/api/docs/guides/decisions
//   OPENROUTER  base_url https://openrouter.ai/api/alpha, model e.g. "openai/gpt-6-luna-decisions"
//               https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request
// Answers are normalized: OpenRouter "noul" answers come back as type "predicate".

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
  OPENROUTER
};

// Probability that the condition is true.
struct PredicateQuestion
{
  std::string name{};
  DecisionInstructions instructions{};
  // OpenRouter only (required there as noul criteria): what counts as true / false.
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
  std::variant<std::string, std::vector<Message>> input{}; // JsonBlock -> input_json (user only)
  std::vector<DecisionQuestion> questions{};               // 1..400
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
