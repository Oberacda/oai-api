#pragma once

// Audio (transcriptions, translations, speech) and image generation.

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace oai {

// ---------------------------------------------------------------------------
// POST /audio/transcriptions, POST /audio/translations (multipart upload)
//
// Provide either file_path or audio (raw bytes). With raw bytes, filename's extension tells the
// server the format. Top-level scalar extra_body entries are sent as additional form fields.
// ---------------------------------------------------------------------------

struct TranscriptionRequest
{
  std::string model{}; // e.g. "whisper-1", "gpt-4o-transcribe"
  std::string file_path{};
  std::string audio{};
  std::string filename{"audio.wav"};
  std::optional<std::string> language{}; // transcriptions only
  std::optional<std::string> prompt{};
  std::optional<std::string> response_format{}; // "json" | "text" | "srt" | "verbose_json" | "vtt"
  std::optional<float> temperature{};
  std::vector<std::string> timestamp_granularities{}; // "word" | "segment" (verbose_json)
  nlohmann::json extra_body{};
};

struct Transcription
{
  std::string text{}; // for text/srt/vtt formats: the raw body
  nlohmann::json raw{};
};

// ---------------------------------------------------------------------------
// POST /audio/speech -> raw audio bytes
// ---------------------------------------------------------------------------

struct SpeechRequest
{
  std::string model{}; // e.g. "gpt-4o-mini-tts", "tts-1"
  std::string input{};
  std::string voice{"alloy"};
  std::optional<std::string> response_format{}; // "mp3" | "opus" | "aac" | "flac" | "wav" | "pcm"
  std::optional<float> speed{};
  std::optional<std::string> instructions{};
  nlohmann::json extra_body{};
};

// ---------------------------------------------------------------------------
// POST /images/generations
// ---------------------------------------------------------------------------

struct ImageRequest
{
  std::string model{};
  std::string prompt{};
  std::optional<int> n{};
  std::optional<std::string> size{};            // e.g. "1024x1024"
  std::optional<std::string> quality{};         // e.g. "high"
  std::optional<std::string> response_format{}; // "b64_json" | "url" (dall-e models)
  std::optional<std::string> output_format{};   // "png" | "jpeg" | "webp" (gpt-image models)
  std::optional<std::string> background{};
  nlohmann::json extra_body{};
};

struct ImageResult
{
  std::string b64_json{};
  std::string url{};
  std::string revised_prompt{};
};

} // namespace oai
