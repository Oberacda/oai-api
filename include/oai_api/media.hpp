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
