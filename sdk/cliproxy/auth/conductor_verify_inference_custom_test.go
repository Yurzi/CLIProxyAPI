package auth

import (
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
)

func TestSelectVerifyInferenceModel_SkipsEndpointOnlyAndPrefersTextOverNanoBanana(t *testing.T) {
	t.Parallel()

	t.Run("skips endpoint-only models completely", func(t *testing.T) {
		t.Parallel()

		models := []*registry.ModelInfo{
			{ID: "text-embedding-3-small", Type: registry.OpenAIEmbeddingModelType},
			{ID: "bge-reranker-v2-m3", Type: registry.OpenAIRerankModelType},
			{ID: "whisper-1", Type: registry.OpenAITranscriptionModelType},
			{ID: "gpt-image-1", Type: registry.OpenAIImageModelType},
		}
		if got := selectVerifyInferenceModel(models); got != "" {
			t.Fatalf("selectVerifyInferenceModel(endpoint-only) = %q, want empty string", got)
		}
	})

	t.Run("skips endpoint-only models before text candidate", func(t *testing.T) {
		t.Parallel()

		models := []*registry.ModelInfo{
			{ID: "text-embedding-3-small", Type: registry.OpenAIEmbeddingModelType},
			{ID: "whisper-1", Type: registry.OpenAITranscriptionModelType},
			{ID: "gpt-5.4", Type: "openai-compatibility"},
		}
		if got := selectVerifyInferenceModel(models); got != "gpt-5.4" {
			t.Fatalf("selectVerifyInferenceModel = %q, want gpt-5.4", got)
		}
	})

	t.Run("treats nano-banana as media fallback behind standard text models", func(t *testing.T) {
		t.Parallel()

		models := []*registry.ModelInfo{
			{
				ID:                        "gemini-nano-banana-2.1",
				SupportedInputModalities:  []string{"text", "image"},
				SupportedOutputModalities: []string{"text", "image"},
			},
			{
				ID:                        "gemini-2.5-pro",
				SupportedInputModalities:  []string{"text"},
				SupportedOutputModalities: []string{"text"},
				Thinking:                  &registry.ThinkingSupport{Min: 128, Max: 32768},
			},
		}
		if got := selectVerifyInferenceModel(models); got != "gemini-2.5-pro" {
			t.Fatalf("selectVerifyInferenceModel = %q, want gemini-2.5-pro", got)
		}

		onlyNanoBanana := models[:1]
		if got := selectVerifyInferenceModel(onlyNanoBanana); got != "gemini-nano-banana-2.1" {
			t.Fatalf("selectVerifyInferenceModel(only nano-banana) = %q, want gemini-nano-banana-2.1", got)
		}
	})
}
