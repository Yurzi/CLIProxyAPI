package registry

import "testing"

func TestGetFirstAvailableModelSkipsEndpointOnlyModels(t *testing.T) {
	r := newTestModelRegistry()
	r.RegisterClient("client-chat", "compat", []*ModelInfo{{ID: "chat-model", Created: 100, Type: "openai-compatibility"}})
	r.RegisterClient("client-endpoint", "compat", []*ModelInfo{
		{ID: "image-model", Created: 200, Type: OpenAIImageModelType},
		{ID: "embedding-model", Created: 300, Type: OpenAIEmbeddingModelType},
		{ID: "rerank-model", Created: 400, Type: OpenAIRerankModelType},
	})

	got, err := r.GetFirstAvailableModel("")
	if err != nil {
		t.Fatalf("GetFirstAvailableModel error: %v", err)
	}
	if got != "chat-model" {
		t.Fatalf("GetFirstAvailableModel = %q, want chat-model", got)
	}
}
