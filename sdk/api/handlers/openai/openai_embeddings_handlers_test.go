package openai

import (
	"context"
	"net/http"
	"sync"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	apihandlers "github.com/router-for-me/CLIProxyAPI/v8/sdk/api/handlers"
	coreauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	coreexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	sdkconfig "github.com/router-for-me/CLIProxyAPI/v8/sdk/config"
)

const jsonEndpointTestProvider = "compat-json-endpoint-test"

type jsonEndpointCaptureExecutor struct {
	mu       sync.Mutex
	models   []string
	sources  []string
	payloads []string
}

func (e *jsonEndpointCaptureExecutor) Identifier() string { return jsonEndpointTestProvider }

func (e *jsonEndpointCaptureExecutor) Execute(_ context.Context, _ *coreauth.Auth, req coreexecutor.Request, opts coreexecutor.Options) (coreexecutor.Response, error) {
	e.mu.Lock()
	e.models = append(e.models, req.Model)
	e.sources = append(e.sources, opts.SourceFormat.String())
	e.payloads = append(e.payloads, string(req.Payload))
	e.mu.Unlock()
	return coreexecutor.Response{Payload: []byte(`{"object":"list","data":[]}`), Headers: http.Header{"Content-Type": []string{"application/json"}}}, nil
}

func (e *jsonEndpointCaptureExecutor) ExecuteStream(context.Context, *coreauth.Auth, coreexecutor.Request, coreexecutor.Options) (*coreexecutor.StreamResult, error) {
	return nil, &coreauth.Error{Code: "not_implemented", Message: "ExecuteStream not implemented"}
}

func (e *jsonEndpointCaptureExecutor) Refresh(_ context.Context, auth *coreauth.Auth) (*coreauth.Auth, error) {
	return auth, nil
}

func (e *jsonEndpointCaptureExecutor) CountTokens(context.Context, *coreauth.Auth, coreexecutor.Request, coreexecutor.Options) (coreexecutor.Response, error) {
	return coreexecutor.Response{}, &coreauth.Error{Code: "not_implemented", Message: "CountTokens not implemented"}
}

func (e *jsonEndpointCaptureExecutor) HttpRequest(context.Context, *coreauth.Auth, *http.Request) (*http.Response, error) {
	return nil, &coreauth.Error{Code: "not_implemented", Message: "HttpRequest not implemented"}
}

func (e *jsonEndpointCaptureExecutor) calls() (models, sources, payloads []string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]string(nil), e.models...), append([]string(nil), e.sources...), append([]string(nil), e.payloads...)
}

func newJSONEndpointTestHandler(t *testing.T, executor *jsonEndpointCaptureExecutor) *OpenAIAPIHandler {
	t.Helper()

	manager := coreauth.NewManager(nil, &coreauth.RoundRobinSelector{}, nil)
	manager.RegisterExecutor(executor)
	auth := &coreauth.Auth{ID: "json-endpoint-routing-auth", Provider: jsonEndpointTestProvider, Status: coreauth.StatusActive}
	if _, errRegister := manager.Register(context.Background(), auth); errRegister != nil {
		t.Fatalf("manager.Register: %v", errRegister)
	}
	registry.GetGlobalRegistry().RegisterClient(auth.ID, auth.Provider, []*registry.ModelInfo{
		{ID: "test-embed-model", Type: registry.OpenAIEmbeddingModelType},
		{ID: "test-rerank-model", Type: registry.OpenAIRerankModelType},
		{ID: "test-chat-model", Type: "openai-compatibility"},
	})
	manager.RefreshSchedulerEntry(auth.ID)
	t.Cleanup(func() {
		registry.GetGlobalRegistry().UnregisterClient(auth.ID)
	})

	base := apihandlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{}, manager)
	return NewOpenAIAPIHandler(base)
}

func TestJSONEndpointsRouteTypedModelsToExecutor(t *testing.T) {
	executor := &jsonEndpointCaptureExecutor{}
	handler := newJSONEndpointTestHandler(t, executor)

	cases := []struct {
		name       string
		path       string
		handler    gin.HandlerFunc
		body       string
		wantModel  string
		wantSource string
	}{
		{name: "embeddings", path: "/v1/embeddings", handler: handler.Embeddings, body: `{"model":"test-embed-model","input":"hello"}`, wantModel: "test-embed-model", wantSource: embeddingsHandlerType},
		{name: "rerank", path: "/v1/rerank", handler: handler.Rerank, body: `{"model":"test-rerank-model","query":"q","documents":["a"]}`, wantModel: "test-rerank-model", wantSource: rerankHandlerType},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			before, _, _ := executor.calls()
			resp := performSpeechRoutingRequest(t, tc.path, tc.body, tc.handler)
			if resp.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d: %s", resp.Code, http.StatusOK, resp.Body.String())
			}
			if resp.Body.String() != `{"object":"list","data":[]}` {
				t.Fatalf("body = %q", resp.Body.String())
			}
			if got := resp.Header().Get("Content-Type"); got != "application/json" {
				t.Fatalf("content type = %q, want application/json", got)
			}
			models, sources, payloads := executor.calls()
			if len(models) != len(before)+1 {
				t.Fatalf("executor calls = %v, want one new call", models)
			}
			if got := models[len(models)-1]; got != tc.wantModel {
				t.Fatalf("executor model = %q, want %q", got, tc.wantModel)
			}
			if got := sources[len(sources)-1]; got != tc.wantSource {
				t.Fatalf("executor source format = %q, want %q", got, tc.wantSource)
			}
			if got := payloads[len(payloads)-1]; got != tc.body {
				t.Fatalf("executor payload = %s, want client body unchanged", got)
			}
		})
	}
}

func TestJSONEndpointsRejectModelsOfOtherTypes(t *testing.T) {
	executor := &jsonEndpointCaptureExecutor{}
	handler := newJSONEndpointTestHandler(t, executor)

	cases := []struct {
		name    string
		path    string
		handler gin.HandlerFunc
		body    string
	}{
		{name: "chat model on embeddings", path: "/v1/embeddings", handler: handler.Embeddings, body: `{"model":"test-chat-model","input":"hello"}`},
		{name: "rerank model on embeddings", path: "/v1/embeddings", handler: handler.Embeddings, body: `{"model":"test-rerank-model","input":"hello"}`},
		{name: "embedding model on rerank", path: "/v1/rerank", handler: handler.Rerank, body: `{"model":"test-embed-model","query":"q","documents":["a"]}`},
		{name: "unknown model on rerank", path: "/v1/rerank", handler: handler.Rerank, body: `{"model":"missing-model","query":"q","documents":["a"]}`},
		{name: "missing model", path: "/v1/embeddings", handler: handler.Embeddings, body: `{"input":"hello"}`},
		{name: "invalid json", path: "/v1/embeddings", handler: handler.Embeddings, body: `{"model":`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp := performSpeechRoutingRequest(t, tc.path, tc.body, tc.handler)
			if resp.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want %d: %s", resp.Code, http.StatusBadRequest, resp.Body.String())
			}
		})
	}
	if models, _, _ := executor.calls(); len(models) != 0 {
		t.Fatalf("executor received rejected requests: %v", models)
	}
}
