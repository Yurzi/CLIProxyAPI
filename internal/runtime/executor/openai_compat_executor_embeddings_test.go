package executor

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v8/sdk/translator"
	"github.com/tidwall/gjson"
)

type openAICompatJSONEndpointCapture struct {
	path          string
	body          []byte
	authorization string
}

func newOpenAICompatJSONEndpointServer(t *testing.T, status int, response string) (*httptest.Server, *openAICompatJSONEndpointCapture) {
	t.Helper()
	captured := &openAICompatJSONEndpointCapture{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured.path = r.URL.Path
		captured.authorization = r.Header.Get("Authorization")
		captured.body, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(response))
	}))
	t.Cleanup(server.Close)
	return server, captured
}

func newOpenAICompatJSONEndpointAuth(baseURL string) *cliproxyauth.Auth {
	return &cliproxyauth.Auth{
		Provider: "openai-compatibility",
		Attributes: map[string]string{
			"base_url":     baseURL + "/v1",
			"api_key":      "test",
			"compat_name":  "compat",
			"provider_key": "compat",
		},
	}
}

func TestOpenAICompatExecutorJSONEndpointsPassthrough(t *testing.T) {
	cases := []struct {
		name     string
		source   string
		body     string
		response string
		wantPath string
		check    string
	}{
		{
			name:     "embeddings",
			source:   "openai-embedding",
			body:     `{"model":"compat-embed","input":["a","b"],"dimensions":256,"encoding_format":"float"}`,
			response: `{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1]}],"model":"upstream-model","usage":{"prompt_tokens":2,"total_tokens":2}}`,
			wantPath: "/v1/embeddings",
			check:    "dimensions",
		},
		{
			name:     "rerank",
			source:   "openai-rerank",
			body:     `{"model":"compat-rerank","query":"q","documents":["a","b"],"top_n":1}`,
			response: `{"results":[{"index":1,"relevance_score":0.9}],"usage":{"total_tokens":5}}`,
			wantPath: "/v1/rerank",
			check:    "top_n",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server, captured := newOpenAICompatJSONEndpointServer(t, http.StatusOK, tc.response)
			executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{
				OpenAICompatibility: []config.OpenAICompatibility{{Name: "compat", SupportPromptCacheKey: true}},
			})
			resp, err := executor.Execute(context.Background(), newOpenAICompatJSONEndpointAuth(server.URL), cliproxyexecutor.Request{
				Model:   "upstream-model",
				Payload: []byte(tc.body),
			}, cliproxyexecutor.Options{
				SourceFormat:    sdktranslator.FromString(tc.source),
				OriginalRequest: []byte(tc.body),
			})
			if err != nil {
				t.Fatalf("Execute error: %v", err)
			}
			if captured.path != tc.wantPath {
				t.Fatalf("path = %q, want %q", captured.path, tc.wantPath)
			}
			if captured.authorization != "Bearer test" {
				t.Fatalf("authorization = %q, want Bearer test", captured.authorization)
			}
			if got := gjson.GetBytes(captured.body, "model").String(); got != "upstream-model" {
				t.Fatalf("model = %q, want upstream-model; body=%s", got, captured.body)
			}
			if want := gjson.Get(tc.body, tc.check).Raw; gjson.GetBytes(captured.body, tc.check).Raw != want {
				t.Fatalf("%s not preserved; body=%s", tc.check, captured.body)
			}
			for _, field := range []string{"prompt_cache_key", "messages", "stream", "reasoning_effort", "max_tokens"} {
				if gjson.GetBytes(captured.body, field).Exists() {
					t.Fatalf("unexpected chat field %q in body: %s", field, captured.body)
				}
			}
			if string(resp.Payload) != tc.response {
				t.Fatalf("response payload = %s, want upstream body", resp.Payload)
			}
		})
	}
}

// Payload rules must be the final semantic barrier: they run after the built-in model
// rewrite, so a configured override of "model" wins.
func TestOpenAICompatExecutorEmbeddingsPayloadRulesAreFinalBarrier(t *testing.T) {
	server, captured := newOpenAICompatJSONEndpointServer(t, http.StatusOK, `{"data":[]}`)
	body := `{"model":"compat-embed","input":"hi","dimensions":1024}`
	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{
		Payload: config.PayloadConfig{
			Default: []config.PayloadRule{{
				Models: []config.PayloadModelRule{{Name: "upstream-embed", FromProtocol: "openai-embedding"}},
				Params: map[string]any{"dimensions": 256, "user": "proxy"},
			}},
			Override: []config.PayloadRule{{
				Models: []config.PayloadModelRule{{Name: "upstream-embed", Protocol: "openai"}},
				Params: map[string]any{"model": "forced-embed", "encoding_format": "base64"},
			}},
		},
	})
	_, err := executor.Execute(context.Background(), newOpenAICompatJSONEndpointAuth(server.URL), cliproxyexecutor.Request{
		Model:   "upstream-embed",
		Payload: []byte(body),
	}, cliproxyexecutor.Options{
		SourceFormat:    sdktranslator.FromString("openai-embedding"),
		OriginalRequest: []byte(body),
	})
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if got := gjson.GetBytes(captured.body, "model").String(); got != "forced-embed" {
		t.Fatalf("model = %q, want payload override forced-embed; body=%s", got, captured.body)
	}
	if got := gjson.GetBytes(captured.body, "encoding_format").String(); got != "base64" {
		t.Fatalf("encoding_format = %q, want base64; body=%s", got, captured.body)
	}
	if got := gjson.GetBytes(captured.body, "dimensions").Int(); got != 1024 {
		t.Fatalf("dimensions = %d, want client value 1024; body=%s", got, captured.body)
	}
	if got := gjson.GetBytes(captured.body, "user").String(); got != "proxy" {
		t.Fatalf("user = %q, want default proxy; body=%s", got, captured.body)
	}
}

func TestOpenAICompatExecutorRerankUpstreamErrorKeepsStatus(t *testing.T) {
	server, _ := newOpenAICompatJSONEndpointServer(t, http.StatusUnprocessableEntity, `{"error":{"message":"documents is required"}}`)
	executor := NewOpenAICompatExecutor("openai-compatibility", &config.Config{})
	_, err := executor.Execute(context.Background(), newOpenAICompatJSONEndpointAuth(server.URL), cliproxyexecutor.Request{
		Model:   "upstream-rerank",
		Payload: []byte(`{"model":"compat-rerank","query":"q"}`),
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-rerank"),
	})
	if err == nil {
		t.Fatal("expected upstream error")
	}
	var status statusErr
	if !errors.As(err, &status) || status.StatusCode() != http.StatusUnprocessableEntity {
		t.Fatalf("error = %#v, want status %d", err, http.StatusUnprocessableEntity)
	}
}
