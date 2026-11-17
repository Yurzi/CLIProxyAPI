package openai

import (
	"bytes"
	"context"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	runtimeexecutor "github.com/router-for-me/CLIProxyAPI/v8/internal/runtime/executor"
	apihandlers "github.com/router-for-me/CLIProxyAPI/v8/sdk/api/handlers"
	coreauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	coreexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	sdkconfig "github.com/router-for-me/CLIProxyAPI/v8/sdk/config"
)

const audioEndpointTestProvider = "compat-audio-endpoint-test"

type audioCaptureCall struct {
	model       string
	source      string
	payload     []byte
	contentType string
}

type audioCaptureExecutor struct {
	mu          sync.Mutex
	response    []byte
	contentType string
	callsLog    []audioCaptureCall
}

func (e *audioCaptureExecutor) Identifier() string { return audioEndpointTestProvider }

func (e *audioCaptureExecutor) Execute(_ context.Context, _ *coreauth.Auth, req coreexecutor.Request, opts coreexecutor.Options) (coreexecutor.Response, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.callsLog = append(e.callsLog, audioCaptureCall{
		model:       req.Model,
		source:      opts.SourceFormat.String(),
		payload:     append([]byte(nil), req.Payload...),
		contentType: opts.Headers.Get("Content-Type"),
	})
	var headers http.Header
	if e.contentType != "" {
		headers = http.Header{"Content-Type": []string{e.contentType}}
	}
	return coreexecutor.Response{Payload: e.response, Headers: headers}, nil
}

func (e *audioCaptureExecutor) ExecuteStream(context.Context, *coreauth.Auth, coreexecutor.Request, coreexecutor.Options) (*coreexecutor.StreamResult, error) {
	return nil, &coreauth.Error{Code: "not_implemented", Message: "ExecuteStream not implemented"}
}

func (e *audioCaptureExecutor) Refresh(_ context.Context, auth *coreauth.Auth) (*coreauth.Auth, error) {
	return auth, nil
}

func (e *audioCaptureExecutor) CountTokens(context.Context, *coreauth.Auth, coreexecutor.Request, coreexecutor.Options) (coreexecutor.Response, error) {
	return coreexecutor.Response{}, &coreauth.Error{Code: "not_implemented", Message: "CountTokens not implemented"}
}

func (e *audioCaptureExecutor) HttpRequest(context.Context, *coreauth.Auth, *http.Request) (*http.Response, error) {
	return nil, &coreauth.Error{Code: "not_implemented", Message: "HttpRequest not implemented"}
}

func (e *audioCaptureExecutor) setResponse(contentType, response string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.contentType = contentType
	e.response = []byte(response)
}

func (e *audioCaptureExecutor) calls() []audioCaptureCall {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]audioCaptureCall(nil), e.callsLog...)
}

func newAudioEndpointTestHandler(t *testing.T, executor *audioCaptureExecutor) *OpenAIAPIHandler {
	t.Helper()

	manager := coreauth.NewManager(nil, &coreauth.RoundRobinSelector{}, nil)
	manager.RegisterExecutor(executor)
	auth := &coreauth.Auth{ID: "audio-endpoint-routing-auth", Provider: audioEndpointTestProvider, Status: coreauth.StatusActive}
	if _, errRegister := manager.Register(context.Background(), auth); errRegister != nil {
		t.Fatalf("manager.Register: %v", errRegister)
	}
	registry.GetGlobalRegistry().RegisterClient(auth.ID, auth.Provider, []*registry.ModelInfo{
		{ID: "test-whisper-model", Type: registry.OpenAITranscriptionModelType},
		{ID: "test-audio-embed-model", Type: registry.OpenAIEmbeddingModelType},
		{ID: "test-audio-chat-model", Type: "openai-compatibility"},
	})
	manager.RefreshSchedulerEntry(auth.ID)
	t.Cleanup(func() {
		registry.GetGlobalRegistry().UnregisterClient(auth.ID)
	})

	base := apihandlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{}, manager)
	return NewOpenAIAPIHandler(base)
}

func newAudioMultipartBody(t *testing.T, fields [][2]string) ([]byte, string) {
	t.Helper()
	var buffer bytes.Buffer
	writer := multipart.NewWriter(&buffer)
	file, err := writer.CreateFormFile("file", "speech.wav")
	if err != nil {
		t.Fatalf("CreateFormFile: %v", err)
	}
	if _, err = file.Write([]byte("RIFF\x00\x01wave-bytes")); err != nil {
		t.Fatalf("write file: %v", err)
	}
	for _, field := range fields {
		if err = writer.WriteField(field[0], field[1]); err != nil {
			t.Fatalf("WriteField: %v", err)
		}
	}
	if err = writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	return buffer.Bytes(), writer.FormDataContentType()
}

func performAudioRequest(t *testing.T, path string, body []byte, contentType string, handler gin.HandlerFunc) *httptest.ResponseRecorder {
	t.Helper()
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST(path, handler)
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
	req.Header.Set("Content-Type", contentType)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	return resp
}

func TestAudioEndpointsRouteTranscriptionModelsToExecutor(t *testing.T) {
	executor := &audioCaptureExecutor{}
	handler := newAudioEndpointTestHandler(t, executor)

	// The response is labeled from the upstream Content-Type, never from request fields:
	// payload rules may rewrite response_format/stream and whisper-1 ignores stream=true.
	cases := []struct {
		name            string
		path            string
		handler         gin.HandlerFunc
		fields          [][2]string
		upstreamType    string
		response        string
		wantSource      string
		wantContentType string
	}{
		{name: "transcriptions json", path: "/v1/audio/transcriptions", handler: handler.AudioTranscriptions, fields: [][2]string{{"model", "test-whisper-model"}}, upstreamType: "application/json", response: `{"text":"hi"}`, wantSource: transcriptionsHandlerType, wantContentType: "application/json"},
		{name: "default json rewritten to text upstream", path: "/v1/audio/transcriptions", handler: handler.AudioTranscriptions, fields: [][2]string{{"model", "test-whisper-model"}}, upstreamType: "text/plain; charset=utf-8", response: "hi\n", wantSource: transcriptionsHandlerType, wantContentType: "text/plain; charset=utf-8"},
		{name: "transcriptions srt", path: "/v1/audio/transcriptions", handler: handler.AudioTranscriptions, fields: [][2]string{{"response_format", "srt"}, {"model", "test-whisper-model"}}, upstreamType: "text/plain; charset=utf-8", response: "1\n00:00:00,000 --> 00:00:01,000\nhi\n", wantSource: transcriptionsHandlerType, wantContentType: "text/plain; charset=utf-8"},
		{name: "transcriptions buffered stream", path: "/v1/audio/transcriptions", handler: handler.AudioTranscriptions, fields: [][2]string{{"model", "test-whisper-model"}, {"stream", "true"}}, upstreamType: "text/event-stream; charset=utf-8", response: "data: {\"type\":\"transcript.text.done\",\"text\":\"hi\"}\n\n", wantSource: transcriptionsHandlerType, wantContentType: "text/event-stream; charset=utf-8"},
		{name: "stream ignored by upstream", path: "/v1/audio/transcriptions", handler: handler.AudioTranscriptions, fields: [][2]string{{"model", "test-whisper-model"}, {"stream", "true"}}, upstreamType: "application/json", response: `{"text":"hi"}`, wantSource: transcriptionsHandlerType, wantContentType: "application/json"},
		{name: "missing upstream type text", path: "/v1/audio/translations", handler: handler.AudioTranslations, fields: [][2]string{{"model", "test-whisper-model"}, {"response_format", "text"}}, response: "data: hello\n", wantSource: translationsHandlerType, wantContentType: "text/plain; charset=utf-8"},
		{name: "missing upstream type json", path: "/v1/audio/translations", handler: handler.AudioTranslations, fields: [][2]string{{"model", "test-whisper-model"}}, response: `{"text":"Hello"}`, wantSource: translationsHandlerType, wantContentType: "application/json"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			executor.setResponse(tc.upstreamType, tc.response)
			before := len(executor.calls())
			body, contentType := newAudioMultipartBody(t, tc.fields)
			resp := performAudioRequest(t, tc.path, body, contentType, tc.handler)
			if resp.Code != http.StatusOK {
				t.Fatalf("status = %d, want %d: %s", resp.Code, http.StatusOK, resp.Body.String())
			}
			if resp.Body.String() != tc.response {
				t.Fatalf("body = %q, want %q", resp.Body.String(), tc.response)
			}
			if got := resp.Header().Get("Content-Type"); got != tc.wantContentType {
				t.Fatalf("content type = %q, want %q", got, tc.wantContentType)
			}
			calls := executor.calls()
			if len(calls) != before+1 {
				t.Fatalf("executor calls = %d, want %d", len(calls), before+1)
			}
			call := calls[len(calls)-1]
			if call.model != "test-whisper-model" {
				t.Fatalf("executor model = %q, want test-whisper-model", call.model)
			}
			if call.source != tc.wantSource {
				t.Fatalf("executor source format = %q, want %q", call.source, tc.wantSource)
			}
			if !bytes.Equal(call.payload, body) {
				t.Fatal("executor payload should be the client multipart body unchanged")
			}
			if call.contentType != contentType {
				t.Fatalf("executor content type = %q, want client %q", call.contentType, contentType)
			}
		})
	}
}

// Exercise the real handler, auth manager and openai-compatibility executor: a payload rule
// that rewrites response_format must be reflected in the downstream body and Content-Type
// even though response header passthrough is disabled.
func TestAudioTranscriptionsPayloadOverrideKeepsUpstreamContentType(t *testing.T) {
	const transcript = "你好，世界\n"
	var gotFormat, gotModel string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/audio/transcriptions" {
			t.Errorf("upstream path = %q", r.URL.Path)
		}
		gotFormat = r.FormValue("response_format")
		gotModel = r.FormValue("model")
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, _ = w.Write([]byte(transcript))
	}))
	t.Cleanup(upstream.Close)

	cfg := &config.Config{Payload: config.PayloadConfig{Override: []config.PayloadRule{{
		Models: []config.PayloadModelRule{{Name: "whisper-integration-test", FromProtocol: transcriptionsHandlerType}},
		Params: map[string]any{"response_format": "text"},
	}}}}
	manager := coreauth.NewManager(nil, &coreauth.RoundRobinSelector{}, nil)
	const providerKey = "audio-compat-test"
	manager.RegisterExecutor(runtimeexecutor.NewOpenAICompatExecutor(providerKey, cfg))
	auth := &coreauth.Auth{ID: "audio-integration-auth", Provider: providerKey, Status: coreauth.StatusActive, Attributes: map[string]string{
		"base_url": upstream.URL + "/v1",
		"api_key":  "test-key",
	}}
	if _, errRegister := manager.Register(context.Background(), auth); errRegister != nil {
		t.Fatalf("manager.Register: %v", errRegister)
	}
	registry.GetGlobalRegistry().RegisterClient(auth.ID, auth.Provider, []*registry.ModelInfo{{ID: "whisper-integration-test", Type: registry.OpenAITranscriptionModelType}})
	manager.RefreshSchedulerEntry(auth.ID)
	t.Cleanup(func() {
		registry.GetGlobalRegistry().UnregisterClient(auth.ID)
	})
	handler := NewOpenAIAPIHandler(apihandlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{}, manager))

	body, contentType := newAudioMultipartBody(t, [][2]string{{"model", "whisper-integration-test"}})
	resp := performAudioRequest(t, "/v1/audio/transcriptions", body, contentType, handler.AudioTranscriptions)
	if resp.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", resp.Code, resp.Body.String())
	}
	if gotFormat != "text" || gotModel != "whisper-integration-test" {
		t.Fatalf("upstream response_format = %q, model = %q", gotFormat, gotModel)
	}
	if resp.Body.String() != transcript {
		t.Fatalf("body = %q, want %q", resp.Body.String(), transcript)
	}
	if got := resp.Header().Get("Content-Type"); got != "text/plain; charset=utf-8" {
		t.Fatalf("content type = %q, want upstream text/plain", got)
	}
}

func TestAudioEndpointsRejectInvalidRequests(t *testing.T) {
	executor := &audioCaptureExecutor{}
	handler := newAudioEndpointTestHandler(t, executor)

	chatBody, chatType := newAudioMultipartBody(t, [][2]string{{"model", "test-audio-chat-model"}})
	embedBody, embedType := newAudioMultipartBody(t, [][2]string{{"model", "test-audio-embed-model"}})
	unknownBody, unknownType := newAudioMultipartBody(t, [][2]string{{"model", "missing-model"}})
	missingBody, missingType := newAudioMultipartBody(t, nil)
	cases := []struct {
		name        string
		handler     gin.HandlerFunc
		body        []byte
		contentType string
	}{
		{name: "chat model", handler: handler.AudioTranscriptions, body: chatBody, contentType: chatType},
		{name: "embedding model", handler: handler.AudioTranslations, body: embedBody, contentType: embedType},
		{name: "unknown model", handler: handler.AudioTranscriptions, body: unknownBody, contentType: unknownType},
		{name: "missing model", handler: handler.AudioTranscriptions, body: missingBody, contentType: missingType},
		{name: "json body", handler: handler.AudioTranscriptions, body: []byte(`{"model":"test-whisper-model"}`), contentType: "application/json"},
		{name: "missing boundary", handler: handler.AudioTranscriptions, body: chatBody, contentType: "multipart/form-data"},
		{name: "malformed multipart", handler: handler.AudioTranscriptions, body: []byte("--x\r\nbroken"), contentType: "multipart/form-data; boundary=x"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp := performAudioRequest(t, "/v1/audio/transcriptions", tc.body, tc.contentType, tc.handler)
			if resp.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want %d: %s", resp.Code, http.StatusBadRequest, resp.Body.String())
			}
		})
	}
	if calls := executor.calls(); len(calls) != 0 {
		t.Fatalf("executor received rejected requests: %d", len(calls))
	}
}
