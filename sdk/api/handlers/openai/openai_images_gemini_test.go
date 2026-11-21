package openai

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"image"
	"image/color"
	"image/png"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/MoeForge/moego/ojson"
	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	runtimeexecutor "github.com/router-for-me/CLIProxyAPI/v8/internal/runtime/executor"
	apihandlers "github.com/router-for-me/CLIProxyAPI/v8/sdk/api/handlers"
	coreauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	coreexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	sdkconfig "github.com/router-for-me/CLIProxyAPI/v8/sdk/config"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/httptransport"
)

type geminiImageCaptureExecutor struct {
	jsonEndpointCaptureExecutor
}

func (e *geminiImageCaptureExecutor) Execute(ctx context.Context, auth *coreauth.Auth, req coreexecutor.Request, opts coreexecutor.Options) (coreexecutor.Response, error) {
	_, _ = e.jsonEndpointCaptureExecutor.Execute(ctx, auth, req, opts)
	return coreexecutor.Response{
		Payload: []byte(`{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":"aW1hZ2U="}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20,"totalTokenCount":30}}`),
	}, nil
}

func testTinyPNGBase64(t *testing.T) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	img.Set(0, 0, color.RGBA{R: 255, G: 0, B: 0, A: 255})
	img.Set(1, 1, color.RGBA{R: 0, G: 0, B: 255, A: 255})
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("png.Encode failed: %v", err)
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes())
}

type geminiTestRedirectTransport struct {
	target *url.URL
	base   http.RoundTripper
}

func (rt geminiTestRedirectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	clone := req.Clone(req.Context())
	clone.URL.Scheme = rt.target.Scheme
	clone.URL.Host = rt.target.Host
	clone.Host = rt.target.Host
	return rt.base.RoundTrip(clone)
}

func testEphemeralServiceAccountMap(t *testing.T, tokenURI string) map[string]any {
	t.Helper()
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("rsa.GenerateKey: %v", err)
	}
	pkcs8, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatalf("x509.MarshalPKCS8PrivateKey: %v", err)
	}
	pemBytes := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: pkcs8})
	return map[string]any{
		"type":                        "service_account",
		"project_id":                  "vertex-test-proj",
		"private_key_id":              "test-key-id",
		"private_key":                 string(pemBytes),
		"client_email":                "vertex-test@vertex-test-proj.iam.gserviceaccount.com",
		"client_id":                   "1234567890",
		"auth_uri":                    "https://accounts.google.com/o/oauth2/auth",
		"token_uri":                   tokenURI,
		"auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
		"client_x509_cert_url":        "https://www.googleapis.com/robot/v1/metadata/x509/vertex-test",
	}
}

func performImagesRequestWithContext(t *testing.T, ctx context.Context, path, contentType string, body io.Reader, handler gin.HandlerFunc) *httptest.ResponseRecorder {
	t.Helper()
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST(path, handler)
	req := httptest.NewRequest(http.MethodPost, path, body)
	if ctx != nil {
		req = req.WithContext(ctx)
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}

func TestGeminiImagesRoutesCollectMultipleResults(t *testing.T) {
	executor := &geminiImageCaptureExecutor{}
	manager := coreauth.NewManager(nil, &coreauth.RoundRobinSelector{}, nil)
	manager.RegisterExecutor(executor)
	auth := &coreauth.Auth{ID: t.Name(), Provider: executor.Identifier(), Status: coreauth.StatusActive}
	if _, errRegister := manager.Register(context.Background(), auth); errRegister != nil {
		t.Fatal(errRegister)
	}
	registry.GetGlobalRegistry().RegisterClient(auth.ID, auth.Provider, []*registry.ModelInfo{{ID: geminiFlashImageModel, Type: "gemini"}})
	manager.RefreshSchedulerEntry(auth.ID)
	t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(auth.ID) })
	handler := NewOpenAIAPIHandler(apihandlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{}, manager))
	for _, tc := range []struct {
		path    string
		body    string
		handler gin.HandlerFunc
	}{
		{imagesGenerationsPath, `{"model":"gemini-3.1-flash-image","prompt":"draw","size":"1536x1024","n":5}`, handler.ImagesGenerations},
		{imagesEditsPath, `{"model":"gemini-3.1-flash-image","prompt":"edit","size":"1536x1024","n":5,"images":[{"image_url":"data:image/png;base64,aW1hZ2U="}]}`, handler.ImagesEdits},
	} {
		t.Run(tc.path, func(t *testing.T) {
			before, _, _ := executor.calls()
			resp := performSpeechRoutingRequest(t, tc.path, tc.body, tc.handler)
			if resp.Code != http.StatusOK {
				t.Fatalf("status = %d: %s", resp.Code, resp.Body.String())
			}
			data := ojson.GetBytesNoCopy(resp.Body.Bytes(), "data").Array()
			if len(data) != 5 {
				t.Fatalf("image count = %d: %s", len(data), resp.Body.String())
			}
			for _, item := range data {
				if item.Get("b64_json").String() != "aW1hZ2U=" || item.Get("output_format").String() != "png" || item.Get("mime_type").String() != "image/png" {
					t.Fatalf("invalid image result: %s", item.Raw)
				}
			}
			if got := ojson.GetBytesNoCopy(resp.Body.Bytes(), "usage.total_tokens").Int(); got != 150 {
				t.Fatalf("usage.total_tokens = %d, want 150", got)
			}
			models, sources, payloads := executor.calls()
			if len(models) != len(before)+5 {
				t.Fatalf("new upstream call count = %d", len(models)-len(before))
			}
			for i := len(before); i < len(models); i++ {
				if models[i] != geminiFlashImageModel || sources[i] != "gemini" || ojson.Get(payloads[i], "generationConfig.imageConfig.aspectRatio").String() != "3:2" {
					t.Fatalf("invalid upstream request: %s (%s, %s)", payloads[i], models[i], sources[i])
				}
			}
		})
	}
}

func TestGeminiExecutorNanoBanana21GenerationsAndEdits(t *testing.T) {
	pngB64 := testTinyPNGBase64(t)

	var mu sync.Mutex
	var recordedPaths []string
	var recordedBodies [][]byte

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		recordedPaths = append(recordedPaths, r.URL.Path)
		recordedBodies = append(recordedBodies, body)
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"candidates":[
				{
					"content":{
						"role":"model",
						"parts":[
							{"thought":true,"inlineData":{"mimeType":"image/png","data":"ZHJhZnQ="}},
							{"inlineData":{"mimeType":"image/png","data":"` + pngB64 + `"}}
						]
					},
					"finishReason":"STOP"
				}
			],
			"usageMetadata":{
				"promptTokenCount":12,
				"candidatesTokenCount":1290,
				"totalTokenCount":1302,
				"promptTokensDetails":[{"modality":"TEXT","tokenCount":12}]
			}
		}`))
	}))
	defer server.Close()

	manager := coreauth.NewManager(nil, &coreauth.RoundRobinSelector{}, nil)
	manager.RegisterExecutor(runtimeexecutor.NewGeminiExecutor(&config.Config{}))
	auth := &coreauth.Auth{
		ID:       t.Name(),
		Provider: "gemini",
		Status:   coreauth.StatusActive,
		Attributes: map[string]string{
			"api_key":  "test-gemini-key",
			"base_url": server.URL,
		},
	}
	if _, err := manager.Register(context.Background(), auth); err != nil {
		t.Fatal(err)
	}
	registry.GetGlobalRegistry().RegisterClient(auth.ID, auth.Provider, []*registry.ModelInfo{{
		ID:                        geminiNanoBanana21Model,
		Type:                      "gemini",
		OwnedBy:                   "google",
		SupportedOutputModalities: []string{"text", "image"},
	}})
	manager.RefreshSchedulerEntry(auth.ID)
	t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(auth.ID) })

	handler := NewOpenAIAPIHandler(apihandlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{}, manager))

	t.Run("generations n=2 with jpeg transcode and 2K resolution", func(t *testing.T) {
		resp := performSpeechRoutingRequest(t, imagesGenerationsPath, `{
			"model":"gemini-nano-banana-2.1",
			"prompt":"minimalist mountain",
			"size":"1536x1024",
			"resolution":"2K",
			"output_format":"jpeg",
			"output_compression":85,
			"n":2
		}`, handler.ImagesGenerations)
		if resp.Code != http.StatusOK {
			t.Fatalf("status = %d: %s", resp.Code, resp.Body.String())
		}
		data := ojson.GetBytesNoCopy(resp.Body.Bytes(), "data").Array()
		if len(data) != 2 {
			t.Fatalf("data len = %d, want 2", len(data))
		}
		for i, item := range data {
			if item.Get("output_format").String() != "jpeg" || item.Get("mime_type").String() != "image/jpeg" {
				t.Fatalf("data[%d] format/mime = %s", i, item.Raw)
			}
			rawImg, errDec := base64.StdEncoding.DecodeString(item.Get("b64_json").String())
			if errDec != nil || len(rawImg) < 2 || rawImg[0] != 0xFF || rawImg[1] != 0xD8 {
				t.Fatalf("data[%d] is not valid JPEG bytes", i)
			}
		}
		if got := ojson.GetBytesNoCopy(resp.Body.Bytes(), "usage.total_tokens").Int(); got != 2604 {
			t.Fatalf("usage.total_tokens = %d, want 2604", got)
		}

		mu.Lock()
		lastPath := recordedPaths[len(recordedPaths)-1]
		lastBody := recordedBodies[len(recordedBodies)-1]
		mu.Unlock()
		if !strings.HasSuffix(lastPath, "/models/gemini-nano-banana-2.1:generateContent") {
			t.Fatalf("upstream path = %q", lastPath)
		}
		if got := ojson.GetBytesNoCopy(lastBody, "generationConfig.imageConfig.aspectRatio").String(); got != "3:2" {
			t.Fatalf("aspectRatio = %q, want 3:2", got)
		}
		if got := ojson.GetBytesNoCopy(lastBody, "generationConfig.imageConfig.imageSize").String(); got != "2K" {
			t.Fatalf("imageSize = %q, want 2K", got)
		}
	})

	t.Run("edits multipart with url response_format", func(t *testing.T) {
		rawPNG, err := base64.StdEncoding.DecodeString(pngB64)
		if err != nil {
			t.Fatal(err)
		}
		var body bytes.Buffer
		w := multipart.NewWriter(&body)
		_ = w.WriteField("model", geminiNanoBanana21Model)
		_ = w.WriteField("prompt", "turn sky purple")
		_ = w.WriteField("aspect_ratio", "16:9")
		_ = w.WriteField("response_format", "url")
		part, err := w.CreateFormFile("image", "input.png")
		if err != nil {
			t.Fatal(err)
		}
		_, _ = part.Write(rawPNG)
		_ = w.Close()

		resp := performImagesEndpointRequest(t, imagesEditsPath, w.FormDataContentType(), &body, handler.ImagesEdits)
		if resp.Code != http.StatusOK {
			t.Fatalf("status = %d: %s", resp.Code, resp.Body.String())
		}
		urlVal := ojson.GetBytesNoCopy(resp.Body.Bytes(), "data.0.url").String()
		if !strings.HasPrefix(urlVal, "data:image/png;base64,") {
			t.Fatalf("data.0.url = %q, want data:image/png;base64,...", urlVal)
		}

		mu.Lock()
		lastBody := recordedBodies[len(recordedBodies)-1]
		mu.Unlock()
		parts := ojson.GetBytesNoCopy(lastBody, "contents.0.parts").Array()
		if len(parts) != 2 || !parts[0].Get("inlineData.data").Exists() || parts[1].Get("text").String() != "turn sky purple" {
			t.Fatalf("edit parts = %s", ojson.GetBytesNoCopy(lastBody, "contents.0.parts").Raw)
		}
		if got := ojson.GetBytesNoCopy(lastBody, "generationConfig.imageConfig.aspectRatio").String(); got != "16:9" {
			t.Fatalf("aspectRatio = %q, want 16:9", got)
		}
	})
}

func TestGeminiInteractionsExecutorNanoBanana21GenerationsEditsAndStream(t *testing.T) {
	pngB64 := testTinyPNGBase64(t)

	var mu sync.Mutex
	var gotPath, gotRevision string
	var gotBody []byte

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		gotPath = r.URL.Path
		gotRevision = r.Header.Get("Api-Revision")
		gotBody = body
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"inter_img_1",
			"object":"interaction",
			"status":"completed",
			"model":"gemini-nano-banana-2.1",
			"steps":[
				{"type":"thought","content":[{"type":"image","mime_type":"image/png","data":"ZHJhZnQ="}]},
				{"type":"model_output","content":[{"type":"image","mime_type":"image/png","data":"` + pngB64 + `"}]}
			],
			"usage":{"total_input_tokens":15,"total_output_tokens":50,"total_tokens":65}
		}`))
	}))
	defer server.Close()

	manager := coreauth.NewManager(nil, &coreauth.RoundRobinSelector{}, nil)
	manager.RegisterExecutor(runtimeexecutor.NewGeminiInteractionsExecutor(&config.Config{}))
	auth := &coreauth.Auth{
		ID:       t.Name(),
		Provider: "gemini-interactions",
		Status:   coreauth.StatusActive,
		Attributes: map[string]string{
			"api_key":      "test-interactions-key",
			"base_url":     server.URL,
			"interactions": "true",
		},
	}
	if _, err := manager.Register(context.Background(), auth); err != nil {
		t.Fatal(err)
	}
	registry.GetGlobalRegistry().RegisterClient(auth.ID, auth.Provider, []*registry.ModelInfo{{
		ID:                        geminiNanoBanana21Model,
		Type:                      "gemini",
		OwnedBy:                   "google",
		SupportedOutputModalities: []string{"text", "image"},
	}})
	manager.RefreshSchedulerEntry(auth.ID)
	t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(auth.ID) })

	handler := NewOpenAIAPIHandler(apihandlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{}, manager))

	t.Run("generations", func(t *testing.T) {
		resp := performSpeechRoutingRequest(t, imagesGenerationsPath, `{
			"model":"gemini-nano-banana-2.1",
			"prompt":"editorial weather card",
			"size":"1792x1024",
			"image_size":"4K"
		}`, handler.ImagesGenerations)
		if resp.Code != http.StatusOK {
			t.Fatalf("status = %d: %s", resp.Code, resp.Body.String())
		}
		data := ojson.GetBytesNoCopy(resp.Body.Bytes(), "data").Array()
		if len(data) != 1 || data[0].Get("b64_json").String() != pngB64 {
			t.Fatalf("unexpected data (must return final image and skip draft): %s", resp.Body.String())
		}
		if got := ojson.GetBytesNoCopy(resp.Body.Bytes(), "usage.total_tokens").Int(); got != 65 {
			t.Fatalf("usage.total_tokens = %d, want 65", got)
		}

		mu.Lock()
		path, rev, body := gotPath, gotRevision, gotBody
		mu.Unlock()
		if !strings.HasSuffix(path, "/interactions") {
			t.Fatalf("path = %q, want suffix /interactions", path)
		}
		if rev != "2026-05-20" {
			t.Fatalf("Api-Revision = %q, want 2026-05-20", rev)
		}
		if got := ojson.GetBytesNoCopy(body, "response_format.type").String(); got != "image" {
			t.Fatalf("response_format = %s", ojson.GetBytesNoCopy(body, "response_format").Raw)
		}
		if got := ojson.GetBytesNoCopy(body, "response_format.aspect_ratio").String(); got != "16:9" {
			t.Fatalf("response_format.aspect_ratio = %q, want 16:9", got)
		}
		if got := ojson.GetBytesNoCopy(body, "response_format.image_size").String(); got != "4K" {
			t.Fatalf("response_format.image_size = %q, want 4K", got)
		}
	})

	t.Run("edits coalesces multimodal turn", func(t *testing.T) {
		resp := performSpeechRoutingRequest(t, imagesEditsPath, `{
			"model":"gemini-nano-banana-2.1",
			"prompt":"add sunglasses",
			"image":"data:image/png;base64,`+pngB64+`"
		}`, handler.ImagesEdits)
		if resp.Code != http.StatusOK {
			t.Fatalf("status = %d: %s", resp.Code, resp.Body.String())
		}
		mu.Lock()
		body := gotBody
		mu.Unlock()
		steps := ojson.GetBytesNoCopy(body, "input").Array()
		if len(steps) != 1 {
			t.Fatalf("input step count = %d, want 1 coalesced user_input step: %s", len(steps), string(body))
		}
		content := steps[0].Get("content").Array()
		if len(content) != 2 || content[0].Get("type").String() != "image" || content[0].Get("data").String() != pngB64 || content[1].Get("type").String() != "text" || content[1].Get("text").String() != "add sunglasses" {
			t.Fatalf("input.0.content = %s", steps[0].Get("content").Raw)
		}
	})

	t.Run("stream sse", func(t *testing.T) {
		resp := performSpeechRoutingRequest(t, imagesGenerationsPath, `{
			"model":"gemini-nano-banana-2.1",
			"prompt":"stream cat",
			"stream":true
		}`, handler.ImagesGenerations)
		if resp.Code != http.StatusOK {
			t.Fatalf("status = %d: %s", resp.Code, resp.Body.String())
		}
		if ct := resp.Header().Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
			t.Fatalf("Content-Type = %q, want text/event-stream", ct)
		}
		bodyStr := resp.Body.String()
		if !strings.Contains(bodyStr, "event: image_generation.completed") || !strings.Contains(bodyStr, pngB64) {
			t.Fatalf("SSE body missing completed event or image: %s", bodyStr)
		}
	})
}

func TestVertexExecutorNanoBanana21APIKeyServiceAccountAndImagen(t *testing.T) {
	pngB64 := testTinyPNGBase64(t)

	t.Run("vertex-api-key", func(t *testing.T) {
		var mu sync.Mutex
		var gotPath, gotKey string
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			gotPath = r.URL.Path
			gotKey = r.Header.Get("x-goog-api-key")
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":"` + pngB64 + `"}}]},"finishReason":"STOP"}]}`))
		}))
		defer server.Close()

		manager := coreauth.NewManager(nil, &coreauth.RoundRobinSelector{}, nil)
		manager.RegisterExecutor(runtimeexecutor.NewGeminiVertexExecutor(&config.Config{}))
		auth := &coreauth.Auth{
			ID:       t.Name(),
			Provider: "vertex",
			Status:   coreauth.StatusActive,
			Attributes: map[string]string{
				"api_key":  "vx-api-key",
				"base_url": server.URL,
			},
		}
		if _, err := manager.Register(context.Background(), auth); err != nil {
			t.Fatal(err)
		}
		registry.GetGlobalRegistry().RegisterClient(auth.ID, auth.Provider, []*registry.ModelInfo{{
			ID:                        geminiNanoBanana21Model,
			Type:                      "vertex",
			OwnedBy:                   "google",
			SupportedOutputModalities: []string{"text", "image"},
		}})
		manager.RefreshSchedulerEntry(auth.ID)
		t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(auth.ID) })

		handler := NewOpenAIAPIHandler(apihandlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{}, manager))
		resp := performSpeechRoutingRequest(t, imagesGenerationsPath, `{"model":"gemini-nano-banana-2.1","prompt":"vertex key cat"}`, handler.ImagesGenerations)
		if resp.Code != http.StatusOK {
			t.Fatalf("status = %d: %s", resp.Code, resp.Body.String())
		}
		if ojson.GetBytesNoCopy(resp.Body.Bytes(), "data.0.b64_json").String() != pngB64 {
			t.Fatalf("unexpected body: %s", resp.Body.String())
		}
		mu.Lock()
		path, key := gotPath, gotKey
		mu.Unlock()
		if !strings.HasSuffix(path, "/publishers/google/models/gemini-nano-banana-2.1:generateContent") || key != "vx-api-key" {
			t.Fatalf("path=%q key=%q", path, key)
		}
	})

	t.Run("vertex-api-key-interactions", func(t *testing.T) {
		var mu sync.Mutex
		var gotPath string
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			gotPath = r.URL.Path
			mu.Unlock()
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"vx_i_1","status":"completed","steps":[{"type":"model_output","content":[{"type":"image","mime_type":"image/png","data":"` + pngB64 + `"}]}]}`))
		}))
		defer server.Close()

		manager := coreauth.NewManager(nil, &coreauth.RoundRobinSelector{}, nil)
		manager.RegisterExecutor(runtimeexecutor.NewGeminiVertexExecutor(&config.Config{}))
		auth := &coreauth.Auth{
			ID:       t.Name(),
			Provider: "vertex",
			Status:   coreauth.StatusActive,
			Attributes: map[string]string{
				"api_key":      "vx-inter-key",
				"base_url":     server.URL,
				"interactions": "true",
			},
		}
		if _, err := manager.Register(context.Background(), auth); err != nil {
			t.Fatal(err)
		}
		registry.GetGlobalRegistry().RegisterClient(auth.ID, auth.Provider, []*registry.ModelInfo{{
			ID:                        geminiNanoBanana21Model,
			Type:                      "vertex",
			OwnedBy:                   "google",
			SupportedOutputModalities: []string{"text", "image"},
		}})
		manager.RefreshSchedulerEntry(auth.ID)
		t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(auth.ID) })

		handler := NewOpenAIAPIHandler(apihandlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{}, manager))
		resp := performSpeechRoutingRequest(t, imagesGenerationsPath, `{"model":"gemini-nano-banana-2.1","prompt":"vertex interactions cat"}`, handler.ImagesGenerations)
		if resp.Code != http.StatusOK {
			t.Fatalf("status = %d: %s", resp.Code, resp.Body.String())
		}
		if ojson.GetBytesNoCopy(resp.Body.Bytes(), "data.0.b64_json").String() != pngB64 {
			t.Fatalf("unexpected body: %s", resp.Body.String())
		}
		mu.Lock()
		path := gotPath
		mu.Unlock()
		if !strings.HasSuffix(path, "/v1beta1/interactions") {
			t.Fatalf("path = %q, want /v1beta1/interactions", path)
		}
	})

	t.Run("vertex-service-account-and-imagen", func(t *testing.T) {
		var mu sync.Mutex
		var paths []string
		var authHeaders []string
		var imagenBody []byte

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/token" {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"access_token":"mock-vx-sa-token","token_type":"Bearer","expires_in":3600}`))
				return
			}
			body, _ := io.ReadAll(r.Body)
			mu.Lock()
			paths = append(paths, r.URL.Path)
			authHeaders = append(authHeaders, r.Header.Get("Authorization"))
			if strings.Contains(r.URL.Path, "imagen-") {
				imagenBody = body
			}
			mu.Unlock()

			w.Header().Set("Content-Type", "application/json")
			if strings.Contains(r.URL.Path, ":predict") {
				_, _ = w.Write([]byte(`{"predictions":[{"bytesBase64Encoded":"` + pngB64 + `","mimeType":"image/png"}]}`))
				return
			}
			_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":"` + pngB64 + `"}}]},"finishReason":"STOP"}]}`))
		}))
		defer server.Close()

		targetURL, err := url.Parse(server.URL)
		if err != nil {
			t.Fatal(err)
		}
		transport := http.DefaultTransport.(*http.Transport).Clone()
		transport.Proxy = nil
		t.Cleanup(transport.CloseIdleConnections)

		ctx := httptransport.WithDecorator(context.Background(), func(next http.RoundTripper) http.RoundTripper {
			return geminiTestRedirectTransport{target: targetURL, base: transport}
		})

		saMap := testEphemeralServiceAccountMap(t, server.URL+"/token")
		manager := coreauth.NewManager(nil, &coreauth.RoundRobinSelector{}, nil)
		manager.RegisterExecutor(runtimeexecutor.NewGeminiVertexExecutor(&config.Config{}))
		auth := &coreauth.Auth{
			ID:       t.Name(),
			Provider: "vertex",
			Status:   coreauth.StatusActive,
			Metadata: map[string]any{
				"project_id":      "vertex-test-proj",
				"location":        "global",
				"service_account": saMap,
			},
		}
		if _, err := manager.Register(context.Background(), auth); err != nil {
			t.Fatal(err)
		}
		registry.GetGlobalRegistry().RegisterClient(auth.ID, auth.Provider, []*registry.ModelInfo{
			{ID: geminiNanoBanana21Model, Type: "gemini", OwnedBy: "google", SupportedOutputModalities: []string{"text", "image"}},
			{ID: "imagen-3.0-generate-002", Type: "gemini", OwnedBy: "google", SupportedOutputModalities: []string{"image"}},
		})
		manager.RefreshSchedulerEntry(auth.ID)
		t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(auth.ID) })

		handler := NewOpenAIAPIHandler(apihandlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{}, manager))

		// Vertex Service Account + gemini-nano-banana-2.1 edit
		editResp := performImagesRequestWithContext(t, ctx, imagesEditsPath, "application/json", strings.NewReader(`{
			"model":"gemini-nano-banana-2.1",
			"prompt":"vertex sa edit",
			"image":"`+pngB64+`"
		}`), handler.ImagesEdits)
		if editResp.Code != http.StatusOK {
			t.Fatalf("vertex sa edit status = %d: %s", editResp.Code, editResp.Body.String())
		}
		if ojson.GetBytesNoCopy(editResp.Body.Bytes(), "data.0.b64_json").String() != pngB64 {
			t.Fatalf("unexpected edit response: %s", editResp.Body.String())
		}

		// Vertex Service Account + imagen-3.0-generate-002 generation
		imgResp := performImagesRequestWithContext(t, ctx, imagesGenerationsPath, "application/json", strings.NewReader(`{
			"model":"imagen-3.0-generate-002",
			"prompt":"imagen landscape",
			"size":"1792x1024"
		}`), handler.ImagesGenerations)
		if imgResp.Code != http.StatusOK {
			t.Fatalf("vertex imagen status = %d: %s", imgResp.Code, imgResp.Body.String())
		}
		if ojson.GetBytesNoCopy(imgResp.Body.Bytes(), "data.0.b64_json").String() != pngB64 {
			t.Fatalf("unexpected imagen response: %s", imgResp.Body.String())
		}

		mu.Lock()
		defer mu.Unlock()
		if len(paths) < 2 {
			t.Fatalf("expected >= 2 upstream calls, got %v", paths)
		}
		if paths[0] != "/v1/projects/vertex-test-proj/locations/global/publishers/google/models/gemini-nano-banana-2.1:generateContent" {
			t.Fatalf("paths[0] = %q", paths[0])
		}
		if paths[1] != "/v1/projects/vertex-test-proj/locations/global/publishers/google/models/imagen-3.0-generate-002:predict" {
			t.Fatalf("paths[1] = %q", paths[1])
		}
		if authHeaders[0] != "Bearer mock-vx-sa-token" || authHeaders[1] != "Bearer mock-vx-sa-token" {
			t.Fatalf("authHeaders = %v", authHeaders)
		}
		if got := ojson.GetBytesNoCopy(imagenBody, "parameters.aspectRatio").String(); got != "16:9" {
			t.Fatalf("imagen parameters.aspectRatio = %q, want 16:9; body=%s", got, string(imagenBody))
		}
	})
}

func TestAntigravityExecutorFlashImageCompatibility(t *testing.T) {
	pngB64 := testTinyPNGBase64(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, `data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"thought":true,"inlineData":{"mimeType":"image/png","data":"ZHJhZnQ="}}]}}]}}`+"\n\n")
		_, _ = io.WriteString(w, `data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":"`+pngB64+`"}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":10,"totalTokenCount":15}}}`+"\n\n")
	}))
	defer server.Close()

	manager := coreauth.NewManager(nil, &coreauth.RoundRobinSelector{}, nil)
	manager.RegisterExecutor(runtimeexecutor.NewAntigravityExecutor(&config.Config{}))
	auth := &coreauth.Auth{
		ID:       t.Name(),
		Provider: "antigravity",
		Prefix:   "antigravity",
		Status:   coreauth.StatusActive,
		Metadata: map[string]any{
			"access_token": "anti-token",
			"expired":      time.Now().Add(time.Hour).Format(time.RFC3339),
			"project_id":   "anti-proj",
		},
		Attributes: map[string]string{
			"base_url": server.URL,
		},
	}
	if _, err := manager.Register(context.Background(), auth); err != nil {
		t.Fatal(err)
	}
	registry.GetGlobalRegistry().RegisterClient(auth.ID, auth.Provider, []*registry.ModelInfo{{
		ID:                        "antigravity/" + geminiFlashImageModel,
		Type:                      "antigravity",
		OwnedBy:                   "antigravity",
		SupportedOutputModalities: []string{"text", "image"},
	}})
	manager.RefreshSchedulerEntry(auth.ID)
	t.Cleanup(func() { registry.GetGlobalRegistry().UnregisterClient(auth.ID) })

	handler := NewOpenAIAPIHandler(apihandlers.NewBaseAPIHandlers(&sdkconfig.SDKConfig{}, manager))
	resp := performSpeechRoutingRequest(t, imagesGenerationsPath, `{"model":"antigravity/gemini-3.1-flash-image","prompt":"draw robot"}`, handler.ImagesGenerations)
	if resp.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", resp.Code, resp.Body.String())
	}
	data := ojson.GetBytesNoCopy(resp.Body.Bytes(), "data").Array()
	if len(data) != 1 || data[0].Get("b64_json").String() != pngB64 {
		t.Fatalf("unexpected antigravity image response: %s", resp.Body.String())
	}
}

func TestGeminiImagesEditsRejectsMaskAndUnsupportedInputsWith400(t *testing.T) {
	handler := &OpenAIAPIHandler{}

	for _, model := range []string{geminiNanoBanana21Model, geminiFlashImageModel} {
		t.Run("json_mask_"+model, func(t *testing.T) {
			body := `{"model":"` + model + `","prompt":"erase","image":"data:image/png;base64,AA==","mask":"data:image/png;base64,BB=="}`
			resp := performImagesEndpointRequest(t, imagesEditsPath, "application/json", strings.NewReader(body), handler.ImagesEdits)
			if resp.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400: %s", resp.Code, resp.Body.String())
			}
			if got := ojson.GetBytesNoCopy(resp.Body.Bytes(), "error.type").String(); got != "invalid_request_error" {
				t.Fatalf("error.type = %q, want invalid_request_error", got)
			}
			if msg := ojson.GetBytesNoCopy(resp.Body.Bytes(), "error.message").String(); !strings.Contains(msg, "mask") {
				t.Fatalf("error.message = %q, want mention of mask", msg)
			}
		})

		t.Run("multipart_mask_"+model, func(t *testing.T) {
			var body bytes.Buffer
			w := multipart.NewWriter(&body)
			_ = w.WriteField("model", model)
			_ = w.WriteField("prompt", "erase")
			imgPart, _ := w.CreateFormFile("image", "img.png")
			_, _ = imgPart.Write([]byte("pngdata"))
			maskPart, _ := w.CreateFormFile("mask", "mask.png")
			_, _ = maskPart.Write([]byte("maskdata"))
			_ = w.Close()

			resp := performImagesEndpointRequest(t, imagesEditsPath, w.FormDataContentType(), &body, handler.ImagesEdits)
			if resp.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400: %s", resp.Code, resp.Body.String())
			}
			if got := ojson.GetBytesNoCopy(resp.Body.Bytes(), "error.type").String(); got != "invalid_request_error" {
				t.Fatalf("error.type = %q, want invalid_request_error", got)
			}
			if msg := ojson.GetBytesNoCopy(resp.Body.Bytes(), "error.message").String(); !strings.Contains(msg, "mask") {
				t.Fatalf("error.message = %q, want mention of mask", msg)
			}
		})
	}

	t.Run("json_remote_url_rejected", func(t *testing.T) {
		body := `{"model":"gemini-nano-banana-2.1","prompt":"edit","image":"https://example.com/test.png"}`
		resp := performImagesEndpointRequest(t, imagesEditsPath, "application/json", strings.NewReader(body), handler.ImagesEdits)
		if resp.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400: %s", resp.Code, resp.Body.String())
		}
	})

	t.Run("json_file_id_rejected", func(t *testing.T) {
		body := `{"model":"gemini-nano-banana-2.1","prompt":"edit","images":[{"file_id":"file-123"}]}`
		resp := performImagesEndpointRequest(t, imagesEditsPath, "application/json", strings.NewReader(body), handler.ImagesEdits)
		if resp.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400: %s", resp.Code, resp.Body.String())
		}
	})
}
