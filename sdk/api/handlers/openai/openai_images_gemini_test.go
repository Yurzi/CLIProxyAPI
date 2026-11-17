package openai

import (
	"context"
	"net/http"
	"testing"

	"github.com/MoeForge/moego/ojson"
	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	apihandlers "github.com/router-for-me/CLIProxyAPI/v8/sdk/api/handlers"
	coreauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	coreexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	sdkconfig "github.com/router-for-me/CLIProxyAPI/v8/sdk/config"
)

type geminiImageCaptureExecutor struct {
	jsonEndpointCaptureExecutor
}

func (e *geminiImageCaptureExecutor) Execute(ctx context.Context, auth *coreauth.Auth, req coreexecutor.Request, opts coreexecutor.Options) (coreexecutor.Response, error) {
	_, _ = e.jsonEndpointCaptureExecutor.Execute(ctx, auth, req, opts)
	return coreexecutor.Response{Payload: []byte(`{"created":123,"choices":[{"message":{"images":[{"image_url":{"url":"data:image/png;base64,aW1hZ2U="}}]}}]}`)}, nil
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
			models, sources, payloads := executor.calls()
			if len(models) != len(before)+5 {
				t.Fatalf("new upstream call count = %d", len(models)-len(before))
			}
			for i := len(before); i < len(models); i++ {
				if models[i] != geminiFlashImageModel || sources[i] != "openai" || ojson.Get(payloads[i], "image_config.aspect_ratio").String() != "3:2" {
					t.Fatalf("invalid upstream request: %s (%s, %s)", payloads[i], models[i], sources[i])
				}
			}
		})
	}
}
