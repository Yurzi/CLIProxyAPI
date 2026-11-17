package executor

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/runtime/executor/helps"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/thinking"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/util"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	log "github.com/sirupsen/logrus"
)

const (
	openAICompatTranscriptionHandlerType = "openai-transcription"
	openAICompatTranslationHandlerType   = "openai-translation"
	openAICompatTranscriptionsPath       = "/audio/transcriptions"
	openAICompatTranslationsPath         = "/audio/translations"
)

// openAICompatAudioEndpointPath returns the upstream path for multipart audio passthrough
// endpoints (transcriptions, translations), or an empty string for other requests.
func openAICompatAudioEndpointPath(opts cliproxyexecutor.Options) string {
	switch opts.SourceFormat.String() {
	case openAICompatTranscriptionHandlerType:
		return openAICompatTranscriptionsPath
	case openAICompatTranslationHandlerType:
		return openAICompatTranslationsPath
	default:
		return ""
	}
}

// executeAudioEndpoint forwards multipart transcription and translation requests to the
// provider base URL. Parts pass through untouched except for the resolved upstream model,
// and payload rules run last as the final semantic barrier. The upstream body (JSON, plain
// text, subtitles, or buffered SSE) is returned verbatim.
func (e *OpenAICompatExecutor) executeAudioEndpoint(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, endpointPath string) (resp cliproxyexecutor.Response, err error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	reporter := helps.NewExecutorUsageReporter(ctx, e, baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	baseURL, apiKey := e.resolveCredentials(auth)
	if baseURL == "" {
		err = statusErr{code: http.StatusUnauthorized, msg: "missing provider baseURL"}
		return resp, err
	}

	payload, contentType := req.Payload, opts.Headers.Get("Content-Type")
	if baseModel != "" {
		var errModel error
		payload, contentType, errModel = helps.SetMultipartFormField(payload, contentType, "model", baseModel)
		if errModel != nil {
			err = statusErr{code: http.StatusBadRequest, msg: fmt.Sprintf("invalid audio request: %v", errModel)}
			return resp, err
		}
	}
	// Payload rules target the "openai" protocol like image endpoints; rules can use
	// from-protocol "openai-transcription" or "openai-translation" to scope to one endpoint.
	payload, contentType, err = helps.ApplyMediaPayloadConfig(e.cfg, e.Identifier(), baseModel, "openai", payload, contentType, req, opts)
	if err != nil {
		return resp, err
	}

	url := strings.TrimSuffix(baseURL, "/") + endpointPath
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return resp, err
	}
	httpReq.Header.Set("Content-Type", contentType)
	if apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	}
	ensureOpenAICompatUserAgent(httpReq, opts.Headers)
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(httpReq, attrs, opts.Headers)
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	helps.RecordAPIRequest(ctx, e.cfg, helps.UpstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      payload,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpClient = reporter.TrackHTTPClient(httpClient)
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("openai compat executor: close response body error: %v", errClose)
		}
	}()
	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())

	body, errRead := io.ReadAll(httpResp.Body)
	if errRead != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, errRead)
		err = errRead
		return resp, err
	}
	helps.AppendAPIResponseChunk(ctx, e.cfg, body)

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		helps.LogWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), body))
		err = newOpenAICompatStatusError(httpResp.StatusCode, httpResp.Header, body)
		return resp, err
	}

	// Token-billed models (gpt-4o-transcribe) report usage; whisper-1 reports duration only.
	reporter.Publish(ctx, helps.ParseOpenAIUsage(body))
	reporter.EnsurePublished(ctx)
	resp = cliproxyexecutor.Response{Payload: body, Headers: httpResp.Header.Clone()}
	return resp, nil
}
