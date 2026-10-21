package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/clienterror"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/api/handlers"
)

const (
	transcriptionsHandlerType = "openai-transcription"
	translationsHandlerType   = "openai-translation"
	transcriptionsPath        = "/v1/audio/transcriptions"
	translationsPath          = "/v1/audio/translations"
)

var (
	transcriptionsEndpoint = typedModelEndpoint{handlerType: transcriptionsHandlerType, modelType: registry.OpenAITranscriptionModelType, path: transcriptionsPath, configFlag: "transcription"}
	translationsEndpoint   = typedModelEndpoint{handlerType: translationsHandlerType, modelType: registry.OpenAITranscriptionModelType, path: translationsPath, configFlag: "transcription"}
)

// AudioTranscriptions handles POST /v1/audio/transcriptions for openai-compatibility models marked transcription: true.
func (h *OpenAIAPIHandler) AudioTranscriptions(c *gin.Context) {
	h.handleAudioEndpoint(c, transcriptionsEndpoint)
}

// AudioTranslations handles POST /v1/audio/translations for openai-compatibility models marked transcription: true.
func (h *OpenAIAPIHandler) AudioTranslations(c *gin.Context) {
	h.handleAudioEndpoint(c, translationsEndpoint)
}

// handleAudioEndpoint forwards the client's multipart body untouched to the executor, which
// rewrites only the upstream model. Responses are buffered, including stream=true SSE bodies,
// and labeled with the upstream Content-Type because payload rules may rewrite
// response_format or stream. Keep-alive is not used: text, subtitle, and SSE bodies must
// reach the client byte-for-byte.
func (h *OpenAIAPIHandler) handleAudioEndpoint(c *gin.Context, endpoint typedModelEndpoint) {
	raw, err := handlers.ReadRequestBody(c)
	if err != nil {
		writeJSONEndpointError(c, clienterror.HTTPStatusFromErrorOr(err, http.StatusBadRequest), fmt.Sprintf("Invalid request: %v", err))
		return
	}
	modelName, err := readAudioFormModel(c.GetHeader("Content-Type"), raw)
	if err != nil {
		writeJSONEndpointError(c, http.StatusBadRequest, fmt.Sprintf("Invalid request: %v", err))
		return
	}
	if !isRegisteredModelOfType(modelName, endpoint.modelType) {
		writeJSONEndpointError(c, http.StatusBadRequest, fmt.Sprintf("Model %s is not supported on %s. Configure an openai-compatibility model with %s: true.", modelName, endpoint.path, endpoint.configFlag))
		return
	}

	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
	resp, upstreamHeaders, errMsg := h.ExecuteVerbatimWithAuthManager(cliCtx, endpoint.handlerType, modelName, raw)
	if errMsg != nil {
		h.WriteErrorResponse(c, errMsg)
		cliCancel(errMsg.Error)
		return
	}
	contentType := upstreamHeaders.Get("Content-Type")
	if contentType == "" {
		contentType = "application/json"
		if !json.Valid(resp) {
			contentType = "text/plain; charset=utf-8"
		}
	}
	c.Header("Content-Type", contentType)
	handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)
	_, _ = c.Writer.Write(resp)
	cliCancel()
}

// readAudioFormModel reads the model field from a multipart/form-data audio request
// without decoding file parts.
func readAudioFormModel(contentType string, body []byte) (string, error) {
	mediaType, params, errParse := mime.ParseMediaType(contentType)
	if errParse != nil || !strings.EqualFold(mediaType, "multipart/form-data") {
		return "", errors.New("Content-Type must be multipart/form-data")
	}
	boundary := params["boundary"]
	if boundary == "" {
		return "", errors.New("multipart boundary is missing")
	}
	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	model := ""
	for {
		part, errNext := reader.NextPart()
		if errNext == io.EOF {
			break
		}
		if errNext != nil {
			return "", fmt.Errorf("malformed multipart body: %w", errNext)
		}
		if part.FormName() != "model" || part.FileName() != "" {
			continue
		}
		value, errRead := io.ReadAll(part)
		if errRead != nil {
			return "", fmt.Errorf("malformed multipart field model: %w", errRead)
		}
		model = strings.TrimSpace(string(value))
	}
	if model == "" {
		return "", errors.New("model is required")
	}
	return model, nil
}
