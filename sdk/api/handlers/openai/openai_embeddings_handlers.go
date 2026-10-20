package openai

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/clienterror"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/api/handlers"
	"github.com/tidwall/gjson"
)

const (
	embeddingsHandlerType = "openai-embedding"
	rerankHandlerType     = "openai-rerank"
	embeddingsPath        = "/v1/embeddings"
	rerankPath            = "/v1/rerank"
)

// jsonEndpoint describes an OpenAI-style JSON endpoint served only by models
// registered with a dedicated model type.
type jsonEndpoint struct {
	handlerType string
	modelType   string
	path        string
	configFlag  string
}

var (
	embeddingsEndpoint = jsonEndpoint{handlerType: embeddingsHandlerType, modelType: registry.OpenAIEmbeddingModelType, path: embeddingsPath, configFlag: "embedding"}
	rerankEndpoint     = jsonEndpoint{handlerType: rerankHandlerType, modelType: registry.OpenAIRerankModelType, path: rerankPath, configFlag: "rerank"}
)

// Embeddings handles POST /v1/embeddings for openai-compatibility models marked embedding: true.
func (h *OpenAIAPIHandler) Embeddings(c *gin.Context) {
	h.handleJSONEndpoint(c, embeddingsEndpoint)
}

// Rerank handles POST /v1/rerank for openai-compatibility models marked rerank: true.
func (h *OpenAIAPIHandler) Rerank(c *gin.Context) {
	h.handleJSONEndpoint(c, rerankEndpoint)
}

func (h *OpenAIAPIHandler) handleJSONEndpoint(c *gin.Context, endpoint jsonEndpoint) {
	rawJSON, err := handlers.ReadRequestBody(c)
	if err != nil {
		writeJSONEndpointError(c, clienterror.HTTPStatusFromErrorOr(err, http.StatusBadRequest), fmt.Sprintf("Invalid request: %v", err))
		return
	}
	if !gjson.ValidBytes(rawJSON) {
		writeJSONEndpointError(c, http.StatusBadRequest, "Invalid request: body must be valid JSON")
		return
	}
	modelName := strings.TrimSpace(gjson.GetBytes(rawJSON, "model").String())
	if !isRegisteredModelOfType(modelName, endpoint.modelType) {
		writeJSONEndpointError(c, http.StatusBadRequest, fmt.Sprintf("Model %s is not supported on %s. Configure an openai-compatibility model with %s: true.", modelName, endpoint.path, endpoint.configFlag))
		return
	}

	c.Header("Content-Type", "application/json")
	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
	stopKeepAlive := h.StartNonStreamingKeepAlive(c, cliCtx)
	resp, upstreamHeaders, errMsg := h.ExecuteWithAuthManager(cliCtx, endpoint.handlerType, modelName, rawJSON, "")
	stopKeepAlive()
	if errMsg != nil {
		h.WriteErrorResponse(c, errMsg)
		cliCancel(errMsg.Error)
		return
	}
	handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)
	_, _ = c.Writer.Write(resp)
	cliCancel()
}

func isRegisteredModelOfType(model, modelType string) bool {
	if model == "" {
		return false
	}
	info := registry.LookupModelInfo(model)
	return info != nil && info.Type == modelType
}

func writeJSONEndpointError(c *gin.Context, status int, message string) {
	c.JSON(status, handlers.ErrorResponse{
		Error: handlers.ErrorDetail{
			Message: message,
			Type:    "invalid_request_error",
		},
	})
}
