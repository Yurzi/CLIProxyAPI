package openai

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/MoeForge/moego/ojson"
	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/interfaces"
	translatorcommon "github.com/router-for-me/CLIProxyAPI/v8/internal/translator/common"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/api/handlers"
)

func geminiImageAspectRatioFromSize(size string) string {
	return translatorcommon.GeminiImageAspectRatioFromSize(size)
}

func buildGeminiChatImagesRequest(prompt, model, size string) []byte {
	req := ojson.NewDocString(`{}`)
	_ = req.Set("model", strings.TrimSpace(model))
	messages := []map[string]any{
		{
			"role":    "user",
			"content": strings.TrimSpace(prompt),
		},
	}
	_ = req.Set("messages", messages)
	_ = req.Set("modalities", []string{"image", "text"})
	if ar := geminiImageAspectRatioFromSize(size); ar != "" {
		_ = req.Set("image_config.aspect_ratio", ar)
	}
	return req.Bytes()
}

func buildGeminiChatImagesEditRequest(prompt, model, size string, images []string) []byte {
	contentParts := make([]map[string]any, 0, len(images)+1)
	for _, img := range images {
		img = strings.TrimSpace(img)
		if img == "" {
			continue
		}
		contentParts = append(contentParts, map[string]any{
			"type": "image_url",
			"image_url": map[string]any{
				"url": img,
			},
		})
	}
	contentParts = append(contentParts, map[string]any{
		"type": "text",
		"text": strings.TrimSpace(prompt),
	})

	req := ojson.NewDocString(`{}`)
	_ = req.Set("model", strings.TrimSpace(model))
	messages := []map[string]any{
		{
			"role":    "user",
			"content": contentParts,
		},
	}
	_ = req.Set("messages", messages)
	_ = req.Set("modalities", []string{"image", "text"})
	if ar := geminiImageAspectRatioFromSize(size); ar != "" {
		_ = req.Set("image_config.aspect_ratio", ar)
	}
	return req.Bytes()
}

func extractImagesFromChatCompletions(resp []byte) ([]imageCallResult, int64, error) {
	createdAt := ojson.GetBytesNoCopy(resp, "created").Int()
	imagesResult := ojson.GetBytesNoCopy(resp, "choices.0.message.images")
	if !imagesResult.Exists() || !imagesResult.IsArray() || len(imagesResult.Array()) == 0 {
		return nil, 0, errors.New("no images returned in chat completions response")
	}

	var results []imageCallResult
	for _, item := range imagesResult.Array() {
		rawURL := strings.TrimSpace(item.Get("image_url.url").String())
		if rawURL == "" {
			rawURL = strings.TrimSpace(item.Get("url").String())
		}
		if rawURL == "" {
			continue
		}

		var b64Data string
		var format string
		if idx := strings.Index(rawURL, ";base64,"); idx >= 0 {
			meta := rawURL[:idx]
			b64Data = rawURL[idx+len(";base64,"):]
			mimeType := strings.TrimPrefix(meta, "data:")
			if semi := strings.Index(mimeType, ";"); semi >= 0 {
				mimeType = mimeType[:semi]
			}
			mimeType = strings.TrimSpace(mimeType)
			format = strings.TrimPrefix(mimeType, "image/")
			if format == "" {
				format = mimeType
			}
		} else {
			b64Data = rawURL
		}

		results = append(results, imageCallResult{
			Result:       b64Data,
			OutputFormat: format,
		})
	}

	if len(results) == 0 {
		return nil, createdAt, errors.New("no valid images found in chat completions response")
	}

	return results, createdAt, nil
}

func (h *OpenAIAPIHandler) collectGeminiChatImages(c *gin.Context, chatReq []byte, model, responseFormat string, n int) {
	c.Header("Content-Type", "application/json")

	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
	stopKeepAlive := h.StartNonStreamingKeepAlive(c, cliCtx)
	defer stopKeepAlive()

	model = strings.TrimSpace(model)
	var allResults []imageCallResult
	var lastCreatedAt int64
	var upstreamHeaders http.Header

	if n <= 1 {
		resp, headers, errMsg := h.ExecuteWithAuthManager(cliCtx, "openai", model, chatReq, "")
		stopKeepAlive()
		if errMsg != nil {
			h.WriteErrorResponse(c, errMsg)
			if errMsg.Error != nil {
				cliCancel(errMsg.Error)
			} else {
				cliCancel(nil)
			}
			return
		}
		upstreamHeaders = headers
		results, createdAt, err := extractImagesFromChatCompletions(resp)
		if err != nil {
			errMsg := &interfaces.ErrorMessage{StatusCode: http.StatusBadGateway, Error: err}
			h.WriteErrorResponse(c, errMsg)
			cliCancel(err)
			return
		}
		allResults = results
		lastCreatedAt = createdAt
	} else {
		count := n
		maxConcurrent := 4
		if count < maxConcurrent {
			maxConcurrent = count
		}
		var wg sync.WaitGroup
		type callResult struct {
			resp    []byte
			headers http.Header
			errMsg  *interfaces.ErrorMessage
		}
		callResults := make([]callResult, count)

		for worker := 0; worker < maxConcurrent; worker++ {
			wg.Add(1)
			go func(first int) {
				defer wg.Done()
				for idx := first; idx < count; idx += maxConcurrent {
					if cliCtx.Err() != nil {
						return
					}
					resp, headers, errMsg := h.ExecuteWithAuthManager(cliCtx, "openai", model, chatReq, "")
					callResults[idx] = callResult{resp: resp, headers: headers, errMsg: errMsg}
				}
			}(worker)
		}
		wg.Wait()
		stopKeepAlive()

		if cliCtx.Err() != nil {
			cliCancel(cliCtx.Err())
			return
		}

		for _, cr := range callResults {
			if cr.errMsg != nil {
				h.WriteErrorResponse(c, cr.errMsg)
				if cr.errMsg.Error != nil {
					cliCancel(cr.errMsg.Error)
				} else {
					cliCancel(nil)
				}
				return
			}
			if cr.headers != nil && upstreamHeaders == nil {
				upstreamHeaders = cr.headers
			}
			results, createdAt, err := extractImagesFromChatCompletions(cr.resp)
			if err != nil {
				errMsg := &interfaces.ErrorMessage{StatusCode: http.StatusBadGateway, Error: err}
				h.WriteErrorResponse(c, errMsg)
				cliCancel(err)
				return
			}
			allResults = append(allResults, results...)
			if createdAt > 0 {
				lastCreatedAt = createdAt
			}
		}
	}

	if len(allResults) == 0 {
		err := errors.New("no images returned")
		h.WriteErrorResponse(c, &interfaces.ErrorMessage{StatusCode: http.StatusBadGateway, Error: err})
		cliCancel(err)
		return
	}
	if lastCreatedAt == 0 {
		lastCreatedAt = time.Now().Unix()
	}

	out, err := buildImagesAPIResponse(allResults, lastCreatedAt, nil, allResults[0], responseFormat)
	if err != nil {
		errMsg := &interfaces.ErrorMessage{StatusCode: http.StatusBadGateway, Error: err}
		h.WriteErrorResponse(c, errMsg)
		cliCancel(err)
		return
	}

	handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)
	_, _ = c.Writer.Write(out)
	cliCancel(nil)
}
