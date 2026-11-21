package openai

import (
	"bytes"
	"context"
	stdBase64 "encoding/base64"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"mime/multipart"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/MoeForge/moego/json"
	"github.com/MoeForge/moego/ojson"
	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/interfaces"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/registry"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/thinking"
	translatorcommon "github.com/router-for-me/CLIProxyAPI/v8/internal/translator/common"
	"github.com/router-for-me/CLIProxyAPI/v8/sdk/api/handlers"
)

const geminiImagesMaxConcurrent = 4

type geminiInlineImage struct {
	MimeType string
	Data     string
}

type geminiImagesOptions struct {
	Model             string
	Prompt            string
	AspectRatio       string
	ImageSize         string
	OutputFormat      string
	OutputCompression int
	ResponseFormat    string
	N                 int
	Stream            bool
}

func isGeminiChatImageModel(model string) bool {
	return isGeminiImagesModel(model)
}

func isGeminiImagesModel(model string) bool {
	trimmed := strings.TrimSpace(model)
	if trimmed == "" || isCodexImagesToolModel(trimmed) || isXAIImagesModel(trimmed) || isOpenAICompatImagesModel(trimmed) {
		return false
	}

	prefix, baseModel := imagesModelParts(trimmed)
	cleanBase := strings.ToLower(strings.TrimSpace(thinking.ParseSuffix(baseModel).ModelName))
	if isGeminiImagesBaseModel(cleanBase) {
		return true
	}

	cleanFull := strings.ToLower(strings.TrimSpace(thinking.ParseSuffix(trimmed).ModelName))
	candidates := make([]string, 0, 4)
	seen := make(map[string]struct{}, 4)
	addCandidate := func(candidate string) {
		candidate = strings.TrimSpace(candidate)
		if candidate == "" {
			return
		}
		if _, ok := seen[candidate]; ok {
			return
		}
		seen[candidate] = struct{}{}
		candidates = append(candidates, candidate)
	}
	addCandidate(trimmed)
	if prefix != "" {
		addCandidate(cleanFull)
	}
	addCandidate(baseModel)
	addCandidate(cleanBase)

	for _, candidate := range candidates {
		info := registry.LookupModelInfo(candidate)
		if info == nil || !isGeminiFamilyModelInfo(candidate, info) {
			continue
		}
		if modelInfoHasImageOutputModality(info) {
			return true
		}
		if metaID := strings.TrimSpace(info.MetadataModelID); metaID != "" {
			cleanMeta := strings.ToLower(strings.TrimSpace(thinking.ParseSuffix(imagesModelBase(metaID)).ModelName))
			if isGeminiImagesBaseModel(cleanMeta) {
				return true
			}
			if metaInfo := registry.LookupModelInfo(metaID); metaInfo != nil && modelInfoHasImageOutputModality(metaInfo) {
				return true
			}
		}
	}
	return false
}

func isGeminiImagesBaseModel(cleanBase string) bool {
	if cleanBase == "" {
		return false
	}
	if strings.Contains(cleanBase, "nano-banana") {
		return true
	}
	if strings.HasPrefix(cleanBase, "imagen") {
		return true
	}
	return strings.HasPrefix(cleanBase, "gemini-") && strings.Contains(cleanBase, "-image")
}

func isGeminiFamilyModelInfo(modelID string, info *registry.ModelInfo) bool {
	if info == nil {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(info.Type)) {
	case "gemini", "vertex", "aistudio", "antigravity", "gemini-interactions":
		return true
	case registry.OpenAIImageModelType:
		return false
	}
	switch strings.ToLower(strings.TrimSpace(info.OwnedBy)) {
	case "google", "antigravity":
		return true
	}
	for _, provider := range registry.GetGlobalRegistry().GetModelProviders(modelID) {
		switch strings.ToLower(strings.TrimSpace(provider)) {
		case "gemini", "vertex", "aistudio", "antigravity", "gemini-interactions":
			return true
		}
	}
	return false
}

func modelInfoHasImageOutputModality(info *registry.ModelInfo) bool {
	if info == nil {
		return false
	}
	for _, mod := range info.SupportedOutputModalities {
		if strings.EqualFold(strings.TrimSpace(mod), "image") {
			return true
		}
	}
	return false
}

func newGeminiImagesOptions(model, prompt, size, aspectRatioRaw, imageSizeRaw, resolution, outputFormatRaw string, outputCompression int, responseFormat string, n int, stream bool) (geminiImagesOptions, error) {
	outputFormat := strings.ToLower(strings.TrimSpace(outputFormatRaw))
	switch outputFormat {
	case "", "png", "jpeg", "webp":
	case "jpg":
		outputFormat = "jpeg"
	default:
		return geminiImagesOptions{}, fmt.Errorf("unsupported output_format %q: expected png, jpeg, or webp", outputFormatRaw)
	}

	aspectRatio := translatorcommon.GeminiImageAspectRatio(aspectRatioRaw)
	if aspectRatio == "" {
		aspectRatio = translatorcommon.GeminiImageAspectRatioFromSize(size)
	}
	imageSize := translatorcommon.GeminiImageSizeFromOptions(imageSizeRaw, resolution, size)

	if n <= 0 {
		n = 1
	}
	if outputCompression < 0 {
		outputCompression = 0
	} else if outputCompression > 100 {
		outputCompression = 100
	}

	return geminiImagesOptions{
		Model:             strings.TrimSpace(model),
		Prompt:            strings.TrimSpace(prompt),
		AspectRatio:       aspectRatio,
		ImageSize:         imageSize,
		OutputFormat:      outputFormat,
		OutputCompression: outputCompression,
		ResponseFormat:    normalizeImagesResponseFormat(responseFormat),
		N:                 n,
		Stream:            stream,
	}, nil
}

func parseGeminiImagesOptionsFromJSON(rawJSON []byte, model, prompt, responseFormat string, stream bool) (geminiImagesOptions, error) {
	var fields [7]ojson.Result
	res := ojson.GetManyBytesNoCopy(fields[:0], rawJSON, "size", "aspect_ratio", "image_size", "resolution", "output_format", "output_compression", "n")
	return newGeminiImagesOptions(
		model,
		prompt,
		res[0].String(),
		res[1].String(),
		res[2].String(),
		res[3].String(),
		res[4].String(),
		int(res[5].Int()),
		responseFormat,
		int(res[6].Int()),
		stream,
	)
}

func parseGeminiImagesOptionsFromMultipart(c *gin.Context, model, prompt, responseFormat string, stream bool) (geminiImagesOptions, error) {
	return newGeminiImagesOptions(
		model,
		prompt,
		c.PostForm("size"),
		c.PostForm("aspect_ratio"),
		c.PostForm("image_size"),
		c.PostForm("resolution"),
		c.PostForm("output_format"),
		int(parseIntField(c.PostForm("output_compression"), 0)),
		responseFormat,
		int(parseIntField(c.PostForm("n"), 1)),
		stream,
	)
}

func hasGeminiMaskMultipart(form *multipart.Form, c *gin.Context) bool {
	if form != nil {
		if len(form.File["mask"]) > 0 || len(form.File["mask[]"]) > 0 {
			return true
		}
	}
	if c != nil && strings.TrimSpace(c.PostForm("mask")) != "" {
		return true
	}
	return false
}

func hasGeminiMaskJSON(rawJSON []byte) bool {
	for _, field := range []string{"mask", "input_image_mask"} {
		val := ojson.GetBytesNoCopy(rawJSON, field)
		if !val.Exists() || val.Type == ojson.Null {
			continue
		}
		if val.Type == ojson.String && strings.TrimSpace(val.String()) == "" {
			continue
		}
		return true
	}
	return false
}

func collectGeminiEditImageRefsFromJSON(rawJSON []byte) ([]string, error) {
	var refs []string
	var sawFileID bool

	appendObj := func(obj ojson.Result) {
		url := strings.TrimSpace(obj.Get("image_url.url").String())
		if url == "" {
			if imageURL := obj.Get("image_url"); imageURL.Type == ojson.String {
				url = strings.TrimSpace(imageURL.String())
			}
		}
		if url == "" {
			url = strings.TrimSpace(obj.Get("url").String())
		}
		if url == "" {
			url = strings.TrimSpace(obj.Get("b64_json").String())
		}
		if url != "" {
			refs = append(refs, url)
			return
		}
		if fileID := obj.Get("file_id"); fileID.Exists() && strings.TrimSpace(fileID.String()) != "" {
			sawFileID = true
		}
	}

	appendEntry := func(entry ojson.Result) {
		switch {
		case !entry.Exists() || entry.Type == ojson.Null:
		case entry.Type == ojson.String:
			if v := strings.TrimSpace(entry.String()); v != "" {
				refs = append(refs, v)
			}
		case entry.IsArray():
			for _, item := range entry.Array() {
				if item.Type == ojson.String {
					if v := strings.TrimSpace(item.String()); v != "" {
						refs = append(refs, v)
					}
				} else if item.IsObject() {
					appendObj(item)
				}
			}
		case entry.IsObject():
			appendObj(entry)
		}
	}

	appendEntry(ojson.GetBytesNoCopy(rawJSON, "image"))
	if imagesNode := ojson.GetBytesNoCopy(rawJSON, "images"); imagesNode.Exists() && imagesNode.IsArray() {
		appendEntry(imagesNode)
	}

	if len(refs) == 0 && sawFileID {
		return nil, errors.New("file_id is not supported; pass base64 or a data: URL")
	}
	if len(refs) == 0 {
		return nil, errors.New("image is required")
	}
	return refs, nil
}

func parseGeminiInlineImages(refs []string) ([]geminiInlineImage, error) {
	if len(refs) == 0 {
		return nil, nil
	}
	out := make([]geminiInlineImage, 0, len(refs))
	for _, rawRef := range refs {
		img, err := parseGeminiInlineImage(rawRef)
		if err != nil {
			return nil, err
		}
		out = append(out, img)
	}
	return out, nil
}

func parseGeminiInlineImage(rawRef string) (geminiInlineImage, error) {
	ref := strings.TrimSpace(rawRef)
	if ref == "" {
		return geminiInlineImage{}, errors.New("empty image entry")
	}
	lower := strings.ToLower(ref)
	if strings.HasPrefix(lower, "http://") || strings.HasPrefix(lower, "https://") {
		return geminiInlineImage{}, errors.New("remote image URLs are not supported; provide base64 or a data: URL")
	}
	if strings.HasPrefix(lower, "data:") {
		idx := strings.Index(lower, ";base64,")
		if idx < len("data:") {
			return geminiInlineImage{}, errors.New("invalid image data URL: expected data:<mime>;base64,<data>")
		}
		mimeType := strings.TrimSpace(ref[len("data:"):idx])
		if semi := strings.Index(mimeType, ";"); semi >= 0 {
			mimeType = strings.TrimSpace(mimeType[:semi])
		}
		if mimeType == "" || strings.EqualFold(mimeType, "application/octet-stream") {
			mimeType = "image/png"
		}
		data := strings.TrimSpace(ref[idx+len(";base64,"):])
		if data == "" {
			return geminiInlineImage{}, errors.New("invalid image data URL: base64 payload is empty")
		}
		return geminiInlineImage{MimeType: mimeType, Data: data}, nil
	}

	cleaned := strings.NewReplacer("\r", "", "\n", "", " ", "", "\t", "").Replace(ref)
	decoded, err := stdBase64.StdEncoding.DecodeString(cleaned)
	if err != nil {
		decoded, err = stdBase64.RawStdEncoding.DecodeString(cleaned)
		if err != nil {
			return geminiInlineImage{}, errors.New("invalid image input: expected base64 or a data: URL")
		}
	}
	if len(decoded) == 0 {
		return geminiInlineImage{}, errors.New("invalid image input: decoded payload is empty")
	}
	mimeType := http.DetectContentType(decoded)
	if !strings.HasPrefix(mimeType, "image/") {
		mimeType = "image/png"
	}
	return geminiInlineImage{
		MimeType: mimeType,
		Data:     stdBase64.StdEncoding.EncodeToString(decoded),
	}, nil
}

func buildGeminiImagesRequest(opts geminiImagesOptions, images []geminiInlineImage) []byte {
	parts := make([]map[string]any, 0, len(images)+1)
	for _, img := range images {
		if strings.TrimSpace(img.Data) == "" {
			continue
		}
		mimeType := strings.TrimSpace(img.MimeType)
		if mimeType == "" {
			mimeType = "image/png"
		}
		parts = append(parts, map[string]any{
			"inlineData": map[string]any{
				"mimeType": mimeType,
				"data":     img.Data,
			},
		})
	}
	parts = append(parts, map[string]any{
		"text": opts.Prompt,
	})

	genConfig := map[string]any{
		"responseModalities": []string{"IMAGE"},
	}
	imageConfig := map[string]any{}
	if opts.AspectRatio != "" {
		imageConfig["aspectRatio"] = opts.AspectRatio
	}
	if opts.ImageSize != "" {
		imageConfig["imageSize"] = opts.ImageSize
	}
	if len(imageConfig) > 0 {
		genConfig["imageConfig"] = imageConfig
	}

	payload := map[string]any{
		"contents": []map[string]any{
			{
				"role":  "user",
				"parts": parts,
			},
		},
		"generationConfig": genConfig,
	}
	encoded, _ := json.Marshal(payload)
	return encoded
}

func imageOutputFormatFromMIMEType(mimeType string) string {
	lower := strings.ToLower(strings.TrimSpace(mimeType))
	if idx := strings.Index(lower, ";"); idx >= 0 {
		lower = strings.TrimSpace(lower[:idx])
	}
	switch lower {
	case "image/jpeg", "image/jpg":
		return "jpeg"
	case "image/webp":
		return "webp"
	case "image/png", "":
		return "png"
	default:
		trimmed := strings.TrimPrefix(lower, "image/")
		if trimmed == "" || trimmed == lower {
			return "png"
		}
		return trimmed
	}
}

func transcodeBase64Image(b64Data, srcFormat, dstFormat string, quality int) (string, string, error) {
	if dstFormat == "" || srcFormat == dstFormat {
		return b64Data, srcFormat, nil
	}
	if dstFormat != "png" && dstFormat != "jpeg" {
		return b64Data, srcFormat, nil
	}
	raw, err := stdBase64.StdEncoding.DecodeString(strings.TrimSpace(b64Data))
	if err != nil {
		raw, err = stdBase64.RawStdEncoding.DecodeString(strings.TrimSpace(b64Data))
		if err != nil {
			return "", "", fmt.Errorf("decode upstream image for %s conversion failed: %w", dstFormat, err)
		}
	}
	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return "", "", fmt.Errorf("decode image pixels for %s conversion failed: %w", dstFormat, err)
	}
	var buf bytes.Buffer
	switch dstFormat {
	case "png":
		if err := png.Encode(&buf, img); err != nil {
			return "", "", fmt.Errorf("encode png image failed: %w", err)
		}
	case "jpeg":
		q := 90
		if quality >= 1 && quality <= 100 {
			q = quality
		}
		if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: q}); err != nil {
			return "", "", fmt.Errorf("encode jpeg image failed: %w", err)
		}
	}
	return stdBase64.StdEncoding.EncodeToString(buf.Bytes()), dstFormat, nil
}

func extractImagesFromGeminiResponse(resp []byte, requestedOutputFormat string, outputCompression int) ([]imageCallResult, []byte, error) {
	if len(bytes.TrimSpace(resp)) == 0 {
		return nil, nil, errors.New("empty response from Gemini upstream")
	}

	var results []imageCallResult
	var textNotes []string

	for _, candidate := range ojson.GetBytesNoCopy(resp, "candidates").Array() {
		for _, part := range candidate.Get("content.parts").Array() {
			if part.Get("thought").Bool() {
				continue
			}
			inline := part.Get("inlineData")
			if !inline.Exists() {
				inline = part.Get("inline_data")
			}
			if !inline.Exists() {
				if text := strings.TrimSpace(part.Get("text").String()); text != "" {
					textNotes = append(textNotes, text)
				}
				continue
			}

			data := strings.TrimSpace(inline.Get("data").String())
			if data == "" {
				continue
			}
			mimeType := strings.TrimSpace(inline.Get("mimeType").String())
			if mimeType == "" {
				mimeType = strings.TrimSpace(inline.Get("mime_type").String())
			}
			outputFmt := imageOutputFormatFromMIMEType(mimeType)
			if requestedOutputFormat != "" && outputFmt != requestedOutputFormat {
				convertedData, convertedFmt, errConv := transcodeBase64Image(data, outputFmt, requestedOutputFormat, outputCompression)
				if errConv != nil {
					return nil, nil, errConv
				}
				data = convertedData
				outputFmt = convertedFmt
			}
			results = append(results, imageCallResult{
				Result:       data,
				OutputFormat: outputFmt,
			})
		}
	}

	if len(results) == 0 {
		if len(textNotes) > 0 {
			msg := strings.Join(textNotes, " ")
			if len(msg) > 200 {
				msg = msg[:200] + "..."
			}
			return nil, nil, fmt.Errorf("no images returned in Gemini response: %s", msg)
		}
		return nil, nil, errors.New("no images returned in Gemini response")
	}

	usageRaw := buildOpenAIImagesUsageFromGeminiResponse(resp)
	return results, usageRaw, nil
}

func firstGeminiResult(root []byte, paths ...string) ojson.Result {
	for _, path := range paths {
		if res := ojson.GetBytesNoCopy(root, path); res.Exists() {
			return res
		}
	}
	return ojson.Result{}
}

func firstGeminiUsageInt(usage ojson.Result, keys ...string) int64 {
	for _, key := range keys {
		if v := usage.Get(key); v.Exists() {
			return v.Int()
		}
	}
	return 0
}

func buildOpenAIImagesUsageFromGeminiResponse(resp []byte) []byte {
	usage := firstGeminiResult(resp, "usageMetadata", "usage_metadata", "cpaUsageMetadata")
	if !usage.Exists() || !usage.IsObject() {
		return nil
	}
	inputTokens := firstGeminiUsageInt(usage, "promptTokenCount", "prompt_token_count")
	outputTokens := firstGeminiUsageInt(usage, "candidatesTokenCount", "candidates_token_count")
	thoughtTokens := firstGeminiUsageInt(usage, "thoughtsTokenCount", "thoughts_token_count")
	totalTokens := firstGeminiUsageInt(usage, "totalTokenCount", "total_token_count")
	if totalTokens == 0 {
		totalTokens = inputTokens + outputTokens + thoughtTokens
	}
	if inputTokens == 0 && outputTokens == 0 && totalTokens == 0 {
		return nil
	}

	var textTokens, imageTokens int64
	details := usage.Get("promptTokensDetails")
	if !details.Exists() {
		details = usage.Get("prompt_tokens_details")
	}
	if details.Exists() && details.IsArray() {
		for _, item := range details.Array() {
			count := item.Get("tokenCount").Int()
			if count == 0 {
				count = item.Get("token_count").Int()
			}
			switch strings.ToUpper(strings.TrimSpace(item.Get("modality").String())) {
			case "TEXT":
				textTokens += count
			case "IMAGE":
				imageTokens += count
			}
		}
	}
	if textTokens == 0 && imageTokens == 0 && inputTokens > 0 {
		textTokens = inputTokens
	}

	usageDoc := ojson.NewDocString(`{"total_tokens":0,"input_tokens":0,"output_tokens":0,"input_tokens_details":{"text_tokens":0,"image_tokens":0}}`)
	_ = usageDoc.Set("total_tokens", totalTokens)
	_ = usageDoc.Set("input_tokens", inputTokens)
	_ = usageDoc.Set("output_tokens", outputTokens)
	_ = usageDoc.Set("input_tokens_details.text_tokens", textTokens)
	_ = usageDoc.Set("input_tokens_details.image_tokens", imageTokens)
	return usageDoc.Bytes()
}

func aggregateGeminiImagesUsage(usages [][]byte) []byte {
	var totalTokens, inputTokens, outputTokens, textTokens, imageTokens int64
	hasAny := false
	for _, raw := range usages {
		if len(raw) == 0 {
			continue
		}
		hasAny = true
		totalTokens += ojson.GetBytesNoCopy(raw, "total_tokens").Int()
		inputTokens += ojson.GetBytesNoCopy(raw, "input_tokens").Int()
		outputTokens += ojson.GetBytesNoCopy(raw, "output_tokens").Int()
		textTokens += ojson.GetBytesNoCopy(raw, "input_tokens_details.text_tokens").Int()
		imageTokens += ojson.GetBytesNoCopy(raw, "input_tokens_details.image_tokens").Int()
	}
	if !hasAny {
		return nil
	}
	usageDoc := ojson.NewDocString(`{"total_tokens":0,"input_tokens":0,"output_tokens":0,"input_tokens_details":{"text_tokens":0,"image_tokens":0}}`)
	_ = usageDoc.Set("total_tokens", totalTokens)
	_ = usageDoc.Set("input_tokens", inputTokens)
	_ = usageDoc.Set("output_tokens", outputTokens)
	_ = usageDoc.Set("input_tokens_details.text_tokens", textTokens)
	_ = usageDoc.Set("input_tokens_details.image_tokens", imageTokens)
	return usageDoc.Bytes()
}

func writeGeminiImagesBadRequest(c *gin.Context, msg string) {
	c.JSON(http.StatusBadRequest, handlers.ErrorResponse{
		Error: handlers.ErrorDetail{
			Message: "Invalid request: " + msg,
			Type:    "invalid_request_error",
		},
	})
}

func (h *OpenAIAPIHandler) handleGeminiImagesGenerationJSON(c *gin.Context, rawJSON []byte, model, prompt, responseFormat string, stream bool) {
	opts, err := parseGeminiImagesOptionsFromJSON(rawJSON, model, prompt, responseFormat, stream)
	if err != nil {
		writeGeminiImagesBadRequest(c, err.Error())
		return
	}
	geminiReq := buildGeminiImagesRequest(opts, nil)
	h.executeGeminiImagesRequest(c, opts, geminiReq, imagesGenerationsPath, "image_generation")
}

func (h *OpenAIAPIHandler) handleGeminiImagesEditMultipart(c *gin.Context, form *multipart.Form, model, prompt string, imageDataURLs []string, responseFormat string, stream bool) {
	if hasGeminiMaskMultipart(form, c) {
		writeGeminiImagesBadRequest(c, "mask is not supported for this model")
		return
	}
	opts, err := parseGeminiImagesOptionsFromMultipart(c, model, prompt, responseFormat, stream)
	if err != nil {
		writeGeminiImagesBadRequest(c, err.Error())
		return
	}
	inlineImages, err := parseGeminiInlineImages(imageDataURLs)
	if err != nil {
		writeGeminiImagesBadRequest(c, err.Error())
		return
	}
	geminiReq := buildGeminiImagesRequest(opts, inlineImages)
	h.executeGeminiImagesRequest(c, opts, geminiReq, imagesEditsPath, "image_edit")
}

func (h *OpenAIAPIHandler) handleGeminiImagesEditJSON(c *gin.Context, rawJSON []byte, model, prompt, responseFormat string, stream bool) {
	if hasGeminiMaskJSON(rawJSON) {
		writeGeminiImagesBadRequest(c, "mask is not supported for this model")
		return
	}
	refs, err := collectGeminiEditImageRefsFromJSON(rawJSON)
	if err != nil {
		writeGeminiImagesBadRequest(c, err.Error())
		return
	}
	inlineImages, err := parseGeminiInlineImages(refs)
	if err != nil {
		writeGeminiImagesBadRequest(c, err.Error())
		return
	}
	opts, err := parseGeminiImagesOptionsFromJSON(rawJSON, model, prompt, responseFormat, stream)
	if err != nil {
		writeGeminiImagesBadRequest(c, err.Error())
		return
	}
	geminiReq := buildGeminiImagesRequest(opts, inlineImages)
	h.executeGeminiImagesRequest(c, opts, geminiReq, imagesEditsPath, "image_edit")
}

func (h *OpenAIAPIHandler) executeGeminiImagesRequest(c *gin.Context, opts geminiImagesOptions, geminiReq []byte, requestPath, streamPrefix string) {
	if opts.Stream {
		h.streamGeminiImages(c, opts, geminiReq, requestPath, streamPrefix)
		return
	}
	h.collectGeminiImages(c, opts, geminiReq, requestPath)
}

func (h *OpenAIAPIHandler) executeGeminiImagesSingle(cliCtx context.Context, model, requestPath string, geminiReq []byte) ([]byte, http.Header, *interfaces.ErrorMessage) {
	execCtx := handlers.WithOriginalRequest(cliCtx, geminiReq)
	resp, errMsg := h.ExecuteProtocolWithAuthManager(execCtx, handlers.ProtocolExecutionRequest{
		EntryProtocol: "gemini",
		ExitProtocol:  "gemini",
		Model:         model,
		Body:          geminiReq,
		Path:          requestPath,
	})
	if errMsg != nil {
		return nil, nil, errMsg
	}
	return resp.Body, resp.Headers, nil
}

func (h *OpenAIAPIHandler) executeGeminiImagesBatch(cliCtx context.Context, opts geminiImagesOptions, geminiReq []byte, requestPath string) ([]imageCallResult, []byte, http.Header, *interfaces.ErrorMessage) {
	count := opts.N
	if count <= 0 {
		count = 1
	}

	if count == 1 {
		resp, headers, errMsg := h.executeGeminiImagesSingle(cliCtx, opts.Model, requestPath, geminiReq)
		if errMsg != nil {
			return nil, nil, nil, errMsg
		}
		results, usageRaw, err := extractImagesFromGeminiResponse(resp, opts.OutputFormat, opts.OutputCompression)
		if err != nil {
			return nil, nil, nil, &interfaces.ErrorMessage{StatusCode: http.StatusBadGateway, Error: err}
		}
		return results, usageRaw, headers, nil
	}

	maxConcurrent := geminiImagesMaxConcurrent
	if count < maxConcurrent {
		maxConcurrent = count
	}
	type callResult struct {
		resp    []byte
		headers http.Header
		errMsg  *interfaces.ErrorMessage
	}
	callResults := make([]callResult, count)
	var wg sync.WaitGroup
	for worker := 0; worker < maxConcurrent; worker++ {
		wg.Add(1)
		go func(first int) {
			defer wg.Done()
			for idx := first; idx < count; idx += maxConcurrent {
				if cliCtx.Err() != nil {
					return
				}
				resp, headers, errMsg := h.executeGeminiImagesSingle(cliCtx, opts.Model, requestPath, geminiReq)
				callResults[idx] = callResult{resp: resp, headers: headers, errMsg: errMsg}
			}
		}(worker)
	}
	wg.Wait()

	if errCtx := cliCtx.Err(); errCtx != nil {
		return nil, nil, nil, &interfaces.ErrorMessage{StatusCode: http.StatusRequestTimeout, Error: errCtx}
	}

	var allResults []imageCallResult
	var usages [][]byte
	var upstreamHeaders http.Header
	for _, cr := range callResults {
		if cr.errMsg != nil {
			return nil, nil, nil, cr.errMsg
		}
		if cr.headers != nil && upstreamHeaders == nil {
			upstreamHeaders = cr.headers
		}
		results, usageRaw, err := extractImagesFromGeminiResponse(cr.resp, opts.OutputFormat, opts.OutputCompression)
		if err != nil {
			return nil, nil, nil, &interfaces.ErrorMessage{StatusCode: http.StatusBadGateway, Error: err}
		}
		allResults = append(allResults, results...)
		if len(usageRaw) > 0 {
			usages = append(usages, usageRaw)
		}
	}

	if len(allResults) == 0 {
		return nil, nil, nil, &interfaces.ErrorMessage{StatusCode: http.StatusBadGateway, Error: errors.New("no images returned")}
	}
	return allResults, aggregateGeminiImagesUsage(usages), upstreamHeaders, nil
}

func (h *OpenAIAPIHandler) collectGeminiImages(c *gin.Context, opts geminiImagesOptions, geminiReq []byte, requestPath string) {
	c.Header("Content-Type", "application/json")

	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
	stopKeepAlive := h.StartNonStreamingKeepAlive(c, cliCtx)
	defer stopKeepAlive()

	allResults, usageRaw, upstreamHeaders, errMsg := h.executeGeminiImagesBatch(cliCtx, opts, geminiReq, requestPath)
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

	out, err := buildImagesAPIResponse(allResults, time.Now().Unix(), usageRaw, allResults[0], opts.ResponseFormat)
	if err != nil {
		errOut := &interfaces.ErrorMessage{StatusCode: http.StatusBadGateway, Error: err}
		h.WriteErrorResponse(c, errOut)
		cliCancel(err)
		return
	}

	handlers.WriteUpstreamHeaders(c.Writer.Header(), upstreamHeaders)
	_, _ = c.Writer.Write(out)
	cliCancel(nil)
}

func (h *OpenAIAPIHandler) streamGeminiImages(c *gin.Context, opts geminiImagesOptions, geminiReq []byte, requestPath, streamPrefix string) {
	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: "Streaming not supported",
				Type:    "server_error",
			},
		})
		return
	}

	cliCtx, cliCancel := h.GetContextWithCancel(h, c, context.Background())
	type batchResult struct {
		results         []imageCallResult
		usageRaw        []byte
		upstreamHeaders http.Header
		errMsg          *interfaces.ErrorMessage
	}
	resultChan := make(chan batchResult, 1)
	go func() {
		results, usageRaw, headers, errMsg := h.executeGeminiImagesBatch(cliCtx, opts, geminiReq, requestPath)
		resultChan <- batchResult{
			results:         results,
			usageRaw:        usageRaw,
			upstreamHeaders: headers,
			errMsg:          errMsg,
		}
	}()

	rawStopKeepAlive, keepAliveC := h.newImagesPreFirstTokenKeepAliveTicker()
	stopKeepAlive := func() {
		if rawStopKeepAlive != nil {
			rawStopKeepAlive()
			rawStopKeepAlive = nil
			keepAliveC = nil
		}
	}
	defer stopKeepAlive()

	streamStarted := false
	writeError := func(errMsg *interfaces.ErrorMessage) {
		if streamStarted {
			writeImagesStreamErrorEvent(c, errMsg)
			flusher.Flush()
		} else {
			h.WriteErrorResponse(c, errMsg)
		}
		if errMsg != nil && errMsg.Error != nil {
			cliCancel(errMsg.Error)
		} else {
			cliCancel(nil)
		}
	}

	for {
		select {
		case <-c.Request.Context().Done():
			cliCancel(c.Request.Context().Err())
			<-resultChan
			return
		case <-keepAliveC:
			setImagesSSEHeaders(c)
			writeImagesStreamKeepAlive(c, flusher)
			streamStarted = true
		case res := <-resultChan:
			stopKeepAlive()
			if res.errMsg != nil {
				writeError(res.errMsg)
				return
			}

			setImagesSSEHeaders(c)
			handlers.WriteUpstreamHeaders(c.Writer.Header(), res.upstreamHeaders)

			eventName := streamPrefix + ".completed"
			responseFormat := normalizeImagesResponseFormat(opts.ResponseFormat)
			for i, img := range res.results {
				dataDoc := ojson.NewDoc([]byte(`{"type":""}`))
				_ = dataDoc.Set("type", eventName)
				if responseFormat == "url" {
					mt := mimeTypeFromOutputFormat(img.OutputFormat)
					_ = dataDoc.Set("url", "data:"+mt+";base64,"+img.Result)
				} else {
					_ = dataDoc.Set("b64_json", img.Result)
				}
				if img.OutputFormat != "" {
					_ = dataDoc.Set("output_format", img.OutputFormat)
				}
				if i == len(res.results)-1 && len(res.usageRaw) > 0 && json.Valid(res.usageRaw) {
					_ = dataDoc.SetRawBytes("usage", res.usageRaw)
				}
				_, _ = fmt.Fprintf(c.Writer, "event: %s\n", eventName)
				_, _ = fmt.Fprintf(c.Writer, "data: %s\n\n", string(dataDoc.Bytes()))
				flusher.Flush()
				streamStarted = true
			}
			cliCancel(nil)
			return
		}
	}
}
