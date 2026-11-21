package chat_completions

import (
	"strings"

	"github.com/MoeForge/moego/ojson"
	translatorcommon "github.com/router-for-me/CLIProxyAPI/v8/internal/translator/common"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/util"
)

func ConvertOpenAIRequestToInteractions(modelName string, inputRawJSON []byte, stream bool) ([]byte, error) {
	return convertOpenAIRequestToInteractions(modelName, inputRawJSON, stream)

}

// convertOpenAIRequestToInteractions also reports a file or audio part that
// Interactions cannot receive when it leaves a user turn with nothing to send.
func convertOpenAIRequestToInteractions(modelName string, inputRawJSON []byte, stream bool) ([]byte, error) {
	root := util.ParseGJSONBytesNoCopy(inputRawJSON)
	outDoc := ojson.NewDocString(`{"model":"","input":[]}`)
	model := firstNonEmpty(modelName, root.Get("model").String())
	_ = outDoc.Set("model", model)
	if streamValue, ok := openAIRequestStreamValue(root, stream); ok {
		_ = outDoc.Set("stream", streamValue)
	}
	if previousResponseID := firstNonEmpty(root.Get("previous_response_id").String(), root.Get("previous_interaction_id").String()); previousResponseID != "" {
		_ = outDoc.Set("previous_interaction_id", previousResponseID)
	}
	if environmentID := firstNonEmpty(root.Get("environment_id").String(), root.Get("environment.id").String()); environmentID != "" {
		_ = outDoc.Set("environment_id", environmentID)
	}
	if agentConfig := root.Get("agent_config"); agentConfig.Exists() {
		_ = outDoc.SetRaw("agent_config", agentConfig.Raw)
	}
	forAntigravity := isAntigravityModel(model)
	errMessages := appendOpenAIMessagesToInteractionsDoc(outDoc, root.Get("messages"), forAntigravity)
	copyOpenAIChatGenerationConfigToInteractionsDoc(outDoc, root, model)
	appendOpenAIChatToolsToInteractionsDoc(outDoc, root.Get("tools"), forAntigravity)
	return outDoc.Bytes(), errMessages
}

func openAIRequestStreamValue(root ojson.Result, stream bool) (bool, bool) {
	if value := root.Get("stream"); value.Exists() {
		return value.Bool(), true
	}
	if stream {
		return true, true
	}
	return false, false
}

func appendOpenAIMessagesToInteractionsDoc(outDoc *ojson.Doc, messages ojson.Result, forAntigravity bool) error {
	if !messages.Exists() || !messages.IsArray() {
		return nil
	}
	var drops translatorcommon.UserTurnDrops
	inputItems := translatorcommon.NewRawArrayItems(messages.Get("#").Int())
	var systemBuilder strings.Builder
	toolNamesByID := make(map[string]string)
	messages.ForEach(func(_, message ojson.Result) bool {
		role := strings.ToLower(strings.TrimSpace(message.Get("role").String()))
		switch role {
		case "system", "developer":
			if text := openAIChatContentText(message.Get("content")); text != "" {
				if systemBuilder.Len() > 0 {
					systemBuilder.WriteByte('\n')
				}
				systemBuilder.WriteString(text)
			}
		default:
			appendOpenAIMessageToInteractions(&inputItems, message, forAntigravity, toolNamesByID, &drops)
		}
		return true
	})
	if systemBuilder.Len() > 0 {
		_ = outDoc.Set("system_instruction", systemBuilder.String())
	}
	translatorcommon.SetRawArrayItemsDoc(outDoc, "input", inputItems)
	return drops.Err()
}

func appendOpenAIMessageToInteractions(items *[][]byte, message ojson.Result, forAntigravity bool, toolNamesByID map[string]string, drops *translatorcommon.UserTurnDrops) {
	role := strings.ToLower(strings.TrimSpace(message.Get("role").String()))
	switch role {
	case "assistant":
		if reasoning := message.Get("reasoning_content"); reasoning.Exists() {
			for _, text := range openAIReasoningTexts(reasoning) {
				*items = append(*items, interactionsTextStep("thought", text))
			}
		}
		if step, ok := openAIChatContentStep("model_output", message.Get("content"), nil); ok {
			*items = append(*items, step)
		}
		if toolCalls := message.Get("tool_calls"); toolCalls.Exists() && toolCalls.IsArray() {
			toolCalls.ForEach(func(_, toolCall ojson.Result) bool {
				if id := toolCall.Get("id").String(); id != "" {
					if name := toolCall.Get("function.name").String(); name != "" && toolNamesByID != nil {
						toolNamesByID[id] = name
					}
				}
				if step, ok := openAIToolCallToInteractionsStep(toolCall, forAntigravity); ok {
					*items = append(*items, step)
				}
				return true
			})
		}
	case "tool", "function":
		*items = append(*items, openAIToolResultToInteractions(message, forAntigravity, toolNamesByID))
	default:
		if step, ok := openAIChatContentStep("user_input", message.Get("content"), drops); ok {
			*items = append(*items, step)
		}
	}
}

// openAIChatContentStep converts one message content into an Interactions step.
// A non-nil drops closes the user turn: an attachment that cannot be sent is
// recorded, and text or any other sendable part beside it keeps the turn alive.
// An empty text part is forwarded but does not count as sendable.
func openAIChatContentStep(stepType string, content ojson.Result, drops *translatorcommon.UserTurnDrops) ([]byte, bool) {
	contentItems := make([][]byte, 0, 4)
	sendable := 0
	if drops != nil {
		defer func() { drops.EndTurn(sendable) }()
	}
	if content.Type == ojson.String {
		if content.String() == "" {
			return nil, false
		}
		part := []byte(`{"type":"text","text":""}`)
		part, _ = ojson.SetBytes(part, "text", content.String())
		contentItems = append(contentItems, part)
		sendable++
	} else {
		appendPart := func(part ojson.Result) {
			converted, ok := openAIChatContentPartToInteractions(part)
			if !ok {
				if partType := openAIChatPartType(part); drops != nil && isOpenAIChatAttachmentType(partType) {
					drops.Drop(partType)
				}
				return
			}
			contentItems = append(contentItems, converted)
			if convertedType := ojson.GetBytesNoCopy(converted, "type").String(); convertedType != "text" || ojson.GetBytesNoCopy(converted, "text").String() != "" {
				sendable++
			}
		}
		if content.IsArray() {
			content.ForEach(func(_, part ojson.Result) bool {
				appendPart(part)
				return true
			})
		} else if content.IsObject() {
			appendPart(content)
		}
	}
	if len(contentItems) == 0 {
		return nil, false
	}
	stepDoc := ojson.NewDocString(`{"type":"","content":[]}`)
	_ = stepDoc.Set("type", stepType)
	_ = stepDoc.SetRawBytes("content", translatorcommon.JoinRawArray(contentItems))
	return stepDoc.Bytes(), true
}

func openAIChatPartType(part ojson.Result) string {
	partType := strings.ToLower(strings.TrimSpace(part.Get("type").String()))
	if partType == "" && part.Get("text").Exists() {
		partType = "text"
	}
	return partType
}

// isOpenAIChatAttachmentType reports the part types whose loss must be refused
// when they leave a user turn empty, instead of being skipped silently.
func isOpenAIChatAttachmentType(partType string) bool {
	switch partType {
	case "file", "input_file", "document", "input_audio", "audio":
		return true
	}
	return false
}

func openAIChatContentPartToInteractions(part ojson.Result) ([]byte, bool) {
	partType := openAIChatPartType(part)
	switch partType {
	case "text", "input_text", "output_text":
		out := []byte(`{"type":"text","text":""}`)
		out, _ = ojson.SetBytes(out, "text", part.Get("text").String())
		return out, true
	case "image_url", "input_image", "image":
		return openAIChatImagePartToInteractions(part), true
	case "input_audio", "audio":
		outDoc := ojson.NewDocString(`{"type":"audio","data":""}`)
		audio := part.Get("input_audio")
		data := firstNonEmpty(audio.Get("data").String(), part.Get("data").String())
		if data == "" {
			return nil, false
		}
		_ = outDoc.Set("data", data)
		if format := firstNonEmpty(audio.Get("format").String(), part.Get("format").String()); format != "" {
			_ = outDoc.Set("mime_type", openAIInputAudioMIMEType(format))
		}
		return outDoc.Bytes(), true
	case "file", "input_file", "document":
		file := part.Get("file")
		filename := firstNonEmpty(file.Get("filename").String(), part.Get("filename").String())
		fallbackMIMEType := firstNonEmpty(file.Get("mime_type").String(), file.Get("mimeType").String(), part.Get("mime_type").String(), part.Get("mimeType").String())
		fileData := firstNonEmpty(file.Get("file_data").String(), part.Get("file_data").String(), part.Get("data").String())
		fileURL := firstNonEmpty(file.Get("file_url").String(), part.Get("file_url").String(), part.Get("url").String())
		outDoc := ojson.NewDocString(`{"type":"document"}`)
		if filename != "" {
			_ = outDoc.Set("filename", filename)
		}
		hasContent := false
		if mimeType, data, ok := translatorcommon.NormalizeOpenAIFileData(filename, fallbackMIMEType, fileData); ok {
			_ = outDoc.Set("mime_type", mimeType)
			_ = outDoc.Set("data", data)
			hasContent = true
		}
		if fileURL != "" {
			_ = outDoc.Set("file_url", fileURL)
			hasContent = true
		}
		return outDoc.Bytes(), hasContent
	}
	return nil, false
}

func openAIChatImagePartToInteractions(part ojson.Result) []byte {
	outDoc := ojson.NewDocString(`{"type":"image"}`)
	imageURL := firstNonEmpty(part.Get("image_url.url").String(), part.Get("image_url").String(), part.Get("url").String())
	if mimeType, data, ok := openAIChatParseDataURL(imageURL); ok {
		_ = outDoc.Set("mime_type", mimeType)
		_ = outDoc.Set("data", data)
		return outDoc.Bytes()
	}
	if data := part.Get("data").String(); data != "" {
		_ = outDoc.Set("data", data)
		if mimeType := part.Get("mime_type").String(); mimeType != "" {
			_ = outDoc.Set("mime_type", mimeType)
		}
		return outDoc.Bytes()
	}
	if imageURL != "" {
		_ = outDoc.Set("image_url", imageURL)
	}
	return outDoc.Bytes()
}

func openAIToolResultToInteractions(message ojson.Result, forAntigravity bool, toolNamesByID map[string]string) []byte {
	outDoc := ojson.NewDocString(`{"type":"function_result","result":""}`)
	callID := firstNonEmpty(message.Get("tool_call_id").String(), message.Get("id").String())
	if callID != "" {
		_ = outDoc.Set("call_id", callID)
	}
	name := message.Get("name").String()
	if name == "" && callID != "" && toolNamesByID != nil {
		name = toolNamesByID[callID]
	}
	if name != "" {
		if forAntigravity {
			name = translatorcommon.AntigravityToolNameToUpstream(name)
		}
		_ = outDoc.Set("name", name)
	}
	content := message.Get("content")
	if content.Exists() && content.Type == ojson.String {
		_ = outDoc.Set("result", content.String())
	} else if content.Exists() {
		_ = outDoc.SetRaw("result", content.Raw)
	}
	return outDoc.Bytes()
}

func isAntigravityModel(model string) bool {
	return strings.Contains(strings.ToLower(model), "antigravity")
}

func copyOpenAIChatGenerationConfigToInteractionsDoc(outDoc *ojson.Doc, root ojson.Result, model string) {
	if isAntigravityModel(model) {
		if maxOutputTokens := firstExisting(root.Get("max_completion_tokens"), root.Get("max_tokens"), root.Get("max_output_tokens")); maxOutputTokens.Exists() && !root.Get("agent_config.max_total_tokens").Exists() {
			_ = outDoc.Set("agent_config.max_total_tokens", maxOutputTokens.Int())
		}
	} else {
		copyNumberDoc(outDoc, "generation_config.max_output_tokens", firstExisting(root.Get("max_completion_tokens"), root.Get("max_tokens")))
		copyNumberDoc(outDoc, "generation_config.temperature", root.Get("temperature"))
		copyNumberDoc(outDoc, "generation_config.top_p", root.Get("top_p"))
		copyNumberDoc(outDoc, "generation_config.presence_penalty", root.Get("presence_penalty"))
		copyNumberDoc(outDoc, "generation_config.frequency_penalty", root.Get("frequency_penalty"))
		copyNumberDoc(outDoc, "generation_config.candidate_count", root.Get("n"))
		if stop := root.Get("stop"); stop.Exists() {
			_ = outDoc.SetRaw("generation_config.stop_sequences", stop.Raw)
		}
	}
	if toolChoice := root.Get("tool_choice"); toolChoice.Exists() && !isOpenAIChatImageGenerationToolChoice(toolChoice) {
		if isAntigravityModel(model) && toolChoice.IsObject() {
			tcRaw := []byte(toolChoice.Raw)
			if fnName := toolChoice.Get("function.name").String(); fnName != "" {
				tcRaw, _ = ojson.SetBytes(tcRaw, "function.name", translatorcommon.AntigravityToolNameToUpstream(fnName))
			} else if name := toolChoice.Get("name").String(); name != "" {
				tcRaw, _ = ojson.SetBytes(tcRaw, "name", translatorcommon.AntigravityToolNameToUpstream(name))
			}
			_ = outDoc.SetRawBytes("generation_config.tool_choice", tcRaw)
		} else {
			_ = outDoc.SetRaw("generation_config.tool_choice", toolChoice.Raw)
		}
	}
	if effort := root.Get("reasoning_effort"); effort.Exists() && effort.Type == ojson.String {
		_ = outDoc.Set("generation_config.thinking_level", strings.ToLower(strings.TrimSpace(effort.String())))
	}
	if responseFormat := root.Get("response_format"); responseFormat.Exists() {
		_ = outDoc.SetRaw("response_format", responseFormat.Raw)
	}
	applyOpenAIChatImageFormatToInteractionsDoc(outDoc, root)
	if serviceTier := root.Get("service_tier"); serviceTier.Exists() && serviceTier.Type == ojson.String {
		_ = outDoc.Set("service_tier", serviceTier.String())
	}
}

func isOpenAIChatImageGenerationToolChoice(toolChoice ojson.Result) bool {
	if toolChoice.Type == ojson.String {
		return strings.EqualFold(strings.TrimSpace(toolChoice.String()), "image_generation")
	}
	return toolChoice.IsObject() && strings.EqualFold(strings.TrimSpace(toolChoice.Get("type").String()), "image_generation")
}

func applyOpenAIChatImageFormatToInteractionsDoc(outDoc *ojson.Doc, root ojson.Result) {
	if outDoc == nil {
		return
	}
	var modalities []string
	seenModalities := make(map[string]bool)
	if mods := root.Get("modalities"); mods.Exists() && mods.IsArray() {
		mods.ForEach(func(_, mod ojson.Result) bool {
			normalized := strings.ToLower(strings.TrimSpace(mod.String()))
			if normalized == "" || seenModalities[normalized] {
				return true
			}
			seenModalities[normalized] = true
			modalities = append(modalities, normalized)
			return true
		})
	}

	imageOpts := make(map[string]string)
	firstOpt := func(v ojson.Result, paths ...string) string {
		for _, path := range paths {
			if s := strings.TrimSpace(v.Get(path).String()); s != "" {
				return s
			}
		}
		return ""
	}
	hasExplicitImageSize := false
	if cfg := root.Get("image_config"); cfg.IsObject() {
		if ar := translatorcommon.GeminiImageAspectRatio(firstOpt(cfg, "aspect_ratio", "aspectRatio")); ar != "" {
			imageOpts["aspect_ratio"] = ar
		}
		if sz := translatorcommon.GeminiImageSizeFromOptions(
			firstOpt(cfg, "image_size", "imageSize"),
			firstOpt(cfg, "resolution"),
			firstOpt(cfg, "size"),
		); sz != "" {
			imageOpts["image_size"] = sz
			hasExplicitImageSize = true
		} else if rawSz := firstOpt(cfg, "image_size", "imageSize"); rawSz != "" {
			imageOpts["image_size"] = rawSz
			hasExplicitImageSize = true
		}
	}

	hasImageGeneration := false
	applyTool := func(tool ojson.Result) {
		hasImageGeneration = true
		ratio := translatorcommon.GeminiImageAspectRatio(firstOpt(tool, "aspect_ratio", "aspectRatio", "image_generation.aspect_ratio", "image_generation.aspectRatio", "image_config.aspect_ratio", "image_config.aspectRatio"))
		sizeOpt := firstOpt(tool, "size", "image_generation.size", "image_config.size")
		if ratio == "" {
			ratio = translatorcommon.GeminiImageAspectRatioFromSize(firstOpt(tool, "size", "image_generation.size", "image_generation.image_size", "image_generation.imageSize", "image_config.size", "image_config.image_size", "image_config.imageSize"))
		}
		if ratio != "" {
			imageOpts["aspect_ratio"] = ratio
		}
		if !hasExplicitImageSize {
			if sz := translatorcommon.GeminiImageSizeFromOptions(
				firstOpt(tool, "image_size", "imageSize", "image_generation.image_size", "image_generation.imageSize", "image_config.image_size", "image_config.imageSize"),
				firstOpt(tool, "resolution", "image_generation.resolution", "image_config.resolution"),
				sizeOpt,
			); sz != "" {
				imageOpts["image_size"] = sz
			}
		}
	}
	if tools := root.Get("tools"); tools.IsArray() {
		tools.ForEach(func(_, tool ojson.Result) bool {
			if strings.EqualFold(strings.TrimSpace(tool.Get("type").String()), "image_generation") {
				applyTool(tool)
			}
			return true
		})
	}
	if tc := root.Get("tool_choice"); isOpenAIChatImageGenerationToolChoice(tc) {
		if tc.IsObject() {
			applyTool(tc)
		} else {
			hasImageGeneration = true
		}
	}
	if hasImageGeneration {
		for _, reqMod := range []string{"image", "text"} {
			if !seenModalities[reqMod] {
				seenModalities[reqMod] = true
				modalities = append(modalities, reqMod)
			}
		}
	}
	if len(modalities) == 0 && len(imageOpts) == 0 {
		return
	}
	buildImageEntry := func() []byte {
		entryDoc := ojson.NewDocString(`{"type":"image"}`)
		if ar := imageOpts["aspect_ratio"]; ar != "" {
			_ = entryDoc.Set("aspect_ratio", ar)
		}
		if sz := imageOpts["image_size"]; sz != "" {
			_ = entryDoc.Set("image_size", sz)
		}
		return entryDoc.Bytes()
	}
	var entries [][]byte
	addedImage := false
	for _, mod := range modalities {
		if mod == "image" {
			entries = append(entries, buildImageEntry())
			addedImage = true
			continue
		}
		entryDoc := ojson.NewDocString(`{"type":""}`)
		_ = entryDoc.Set("type", mod)
		entries = append(entries, entryDoc.Bytes())
	}
	if len(imageOpts) > 0 && !addedImage {
		entries = append(entries, buildImageEntry())
	}
	if len(entries) == 1 {
		_ = outDoc.SetRawBytes("response_format", entries[0])
	} else if len(entries) > 1 {
		translatorcommon.SetRawArrayItemsDoc(outDoc, "response_format", entries)
	}
}

func appendOpenAIChatToolsToInteractionsDoc(outDoc *ojson.Doc, tools ojson.Result, forAntigravity bool) {
	if !tools.Exists() || !tools.IsArray() {
		return
	}
	var toolItems [][]byte
	tools.ForEach(func(_, tool ojson.Result) bool {
		if converted, ok := openAIChatToolToInteractions(tool, forAntigravity); ok {
			toolItems = append(toolItems, converted)
		}
		return true
	})
	if len(toolItems) > 0 {
		_ = outDoc.SetRawBytes("tools", translatorcommon.JoinRawArray(toolItems))
	}
}

func openAIChatToolToInteractions(tool ojson.Result, forAntigravity bool) ([]byte, bool) {
	toolType := strings.ToLower(strings.TrimSpace(tool.Get("type").String()))
	if toolType != "" && toolType != "function" {
		return nil, false
	}
	name := firstNonEmpty(tool.Get("function.name").String(), tool.Get("name").String())
	if name == "" {
		return nil, false
	}
	if forAntigravity {
		name = translatorcommon.AntigravityToolNameToUpstream(name)
	}
	outDoc := ojson.NewDocString(`{"type":"function","name":""}`)
	_ = outDoc.Set("name", name)
	if desc := firstExisting(tool.Get("function.description"), tool.Get("description")); desc.Exists() {
		_ = outDoc.Set("description", desc.String())
	}
	if parameters := firstExisting(tool.Get("function.parameters"), tool.Get("parameters")); parameters.Exists() {
		_ = outDoc.SetRaw("parameters", parameters.Raw)
	}
	return outDoc.Bytes(), true
}

func openAIChatContentText(content ojson.Result) string {
	if content.Type == ojson.String {
		return content.String()
	}
	if content.IsObject() {
		return content.Get("text").String()
	}
	if !content.IsArray() {
		return ""
	}
	var builder strings.Builder
	content.ForEach(func(_, part ojson.Result) bool {
		if text := part.Get("text").String(); text != "" {
			builder.WriteString(text)
		}
		return true
	})
	return builder.String()
}

func openAIInputAudioMIMEType(format string) string {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "wav":
		return "audio/wav"
	case "flac":
		return "audio/flac"
	case "opus":
		return "audio/opus"
	case "pcm16":
		return "audio/pcm"
	default:
		return "audio/mpeg"
	}
}

func openAIChatParseDataURL(value string) (string, string, bool) {
	if !strings.HasPrefix(value, "data:") {
		return "", "", false
	}
	meta, data, ok := strings.Cut(strings.TrimPrefix(value, "data:"), ",")
	if !ok {
		return "", "", false
	}
	mimeType, encoding, _ := strings.Cut(meta, ";")
	if !strings.EqualFold(encoding, "base64") || strings.TrimSpace(mimeType) == "" || data == "" {
		return "", "", false
	}
	return mimeType, data, true
}
