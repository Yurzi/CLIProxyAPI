package common

import (
	"strings"

	"github.com/MoeForge/moego/ojson"
)

// ApplyGeminiImageConfig maps OpenAI image options into a Gemini generation
// config. generationPath also supports Antigravity's request envelope. Explicit
// tool_choice options take precedence over the corresponding tools entry.
func ApplyGeminiImageConfig(doc *ojson.Doc, root ojson.Result, generationPath string) {
	var modalities []string
	mods := root.Get("modalities")
	if !mods.IsArray() || len(mods.Array()) == 0 {
		mods = doc.Get(generationPath + ".responseModalities")
	}
	if mods.IsArray() {
		for _, mod := range mods.Array() {
			if value := strings.ToUpper(strings.TrimSpace(mod.String())); value != "" {
				modalities = append(modalities, value)
			}
		}
	}
	if cfg := root.Get("image_config"); cfg.IsObject() {
		if ratio := firstImageOption(cfg, "aspect_ratio", "aspectRatio"); ratio != "" {
			_ = doc.Set(generationPath+".imageConfig.aspectRatio", ratio)
		}
		if size := firstImageOption(cfg, "image_size", "imageSize"); size != "" {
			_ = doc.Set(generationPath+".imageConfig.imageSize", size)
		}
	}

	hasImageGeneration := false
	applyTool := func(tool ojson.Result) {
		hasImageGeneration = true
		ratio := firstImageOption(tool, "aspect_ratio", "aspectRatio", "image_generation.aspect_ratio", "image_generation.aspectRatio", "image_config.aspect_ratio", "image_config.aspectRatio")
		if ratio == "" {
			ratio = GeminiImageAspectRatioFromSize(firstImageOption(tool, "size", "image_generation.size", "image_generation.image_size", "image_generation.imageSize", "image_config.size", "image_config.image_size", "image_config.imageSize"))
		}
		if ratio != "" {
			_ = doc.Set(generationPath+".imageConfig.aspectRatio", ratio)
		}
	}
	if tools := root.Get("tools"); tools.IsArray() {
		for _, tool := range tools.Array() {
			if strings.EqualFold(tool.Get("type").String(), "image_generation") {
				applyTool(tool)
			}
		}
	}
	choice := root.Get("tool_choice")
	if choice.Type == ojson.String && strings.EqualFold(choice.String(), "image_generation") {
		hasImageGeneration = true
	} else if choice.IsObject() && strings.EqualFold(choice.Get("type").String(), "image_generation") {
		applyTool(choice)
	}
	if hasImageGeneration {
		for _, required := range []string{"IMAGE", "TEXT"} {
			found := false
			for _, mod := range modalities {
				found = found || mod == required
			}
			if !found {
				modalities = append(modalities, required)
			}
		}
	}
	if len(modalities) > 0 {
		_ = doc.Set(generationPath+".responseModalities", modalities)
	}
}

func firstImageOption(value ojson.Result, paths ...string) string {
	for _, path := range paths {
		if text := strings.TrimSpace(value.Get(path).String()); text != "" {
			return text
		}
	}
	return ""
}

// GeminiImageAspectRatioFromSize maps OpenAI image dimensions to Gemini ratios.
// Unknown non-empty dimensions retain the historical square fallback.
func GeminiImageAspectRatioFromSize(size string) string {
	size = strings.ToLower(strings.TrimSpace(size))
	switch size {
	case "":
		return ""
	case "1024x1024", "2048x2048", "512x512", "1:1":
		return "1:1"
	case "1792x1024", "1280x720", "16:9":
		return "16:9"
	case "1024x1792", "720x1280", "9:16":
		return "9:16"
	case "1024x768", "4:3":
		return "4:3"
	case "768x1024", "3:4":
		return "3:4"
	case "1536x1024", "3:2":
		return "3:2"
	case "1024x1536", "2:3":
		return "2:3"
	default:
		if strings.Contains(size, ":") {
			return size
		}
		return "1:1"
	}
}
