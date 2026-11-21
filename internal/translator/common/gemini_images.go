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
	hasExplicitImageSize := false
	if cfg := root.Get("image_config"); cfg.IsObject() {
		if ratio := firstImageOption(cfg, "aspect_ratio", "aspectRatio"); ratio != "" {
			_ = doc.Set(generationPath+".imageConfig.aspectRatio", ratio)
		}
		if size := firstImageOption(cfg, "image_size", "imageSize"); size != "" {
			_ = doc.Set(generationPath+".imageConfig.imageSize", size)
			hasExplicitImageSize = true
		}
	}

	hasImageGeneration := false
	applyTool := func(tool ojson.Result) {
		hasImageGeneration = true
		ratio := GeminiImageAspectRatio(firstImageOption(tool, "aspect_ratio", "aspectRatio", "image_generation.aspect_ratio", "image_generation.aspectRatio", "image_config.aspect_ratio", "image_config.aspectRatio"))
		sizeOpt := firstImageOption(tool, "size", "image_generation.size", "image_config.size")
		if ratio == "" {
			ratio = GeminiImageAspectRatioFromSize(firstImageOption(tool, "size", "image_generation.size", "image_generation.image_size", "image_generation.imageSize", "image_config.size", "image_config.image_size", "image_config.imageSize"))
		}
		if ratio != "" {
			_ = doc.Set(generationPath+".imageConfig.aspectRatio", ratio)
		}
		if !hasExplicitImageSize {
			imageSize := GeminiImageSizeFromOptions(
				firstImageOption(tool, "image_size", "imageSize", "image_generation.image_size", "image_generation.imageSize", "image_config.image_size", "image_config.imageSize"),
				firstImageOption(tool, "resolution", "image_generation.resolution", "image_config.resolution"),
				sizeOpt,
			)
			if imageSize != "" {
				_ = doc.Set(generationPath+".imageConfig.imageSize", imageSize)
			}
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

// GeminiImageAspectRatio normalizes explicit Gemini image aspect ratio strings.
func GeminiImageAspectRatio(raw string) string {
	trimmed := strings.TrimSpace(raw)
	switch strings.ToLower(trimmed) {
	case "", "auto":
		return ""
	case "1:1", "square":
		return "1:1"
	case "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "4:5", "5:4", "21:9", "1:4", "4:1", "1:8", "8:1":
		return strings.ToLower(trimmed)
	default:
		if strings.Contains(trimmed, ":") {
			return trimmed
		}
		return ""
	}
}

// GeminiImageAspectRatioFromSize maps OpenAI image dimensions to Gemini ratios.
// Empty or "auto" dimensions leave aspectRatio unset; unknown explicit
// dimensions retain the historical square fallback.
func GeminiImageAspectRatioFromSize(size string) string {
	size = strings.ToLower(strings.TrimSpace(size))
	switch size {
	case "", "auto":
		return ""
	case "1024x1024", "2048x2048", "4096x4096", "512x512", "1:1", "square":
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

// GeminiImageSizeFromOptions normalizes explicit image_size / resolution tiers
// or high-resolution size dimensions (2K/4K) into Gemini imageConfig.imageSize.
func GeminiImageSizeFromOptions(imageSize, resolution, size string) string {
	if tier := normalizeGeminiImageSizeTier(imageSize); tier != "" {
		return tier
	}
	if tier := normalizeGeminiImageSizeTier(resolution); tier != "" {
		return tier
	}
	switch strings.ToLower(strings.TrimSpace(size)) {
	case "2048x2048", "2k":
		return "2K"
	case "4096x4096", "4k":
		return "4K"
	default:
		return ""
	}
}

func normalizeGeminiImageSizeTier(raw string) string {
	trimmed := strings.TrimSpace(raw)
	switch strings.ToLower(trimmed) {
	case "":
		return ""
	case "512", "512px", "0.5k":
		return "512px"
	case "1k", "1024":
		return "1K"
	case "2k", "2048":
		return "2K"
	case "4k", "4096":
		return "4K"
	default:
		return ""
	}
}
