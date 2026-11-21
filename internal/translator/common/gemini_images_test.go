package common

import (
	"testing"

	"github.com/MoeForge/moego/ojson"
)

func TestGeminiImageConfigPreservesModalitiesAndChoicePrecedence(t *testing.T) {
	for _, prefix := range []string{"generationConfig", "request.generationConfig"} {
		t.Run(prefix, func(t *testing.T) {
			doc := ojson.NewDocString(`{}`)
			_ = doc.Set(prefix+".responseModalities", []string{"AUDIO"})
			root := ojson.Parse(`{"tools":[{"type":"image_generation","size":"1024x1024"}],"tool_choice":{"type":"image_generation","image_config":{"aspectRatio":"3:2"}},"image_config":{"imageSize":"2K"}}`)
			ApplyGeminiImageConfig(doc, root, prefix)
			if got := doc.Get(prefix + ".responseModalities").Raw; got != `["AUDIO","IMAGE","TEXT"]` {
				t.Fatalf("modalities = %s", got)
			}
			if got := doc.Get(prefix + ".imageConfig.aspectRatio").String(); got != "3:2" {
				t.Fatalf("tool choice aspect ratio = %q", got)
			}
			if got := doc.Get(prefix + ".imageConfig.imageSize").String(); got != "2K" {
				t.Fatalf("image size = %q", got)
			}
		})
	}
}

func TestGeminiImageConfigExplicitModalitiesAndMixedTools(t *testing.T) {
	doc := ojson.NewDocString(`{"generationConfig":{"responseModalities":["AUDIO"]},"toolConfig":{"functionCallingConfig":{"mode":"AUTO"}}}`)
	root := ojson.Parse(`{"modalities":["text"],"tools":[{"type":"function","function":{"name":"lookup"}},{"type":"image_generation","size":"1536x1024"}]}`)
	ApplyGeminiImageConfig(doc, root, "generationConfig")
	if got := doc.Get("generationConfig.responseModalities").Raw; got != `["TEXT","IMAGE"]` {
		t.Fatalf("modalities = %s", got)
	}
	if got := doc.Get("generationConfig.imageConfig.aspectRatio").String(); got != "3:2" {
		t.Fatalf("aspect ratio = %q", got)
	}
	if got := doc.Get("toolConfig.functionCallingConfig.mode").String(); got != "AUTO" {
		t.Fatalf("function calling mode changed to %q", got)
	}
}

func TestGeminiImageAspectRatioAndSizeHelpers(t *testing.T) {
	for _, tc := range []struct {
		raw  string
		want string
	}{
		{"", ""},
		{"auto", ""},
		{"AUTO", ""},
		{"square", "1:1"},
		{"21:9", "21:9"},
		{"4:5", "4:5"},
		{"invalid", ""},
	} {
		if got := GeminiImageAspectRatio(tc.raw); got != tc.want {
			t.Errorf("GeminiImageAspectRatio(%q) = %q, want %q", tc.raw, got, tc.want)
		}
	}

	for _, tc := range []struct {
		size string
		want string
	}{
		{"", ""},
		{"auto", ""},
		{"4096x4096", "1:1"},
		{"2048x2048", "1:1"},
		{"512x512", "1:1"},
		{"1536x1024", "3:2"},
		{"800x600", "1:1"},
	} {
		if got := GeminiImageAspectRatioFromSize(tc.size); got != tc.want {
			t.Errorf("GeminiImageAspectRatioFromSize(%q) = %q, want %q", tc.size, got, tc.want)
		}
	}

	for _, tc := range []struct {
		imageSize  string
		resolution string
		size       string
		want       string
	}{
		{"512px", "", "", "512px"},
		{"", "2k", "", "2K"},
		{"", "4K", "2048x2048", "4K"},
		{"", "", "2048x2048", "2K"},
		{"", "", "4096x4096", "4K"},
		{"", "", "1024x1024", ""},
		{"", "", "auto", ""},
		{"1k", "", "", "1K"},
	} {
		if got := GeminiImageSizeFromOptions(tc.imageSize, tc.resolution, tc.size); got != tc.want {
			t.Errorf("GeminiImageSizeFromOptions(%q, %q, %q) = %q, want %q", tc.imageSize, tc.resolution, tc.size, got, tc.want)
		}
	}

	doc := ojson.NewDocString(`{}`)
	root := ojson.Parse(`{"tools":[{"type":"image_generation","aspect_ratio":"4:5","resolution":"4k"}]}`)
	ApplyGeminiImageConfig(doc, root, "generationConfig")
	if got := doc.Get("generationConfig.imageConfig.aspectRatio").String(); got != "4:5" {
		t.Fatalf("aspectRatio = %q, want 4:5", got)
	}
	if got := doc.Get("generationConfig.imageConfig.imageSize").String(); got != "4K" {
		t.Fatalf("imageSize = %q, want 4K", got)
	}
}
