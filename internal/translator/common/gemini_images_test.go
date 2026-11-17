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
