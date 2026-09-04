package chat_completions

import (
	"testing"

	"github.com/tidwall/gjson"
)

func TestConvertOpenAIRequestToAntigravitySkipsEmptyTextPartsWithoutNulls(t *testing.T) {
	inputJSON := `{
		"model": "gemini-3-flash",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": ""},
					{"type": "input_audio", "input_audio": {"data": "SUQzBA==", "format": "mp3"}}
				]
			},
			{
				"role": "assistant",
				"content": [{"type": "text", "text": ""}],
				"tool_calls": [{
					"id": "call_1",
					"type": "function",
					"function": {"name": "read_file", "arguments": "{\"path\":\"a.txt\"}"}
				}]
			},
			{"role": "tool", "tool_call_id": "call_1", "content": "{\"output\":\"ok\"}"},
			{"role": "user", "content": "done"}
		]
	}`

	result := ConvertOpenAIRequestToAntigravity("gemini-3-flash", []byte(inputJSON), false)
	userParts := gjson.GetBytes(result, "request.contents.0.parts").Array()
	if len(userParts) != 1 {
		t.Fatalf("user parts length = %d, want 1. Output: %s", len(userParts), result)
	}
	if userParts[0].Type == gjson.Null {
		t.Fatalf("user parts.0 is null. Output: %s", result)
	}
	if got := userParts[0].Get("inlineData.mime_type").String(); got != "audio/mpeg" {
		t.Fatalf("audio mime_type = %q, want audio/mpeg. Output: %s", got, result)
	}

	assistantParts := gjson.GetBytes(result, "request.contents.1.parts").Array()
	if len(assistantParts) != 1 {
		t.Fatalf("assistant parts length = %d, want 1. Output: %s", len(assistantParts), result)
	}
	if assistantParts[0].Type == gjson.Null {
		t.Fatalf("assistant parts.0 is null. Output: %s", result)
	}
	if !assistantParts[0].Get("functionCall").Exists() {
		t.Fatalf("functionCall missing. Output: %s", result)
	}
}

func TestConvertOpenAIRequestToAntigravity_ClaudeModelSanitizesUnsignedReasoningContent(t *testing.T) {
	inputJSON := `{
		"model": "claude-sonnet-4-6",
		"messages": [
			{"role": "user", "content": "hi"},
			{"role": "assistant", "content": "visible text", "reasoning_content": "unsigned reasoning"},
			{"role": "user", "content": "say ok"}
		]
	}`

	result := ConvertOpenAIRequestToAntigravity("claude-sonnet-4-6", []byte(inputJSON), false)
	contents := gjson.GetBytes(result, "request.contents").Array()
	if len(contents) != 3 {
		t.Fatalf("contents length = %d, want 3. Output: %s", len(contents), result)
	}
	parts := contents[1].Get("parts").Array()
	if len(parts) != 1 {
		t.Fatalf("model parts length = %d, want 1 (thinking part dropped). Output: %s", len(parts), result)
	}
	if got := parts[0].Get("text").String(); got != "visible text" {
		t.Fatalf("parts[0].text = %q, want visible text. Output: %s", got, result)
	}
	if parts[0].Get("thought").Exists() {
		t.Fatalf("parts[0] should not be thought part. Output: %s", result)
	}
}

func TestConvertOpenAIRequestToAntigravity_ClaudeModelDropsEmptyAssistantTurnAfterSanitizingReasoningContent(t *testing.T) {
	inputJSON := `{
		"model": "claude-sonnet-4-6",
		"messages": [
			{"role": "user", "content": "hi"},
			{"role": "assistant", "content": "", "reasoning_content": "unsigned reasoning"},
			{"role": "user", "content": "say ok"}
		]
	}`

	result := ConvertOpenAIRequestToAntigravity("claude-sonnet-4-6", []byte(inputJSON), false)
	contents := gjson.GetBytes(result, "request.contents").Array()
	if len(contents) != 2 {
		t.Fatalf("contents length = %d, want 2 (empty model turn dropped). Output: %s", len(contents), result)
	}
	if got := contents[0].Get("role").String(); got != "user" {
		t.Fatalf("contents[0].role = %q, want user. Output: %s", got, result)
	}
	if got := contents[1].Get("role").String(); got != "user" {
		t.Fatalf("contents[1].role = %q, want user. Output: %s", got, result)
	}
}

func TestConvertOpenAIRequestToAntigravityPreservesReasoningContent(t *testing.T) {
	inputJSON := `{
		"model": "gemini-3-flash",
		"messages": [
			{"role": "user", "content": "hi"},
			{"role": "assistant", "content": "", "reasoning_content": "thinking only"},
			{"role": "user", "content": "say ok"}
		]
	}`

	result := ConvertOpenAIRequestToAntigravity("gemini-3-flash", []byte(inputJSON), true)
	contents := gjson.GetBytes(result, "request.contents").Array()
	if len(contents) != 3 {
		t.Fatalf("contents length = %d, want 3. Output: %s", len(contents), result)
	}
	part := contents[1].Get("parts.0")
	if got := contents[1].Get("role").String(); got != "model" {
		t.Fatalf("contents.1.role = %q, want model. Output: %s", got, result)
	}
	if got := part.Get("text").String(); got != "thinking only" {
		t.Fatalf("reasoning text = %q, want thinking only. Output: %s", got, result)
	}
	if !part.Get("thought").Bool() {
		t.Fatalf("reasoning part should be marked as thought. Output: %s", result)
	}
	if got := part.Get("thoughtSignature").String(); got != antigravityFunctionThoughtSignature {
		t.Fatalf("thoughtSignature = %q, want bypass sentinel. Output: %s", got, result)
	}
}

func TestConvertOpenAIRequestToAntigravityPreservesReasoningBeforeVisibleContentAndToolCall(t *testing.T) {
	inputJSON := `{
		"model": "gemini-3-flash",
		"messages": [
			{"role": "user", "content": "hi"},
			{"role": "assistant", "content": "visible answer", "reasoning_content": "thinking only", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
			{"role": "tool", "tool_call_id": "call_1", "content": "{\"output\":\"ok\"}"},
			{"role": "user", "content": "say ok"}
		]
	}`

	result := ConvertOpenAIRequestToAntigravity("gemini-3-flash", []byte(inputJSON), true)
	contents := gjson.GetBytes(result, "request.contents").Array()
	if len(contents) != 4 {
		t.Fatalf("contents length = %d, want 4. Output: %s", len(contents), result)
	}
	parts := contents[1].Get("parts").Array()
	if len(parts) != 3 {
		t.Fatalf("model parts length = %d, want 3. Output: %s", len(parts), result)
	}
	if got := parts[0].Get("text").String(); got != "thinking only" || !parts[0].Get("thought").Bool() {
		t.Fatalf("first part should be the reasoning thought. Output: %s", result)
	}
	if got := parts[1].Get("text").String(); got != "visible answer" || parts[1].Get("thought").Bool() {
		t.Fatalf("second part should be visible assistant content. Output: %s", result)
	}
	if got := parts[2].Get("functionCall.name").String(); got != "read_file" {
		t.Fatalf("functionCall.name = %q, want read_file. Output: %s", got, result)
	}
	if got := parts[2].Get("thoughtSignature").String(); got != antigravityFunctionThoughtSignature {
		t.Fatalf("functionCall thoughtSignature = %q, want bypass sentinel. Output: %s", got, result)
	}
	if got := contents[2].Get("parts.0.functionResponse.name").String(); got != "read_file" {
		t.Fatalf("functionResponse.name = %q, want read_file. Output: %s", got, result)
	}
}

func TestConvertOpenAIRequestToAntigravitySkipsEmptyAssistantMessages(t *testing.T) {
	inputJSON := `{
		"model": "gemini-3-flash",
		"messages": [
			{"role": "user", "content": "hi"},
			{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "", "arguments": "{}"}}, {"type": "custom"}]},
			{"role": "user", "content": "say ok"}
		]
	}`

	result := ConvertOpenAIRequestToAntigravity("gemini-3-flash", []byte(inputJSON), true)
	contents := gjson.GetBytes(result, "request.contents").Array()
	if len(contents) != 2 {
		t.Fatalf("contents length = %d, want 2. Output: %s", len(contents), result)
	}
}

func TestConvertOpenAIRequestToAntigravity_MidSessionDeveloperMessageDoesNotMutateSystemInstruction(t *testing.T) {
	inputJSON := `{
		"model": "gemini-3-flash",
		"messages": [
			{"role": "system", "content": "You are a helpful assistant"},
			{"role": "user", "content": "Turn 1 user"},
			{"role": "assistant", "content": "Turn 1 assistant"},
			{"role": "developer", "content": "<image_resize_notice>Image 1 was resized to 800x600</image_resize_notice>"},
			{"role": "user", "content": "Turn 2 user"}
		]
	}`

	result := ConvertOpenAIRequestToAntigravity("gemini-3-flash", []byte(inputJSON), false)
	output := gjson.ParseBytes(result)

	// request.systemInstruction must contain only original system prompt
	sysParts := output.Get("request.systemInstruction.parts").Array()
	if len(sysParts) != 1 {
		t.Fatalf("request.systemInstruction parts = %d, want 1. Output: %s", len(sysParts), result)
	}
	if got := sysParts[0].Get("text").String(); got != "You are a helpful assistant" {
		t.Fatalf("systemInstruction text = %q, want %q", got, "You are a helpful assistant")
	}

	// contents must contain user, model, user (demoted dev message), user
	contents := output.Get("request.contents").Array()
	if len(contents) != 4 {
		t.Fatalf("contents length = %d, want 4. Output: %s", len(contents), result)
	}
	if contents[0].Get("role").String() != "user" || contents[0].Get("parts.0.text").String() != "Turn 1 user" {
		t.Fatalf("turn 0 mismatch: %s", contents[0].Raw)
	}
	if contents[1].Get("role").String() != "model" || contents[1].Get("parts.0.text").String() != "Turn 1 assistant" {
		t.Fatalf("turn 1 mismatch: %s", contents[1].Raw)
	}
	if contents[2].Get("role").String() != "user" || contents[2].Get("parts.0.text").String() != "<image_resize_notice>Image 1 was resized to 800x600</image_resize_notice>" {
		t.Fatalf("turn 2 mismatch: %s", contents[2].Raw)
	}
	if contents[3].Get("role").String() != "user" || contents[3].Get("parts.0.text").String() != "Turn 2 user" {
		t.Fatalf("turn 3 mismatch: %s", contents[3].Raw)
	}
}

func TestConvertOpenAIRequestToAntigravityThinkingAliases(t *testing.T) {
	tests := []struct {
		name       string
		body       string
		wantExists bool
		want       bool
	}{
		{
			name: "Missing summary intent leaves include thoughts absent",
			body: `{
				"model":"gemini-3.1-pro-low",
				"messages":[{"role":"user","content":"hi"}]
			}`,
		},
		{
			name: "Reasoning effort enables thoughts",
			body: `{
				"model":"gemini-3.1-pro-low",
				"messages":[{"role":"user","content":"hi"}],
				"reasoning_effort":"high"
			}`,
			wantExists: true,
			want:       true,
		},
		{
			name: "GenerationConfig snake include thoughts",
			body: `{
				"model":"gemini-3.1-pro-low",
				"messages":[{"role":"user","content":"hi"}],
				"generationConfig":{"thinkingConfig":{"include_thoughts":true}}
			}`,
			wantExists: true,
			want:       true,
		},
		{
			name: "String include thoughts is ignored",
			body: `{
				"model":"gemini-3.1-pro-low",
				"messages":[{"role":"user","content":"hi"}],
				"generationConfig":{"thinkingConfig":{"includeThoughts":"true"}}
			}`,
		},
		{
			name: "Top-level thinking include thoughts",
			body: `{
				"model":"gemini-3.1-pro-low",
				"messages":[{"role":"user","content":"hi"}],
				"thinking":{"include_thoughts":true}
			}`,
			wantExists: true,
			want:       true,
		},
		{
			name: "Reasoning exclude false includes thoughts",
			body: `{
				"model":"gemini-3.1-pro-low",
				"messages":[{"role":"user","content":"hi"}],
				"reasoning":{"exclude":false}
			}`,
			wantExists: true,
			want:       true,
		},
		{
			name: "Reasoning exclude true hides thoughts",
			body: `{
				"model":"gemini-3.1-pro-low",
				"messages":[{"role":"user","content":"hi"}],
				"reasoning":{"exclude":true}
			}`,
			wantExists: true,
			want:       false,
		},
		{
			name: "Google extension disables thoughts",
			body: `{
				"model":"gemini-3.1-pro-low",
				"messages":[{"role":"user","content":"hi"}],
				"reasoning_effort":"high",
				"extra_body":{"google":{"thinking_config":{"include_thoughts":false}}}
			}`,
			wantExists: true,
			want:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ConvertOpenAIRequestToAntigravity("gemini-3.1-pro-low", []byte(tt.body), false)
			includeThoughts := gjson.GetBytes(result, "request.generationConfig.thinkingConfig.includeThoughts")
			if includeThoughts.Exists() != tt.wantExists {
				t.Fatalf("includeThoughts exists = %v, want %v. Output: %s", includeThoughts.Exists(), tt.wantExists, result)
			}
			if tt.wantExists {
				if got := includeThoughts.Bool(); got != tt.want {
					t.Fatalf("includeThoughts = %v, want %v. Output: %s", got, tt.want, result)
				}
			}
			if snake := gjson.GetBytes(result, "request.generationConfig.thinkingConfig.include_thoughts"); snake.Exists() {
				t.Fatalf("include_thoughts should be normalized away. Output: %s", result)
			}
		})
	}
}

func TestConvertOpenAIRequestToAntigravityDeduplicatesAndDisambiguatesTools(t *testing.T) {
	first := "mcp__plugin_cloudflare_cloudflare-builds__workers_builds_get_build"
	second := "mcp__plugin_cloudflare_cloudflare-builds__workers_builds_get_build_logs"
	inputJSON := `{
		"messages":[
			{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"` + second + `","arguments":"{}"}}]},
			{"role":"tool","tool_call_id":"call_1","content":"{}"}
		],
		"tools":[
			{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}},
			{"type":"function","function":{"name":"lookup","description":"duplicate","parameters":{"type":"object"}}},
			{"type":"function","function":{"name":"` + first + `","parameters":{"type":"object"}}},
			{"type":"function","function":{"name":"` + second + `","parameters":{"type":"object"}}}
		],
		"tool_choice":{"type":"function","function":{"name":"` + second + `"}}
	}`

	out := ConvertOpenAIRequestToAntigravity("gemini-3-flash", []byte(inputJSON), false)
	declarations := gjson.GetBytes(out, "request.tools.0.functionDeclarations").Array()
	if len(declarations) != 3 {
		t.Fatalf("declaration count = %d, want 3. Output: %s", len(declarations), out)
	}
	firstMapped := declarations[1].Get("name").String()
	secondMapped := declarations[2].Get("name").String()
	if firstMapped == secondMapped || len(secondMapped) > 64 {
		t.Fatalf("collision names = %q and %q, want distinct names <= 64 chars", firstMapped, secondMapped)
	}
	if got := gjson.GetBytes(out, "request.contents.0.parts.0.functionCall.name").String(); got != secondMapped {
		t.Fatalf("functionCall.name = %q, want %q. Output: %s", got, secondMapped, out)
	}
	if got := gjson.GetBytes(out, "request.contents.1.parts.0.functionResponse.name").String(); got != secondMapped {
		t.Fatalf("functionResponse.name = %q, want %q. Output: %s", got, secondMapped, out)
	}
	if got := gjson.GetBytes(out, "request.toolConfig.functionCallingConfig.allowedFunctionNames.0").String(); got != secondMapped {
		t.Fatalf("allowedFunctionNames.0 = %q, want %q. Output: %s", got, secondMapped, out)
	}
}

func TestConvertOpenAIRequestToAntigravityMapsToolChoiceModes(t *testing.T) {
	for _, tt := range []struct {
		choice string
		mode   string
	}{
		{choice: `"none"`, mode: "NONE"},
		{choice: `"auto"`, mode: "AUTO"},
		{choice: `"required"`, mode: "ANY"},
	} {
		t.Run(tt.mode+tt.choice, func(t *testing.T) {
			inputJSON := []byte(`{"messages":[{"role":"user","content":"hi"}],"tool_choice":` + tt.choice + `}`)
			out := ConvertOpenAIRequestToAntigravity("gemini-3-flash", inputJSON, false)
			if got := gjson.GetBytes(out, "request.toolConfig.functionCallingConfig.mode").String(); got != tt.mode {
				t.Fatalf("tool choice mode = %q, want %q. Output: %s", got, tt.mode, out)
			}
		})
	}
}

func TestConvertOpenAIRequestToAntigravityMapsResponseFormatJSONObject(t *testing.T) {
	inputJSON := []byte(`{
		"model":"gemini-3.6-flash-high",
		"messages":[{"role":"user","content":"hi"}],
		"generationConfig":{
			"responseSchema":{"type":"string","description":"stale"},
			"responseJsonSchema":{"type":"string"},
			"response_schema":{"type":"string"},
			"response_json_schema":{"type":"string"}
		},
		"response_format":{"type":"json_object"}
	}`)

	out := ConvertOpenAIRequestToAntigravity("gemini-3.6-flash-high", inputJSON, false)
	if got := gjson.GetBytes(out, "request.generationConfig.responseMimeType").String(); got != "application/json" {
		t.Fatalf("responseMimeType = %q, want application/json. Output: %s", got, out)
	}
	if gjson.GetBytes(out, "request.generationConfig.responseSchema").Exists() {
		t.Fatalf("responseSchema should not be set for json_object. Output: %s", out)
	}
	assertNoResponseSchemaAliases(t, out)
}

func TestConvertOpenAIRequestToAntigravityMapsResponseFormatJSONSchema(t *testing.T) {
	inputJSON := []byte(`{
		"model":"gemini-3.6-flash-high",
		"messages":[{"role":"user","content":"hi"}],
		"generationConfig":{
			"responseSchema":{"type":"string","description":"stale"},
			"responseJsonSchema":{"type":"string"},
			"response_schema":{"type":"string"},
			"response_json_schema":{"type":"string"}
		},
		"response_format":{
			"type":"json_schema",
			"json_schema":{
				"name":"verdict",
				"schema":{
					"type":"object",
					"properties":{"score":{"type":"integer"}},
					"required":["score"]
				}
			}
		}
	}`)

	out := ConvertOpenAIRequestToAntigravity("gemini-3.6-flash-high", inputJSON, false)
	if got := gjson.GetBytes(out, "request.generationConfig.responseMimeType").String(); got != "application/json" {
		t.Fatalf("responseMimeType = %q, want application/json. Output: %s", got, out)
	}
	schema := gjson.GetBytes(out, "request.generationConfig.responseSchema")
	if !schema.Exists() {
		t.Fatalf("responseSchema missing. Output: %s", out)
	}
	if got := schema.Get("properties.score.type").String(); got != "integer" {
		t.Fatalf("responseSchema.properties.score.type = %q, want integer. Output: %s", got, out)
	}
	if schema.Get("description").Exists() {
		t.Fatalf("stale responseSchema survived. Output: %s", out)
	}
	assertNoResponseSchemaAliases(t, out)
}

func assertNoResponseSchemaAliases(t *testing.T, out []byte) {
	t.Helper()
	for _, schemaKey := range []string{"responseJsonSchema", "response_schema", "response_json_schema"} {
		if gjson.GetBytes(out, "request.generationConfig."+schemaKey).Exists() {
			t.Errorf("stale %s survived response_format mapping. Output: %s", schemaKey, out)
		}
	}
}

func TestConvertOpenAIRequestToAntigravityTranslatesVideoURL(t *testing.T) {
	inputJSON := []byte(`{
		"model": "gemini-3.7-flash-high",
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "Name the colours in order"},
				{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAAIGZ0eXBtcDQy"}}
			]
		}]
	}`)

	out := ConvertOpenAIRequestToAntigravity("gemini-3.7-flash-high", inputJSON, false)
	parts := gjson.GetBytes(out, "request.contents.0.parts").Array()
	if len(parts) != 2 {
		t.Fatalf("parts length = %d, want 2. Output: %s", len(parts), out)
	}

	if got := parts[0].Get("text").String(); got != "Name the colours in order" {
		t.Fatalf("parts[0].text = %q, want 'Name the colours in order'", got)
	}

	inlineData := parts[1].Get("inlineData")
	if !inlineData.Exists() {
		t.Fatalf("parts[1].inlineData missing. Output: %s", out)
	}
	if got := inlineData.Get("mimeType").String(); got != "video/mp4" {
		t.Fatalf("inlineData.mimeType = %q, want video/mp4. Output: %s", got, out)
	}
	if got := inlineData.Get("data").String(); got != "AAAAIGZ0eXBtcDQy" {
		t.Fatalf("inlineData.data = %q, want AAAAIGZ0eXBtcDQy. Output: %s", got, out)
	}
}

func TestConvertOpenAIRequestToAntigravity_MaxCompletionTokens(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		expected float64
	}{
		{
			name:     "only max_tokens",
			body:     `{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":100}`,
			expected: 100,
		},
		{
			name:     "only max_completion_tokens",
			body:     `{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":200}`,
			expected: 200,
		},
		{
			name:     "max_tokens preferred over max_completion_tokens",
			body:     `{"model":"gemini-2.5-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":100,"max_completion_tokens":200}`,
			expected: 100,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := ConvertOpenAIRequestToAntigravity("gemini-2.5-flash", []byte(tt.body), false)
			got := gjson.GetBytes(out, "request.generationConfig.maxOutputTokens")
			if !got.Exists() {
				t.Fatalf("request.generationConfig.maxOutputTokens missing. Output: %s", out)
			}
			if got.Float() != tt.expected {
				t.Fatalf("maxOutputTokens = %v, want %v. Output: %s", got.Float(), tt.expected, out)
			}
		})
	}
}

func TestConvertOpenAIRequestToAntigravityPreservesToolResponseAsString(t *testing.T) {
	inputJSON := `{
		"model": "gemini-3-flash",
		"messages": [
			{
				"role": "user",
				"content": "read file"
			},
			{
				"role": "assistant",
				"tool_calls": [{
					"id": "call_1",
					"type": "function",
					"function": {"name": "read_file", "arguments": "{\"path\":\"config.json\"}"}
				}]
			},
			{
				"role": "tool",
				"tool_call_id": "call_1",
				"content": "{\"key\":\"value\",\"items\":[1,2,3]}"
			}
		]
	}`

	result := ConvertOpenAIRequestToAntigravity("gemini-3-flash", []byte(inputJSON), false)
	contents := gjson.GetBytes(result, "request.contents").Array()
	if len(contents) < 3 {
		t.Fatalf("expected at least 3 contents, got %d. Output: %s", len(contents), result)
	}
	frResult := contents[2].Get("parts.0.functionResponse.response.result")
	if frResult.Type != gjson.String {
		t.Fatalf("expected functionResponse.response.result to be string, got type %s (raw: %s)", frResult.Type, frResult.Raw)
	}
	expected := `{"key":"value","items":[1,2,3]}`
	if got := frResult.String(); got != expected {
		t.Fatalf("expected %q, got %q", expected, got)
	}
}

func TestConvertOpenAIRequestToAntigravityImageGenerationTool(t *testing.T) {
	t.Run("tools with image_generation sets responseModalities without function declarations", func(t *testing.T) {
		inputJSON := []byte(`{
			"model": "gemini-3-flash",
			"messages": [{"role": "user", "content": "Draw a sunset"}],
			"tools": [{"type": "image_generation"}]
		}`)
		out := ConvertOpenAIRequestToAntigravity("gemini-3-flash", inputJSON, false)

		mods := gjson.GetBytes(out, "request.generationConfig.responseModalities").Array()
		if len(mods) != 2 {
			t.Fatalf("responseModalities length = %d, want 2. Output: %s", len(mods), out)
		}
		if mods[0].String() != "IMAGE" || mods[1].String() != "TEXT" {
			t.Fatalf("responseModalities = %v, want [IMAGE, TEXT]. Output: %s", mods, out)
		}
		if gjson.GetBytes(out, "request.tools").Exists() {
			t.Fatalf("request.tools should not be present when only image_generation is provided. Output: %s", out)
		}
		if gjson.GetBytes(out, "request.toolConfig").Exists() {
			t.Fatalf("request.toolConfig should not be present. Output: %s", out)
		}
	})

	t.Run("tools with image_generation and aspect_ratio", func(t *testing.T) {
		inputJSON := []byte(`{
			"model": "gemini-3-flash",
			"messages": [{"role": "user", "content": "Draw a landscape"}],
			"tools": [{"type": "image_generation", "aspect_ratio": "16:9"}]
		}`)
		out := ConvertOpenAIRequestToAntigravity("gemini-3-flash", inputJSON, false)

		if got := gjson.GetBytes(out, "request.generationConfig.imageConfig.aspectRatio").String(); got != "16:9" {
			t.Fatalf("aspectRatio = %q, want 16:9. Output: %s", got, out)
		}
		mods := gjson.GetBytes(out, "request.generationConfig.responseModalities").Array()
		if len(mods) != 2 || mods[0].String() != "IMAGE" || mods[1].String() != "TEXT" {
			t.Fatalf("responseModalities = %v, want [IMAGE, TEXT]. Output: %s", mods, out)
		}
	})

	t.Run("tools with size mapping", func(t *testing.T) {
		tests := []struct {
			size   string
			wantAR string
		}{
			{"1024x1024", "1:1"},
			{"1792x1024", "16:9"},
			{"1024x1792", "9:16"},
			{"1024x768", "4:3"},
			{"768x1024", "3:4"},
			{"1536x1024", "3:2"},
			{"1024x1536", "2:3"},
		}
		for _, tc := range tests {
			t.Run(tc.size, func(t *testing.T) {
				inputJSON := []byte(`{"messages":[{"role":"user","content":"Draw an image"}],"tools":[{"type":"image_generation","size":"` + tc.size + `"}]}`)
				out := ConvertOpenAIRequestToAntigravity("gemini-3-flash", inputJSON, false)
				if got := gjson.GetBytes(out, "request.generationConfig.imageConfig.aspectRatio").String(); got != tc.wantAR {
					t.Fatalf("aspectRatio for size %q = %q, want %q. Output: %s", tc.size, got, tc.wantAR, out)
				}
			})
		}
	})

	t.Run("tool_choice with type image_generation", func(t *testing.T) {
		inputJSON := []byte(`{
			"model": "gemini-3-flash",
			"messages": [{"role": "user", "content": "Draw an image"}],
			"tool_choice": {"type": "image_generation", "aspect_ratio": "4:3"}
		}`)
		out := ConvertOpenAIRequestToAntigravity("gemini-3-flash", inputJSON, false)

		if got := gjson.GetBytes(out, "request.generationConfig.imageConfig.aspectRatio").String(); got != "4:3" {
			t.Fatalf("aspectRatio = %q, want 4:3. Output: %s", got, out)
		}
		mods := gjson.GetBytes(out, "request.generationConfig.responseModalities").Array()
		if len(mods) != 2 || mods[0].String() != "IMAGE" || mods[1].String() != "TEXT" {
			t.Fatalf("responseModalities = %v, want [IMAGE, TEXT]. Output: %s", mods, out)
		}
		if gjson.GetBytes(out, "request.toolConfig").Exists() {
			t.Fatalf("toolConfig should not be set for image_generation tool_choice. Output: %s", out)
		}
	})

	t.Run("tool_choice string image_generation", func(t *testing.T) {
		inputJSON := []byte(`{
			"model": "gemini-3-flash",
			"messages": [{"role": "user", "content": "Draw an image"}],
			"tool_choice": "image_generation"
		}`)
		out := ConvertOpenAIRequestToAntigravity("gemini-3-flash", inputJSON, false)

		mods := gjson.GetBytes(out, "request.generationConfig.responseModalities").Array()
		if len(mods) != 2 || mods[0].String() != "IMAGE" || mods[1].String() != "TEXT" {
			t.Fatalf("responseModalities = %v, want [IMAGE, TEXT]. Output: %s", mods, out)
		}
		if gjson.GetBytes(out, "request.toolConfig").Exists() {
			t.Fatalf("toolConfig should not be set for string image_generation tool_choice. Output: %s", out)
		}
	})

	t.Run("preserves existing modalities when image_generation is detected", func(t *testing.T) {
		inputJSON := []byte(`{
			"model": "gemini-3-flash",
			"messages": [{"role": "user", "content": "Draw an image"}],
			"modalities": ["TEXT"],
			"tools": [{"type": "image_generation"}]
		}`)
		out := ConvertOpenAIRequestToAntigravity("gemini-3-flash", inputJSON, false)

		mods := gjson.GetBytes(out, "request.generationConfig.responseModalities").Array()
		if len(mods) != 2 {
			t.Fatalf("responseModalities length = %d, want 2. Output: %s", len(mods), out)
		}
		if mods[0].String() != "TEXT" || mods[1].String() != "IMAGE" {
			t.Fatalf("responseModalities = %v, want [TEXT, IMAGE]. Output: %s", mods, out)
		}
	})

	t.Run("tools with both function and image_generation does not declare image_generation as function", func(t *testing.T) {
		inputJSON := []byte(`{
			"model": "gemini-3-flash",
			"messages": [{"role": "user", "content": "Search and draw"}],
			"tools": [
				{
					"type": "function",
					"function": {
						"name": "search_web",
						"description": "Search web",
						"parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
					}
				},
				{
					"type": "image_generation",
					"aspect_ratio": "16:9"
				}
			],
			"tool_choice": {"type": "image_generation"}
		}`)
		out := ConvertOpenAIRequestToAntigravity("gemini-3-flash", inputJSON, false)

		if got := gjson.GetBytes(out, "request.generationConfig.imageConfig.aspectRatio").String(); got != "16:9" {
			t.Fatalf("aspectRatio = %q, want 16:9. Output: %s", got, out)
		}
		mods := gjson.GetBytes(out, "request.generationConfig.responseModalities").Array()
		if len(mods) != 2 || mods[0].String() != "IMAGE" || mods[1].String() != "TEXT" {
			t.Fatalf("responseModalities = %v, want [IMAGE, TEXT]. Output: %s", mods, out)
		}

		// Function declarations should only contain search_web
		fnDecls := gjson.GetBytes(out, "request.tools.0.functionDeclarations").Array()
		if len(fnDecls) != 1 {
			t.Fatalf("functionDeclarations length = %d, want 1. Output: %s", len(fnDecls), out)
		}
		if got := fnDecls[0].Get("name").String(); got != "search_web" {
			t.Fatalf("functionDeclarations[0].name = %q, want search_web. Output: %s", got, out)
		}

		// tool_choice was image_generation, so request.toolConfig should NOT be set
		if gjson.GetBytes(out, "request.toolConfig").Exists() {
			t.Fatalf("request.toolConfig should not be set when tool_choice is image_generation. Output: %s", out)
		}
	})
}
