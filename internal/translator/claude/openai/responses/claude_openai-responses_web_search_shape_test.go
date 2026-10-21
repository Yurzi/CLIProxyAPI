package responses

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

// webSearchCitationChunks is a Claude stream with one search followed by a
// message of three adjacent text blocks; the first and last are cited. Claude
// sends the citation before the text in block 2 and after it in block 4, and
// the text mixes multi-byte characters so byte and code point offsets differ.
func webSearchCitationChunks() [][]byte {
	return [][]byte{
		[]byte(`data: {"type":"message_start","message":{"id":"msg_cite","usage":{"input_tokens":1,"output_tokens":0}}}`),
		[]byte(`data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_cite","name":"web_search","input":{}}}`),
		[]byte(`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"DeepSeek V4 release date\"}"}}`),
		[]byte(`data: {"type":"content_block_stop","index":0}`),
		[]byte(`data: {"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_cite","content":[{"type":"web_search_result","title":"A","url":"https://example.com/a","encrypted_content":"ENC_A","page_age":"1 day"},{"type":"web_search_result","title":"B","url":"https://example.com/b","encrypted_content":"ENC_B"},{"type":"web_search_result","title":"A again","url":"https://example.com/a","encrypted_content":"ENC_A2"}]}}`),
		[]byte(`data: {"type":"content_block_stop","index":1}`),
		[]byte(`data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":"","citations":[]}}`),
		[]byte(`data: {"type":"content_block_delta","index":2,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","cited_text":"source A","url":"https://example.com/a","title":"A","encrypted_index":"IDX_A"}}}`),
		[]byte(`data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"前🙂"}}`),
		[]byte(`data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"引用"}}`),
		[]byte(`data: {"type":"content_block_stop","index":2}`),
		[]byte(`data: {"type":"content_block_start","index":3,"content_block":{"type":"text","text":""}}`),
		[]byte(`data: {"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":" and "}}`),
		[]byte(`data: {"type":"content_block_stop","index":3}`),
		[]byte(`data: {"type":"content_block_start","index":4,"content_block":{"type":"text","text":""}}`),
		[]byte(`data: {"type":"content_block_delta","index":4,"delta":{"type":"text_delta","text":"B fact."}}`),
		[]byte(`data: {"type":"content_block_delta","index":4,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","cited_text":"source B","url":"https://example.com/b","title":"B","encrypted_index":"IDX_B"}}}`),
		[]byte(`data: {"type":"content_block_stop","index":4}`),
		[]byte(`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`),
		[]byte(`data: {"type":"message_stop"}`),
	}
}

const webSearchCitationText = "前🙂引用 and B fact."

func translateWebSearchCitationStream(t *testing.T, request []byte) (map[string][]gjson.Result, gjson.Result) {
	t.Helper()
	var param any
	events := map[string][]gjson.Result{}
	var completed gjson.Result
	for _, chunk := range webSearchCitationChunks() {
		for _, output := range ConvertClaudeResponseToOpenAIResponses(context.Background(), "claude-test", request, nil, chunk, &param) {
			event, data := parseClaudeResponsesSSEEvent(t, output)
			events[event] = append(events[event], data)
			if event == "response.completed" {
				completed = data.Get("response")
			}
		}
	}
	return events, completed
}

func translateWebSearchCitationNonStream(request []byte) gjson.Result {
	var lines []string
	for _, chunk := range webSearchCitationChunks() {
		lines = append(lines, string(chunk))
	}
	return gjson.ParseBytes(ConvertClaudeResponseToOpenAIResponsesNonStream(context.Background(), "claude-test", request, nil, []byte(strings.Join(lines, "\n")), nil))
}

func checkWebSearchCitationAnnotations(t *testing.T, label string, annotations gjson.Result) {
	t.Helper()
	items := annotations.Array()
	if len(items) != 2 {
		t.Fatalf("%s: annotations = %s, want 2", label, annotations.Raw)
	}
	want := []struct {
		url, title, citedText, encryptedIndex string
		start, end                            int64
	}{
		{"https://example.com/a", "A", "source A", "IDX_A", 0, 4},
		{"https://example.com/b", "B", "source B", "IDX_B", 9, 16},
	}
	runes := []rune(webSearchCitationText)
	for i, w := range want {
		a := items[i]
		if a.Get("type").String() != "url_citation" || a.Get("url").String() != w.url || a.Get("title").String() != w.title {
			t.Fatalf("%s: annotation %d = %s, want url_citation for %s", label, i, a.Raw, w.url)
		}
		if a.Get("start_index").Int() != w.start || a.Get("end_index").Int() != w.end {
			t.Fatalf("%s: annotation %d span = [%d,%d), want [%d,%d)", label, i, a.Get("start_index").Int(), a.Get("end_index").Int(), w.start, w.end)
		}
		if a.Get("cited_text").String() != w.citedText || a.Get("encrypted_index").String() != w.encryptedIndex {
			t.Fatalf("%s: annotation %d lost replay fields: %s", label, i, a.Raw)
		}
		if i == 1 && string(runes[w.start:w.end]) != "B fact." {
			t.Fatalf("%s: annotation %d covers %q", label, i, string(runes[w.start:w.end]))
		}
	}
}

func checkWebSearchCallSources(t *testing.T, label string, item gjson.Result, wantSources bool) {
	t.Helper()
	if item.Get("type").String() != "web_search_call" {
		t.Fatalf("%s: item = %s, want web_search_call", label, item.Raw)
	}
	if item.Get("action.query").String() != "DeepSeek V4 release date" {
		t.Fatalf("%s: action.query = %q", label, item.Get("action.query").String())
	}
	if got := item.Get("results.#").Int(); got != 3 || item.Get("results.0.encrypted_content").String() != "ENC_A" {
		t.Fatalf("%s: results must stay verbatim for replay: %s", label, item.Get("results").Raw)
	}
	sources := item.Get("action.sources")
	if !wantSources {
		if sources.Exists() {
			t.Fatalf("%s: action.sources = %s, want absent without include", label, sources.Raw)
		}
		return
	}
	if sources.Raw != `[{"type":"url","url":"https://example.com/a"},{"type":"url","url":"https://example.com/b"}]` {
		t.Fatalf("%s: action.sources = %s", label, sources.Raw)
	}
}

func TestClaudeWebSearchStandardShapeNonStream(t *testing.T) {
	request := []byte(`{"model":"claude-test","tools":[{"type":"web_search"}],"include":["web_search_call.action.sources"]}`)
	out := translateWebSearchCitationNonStream(request)
	output := out.Get("output").Array()
	if len(output) != 2 {
		t.Fatalf("output = %s, want web_search_call and message", out.Get("output").Raw)
	}
	checkWebSearchCallSources(t, "nonstream", output[0], true)
	if got := output[1].Get("content.0.text").String(); got != webSearchCitationText {
		t.Fatalf("message text = %q", got)
	}
	checkWebSearchCitationAnnotations(t, "nonstream", output[1].Get("content.0.annotations"))

	withoutInclude := translateWebSearchCitationNonStream([]byte(`{"model":"claude-test","tools":[{"type":"web_search"}]}`))
	checkWebSearchCallSources(t, "nonstream without include", withoutInclude.Get("output.0"), false)
}

func TestClaudeWebSearchStandardShapeStream(t *testing.T) {
	request := []byte(`{"model":"claude-test","tools":[{"type":"web_search"}],"include":["reasoning.encrypted_content","web_search_call.action.sources"]}`)
	events, completed := translateWebSearchCitationStream(t, request)

	added := events["response.output_text.annotation.added"]
	if len(added) != 2 {
		t.Fatalf("annotation.added events = %d, want 2", len(added))
	}
	var addedAnnotations []json.RawMessage
	for i, event := range added {
		if event.Get("annotation_index").Int() != int64(i) || event.Get("output_index").Int() != 1 || event.Get("content_index").Int() != 0 || event.Get("item_id").String() == "" {
			t.Fatalf("annotation.added %d = %s", i, event.Raw)
		}
		addedAnnotations = append(addedAnnotations, json.RawMessage(event.Get("annotation").Raw))
	}
	addedJSON, _ := json.Marshal(addedAnnotations)
	checkWebSearchCitationAnnotations(t, "annotation.added", gjson.ParseBytes(addedJSON))

	if parts := events["response.content_part.done"]; len(parts) != 1 {
		t.Fatalf("content_part.done events = %d", len(parts))
	} else {
		checkWebSearchCitationAnnotations(t, "content_part.done", parts[0].Get("part.annotations"))
	}
	var webSearchDone, messageDone gjson.Result
	for _, event := range events["response.output_item.done"] {
		switch event.Get("item.type").String() {
		case "web_search_call":
			webSearchDone = event.Get("item")
		case "message":
			messageDone = event.Get("item")
		}
	}
	checkWebSearchCallSources(t, "output_item.done", webSearchDone, true)
	checkWebSearchCitationAnnotations(t, "output_item.done", messageDone.Get("content.0.annotations"))

	output := completed.Get("output").Array()
	if len(output) != 2 {
		t.Fatalf("completed output = %s", completed.Get("output").Raw)
	}
	checkWebSearchCallSources(t, "completed", output[0], true)
	checkWebSearchCitationAnnotations(t, "completed", output[1].Get("content.0.annotations"))

	_, withoutInclude := translateWebSearchCitationStream(t, []byte(`{"model":"claude-test","include":["reasoning.encrypted_content"]}`))
	checkWebSearchCallSources(t, "completed without include", withoutInclude.Get("output.0"), false)
}

// A url_citation produced on the way out must replay as the Claude citation
// Anthropic expects: the original fields, without the Responses-only offsets.
func TestURLCitationReplaysAsClaudeCitation(t *testing.T) {
	out := translateWebSearchCitationNonStream([]byte(`{"model":"claude-test","include":["web_search_call.action.sources"]}`))
	var items []json.RawMessage
	items = append(items, json.RawMessage(`{"type":"message","role":"user","content":[{"type":"input_text","text":"search"}]}`))
	for _, item := range out.Get("output").Array() {
		items = append(items, json.RawMessage(item.Raw))
	}
	request, _ := json.Marshal(map[string]any{"model": "claude-test", "input": items})
	replayed, _ := ConvertOpenAIResponsesRequestToClaude("claude-test", request, false)

	if got := claudeAssistantBlockTypes(t, replayed); strings.Join(got, ",") != "server_tool_use,web_search_tool_result,text" {
		t.Fatalf("replayed blocks = %v", got)
	}
	assistant := gjson.GetBytes(replayed, "messages.1.content")
	if got := assistant.Get("1.content.#").Int(); got != 3 {
		t.Fatalf("replayed search results = %d, want 3: %s", got, assistant.Get("1").Raw)
	}
	citations := assistant.Get("2.citations")
	want := `[{"type":"web_search_result_location","cited_text":"source A","url":"https://example.com/a","title":"A","encrypted_index":"IDX_A"},{"type":"web_search_result_location","cited_text":"source B","url":"https://example.com/b","title":"B","encrypted_index":"IDX_B"}]`
	if citations.Raw != want {
		t.Fatalf("replayed citations = %s, want %s", citations.Raw, want)
	}
}

func TestNativeMessagesJSONCitationSpansText(t *testing.T) {
	raw := []byte(`{"id":"msg_native","type":"message","model":"claude-native","content":[{"type":"text","text":"Intro. "},{"type":"text","text":"Cited.","citations":[{"type":"web_search_result_location","url":"https://example.com","title":"Source","cited_text":"Cited.","encrypted_index":"IDX"}]}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":2}}`)
	out := ConvertClaudeResponseToOpenAIResponsesNonStream(context.Background(), "", nil, nil, raw, nil)
	annotation := gjson.GetBytes(out, "output.0.content.0.annotations.0")
	if annotation.Get("type").String() != "url_citation" || annotation.Get("start_index").Int() != 7 || annotation.Get("end_index").Int() != 13 {
		t.Fatalf("annotation = %s, want url_citation over [7,13)", annotation.Raw)
	}
}

// A citation names its text block by index, so one that arrives after its block
// ended - before or during the next adjacent block - still annotates that block.
func TestLateClaudeCitationAnnotatesItsOwnBlock(t *testing.T) {
	chunks := [][]byte{
		[]byte(`data: {"type":"message_start","message":{"id":"msg_late","usage":{"input_tokens":1,"output_tokens":0}}}`),
		[]byte(`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		[]byte(`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"First."}}`),
		[]byte(`data: {"type":"content_block_stop","index":0}`),
		[]byte(`data: {"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","cited_text":"a","url":"https://example.com/a","title":"A","encrypted_index":"IDX_A"}}}`),
		[]byte(`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`),
		[]byte(`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":" Second."}}`),
		[]byte(`data: {"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","cited_text":"b","url":"https://example.com/b","title":"B","encrypted_index":"IDX_B"}}}`),
		[]byte(`data: {"type":"content_block_stop","index":1}`),
		[]byte(`data: {"type":"message_stop"}`),
	}
	check := func(t *testing.T, label string, annotations gjson.Result) {
		t.Helper()
		items := annotations.Array()
		if len(items) != 2 {
			t.Fatalf("%s: annotations = %s, want 2", label, annotations.Raw)
		}
		for _, a := range items {
			if a.Get("start_index").Int() != 0 || a.Get("end_index").Int() != 6 {
				t.Fatalf("%s: annotation %s, want span [0,6) of the first block", label, a.Raw)
			}
		}
	}

	var param any
	var completed gjson.Result
	var added int
	for _, chunk := range chunks {
		for _, output := range ConvertClaudeResponseToOpenAIResponses(context.Background(), "claude-test", nil, nil, chunk, &param) {
			event, data := parseClaudeResponsesSSEEvent(t, output)
			switch event {
			case "response.output_text.annotation.added":
				if data.Get("annotation_index").Int() != int64(added) {
					t.Fatalf("annotation_index = %d, want %d", data.Get("annotation_index").Int(), added)
				}
				added++
			case "response.completed":
				completed = data
			}
		}
	}
	if added != 2 {
		t.Fatalf("annotation.added events = %d, want 2", added)
	}
	check(t, "stream", completed.Get("response.output.0.content.0.annotations"))

	var lines []string
	for _, chunk := range chunks {
		lines = append(lines, string(chunk))
	}
	out := ConvertClaudeResponseToOpenAIResponsesNonStream(context.Background(), "claude-test", nil, nil, []byte(strings.Join(lines, "\n")), nil)
	check(t, "nonstream", gjson.GetBytes(out, "output.0.content.0.annotations"))
}

// A citation for a text block of a message already finalized by a later
// non-text block still annotates that message, in both paths.
func TestLateClaudeCitationAfterMessageFinalized(t *testing.T) {
	chunks := [][]byte{
		[]byte(`data: {"type":"message_start","message":{"id":"msg_late2","usage":{"input_tokens":1,"output_tokens":0}}}`),
		[]byte(`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		[]byte(`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"First."}}`),
		[]byte(`data: {"type":"content_block_stop","index":0}`),
		[]byte(`data: {"type":"content_block_start","index":1,"content_block":{"type":"server_tool_use","id":"srvtoolu_late","name":"web_search","input":{}}}`),
		[]byte(`data: {"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","cited_text":"a","url":"https://example.com/a","title":"A","encrypted_index":"IDX_A"}}}`),
		[]byte(`data: {"type":"content_block_stop","index":1}`),
		[]byte(`data: {"type":"content_block_start","index":2,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_late","content":[]}}`),
		[]byte(`data: {"type":"content_block_stop","index":2}`),
		[]byte(`data: {"type":"content_block_start","index":3,"content_block":{"type":"text","text":""}}`),
		[]byte(`data: {"type":"content_block_delta","index":3,"delta":{"type":"text_delta","text":"Second."}}`),
		[]byte(`data: {"type":"content_block_stop","index":3}`),
		[]byte(`data: {"type":"message_stop"}`),
	}
	check := func(t *testing.T, label string, output gjson.Result) {
		t.Helper()
		first := output.Get("0.content.0.annotations")
		if len(first.Array()) != 1 || first.Get("0.url").String() != "https://example.com/a" || first.Get("0.start_index").Int() != 0 || first.Get("0.end_index").Int() != 6 {
			t.Fatalf("%s: first message annotations = %s, want the late citation over [0,6)", label, first.Raw)
		}
		if got := output.Get("2.content.0.annotations.#").Int(); got != 0 {
			t.Fatalf("%s: second message annotations = %d, want 0", label, got)
		}
	}

	var param any
	var completed, added gjson.Result
	for _, chunk := range chunks {
		for _, output := range ConvertClaudeResponseToOpenAIResponses(context.Background(), "claude-test", nil, nil, chunk, &param) {
			switch event, data := parseClaudeResponsesSSEEvent(t, output); event {
			case "response.output_text.annotation.added":
				added = data
			case "response.completed":
				completed = data
			}
		}
	}
	if added.Get("item_id").String() != completed.Get("response.output.0.id").String() || added.Get("output_index").Int() != 0 || added.Get("annotation_index").Int() != 0 {
		t.Fatalf("annotation.added = %s, want it on the first message", added.Raw)
	}
	check(t, "stream", completed.Get("response.output"))

	var lines []string
	for _, chunk := range chunks {
		lines = append(lines, string(chunk))
	}
	out := ConvertClaudeResponseToOpenAIResponsesNonStream(context.Background(), "claude-test", nil, nil, []byte(strings.Join(lines, "\n")), nil)
	check(t, "nonstream", gjson.GetBytes(out, "output"))
}

// With sources included, a failed or empty search yields an empty list and a
// search paused before its result yields neither sources nor results.
func TestClaudeWebSearchSourcesEdgeCases(t *testing.T) {
	request := []byte(`{"model":"claude-test","include":["web_search_call.action.sources"]}`)
	cases := []struct {
		name, result, stopReason, wantStatus, wantSources string
	}{
		{"error", `{"type":"web_search_tool_result_error","error_code":"max_uses_exceeded"}`, "end_turn", "completed", "[]"},
		{"empty", `[]`, "end_turn", "completed", "[]"},
		{"paused", "", "pause_turn", "in_progress", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			chunks := [][]byte{
				[]byte(`data: {"type":"message_start","message":{"id":"msg_src","usage":{"input_tokens":1,"output_tokens":0}}}`),
				[]byte(`data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_src","name":"web_search","input":{"query":"q"}}}`),
				[]byte(`data: {"type":"content_block_stop","index":0}`),
			}
			if tc.result != "" {
				chunks = append(chunks,
					[]byte(`data: {"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_src","content":`+tc.result+`}}`),
					[]byte(`data: {"type":"content_block_stop","index":1}`))
			}
			chunks = append(chunks,
				[]byte(`data: {"type":"message_delta","delta":{"stop_reason":"`+tc.stopReason+`"},"usage":{"output_tokens":1}}`),
				[]byte(`data: {"type":"message_stop"}`))

			var param any
			var streamed gjson.Result
			for _, chunk := range chunks {
				for _, output := range ConvertClaudeResponseToOpenAIResponses(context.Background(), "claude-test", request, nil, chunk, &param) {
					if event, data := parseClaudeResponsesSSEEvent(t, output); event == "response.completed" {
						streamed = data.Get("response.output.0")
					}
				}
			}
			var lines []string
			for _, chunk := range chunks {
				lines = append(lines, string(chunk))
			}
			nonStream := gjson.GetBytes(ConvertClaudeResponseToOpenAIResponsesNonStream(context.Background(), "claude-test", request, nil, []byte(strings.Join(lines, "\n")), nil), "output.0")

			for label, item := range map[string]gjson.Result{"stream": streamed, "nonstream": nonStream} {
				if got := item.Get("status").String(); got != tc.wantStatus {
					t.Fatalf("%s: status = %q, want %q", label, got, tc.wantStatus)
				}
				if got := item.Get("action.sources").Raw; got != tc.wantSources {
					t.Fatalf("%s: action.sources = %q, want %q", label, got, tc.wantSources)
				}
				if got, want := item.Get("results").Exists(), tc.result != ""; got != want {
					t.Fatalf("%s: results present = %v, want %v: %s", label, got, want, item.Raw)
				}
			}
		})
	}
}

// Citations still get the right spans when text blocks never see their stop
// event, and the stream keeps strictly increasing sequence numbers.
func TestClaudeCitationsWithoutBlockStop(t *testing.T) {
	chunks := [][]byte{
		[]byte(`data: {"type":"message_start","message":{"id":"msg_nostop","usage":{"input_tokens":1,"output_tokens":0}}}`),
		[]byte(`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`),
		[]byte(`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"AB"}}`),
		[]byte(`data: {"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","cited_text":"a","url":"https://example.com/a","title":"A","encrypted_index":"IDX_A"}}}`),
		[]byte(`data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`),
		[]byte(`data: {"type":"content_block_delta","index":1,"delta":{"type":"citations_delta","citation":{"type":"web_search_result_location","cited_text":"b","url":"https://example.com/b","title":"B","encrypted_index":"IDX_B"}}}`),
		[]byte(`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"CD"}}`),
		[]byte(`data: {"type":"message_stop"}`),
	}
	check := func(t *testing.T, label string, annotations gjson.Result) {
		t.Helper()
		if annotations.Raw == "" || annotations.Get("#").Int() != 2 ||
			annotations.Get("0.start_index").Int() != 0 || annotations.Get("0.end_index").Int() != 2 ||
			annotations.Get("1.start_index").Int() != 2 || annotations.Get("1.end_index").Int() != 4 {
			t.Fatalf("%s: annotations = %s, want [0,2) and [2,4)", label, annotations.Raw)
		}
	}

	var param any
	var completed gjson.Result
	lastSeq := int64(0)
	for _, chunk := range chunks {
		for _, output := range ConvertClaudeResponseToOpenAIResponses(context.Background(), "claude-test", nil, nil, chunk, &param) {
			event, data := parseClaudeResponsesSSEEvent(t, output)
			if seq := data.Get("sequence_number").Int(); seq <= lastSeq {
				t.Fatalf("%s sequence_number = %d after %d", event, seq, lastSeq)
			} else {
				lastSeq = seq
			}
			if event == "response.completed" {
				completed = data
			}
		}
	}
	check(t, "stream", completed.Get("response.output.0.content.0.annotations"))

	var lines []string
	for _, chunk := range chunks {
		lines = append(lines, string(chunk))
	}
	out := ConvertClaudeResponseToOpenAIResponsesNonStream(context.Background(), "claude-test", nil, nil, []byte(strings.Join(lines, "\n")), nil)
	check(t, "nonstream", gjson.GetBytes(out, "output.0.content.0.annotations"))
}
