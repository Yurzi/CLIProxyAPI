package executor

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/gorilla/websocket"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	_ "github.com/router-for-me/CLIProxyAPI/v8/internal/translator"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v8/sdk/translator"
	"github.com/tidwall/gjson"
)

const codexInstructionRoleResponsesPayload = `{"model":"gpt-5.5","input":[{"type":"message","role":"system","content":[{"type":"input_text","text":"sys"}]},{"type":"message","role":"developer","content":[{"type":"input_text","text":"dev"}]},{"type":"additional_tools","role":"developer","tools":[{"type":"custom","name":"exec"}]},{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`

func codexInstructionRoleTestAuth(baseURL, role string) *cliproxyauth.Auth {
	auth := codexAPIKeyTestAuth(baseURL)
	auth.ID = "codex-instruction-role-" + role
	if role != "" {
		auth.Attributes[cliproxyauth.AttributeCodexInstructionRole] = role
	}
	return auth
}

func codexInstructionRoleTestConfig() *config.Config {
	return &config.Config{SDKConfig: config.SDKConfig{DisableImageGeneration: config.DisableImageGenerationAll}}
}

func newCodexInstructionRoleHTTPServer(t *testing.T, captured *[]byte) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, errRead := io.ReadAll(r.Body)
		if errRead != nil {
			t.Errorf("read upstream body: %v", errRead)
		}
		*captured = body
		if r.URL.Path == "/responses/compact" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"id":"cmp_1","object":"response.compaction","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"type":"response.completed","response":{"id":"resp_1","object":"response","status":"completed","model":"gpt-5.5","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}` + "\n\n"))
	}))
}

type codexInstructionRoleFrames struct {
	mu     sync.Mutex
	frames [][]byte
}

func newCodexInstructionRoleWebsocketServer(t *testing.T, rec *codexInstructionRoleFrames) *httptest.Server {
	t.Helper()
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, errUpgrade := upgrader.Upgrade(w, r, nil)
		if errUpgrade != nil {
			t.Errorf("upgrade websocket: %v", errUpgrade)
			return
		}
		defer func() { _ = conn.Close() }()
		for {
			_, payload, errRead := conn.ReadMessage()
			if errRead != nil {
				return
			}
			rec.mu.Lock()
			rec.frames = append(rec.frames, payload)
			rec.mu.Unlock()
			completed := []byte(`{"type":"response.completed","response":{"id":"resp-1","object":"response","status":"completed","model":"gpt-5.5","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`)
			if errWrite := conn.WriteMessage(websocket.TextMessage, completed); errWrite != nil {
				return
			}
		}
	}))
}

func (rec *codexInstructionRoleFrames) single(t *testing.T) []byte {
	t.Helper()
	rec.mu.Lock()
	defer rec.mu.Unlock()
	if len(rec.frames) != 1 {
		t.Fatalf("websocket frames = %d, want 1", len(rec.frames))
	}
	return rec.frames[0]
}

func assertCodexInputRoles(t *testing.T, body []byte, want ...string) {
	t.Helper()
	items := gjson.GetBytes(body, "input").Array()
	if len(items) != len(want) {
		t.Fatalf("upstream input length = %d, want %d: %s", len(items), len(want), body)
	}
	for i, role := range want {
		if got := items[i].Get("role").String(); got != role {
			t.Errorf("upstream input[%d].role = %q, want %q: %s", i, got, role, body)
		}
	}
}

func drainCodexInstructionRoleStream(t *testing.T, result *cliproxyexecutor.StreamResult, errStream error) {
	t.Helper()
	if errStream != nil {
		t.Fatalf("ExecuteStream error: %v", errStream)
	}
	for chunk := range result.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
	}
}

func TestCodexExecutorInstructionRoleRewritesHTTPRequests(t *testing.T) {
	responsesOpts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAIResponse}
	cases := []struct {
		name string
		role string
		run  func(t *testing.T, exec *CodexExecutor, auth *cliproxyauth.Auth)
		want []string
	}{
		{
			name: "execute system",
			role: config.CodexInstructionRoleSystem,
			run: func(t *testing.T, exec *CodexExecutor, auth *cliproxyauth.Auth) {
				if _, err := exec.Execute(context.Background(), auth, cliproxyexecutor.Request{Model: "gpt-5.5", Payload: []byte(codexInstructionRoleResponsesPayload)}, responsesOpts); err != nil {
					t.Fatalf("Execute error: %v", err)
				}
			},
			want: []string{"system", "developer", "user"},
		},
		{
			name: "execute default keeps developer conversion",
			run: func(t *testing.T, exec *CodexExecutor, auth *cliproxyauth.Auth) {
				if _, err := exec.Execute(context.Background(), auth, cliproxyexecutor.Request{Model: "gpt-5.5", Payload: []byte(codexInstructionRoleResponsesPayload)}, responsesOpts); err != nil {
					t.Fatalf("Execute error: %v", err)
				}
			},
			want: []string{"developer", "developer", "developer", "user"},
		},
		{
			name: "stream from chat completions system",
			role: config.CodexInstructionRoleSystem,
			run: func(t *testing.T, exec *CodexExecutor, auth *cliproxyauth.Auth) {
				result, err := exec.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{
					Model:   "gpt-5.5",
					Payload: []byte(`{"model":"gpt-5.5","stream":true,"messages":[{"role":"system","content":"sys"},{"role":"user","content":"hi"}]}`),
				}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAI, Stream: true})
				drainCodexInstructionRoleStream(t, result, err)
			},
			want: []string{"system", "user"},
		},
		{
			name: "compact developer",
			role: config.CodexInstructionRoleDeveloper,
			run: func(t *testing.T, exec *CodexExecutor, auth *cliproxyauth.Auth) {
				opts := responsesOpts
				opts.Alt = "responses/compact"
				if _, err := exec.Execute(context.Background(), auth, cliproxyexecutor.Request{Model: "gpt-5.5", Payload: []byte(codexInstructionRoleResponsesPayload)}, opts); err != nil {
					t.Fatalf("Execute compact error: %v", err)
				}
			},
			want: []string{"developer", "developer", "developer", "user"},
		},
		{
			name: "compact system",
			role: config.CodexInstructionRoleSystem,
			run: func(t *testing.T, exec *CodexExecutor, auth *cliproxyauth.Auth) {
				opts := responsesOpts
				opts.Alt = "responses/compact"
				if _, err := exec.Execute(context.Background(), auth, cliproxyexecutor.Request{Model: "gpt-5.5", Payload: []byte(codexInstructionRoleResponsesPayload)}, opts); err != nil {
					t.Fatalf("Execute compact error: %v", err)
				}
			},
			want: []string{"system", "developer", "user"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var upstreamBody []byte
			server := newCodexInstructionRoleHTTPServer(t, &upstreamBody)
			defer server.Close()

			tc.run(t, NewCodexExecutor(codexInstructionRoleTestConfig()), codexInstructionRoleTestAuth(server.URL, tc.role))
			assertCodexInputRoles(t, upstreamBody, tc.want...)
		})
	}
}

func TestCodexWebsocketsInstructionRoleRewritesRequests(t *testing.T) {
	req := cliproxyexecutor.Request{Model: "gpt-5.5", Payload: []byte(codexInstructionRoleResponsesPayload)}
	t.Run("execute", func(t *testing.T) {
		var rec codexInstructionRoleFrames
		server := newCodexInstructionRoleWebsocketServer(t, &rec)
		defer server.Close()

		exec := NewCodexWebsocketsExecutor(codexInstructionRoleTestConfig())
		if _, err := exec.Execute(context.Background(), codexInstructionRoleTestAuth(server.URL, config.CodexInstructionRoleSystem), req, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAIResponse}); err != nil {
			t.Fatalf("Execute error: %v", err)
		}
		assertCodexInputRoles(t, rec.single(t), "system", "developer", "user")
	})
	t.Run("stream", func(t *testing.T) {
		var rec codexInstructionRoleFrames
		server := newCodexInstructionRoleWebsocketServer(t, &rec)
		defer server.Close()

		exec := NewCodexWebsocketsExecutor(codexInstructionRoleTestConfig())
		result, err := exec.ExecuteStream(context.Background(), codexInstructionRoleTestAuth(server.URL, config.CodexInstructionRoleSystem), req, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAIResponse, Stream: true})
		drainCodexInstructionRoleStream(t, result, err)
		assertCodexInputRoles(t, rec.single(t), "system", "developer", "user")
	})
}

// Upstreams that accept a single leading system message must never receive a
// second system message, whether it comes from several downstream system
// prompts, a mid-conversation prompt, or the top-level instructions field.
func TestCodexExecutorInstructionRoleSystemSendsSingleLeadingSystem(t *testing.T) {
	t.Run("chat completions with several system prompts", func(t *testing.T) {
		var upstreamBody []byte
		server := newCodexInstructionRoleHTTPServer(t, &upstreamBody)
		defer server.Close()

		result, err := NewCodexExecutor(codexInstructionRoleTestConfig()).ExecuteStream(context.Background(), codexInstructionRoleTestAuth(server.URL, config.CodexInstructionRoleSystem), cliproxyexecutor.Request{
			Model:   "gpt-5.5",
			Payload: []byte(`{"model":"gpt-5.5","stream":true,"messages":[{"role":"system","content":"a"},{"role":"system","content":"b"},{"role":"user","content":"hi"},{"role":"assistant","content":"hello"},{"role":"system","content":"c"},{"role":"user","content":"again"}]}`),
		}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAI, Stream: true})
		drainCodexInstructionRoleStream(t, result, err)

		assertCodexInputRoles(t, upstreamBody, "system", "user", "assistant", "user", "user")
		if texts := gjson.GetBytes(upstreamBody, "input.0.content.#.text").String(); texts != `["a","b"]` {
			t.Fatalf("leading system texts = %s, want [\"a\",\"b\"]", texts)
		}
		if text := gjson.GetBytes(upstreamBody, "input.3.content.0.text").String(); text != "<system-reminder>\nc\n</system-reminder>" {
			t.Fatalf("demoted system text = %q", text)
		}
	})
	t.Run("responses instructions with developer prompts", func(t *testing.T) {
		var upstreamBody []byte
		server := newCodexInstructionRoleHTTPServer(t, &upstreamBody)
		defer server.Close()

		payload := `{"model":"gpt-5.5","instructions":"base","input":[{"type":"message","role":"developer","content":[{"type":"input_text","text":"dev1"}]},{"type":"message","role":"developer","content":[{"type":"input_text","text":"dev2"}]},{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}`
		if _, err := NewCodexExecutor(codexInstructionRoleTestConfig()).Execute(context.Background(), codexInstructionRoleTestAuth(server.URL, config.CodexInstructionRoleSystem), cliproxyexecutor.Request{Model: "gpt-5.5", Payload: []byte(payload)}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAIResponse}); err != nil {
			t.Fatalf("Execute error: %v", err)
		}

		assertCodexInputRoles(t, upstreamBody, "system", "user")
		if instructions := gjson.GetBytes(upstreamBody, "instructions").String(); instructions != "" {
			t.Fatalf("instructions = %q, want merged into the system message", instructions)
		}
		if texts := gjson.GetBytes(upstreamBody, "input.0.content.#.text").String(); texts != `["base","dev1","dev2"]` {
			t.Fatalf("leading system texts = %s", texts)
		}
	})
}

// Payload rules are the final barrier: an override of an instruction message role
// must survive the instruction-role rewrite on every transport.
func TestCodexInstructionRoleRunsBeforePayloadRules(t *testing.T) {
	cfg := codexInstructionRoleTestConfig()
	cfg.Payload = config.PayloadConfig{Override: []config.PayloadRule{{
		Models: []config.PayloadModelRule{{Name: "gpt-5.5", Protocol: "codex"}},
		Params: map[string]any{"input.0.role": "developer"},
	}}}
	req := cliproxyexecutor.Request{Model: "gpt-5.5", Payload: []byte(codexInstructionRoleResponsesPayload)}
	opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatOpenAIResponse}
	want := []string{"developer", "developer", "user"}

	t.Run("http", func(t *testing.T) {
		var upstreamBody []byte
		server := newCodexInstructionRoleHTTPServer(t, &upstreamBody)
		defer server.Close()

		if _, err := NewCodexExecutor(cfg).Execute(context.Background(), codexInstructionRoleTestAuth(server.URL, config.CodexInstructionRoleSystem), req, opts); err != nil {
			t.Fatalf("Execute error: %v", err)
		}
		assertCodexInputRoles(t, upstreamBody, want...)
	})
	t.Run("websocket", func(t *testing.T) {
		var rec codexInstructionRoleFrames
		server := newCodexInstructionRoleWebsocketServer(t, &rec)
		defer server.Close()

		if _, err := NewCodexWebsocketsExecutor(cfg).Execute(context.Background(), codexInstructionRoleTestAuth(server.URL, config.CodexInstructionRoleSystem), req, opts); err != nil {
			t.Fatalf("Execute error: %v", err)
		}
		assertCodexInputRoles(t, rec.single(t), want...)
	})
}
