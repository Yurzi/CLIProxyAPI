package executor

import (
	"bytes"
	"context"
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v8/sdk/translator"
)

const openAICompatAudioTestFile = "ID3\x04\x00binary-audio\r\n\xff\xfe"

type openAICompatAudioCapture struct {
	path          string
	authorization string
	contentType   string
	fields        map[string][]string
	files         map[string]string
	fileTypes     map[string]string
}

func newOpenAICompatAudioServer(t *testing.T, status int, responseType, response string) (*httptest.Server, *openAICompatAudioCapture) {
	t.Helper()
	captured := &openAICompatAudioCapture{fields: map[string][]string{}, files: map[string]string{}, fileTypes: map[string]string{}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured.path = r.URL.Path
		captured.authorization = r.Header.Get("Authorization")
		captured.contentType = r.Header.Get("Content-Type")
		if errParse := r.ParseMultipartForm(32 << 20); errParse != nil {
			t.Errorf("upstream ParseMultipartForm: %v", errParse)
		} else {
			for key, values := range r.MultipartForm.Value {
				captured.fields[key] = values
			}
			for key, headers := range r.MultipartForm.File {
				file, errOpen := headers[0].Open()
				if errOpen != nil {
					t.Errorf("open upload: %v", errOpen)
					continue
				}
				data, _ := io.ReadAll(file)
				_ = file.Close()
				captured.files[key] = headers[0].Filename + ":" + string(data)
				captured.fileTypes[key] = headers[0].Header.Get("Content-Type")
			}
		}
		w.Header().Set("Content-Type", responseType)
		w.WriteHeader(status)
		_, _ = w.Write([]byte(response))
	}))
	t.Cleanup(server.Close)
	return server, captured
}

func newOpenAICompatAudioBody(t *testing.T, fields [][2]string) ([]byte, string) {
	t.Helper()
	var buffer bytes.Buffer
	writer := multipart.NewWriter(&buffer)
	header := make(textproto.MIMEHeader)
	header.Set("Content-Disposition", multipart.FileContentDisposition("file", "speech.mp3"))
	header.Set("Content-Type", "audio/mpeg")
	part, err := writer.CreatePart(header)
	if err != nil {
		t.Fatalf("CreatePart: %v", err)
	}
	if _, err = part.Write([]byte(openAICompatAudioTestFile)); err != nil {
		t.Fatalf("write file: %v", err)
	}
	for _, field := range fields {
		if err = writer.WriteField(field[0], field[1]); err != nil {
			t.Fatalf("WriteField: %v", err)
		}
	}
	if err = writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	return buffer.Bytes(), writer.FormDataContentType()
}

func executeOpenAICompatAudio(t *testing.T, cfg *config.Config, baseURL, source, model string, body []byte, contentType string) (cliproxyexecutor.Response, error) {
	t.Helper()
	executor := NewOpenAICompatExecutor("openai-compatibility", cfg)
	return executor.Execute(context.Background(), newOpenAICompatJSONEndpointAuth(baseURL), cliproxyexecutor.Request{
		Model:   model,
		Payload: body,
	}, cliproxyexecutor.Options{
		SourceFormat:    sdktranslator.FromString(source),
		OriginalRequest: body,
		Headers:         http.Header{"Content-Type": []string{contentType}},
	})
}

func TestOpenAICompatExecutorAudioEndpointsPassthrough(t *testing.T) {
	cases := []struct {
		name         string
		source       string
		wantPath     string
		responseType string
		response     string
	}{
		{name: "transcriptions text", source: "openai-transcription", wantPath: "/v1/audio/transcriptions", responseType: "text/plain; charset=utf-8", response: "你好，世界\n"},
		{name: "translations json", source: "openai-translation", wantPath: "/v1/audio/translations", responseType: "application/json", response: `{"text":"Hello, world"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server, captured := newOpenAICompatAudioServer(t, http.StatusOK, tc.responseType, tc.response)
			body, contentType := newOpenAICompatAudioBody(t, [][2]string{
				{"model", "whisper-alias"},
				{"response_format", "text"},
				{"temperature", "0.2"},
				{"timestamp_granularities[]", "word"},
				{"timestamp_granularities[]", "segment"},
			})
			resp, err := executeOpenAICompatAudio(t, &config.Config{}, server.URL, tc.source, "whisper-1", body, contentType)
			if err != nil {
				t.Fatalf("Execute error: %v", err)
			}
			if captured.path != tc.wantPath {
				t.Fatalf("path = %q, want %q", captured.path, tc.wantPath)
			}
			if captured.authorization != "Bearer test" {
				t.Fatalf("authorization = %q, want Bearer test", captured.authorization)
			}
			if mediaType, _, _ := mime.ParseMediaType(captured.contentType); mediaType != "multipart/form-data" {
				t.Fatalf("content type = %q, want multipart/form-data", captured.contentType)
			}
			if got := captured.fields["model"]; len(got) != 1 || got[0] != "whisper-1" {
				t.Fatalf("model = %v, want [whisper-1]", got)
			}
			if got := captured.fields["timestamp_granularities[]"]; len(got) != 2 || got[0] != "word" || got[1] != "segment" {
				t.Fatalf("timestamp_granularities[] = %v, want [word segment]", got)
			}
			if got := captured.fields["temperature"]; len(got) != 1 || got[0] != "0.2" {
				t.Fatalf("temperature = %v, want [0.2]", got)
			}
			if got := captured.files["file"]; got != "speech.mp3:"+openAICompatAudioTestFile {
				t.Fatalf("file = %q, want original upload", got)
			}
			if got := captured.fileTypes["file"]; got != "audio/mpeg" {
				t.Fatalf("file content type = %q, want audio/mpeg", got)
			}
			if string(resp.Payload) != tc.response {
				t.Fatalf("response payload = %q, want upstream body %q", resp.Payload, tc.response)
			}
		})
	}
}

// Payload rules must be the final semantic barrier: they run after the built-in model
// rewrite, so a configured override of "model" wins and file parts survive the rebuild.
func TestOpenAICompatExecutorAudioPayloadRulesAreFinalBarrier(t *testing.T) {
	server, captured := newOpenAICompatAudioServer(t, http.StatusOK, "application/json", `{"text":"ok"}`)
	body, contentType := newOpenAICompatAudioBody(t, [][2]string{{"model", "whisper-alias"}, {"language", "zh"}})
	cfg := &config.Config{
		Payload: config.PayloadConfig{
			Default: []config.PayloadRule{{
				Models: []config.PayloadModelRule{{Name: "whisper-upstream", FromProtocol: "openai-transcription"}},
				Params: map[string]any{"language": "en", "prompt": "glossary"},
			}},
			Override: []config.PayloadRule{{
				Models: []config.PayloadModelRule{{Name: "whisper-upstream", Protocol: "openai"}},
				Params: map[string]any{"model": "forced-whisper", "response_format": "verbose_json"},
			}},
		},
	}
	if _, err := executeOpenAICompatAudio(t, cfg, server.URL, "openai-transcription", "whisper-upstream", body, contentType); err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	want := map[string]string{"model": "forced-whisper", "response_format": "verbose_json", "language": "zh", "prompt": "glossary"}
	for key, value := range want {
		if got := captured.fields[key]; len(got) != 1 || got[0] != value {
			t.Fatalf("%s = %v, want [%s]; fields=%v", key, got, value, captured.fields)
		}
	}
	if got := captured.files["file"]; got != "speech.mp3:"+openAICompatAudioTestFile {
		t.Fatalf("file = %q, want original upload", got)
	}
}

func TestOpenAICompatExecutorAudioUpstreamErrorKeepsStatus(t *testing.T) {
	server, _ := newOpenAICompatAudioServer(t, http.StatusRequestEntityTooLarge, "application/json", `{"error":{"message":"file too large"}}`)
	body, contentType := newOpenAICompatAudioBody(t, [][2]string{{"model", "whisper-1"}})
	_, err := executeOpenAICompatAudio(t, &config.Config{}, server.URL, "openai-translation", "whisper-1", body, contentType)
	var status statusErr
	if !errors.As(err, &status) || status.StatusCode() != http.StatusRequestEntityTooLarge {
		t.Fatalf("error = %#v, want status %d", err, http.StatusRequestEntityTooLarge)
	}
}

func TestOpenAICompatExecutorAudioRejectsNonMultipart(t *testing.T) {
	server, captured := newOpenAICompatAudioServer(t, http.StatusOK, "application/json", `{}`)
	_, err := executeOpenAICompatAudio(t, &config.Config{}, server.URL, "openai-transcription", "whisper-1", []byte(`{"model":"whisper-1"}`), "application/json")
	var status statusErr
	if !errors.As(err, &status) || status.StatusCode() != http.StatusBadRequest {
		t.Fatalf("error = %#v, want status %d", err, http.StatusBadRequest)
	}
	if captured.path != "" {
		t.Fatalf("upstream should not be called, got path %q", captured.path)
	}
}
