package helps

import (
	"bytes"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v8/sdk/cliproxy/executor"
)

type multipartTestPart struct {
	name     string
	filename string
	value    string
}

func buildMultipartTestBody(t *testing.T, parts []multipartTestPart) ([]byte, string) {
	t.Helper()
	var buffer bytes.Buffer
	writer := multipart.NewWriter(&buffer)
	for _, part := range parts {
		if part.filename == "" {
			if err := writer.WriteField(part.name, part.value); err != nil {
				t.Fatalf("WriteField: %v", err)
			}
			continue
		}
		header := make(textproto.MIMEHeader)
		header.Set("Content-Disposition", multipart.FileContentDisposition(part.name, part.filename))
		header.Set("Content-Type", "audio/mpeg")
		dst, err := writer.CreatePart(header)
		if err != nil {
			t.Fatalf("CreatePart: %v", err)
		}
		if _, err = dst.Write([]byte(part.value)); err != nil {
			t.Fatalf("write part: %v", err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	return buffer.Bytes(), writer.FormDataContentType()
}

func readMultipartTestBody(t *testing.T, body []byte, contentType string) []multipartTestPart {
	t.Helper()
	_, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		t.Fatalf("ParseMediaType: %v", err)
	}
	reader := multipart.NewReader(bytes.NewReader(body), params["boundary"])
	var parts []multipartTestPart
	for {
		part, errNext := reader.NextPart()
		if errNext == io.EOF {
			return parts
		}
		if errNext != nil {
			t.Fatalf("NextPart: %v", errNext)
		}
		data, errRead := io.ReadAll(part)
		if errRead != nil {
			t.Fatalf("ReadAll: %v", errRead)
		}
		if part.FileName() != "" && part.Header.Get("Content-Type") != "audio/mpeg" {
			t.Fatalf("file part content type = %q, want audio/mpeg", part.Header.Get("Content-Type"))
		}
		parts = append(parts, multipartTestPart{name: part.FormName(), filename: part.FileName(), value: string(data)})
	}
}

func TestSetMultipartFormFieldReplacesValueAndPreservesParts(t *testing.T) {
	audio := "ID3\x00\x01binary\r\n--not-a-boundary\xff"
	body, contentType := buildMultipartTestBody(t, []multipartTestPart{
		{name: "file", filename: "speech.mp3", value: audio},
		{name: "model", value: "whisper-alias"},
		{name: "language", value: "zh"},
		{name: "timestamp_granularities[]", value: "word"},
		{name: "timestamp_granularities[]", value: "segment"},
	})

	out, outContentType, err := SetMultipartFormField(body, contentType, "model", "whisper-1")
	if err != nil {
		t.Fatalf("SetMultipartFormField: %v", err)
	}
	want := []multipartTestPart{
		{name: "file", filename: "speech.mp3", value: audio},
		{name: "model", value: "whisper-1"},
		{name: "language", value: "zh"},
		{name: "timestamp_granularities[]", value: "word"},
		{name: "timestamp_granularities[]", value: "segment"},
	}
	got := readMultipartTestBody(t, out, outContentType)
	if len(got) != len(want) {
		t.Fatalf("parts = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("part %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestSetMultipartFormFieldKeepsBodyWhenUnchanged(t *testing.T) {
	body, contentType := buildMultipartTestBody(t, []multipartTestPart{
		{name: "model", value: "whisper-1"},
		{name: "file", filename: "a.wav", value: "RIFF"},
	})
	out, outContentType, err := SetMultipartFormField(body, contentType, "model", "whisper-1")
	if err != nil {
		t.Fatalf("SetMultipartFormField: %v", err)
	}
	if &out[0] != &body[0] || len(out) != len(body) || outContentType != contentType {
		t.Fatal("unchanged field should return the original body without copying")
	}
}

func TestApplyMediaPayloadConfigSkipsMultipartWithoutRules(t *testing.T) {
	body, contentType := buildMultipartTestBody(t, []multipartTestPart{
		{name: "model", value: "whisper-1"},
		{name: "file", filename: "a.wav", value: "RIFF"},
	})
	opts := cliproxyexecutor.Options{OriginalRequest: body, Headers: http.Header{"Content-Type": []string{contentType}}}
	out, outContentType, err := ApplyMediaPayloadConfig(&config.Config{}, "compat", "whisper-1", "openai", body, contentType, cliproxyexecutor.Request{Model: "whisper-1", Payload: body}, opts)
	if err != nil {
		t.Fatalf("ApplyMediaPayloadConfig: %v", err)
	}
	if &out[0] != &body[0] || len(out) != len(body) || outContentType != contentType {
		t.Fatal("multipart body without payload rules should pass through without decoding")
	}
}

func TestSetMultipartFormFieldAppendsMissingAndCollapsesDuplicates(t *testing.T) {
	missing, missingType := buildMultipartTestBody(t, []multipartTestPart{{name: "file", filename: "a.wav", value: "RIFF"}})
	out, outType, err := SetMultipartFormField(missing, missingType, "model", "whisper-1")
	if err != nil {
		t.Fatalf("SetMultipartFormField missing: %v", err)
	}
	if got := readMultipartTestBody(t, out, outType); len(got) != 2 || got[1] != (multipartTestPart{name: "model", value: "whisper-1"}) {
		t.Fatalf("parts = %+v, want appended model", got)
	}

	duplicated, duplicatedType := buildMultipartTestBody(t, []multipartTestPart{
		{name: "model", value: "whisper-1"},
		{name: "model", value: "other"},
	})
	out, outType, err = SetMultipartFormField(duplicated, duplicatedType, "model", "whisper-1")
	if err != nil {
		t.Fatalf("SetMultipartFormField duplicated: %v", err)
	}
	if got := readMultipartTestBody(t, out, outType); len(got) != 1 || got[0].value != "whisper-1" {
		t.Fatalf("parts = %+v, want single model field", got)
	}
}

func TestSetMultipartFormFieldRejectsNonMultipart(t *testing.T) {
	for _, contentType := range []string{"application/json", "", "multipart/form-data"} {
		if _, _, err := SetMultipartFormField([]byte(`{"model":"x"}`), contentType, "model", "whisper-1"); err == nil {
			t.Fatalf("content type %q: expected error", contentType)
		}
	}
}
