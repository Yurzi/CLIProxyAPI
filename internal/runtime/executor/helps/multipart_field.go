package helps

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"strings"
)

// SetMultipartFormField sets the non-file form field name to value in a multipart/form-data
// body. All other parts keep their order, headers, and raw bytes. Duplicate fields collapse
// into one and a missing field is appended. When the body already carries exactly that
// value, the original body and content type are returned without copying.
func SetMultipartFormField(body []byte, contentType, name, value string) ([]byte, string, error) {
	mediaType, params, errParse := mime.ParseMediaType(contentType)
	if errParse != nil || !strings.EqualFold(mediaType, "multipart/form-data") {
		return nil, "", fmt.Errorf("content type %q is not multipart/form-data", contentType)
	}
	boundary := params["boundary"]
	if boundary == "" {
		return nil, "", errors.New("multipart boundary is missing")
	}
	unchanged, errScan := multipartFieldEquals(body, boundary, name, value)
	if errScan != nil {
		return nil, "", errScan
	}
	if unchanged {
		return body, contentType, nil
	}

	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	var buffer bytes.Buffer
	buffer.Grow(len(body) + len(value))
	writer := multipart.NewWriter(&buffer)
	written := false
	for {
		// NextRawPart keeps quoted-printable parts byte-for-byte with their headers.
		part, errNext := reader.NextRawPart()
		if errNext == io.EOF {
			break
		}
		if errNext != nil {
			return nil, "", fmt.Errorf("read multipart part: %w", errNext)
		}
		if part.FormName() == name && part.FileName() == "" {
			if !written {
				written = true
				if errWrite := writer.WriteField(name, value); errWrite != nil {
					return nil, "", fmt.Errorf("write multipart field %s: %w", name, errWrite)
				}
			}
			continue
		}
		dst, errCreate := writer.CreatePart(part.Header)
		if errCreate != nil {
			return nil, "", fmt.Errorf("create multipart part %s: %w", part.FormName(), errCreate)
		}
		if _, errCopy := io.Copy(dst, part); errCopy != nil {
			return nil, "", fmt.Errorf("copy multipart part %s: %w", part.FormName(), errCopy)
		}
	}
	if !written {
		if errWrite := writer.WriteField(name, value); errWrite != nil {
			return nil, "", fmt.Errorf("write multipart field %s: %w", name, errWrite)
		}
	}
	if errClose := writer.Close(); errClose != nil {
		return nil, "", fmt.Errorf("close multipart writer: %w", errClose)
	}
	return buffer.Bytes(), writer.FormDataContentType(), nil
}

// multipartFieldEquals reports whether the body has exactly one non-file field name whose
// raw value equals value. File contents are skipped, not buffered.
func multipartFieldEquals(body []byte, boundary, name, value string) (bool, error) {
	reader := multipart.NewReader(bytes.NewReader(body), boundary)
	matches := 0
	equal := false
	for {
		part, errNext := reader.NextRawPart()
		if errNext == io.EOF {
			return matches == 1 && equal, nil
		}
		if errNext != nil {
			return false, fmt.Errorf("read multipart part: %w", errNext)
		}
		if part.FormName() != name || part.FileName() != "" {
			continue
		}
		current, errRead := io.ReadAll(part)
		if errRead != nil {
			return false, fmt.Errorf("read multipart field %s: %w", name, errRead)
		}
		matches++
		equal = string(current) == value
	}
}
