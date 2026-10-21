package helps

import (
	"strconv"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	translatorcommon "github.com/router-for-me/CLIProxyAPI/v8/internal/translator/common"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// RewriteCodexInstructionRole applies an API-key instruction-role to input
// messages whose role is "system" or "developer". Non-message items that carry a
// role, such as Codex additional_tools, are left untouched. Empty or unsupported
// role values return body unchanged.
//
// "developer" only rewrites roles, because developer messages may appear any
// number of times and anywhere. "system" targets upstreams that accept a single
// system message at the very beginning (for example strict chat templates): the
// leading system/developer messages and a non-empty top-level instructions value
// are merged into one leading system message, and later system/developer
// messages are demoted to user messages wrapped in <system-reminder>.
func RewriteCodexInstructionRole(body []byte, role string) []byte {
	switch config.NormalizeCodexInstructionRole(role) {
	case config.CodexInstructionRoleDeveloper:
		return rewriteCodexInstructionMessagesToDeveloper(body)
	case config.CodexInstructionRoleSystem:
		return rewriteCodexInstructionMessagesToLeadingSystem(body)
	default:
		return body
	}
}

func rewriteCodexInstructionMessagesToDeveloper(body []byte) []byte {
	input := gjson.GetBytes(body, "input")
	if !input.IsArray() {
		return body
	}
	var indexes []int
	index := 0
	input.ForEach(func(_, item gjson.Result) bool {
		if codexInstructionMessageRole(item) == config.CodexInstructionRoleSystem {
			indexes = append(indexes, index)
		}
		index++
		return true
	})
	rewritten := body
	for _, i := range indexes {
		updated, errSet := sjson.SetBytes(rewritten, "input."+strconv.Itoa(i)+".role", config.CodexInstructionRoleDeveloper)
		if errSet != nil {
			return body
		}
		rewritten = updated
	}
	return rewritten
}

func rewriteCodexInstructionMessagesToLeadingSystem(body []byte) []byte {
	input := gjson.GetBytes(body, "input")
	if !input.IsArray() {
		return body
	}
	items := input.Array()
	leading := 0
	for leading < len(items) && codexInstructionMessageRole(items[leading]) != "" {
		leading++
	}
	// Responses upstreams render a non-empty instructions value as the first
	// system message, so it must join the leading system message.
	instructions := gjson.GetBytes(body, "instructions")
	mergeInstructions := leading > 0 && instructions.Type == gjson.String && strings.TrimSpace(instructions.String()) != ""

	changed := false
	rebuilt := make([][]byte, 0, len(items))
	switch {
	case leading == 1 && !mergeInstructions:
		item := []byte(items[0].Raw)
		if codexInstructionMessageRole(items[0]) != config.CodexInstructionRoleSystem {
			updated, errSet := sjson.SetBytes(item, "role", config.CodexInstructionRoleSystem)
			if errSet != nil {
				return body
			}
			item = updated
			changed = true
		}
		rebuilt = append(rebuilt, item)
	case leading > 0:
		var parts [][]byte
		if mergeInstructions {
			parts = append(parts, codexInputTextPart(instructions.String()))
		}
		for _, item := range items[:leading] {
			parts = append(parts, codexMessageContentParts(item.Get("content"))...)
		}
		message, errSet := sjson.SetRawBytes([]byte(`{"type":"message","role":"system"}`), "content", translatorcommon.JoinRawArray(parts))
		if errSet != nil {
			return body
		}
		rebuilt = append(rebuilt, message)
		changed = true
	}
	for _, item := range items[leading:] {
		if codexInstructionMessageRole(item) == "" {
			rebuilt = append(rebuilt, []byte(item.Raw))
			continue
		}
		demoted, ok := demoteCodexInstructionMessage(item)
		if !ok {
			return body
		}
		rebuilt = append(rebuilt, demoted)
		changed = true
	}
	if !changed {
		return body
	}
	updated, errSet := sjson.SetRawBytes(body, "input", translatorcommon.JoinRawArray(rebuilt))
	if errSet != nil {
		return body
	}
	if mergeInstructions {
		if updated, errSet = sjson.SetBytes(updated, "instructions", ""); errSet != nil {
			return body
		}
	}
	return updated
}

// demoteCodexInstructionMessage turns a non-leading system/developer message into
// a user message whose text is wrapped in <system-reminder>.
func demoteCodexInstructionMessage(item gjson.Result) ([]byte, bool) {
	raw, errSet := sjson.SetBytes([]byte(item.Raw), "role", "user")
	if errSet != nil {
		return nil, false
	}
	content := item.Get("content")
	switch {
	case content.Type == gjson.String:
		if strings.TrimSpace(content.String()) == "" {
			return raw, true
		}
		raw, errSet = sjson.SetBytes(raw, "content", translatorcommon.SystemReminderText(content.String()))
		return raw, errSet == nil
	case content.IsArray():
		index := 0
		ok := true
		content.ForEach(func(_, part gjson.Result) bool {
			partType := part.Get("type").String()
			text := part.Get("text").String()
			if (partType == "input_text" || partType == "text") && strings.TrimSpace(text) != "" {
				raw, errSet = sjson.SetBytes(raw, "content."+strconv.Itoa(index)+".text", translatorcommon.SystemReminderText(text))
				if errSet != nil {
					ok = false
					return false
				}
			}
			index++
			return true
		})
		return raw, ok
	default:
		return raw, true
	}
}

// codexMessageContentParts returns the content parts of a message, converting
// string content to a single input_text part.
func codexMessageContentParts(content gjson.Result) [][]byte {
	switch {
	case content.Type == gjson.String:
		if content.String() == "" {
			return nil
		}
		return [][]byte{codexInputTextPart(content.String())}
	case content.IsArray():
		var parts [][]byte
		content.ForEach(func(_, part gjson.Result) bool {
			parts = append(parts, []byte(part.Raw))
			return true
		})
		return parts
	default:
		return nil
	}
}

func codexInputTextPart(text string) []byte {
	part, _ := sjson.SetBytes([]byte(`{"type":"input_text","text":""}`), "text", text)
	return part
}

// codexInstructionMessageRole returns the role of a system or developer message item.
func codexInstructionMessageRole(item gjson.Result) string {
	if !item.IsObject() {
		return ""
	}
	if itemType := item.Get("type").String(); itemType != "" && itemType != "message" {
		return ""
	}
	switch role := item.Get("role").String(); role {
	case config.CodexInstructionRoleDeveloper, config.CodexInstructionRoleSystem:
		return role
	default:
		return ""
	}
}
