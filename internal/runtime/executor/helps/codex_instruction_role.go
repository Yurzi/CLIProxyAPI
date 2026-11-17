package helps

import (
	"strconv"
	"strings"

	"github.com/MoeForge/moego/ojson"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/config"
	translatorcommon "github.com/router-for-me/CLIProxyAPI/v8/internal/translator/common"
)

// RewriteCodexInstructionRoleDoc applies the API-key instruction role in place.
// Developer mode only rewrites message roles. System mode merges leading
// instructions and demotes later instruction messages to user reminders.
// Call before the payload finalizer so user rules retain the final say.
func RewriteCodexInstructionRoleDoc(doc *ojson.Doc, role string) {
	if doc == nil {
		return
	}
	role = config.NormalizeCodexInstructionRole(role)
	input := doc.Get("input")
	if role == "" || !input.IsArray() {
		return
	}
	items := input.Array()
	if role == config.CodexInstructionRoleDeveloper {
		for i, item := range items {
			if codexInstructionMessageRole(item) == config.CodexInstructionRoleSystem {
				_ = doc.Set("input."+strconv.Itoa(i)+".role", role)
			}
		}
		return
	}

	leading := 0
	for leading < len(items) && codexInstructionMessageRole(items[leading]) != "" {
		leading++
	}
	instructions := doc.Get("instructions")
	mergeInstructions := leading > 0 && instructions.Type == ojson.String && strings.TrimSpace(instructions.String()) != ""
	if leading == 1 && !mergeInstructions {
		_ = doc.Set("input.0.role", config.CodexInstructionRoleSystem)
	} else if leading > 0 {
		var parts [][]byte
		if mergeInstructions {
			parts = append(parts, codexInputTextPart(instructions.String()))
		}
		for _, item := range items[:leading] {
			parts = append(parts, codexMessageContentParts(item.Get("content"))...)
		}
		_ = doc.SetRaw("input.0", `{"type":"message","role":"system"}`)
		_ = doc.SetRawBytes("input.0.content", translatorcommon.JoinRawArray(parts))
		if mergeInstructions {
			_ = doc.Set("instructions", "")
		}
	}

	for i := leading; i < len(items); i++ {
		item := items[i]
		if codexInstructionMessageRole(item) == "" {
			continue
		}
		path := "input." + strconv.Itoa(i)
		_ = doc.Set(path+".role", "user")
		content := item.Get("content")
		switch {
		case content.Type == ojson.String:
			if strings.TrimSpace(content.String()) != "" {
				_ = doc.Set(path+".content", translatorcommon.SystemReminderText(content.String()))
			}
		case content.IsArray():
			for j, part := range content.Array() {
				partType := part.Get("type").String()
				text := part.Get("text").String()
				if (partType == "input_text" || partType == "text") && strings.TrimSpace(text) != "" {
					_ = doc.Set(path+".content."+strconv.Itoa(j)+".text", translatorcommon.SystemReminderText(text))
				}
			}
		}
	}
	// Delete backwards after rewriting so paths still refer to the original indexes.
	for i := leading - 1; i > 0; i-- {
		_ = doc.Delete("input." + strconv.Itoa(i))
	}
}

// codexMessageContentParts returns the content parts of a message, converting
// string content to a single input_text part.
func codexMessageContentParts(content ojson.Result) [][]byte {
	switch {
	case content.Type == ojson.String:
		if content.String() == "" {
			return nil
		}
		return [][]byte{codexInputTextPart(content.String())}
	case content.IsArray():
		var parts [][]byte
		content.ForEach(func(_, part ojson.Result) bool {
			parts = append(parts, []byte(part.Raw))
			return true
		})
		return parts
	default:
		return nil
	}
}

func codexInputTextPart(text string) []byte {
	part := ojson.NewDocString(`{"type":"input_text","text":""}`)
	_ = part.Set("text", text)
	return part.Bytes()
}

// codexInstructionMessageRole returns the role of a system or developer message item.
func codexInstructionMessageRole(item ojson.Result) string {
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
