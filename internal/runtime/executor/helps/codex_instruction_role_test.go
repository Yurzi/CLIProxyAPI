package helps

import (
	"testing"

	"github.com/tidwall/gjson"
)

const codexInstructionRoleBody = `{"instructions":"","input":[{"type":"message","role":"system","content":[{"type":"input_text","text":"sys"}]},{"role":"developer","content":"dev"},{"type":"additional_tools","role":"developer","tools":[]},{"type":"message","role":"user","content":"hi"},{"type":"message","role":"developer","content":[{"type":"input_text","text":"mid"}]},{"type":"message","role":"system","content":"late"},{"type":"function_call_output","call_id":"c1","output":"ok"}]}`

func codexInputRoles(body []byte) []string {
	var roles []string
	for _, item := range gjson.GetBytes(body, "input").Array() {
		roles = append(roles, item.Get("role").String())
	}
	return roles
}

func assertRoles(t *testing.T, body []byte, want ...string) {
	t.Helper()
	got := codexInputRoles(body)
	if len(got) != len(want) {
		t.Fatalf("input roles = %q, want %q: %s", got, want, body)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("input roles = %q, want %q: %s", got, want, body)
		}
	}
}

func TestRewriteCodexInstructionRoleDeveloperOnlyRewritesRoles(t *testing.T) {
	got := RewriteCodexInstructionRole([]byte(codexInstructionRoleBody), "Developer")
	assertRoles(t, got, "developer", "developer", "developer", "user", "developer", "developer", "")
	if text := gjson.GetBytes(got, "input.5.content").String(); text != "late" {
		t.Fatalf("developer mode changed content: %s", got)
	}
}

func TestRewriteCodexInstructionRoleSystemKeepsSingleLeadingSystem(t *testing.T) {
	got := RewriteCodexInstructionRole([]byte(codexInstructionRoleBody), "system")
	assertRoles(t, got, "system", "developer", "user", "user", "user", "")

	merged := gjson.GetBytes(got, "input.0.content").Array()
	if len(merged) != 2 || merged[0].Get("text").String() != "sys" || merged[1].Get("type").String() != "input_text" || merged[1].Get("text").String() != "dev" {
		t.Fatalf("merged system content = %s", gjson.GetBytes(got, "input.0.content").Raw)
	}
	if itemType := gjson.GetBytes(got, "input.1.type").String(); itemType != "additional_tools" {
		t.Fatalf("input[1] = %s, want untouched additional_tools", gjson.GetBytes(got, "input.1").Raw)
	}
	if text := gjson.GetBytes(got, "input.3.content.0.text").String(); text != "<system-reminder>\nmid\n</system-reminder>" {
		t.Fatalf("demoted array text = %q", text)
	}
	if text := gjson.GetBytes(got, "input.4.content").String(); text != "<system-reminder>\nlate\n</system-reminder>" {
		t.Fatalf("demoted string text = %q", text)
	}
	systems := 0
	for _, role := range codexInputRoles(got) {
		if role == "system" {
			systems++
		}
	}
	if systems != 1 {
		t.Fatalf("system message count = %d, want 1: %s", systems, got)
	}
}

func TestRewriteCodexInstructionRoleSystemMergesInstructions(t *testing.T) {
	body := `{"instructions":"base","input":[{"type":"message","role":"developer","content":[{"type":"input_text","text":"dev"}]},{"type":"message","role":"user","content":"hi"}]}`
	got := RewriteCodexInstructionRole([]byte(body), "system")
	assertRoles(t, got, "system", "user")
	if instructions := gjson.GetBytes(got, "instructions"); instructions.String() != "" || !instructions.Exists() {
		t.Fatalf("instructions = %s, want empty string", instructions.Raw)
	}
	parts := gjson.GetBytes(got, "input.0.content").Array()
	if len(parts) != 2 || parts[0].Get("text").String() != "base" || parts[1].Get("text").String() != "dev" {
		t.Fatalf("merged system content = %s", gjson.GetBytes(got, "input.0.content").Raw)
	}
}

func TestRewriteCodexInstructionRoleSystemKeepsInstructionsWithoutLeadingMessage(t *testing.T) {
	body := `{"instructions":"base","input":[{"type":"message","role":"user","content":"hi"},{"type":"message","role":"system","content":"late"}]}`
	got := RewriteCodexInstructionRole([]byte(body), "system")
	assertRoles(t, got, "user", "user")
	if instructions := gjson.GetBytes(got, "instructions").String(); instructions != "base" {
		t.Fatalf("instructions = %q, want base", instructions)
	}
}

func TestRewriteCodexInstructionRoleSystemSingleLeadingMessageOnlyChangesRole(t *testing.T) {
	body := `{"instructions":"","input":[{"type":"message","role":"developer","id":"msg_1","content":[{"type":"input_text","text":"dev"}]},{"type":"message","role":"user","content":"hi"}]}`
	got := RewriteCodexInstructionRole([]byte(body), "system")
	want := `{"instructions":"","input":[{"type":"message","role":"system","id":"msg_1","content":[{"type":"input_text","text":"dev"}]},{"type":"message","role":"user","content":"hi"}]}`
	if string(got) != want {
		t.Fatalf("got %s, want %s", got, want)
	}
	already := `{"input":[{"type":"message","role":"system","content":"sys"},{"type":"message","role":"user","content":"hi"}]}`
	if got := RewriteCodexInstructionRole([]byte(already), "system"); string(got) != already {
		t.Fatalf("got %s, want unchanged", got)
	}
}

func TestRewriteCodexInstructionRoleUnsupportedOrMissingInput(t *testing.T) {
	for _, role := range []string{"", "assistant"} {
		if got := RewriteCodexInstructionRole([]byte(codexInstructionRoleBody), role); string(got) != codexInstructionRoleBody {
			t.Errorf("role %q changed body: %s", role, got)
		}
	}
	for _, body := range []string{`{"input":"hi"}`, `{"model":"m"}`} {
		for _, role := range []string{"system", "developer"} {
			if got := RewriteCodexInstructionRole([]byte(body), role); string(got) != body {
				t.Errorf("RewriteCodexInstructionRole(%s, %s) = %s, want unchanged", body, role, got)
			}
		}
	}
}
