package config

import (
	"testing"

	"gopkg.in/yaml.v3"
)

func TestCodexInstructionRoleGroupFieldAppliesToKeys(t *testing.T) {
	cfg, errParse := ParseConfigBytes([]byte(`api-keys:
  codex:
    - name: compat
      base-url: https://compat.example.invalid
      instruction-role: System
      keys:
        - api-key: key-a
        - api-key: key-b
          instruction-role: developer
    - name: native
      base-url: https://native.example.invalid
      keys:
        - api-key: key-c
          instruction-role: assistant
`))
	if errParse != nil {
		t.Fatalf("ParseConfigBytes() error = %v", errParse)
	}
	if len(cfg.CodexKey) != 3 {
		t.Fatalf("codex key count = %d, want 3", len(cfg.CodexKey))
	}
	want := []string{CodexInstructionRoleSystem, CodexInstructionRoleDeveloper, ""}
	for i, role := range want {
		if got := cfg.CodexKey[i].InstructionRole; got != role {
			t.Errorf("codex[%d].instruction-role = %q, want %q", i, got, role)
		}
	}
}

func TestCodexInstructionRoleRejectedForClaudeGroup(t *testing.T) {
	_, errParse := ParseConfigBytes([]byte(`api-keys:
  claude:
    - name: claude
      base-url: https://claude.example.invalid
      instruction-role: system
      keys:
        - api-key: key-a
`))
	if errParse == nil {
		t.Fatal("ParseConfigBytes() error = nil, want unsupported group field error for claude instruction-role")
	}
}

func TestCodexInstructionRoleMigratesToGroupField(t *testing.T) {
	migrated, _, errMigrate := NormalizeConfigLayout([]byte(`codex-api-key:
  - api-key: key-a
    base-url: https://compat.example.invalid
    instruction-role: system
`), true)
	if errMigrate != nil {
		t.Fatalf("NormalizeConfigLayout() error = %v", errMigrate)
	}
	var doc struct {
		APIKeys struct {
			Codex []map[string]any `yaml:"codex"`
		} `yaml:"api-keys"`
	}
	if errUnmarshal := yaml.Unmarshal(migrated, &doc); errUnmarshal != nil {
		t.Fatalf("unmarshal migrated config: %v", errUnmarshal)
	}
	if len(doc.APIKeys.Codex) != 1 || doc.APIKeys.Codex[0]["instruction-role"] != "system" {
		t.Fatalf("migrated codex groups = %v, want group-level instruction-role system", doc.APIKeys.Codex)
	}
	cfg, errParse := ParseConfigBytes(migrated)
	if errParse != nil {
		t.Fatalf("ParseConfigBytes(migrated) error = %v", errParse)
	}
	if len(cfg.CodexKey) != 1 || cfg.CodexKey[0].InstructionRole != CodexInstructionRoleSystem {
		t.Fatalf("migrated codex keys = %+v, want instruction-role system", cfg.CodexKey)
	}
}
