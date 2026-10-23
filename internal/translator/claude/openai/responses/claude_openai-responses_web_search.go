package responses

import (
	"context"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/MoeForge/moego/ojson"
	translatorcommon "github.com/router-for-me/CLIProxyAPI/v8/internal/translator/common"
	"github.com/router-for-me/CLIProxyAPI/v8/internal/util"
)

// Claude reports server-side web search as a pair of assistant content blocks: a
// `server_tool_use` block carrying the query and a `web_search_tool_result` block
// carrying the hits. OpenAI Responses models the same exchange as a single
// `web_search_call` item. The representations are isomorphic, so the pair folds
// into one item on the way out and expands back on the way in.
//
// Dropping the pair instead is not merely cosmetic: the replayed turn then shows
// a conclusion with no search behind it, which makes the model treat its own
// previous answer as unsourced and search again.
const (
	claudeWebSearchToolName = "web_search"

	// responsesWebSearchIDPrefix namespaces the Claude server_tool_use id inside
	// the Responses item id, mirroring the fc_/ctc_ prefixes used for tool calls
	// so the original id can be recovered on replay.
	responsesWebSearchIDPrefix = "ws_"

	// claudeServerToolIDPrefix is mandated by Anthropic: server tool ids must
	// match ^srvtoolu_[a-zA-Z0-9_]+$.
	claudeServerToolIDPrefix = "srvtoolu_"

	// responsesIncludeWebSearchSources is the Responses `include` entry that
	// asks for web_search_call.action.sources.
	responsesIncludeWebSearchSources = "web_search_call.action.sources"
)

// claudeServerToolIDSanitizer matches the characters Anthropic forbids in a
// server tool id. Note it is stricter than util.SanitizeClaudeToolID, which
// targets tool_use ids and allows '-'.
var claudeServerToolIDSanitizer = regexp.MustCompile(`[^a-zA-Z0-9_]`)

func responsesWebSearchCallID(claudeToolUseID string) string {
	return responsesWebSearchIDPrefix + claudeToolUseID
}

// claudeWebSearchToolUseID recovers the Claude server_tool_use id from a
// Responses item id, normalising it to Anthropic's required shape. A Responses
// history need not come from Claude at all - a native OpenAI web_search_call
// carries an id like "ws_00112233aabb" - and replaying such an id verbatim is
// rejected outright, so the body is always sanitised and re-prefixed. The
// transform is idempotent, leaving genuine Claude ids untouched.
func claudeWebSearchToolUseID(responsesItemID string) string {
	body := strings.TrimPrefix(strings.TrimSpace(responsesItemID), responsesWebSearchIDPrefix)
	body = claudeServerToolIDSanitizer.ReplaceAllString(strings.TrimPrefix(body, claudeServerToolIDPrefix), "_")
	if body == "" {
		return ""
	}
	return claudeServerToolIDPrefix + body
}

// claudeWebSearchQuery extracts the query from a Claude server_tool_use input,
// accepting either the accumulated streaming JSON or an already parsed object.
func claudeWebSearchQuery(input string) string {
	if input == "" {
		return ""
	}
	return strings.TrimSpace(ojson.Get(input, "query").String())
}

// claudeWebSearchResultsToResponses converts the content of a Claude
// `web_search_tool_result` block into the Responses `results` array.
//
// Entries ride through verbatim rather than being rebuilt field by field:
// Anthropic requires a genuine `encrypted_content` on every result it is asked
// to replay and rejects fabricated values, so anything we drop here can never be
// reconstructed later.
func claudeWebSearchResultsToResponses(content ojson.Result) []byte {
	if content.IsObject() {
		return []byte(content.Raw)
	}
	if !content.IsArray() {
		return nil
	}
	var results [][]byte
	content.ForEach(func(_, entry ojson.Result) bool {
		if entry.Get("type").String() == "web_search_tool_result_error" || strings.TrimSpace(entry.Get("url").String()) != "" {
			results = append(results, []byte(entry.Raw))
		}
		return true
	})
	if len(results) == 0 {
		return []byte(`[]`)
	}
	return translatorcommon.JoinRawArray(results)
}

// responsesRequestIncludesWebSearchSources reports whether a Responses request
// asked for web_search_call.action.sources through `include`.
func responsesRequestIncludesWebSearchSources(requestJSON []byte) bool {
	include := ojson.GetBytesNoCopy(requestJSON, "include")
	if !include.IsArray() {
		return false
	}
	found := false
	include.ForEach(func(_, value ojson.Result) bool {
		found = value.Type == ojson.String && strings.TrimSpace(value.String()) == responsesIncludeWebSearchSources
		return !found
	})
	return found
}

// responsesWebSearchSources derives the standard `action.sources` list from the
// Claude results carried in `results`. A failed or empty search yields an empty
// list.
func responsesWebSearchSources(results []byte) []byte {
	var sources [][]byte
	seen := map[string]bool{}
	ojson.ParseBytes(results).ForEach(func(_, entry ojson.Result) bool {
		url := strings.TrimSpace(entry.Get("url").String())
		if url == "" || seen[url] {
			return true
		}
		seen[url] = true
		sourceDoc := ojson.NewDocString(`{"type":"url","url":""}`)
		_ = sourceDoc.Set("url", url)
		sources = append(sources, sourceDoc.Bytes())
		return true
	})
	return translatorcommon.JoinRawArray(sources)
}

// buildResponsesWebSearchCallItem renders the Responses item that stands for one
// Claude server-side search. results may be nil when the upstream turn ended
// before the result block arrived.
func buildResponsesWebSearchCallItem(claudeToolUseID, query string, results []byte, includeSources bool) []byte {
	return buildResponsesWebSearchCallItemWithStatus(claudeToolUseID, query, "completed", results, includeSources)
}

func buildResponsesWebSearchCallItemWithStatus(claudeToolUseID, query, status string, results []byte, includeSources bool) []byte {
	if status == "" {
		status = "completed"
	}
	itemDoc := ojson.NewDocString(`{"id":"","type":"web_search_call","status":"completed","action":{"type":"search","query":""}}`)
	_ = itemDoc.Set("id", responsesWebSearchCallID(claudeToolUseID))
	_ = itemDoc.Set("status", status)
	_ = itemDoc.Set("action.query", query)
	if len(results) > 0 {
		if includeSources {
			_ = itemDoc.SetRawBytes("action.sources", responsesWebSearchSources(results))
		}
		_ = itemDoc.SetRawBytes("results", results)
	}
	return itemDoc.Bytes()
}

type webSearchResultLookupContextKey struct{}

// WithWebSearchResultLookup attaches a request-scoped lookup that restores
// cached Claude web_search_tool_result content for a given server_tool_use id
// when a Responses web_search_call item arrives without results.
func WithWebSearchResultLookup(ctx context.Context, lookup func(toolUseID string) ([]byte, bool)) context.Context {
	if lookup == nil {
		return ctx
	}
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, webSearchResultLookupContextKey{}, lookup)
}

func lookupWebSearchResultFor(ctx context.Context, toolUseID string) []byte {
	if ctx == nil || toolUseID == "" {
		return nil
	}
	lookup, ok := ctx.Value(webSearchResultLookupContextKey{}).(func(string) ([]byte, bool))
	if !ok || lookup == nil {
		return nil
	}
	content, found := lookup(toolUseID)
	if !found || len(content) == 0 {
		return nil
	}
	return content
}

// claudeCitationTracker resolves the Claude citations of one Responses
// output_text into annotations. Claude cites a whole text block, and the
// citations_delta may arrive before or after the block's text, so a citation is
// resolved once the block its event index names has ended, spanning exactly
// that block's text. A citation whose index names no text block of this
// output_text belongs to the next text block, or to the last one when none
// follows. Offsets count Unicode code points, like the other Responses
// translators.
type claudeCitationTracker struct {
	textLen int
	// spans holds [start, end) of each closed text block by Claude block index.
	spans map[int][2]int
	last  [2]int
	// open is the Claude index of the open text block, valid while isOpen.
	open      int
	openStart int
	isOpen    bool
	// openPending belongs to the open block; orphans name no known text block.
	openPending []string
	orphans     []string
}

// startBlock opens the text block with the given Claude index at the current end
// of the text, adopting the orphan citations. A previous block that never saw
// its stop event is closed first, and its resolved annotations are returned.
func (t *claudeCitationTracker) startBlock(index int) [][]byte {
	var resolved [][]byte
	if t.isOpen {
		resolved = t.endBlock(t.open)
	}
	t.open, t.openStart, t.isOpen = index, t.textLen, true
	t.openPending, t.orphans = t.orphans, nil
	return resolved
}

func (t *claudeCitationTracker) addText(text string) {
	t.textLen += utf8.RuneCountInString(text)
}

// addCitation records a citation for the text block with the given Claude
// index. A citation for a block that already ended resolves immediately.
func (t *claudeCitationTracker) addCitation(index int, citation ojson.Result) [][]byte {
	raw := strings.Clone(citation.Raw)
	switch span, closed := t.spans[index]; {
	case t.isOpen && index == t.open:
		t.openPending = append(t.openPending, raw)
	case closed:
		return resolveClaudeCitations([]string{raw}, span)
	default:
		t.orphans = append(t.orphans, raw)
	}
	return nil
}

// endBlock closes the open text block when index names it and resolves its
// citations.
func (t *claudeCitationTracker) endBlock(index int) [][]byte {
	if !t.isOpen || index != t.open {
		return nil
	}
	if t.spans == nil {
		t.spans = map[int][2]int{}
	}
	t.last = [2]int{t.openStart, t.textLen}
	t.spans[index] = t.last
	t.isOpen = false
	resolved := resolveClaudeCitations(t.openPending, t.last)
	t.openPending = nil
	return resolved
}

// finish resolves every remaining citation once the output_text is complete.
func (t *claudeCitationTracker) finish() [][]byte {
	var resolved [][]byte
	if t.isOpen {
		resolved = t.endBlock(t.open)
	}
	resolved = append(resolved, resolveClaudeCitations(t.orphans, t.last)...)
	t.orphans = nil
	return resolved
}

func resolveClaudeCitations(citations []string, span [2]int) [][]byte {
	if len(citations) == 0 {
		return nil
	}
	annotations := make([][]byte, 0, len(citations))
	for _, raw := range citations {
		annotations = append(annotations, claudeCitationToResponsesAnnotation(ojson.Parse(raw), span[0], span[1]))
	}
	return annotations
}

// claudeCitationToResponsesAnnotation renders one Claude citation as a Responses
// annotation over [start, end) of the output text.
//
// web_search_result_location becomes the standard url_citation. cited_text and
// encrypted_index ride along as extra fields, which OpenAI clients ignore,
// because Anthropic requires encrypted_index to replay the citation (see
// responsesAnnotationToClaudeCitation). Other citation kinds have no Responses
// counterpart and pass through verbatim.
func claudeCitationToResponsesAnnotation(citation ojson.Result, start, end int) []byte {
	if citation.Get("type").String() != "web_search_result_location" {
		return []byte(citation.Raw)
	}
	annotationDoc := ojson.NewDocString(`{"type":"url_citation","url":"","title":"","start_index":0,"end_index":0}`)
	_ = annotationDoc.Set("url", citation.Get("url").String())
	_ = annotationDoc.Set("title", citation.Get("title").String())
	_ = annotationDoc.Set("start_index", start)
	_ = annotationDoc.Set("end_index", end)
	for _, field := range []string{"cited_text", "encrypted_index"} {
		if value := citation.Get(field); value.Exists() {
			_ = annotationDoc.SetRawBytes(field, []byte(value.Raw))
		}
	}
	return annotationDoc.Bytes()
}

// convertResponsesWebSearchCallToClaudeBlocks is the inverse of
// buildResponsesWebSearchCallItem: it rebuilds the Claude block pair so a
// replayed turn still shows that the search happened and what it returned.
func convertResponsesWebSearchCallToClaudeBlocks(item ojson.Result) [][]byte {
	return convertResponsesWebSearchCallToClaudeBlocksWithQuery(item, "", true)
}

// convertResponsesWebSearchCallToClaudeBlocksWithQuery rebuilds the server_tool_use
// block, using fallbackQuery when the item carries none, and the result block
// only when withResult is set.
func convertResponsesWebSearchCallToClaudeBlocksWithQuery(item ojson.Result, fallbackQuery string, withResult bool) [][]byte {
	return convertResponsesWebSearchCallToClaudeBlocksWithContext(context.Background(), item, fallbackQuery, withResult)
}

func convertResponsesWebSearchCallToClaudeBlocksWithContext(ctx context.Context, item ojson.Result, fallbackQuery string, withResult bool) [][]byte {
	toolUseID := claudeWebSearchToolUseID(strings.TrimSpace(item.Get("id").String()))
	if toolUseID == "" {
		return nil
	}

	useDoc := ojson.NewDocString(`{"type":"server_tool_use","id":"","name":"","input":{}}`)
	_ = useDoc.Set("id", toolUseID)
	_ = useDoc.Set("name", claudeWebSearchToolName)
	query := responsesWebSearchCallQuery(item)
	if query == "" {
		query = fallbackQuery
	}
	if query != "" {
		_ = useDoc.Set("input.query", query)
	}
	if !withResult {
		return [][]byte{useDoc.Bytes()}
	}

	resultDoc := ojson.NewDocString(`{"type":"web_search_tool_result","tool_use_id":"","content":[]}`)
	_ = resultDoc.Set("tool_use_id", toolUseID)
	content := responsesWebSearchResultsToClaude(item.Get("results"))
	if len(content) == 0 {
		content = lookupWebSearchResultFor(ctx, toolUseID)
	}
	if len(content) > 0 {
		_ = resultDoc.SetRawBytes("content", content)
	}
	return [][]byte{useDoc.Bytes(), resultDoc.Bytes()}
}

// responsesWebSearchHistory indexes web_search_call items of one request so a
// search paused by pause_turn or deferred by tool_use replays faithfully:
//   - while the open search turn is the tail of the history, the search is sent
//     back as a server_tool_use without a result, unchanged, as Anthropic requires;
//   - when the continuation delivered the result under the same id within the
//     same assistant run (no user turn in between), the open item is skipped and
//     the continuation item emits the full pair, keeping the query recorded by
//     the open item.
//
// Every other web_search_call, including repeated ids across separate turns,
// keeps its previous one-pair-per-item replay.
type responsesWebSearchHistory struct {
	// boundaryAfter[i] reports whether a user turn follows index i.
	boundaryAfter []bool
	// skip marks paused items that a later continuation item replaces.
	skip map[int]bool
	// pausedQuery carries the paused item's query to its continuation item.
	pausedQuery map[int]string
}

func responsesWebSearchItemIsPaused(item ojson.Result) bool {
	return item.Get("type").String() == "web_search_call" &&
		strings.EqualFold(strings.TrimSpace(item.Get("status").String()), "in_progress") &&
		!item.Get("results").Exists()
}

// responsesItemIsTurnBoundary reports a user turn. Tool outputs do not end the
// assistant turn: Anthropic runs a deferred search right after them.
func responsesItemIsTurnBoundary(item ojson.Result) bool {
	typ := item.Get("type").String()
	role := item.Get("role").String()
	if typ == "" && role != "" {
		typ = "message"
	}
	return typ == "message" && role != "assistant"
}

func scanResponsesWebSearchHistory(items []ojson.Result) responsesWebSearchHistory {
	history := responsesWebSearchHistory{
		boundaryAfter: make([]bool, len(items)),
		skip:          map[int]bool{},
		pausedQuery:   map[int]string{},
	}
	boundary := false
	for index := len(items) - 1; index >= 0; index-- {
		history.boundaryAfter[index] = boundary
		if responsesItemIsTurnBoundary(items[index]) {
			boundary = true
		}
	}
	for index, item := range items {
		if !responsesWebSearchItemIsPaused(item) {
			continue
		}
		id := strings.TrimSpace(item.Get("id").String())
		if id == "" {
			continue
		}
		for next := index + 1; next < len(items); next++ {
			if responsesItemIsTurnBoundary(items[next]) {
				break
			}
			if items[next].Get("type").String() == "web_search_call" && strings.TrimSpace(items[next].Get("id").String()) == id {
				history.skip[index] = true
				query := responsesWebSearchCallQuery(item)
				if previous, ok := history.pausedQuery[index]; ok && query == "" {
					query = previous
				}
				history.pausedQuery[next] = query
				break
			}
		}
	}
	return history
}

func (h responsesWebSearchHistory) claudeBlocks(index int, item ojson.Result) [][]byte {
	return h.claudeBlocksWithContext(context.Background(), index, item)
}

func (h responsesWebSearchHistory) claudeBlocksWithContext(ctx context.Context, index int, item ojson.Result) [][]byte {
	if h.skip[index] {
		return nil
	}
	query := h.pausedQuery[index]
	tail := index >= len(h.boundaryAfter) || !h.boundaryAfter[index]
	if responsesWebSearchItemIsPaused(item) && tail {
		return convertResponsesWebSearchCallToClaudeBlocksWithContext(ctx, item, query, false)
	}
	return convertResponsesWebSearchCallToClaudeBlocksWithContext(ctx, item, query, true)
}

// responsesWebSearchCallQuery reads the query from a Responses web_search_call,
// tolerating the `queries` array and the `open_page` action shape that OpenAI
// clients emit.
func responsesWebSearchCallQuery(item ojson.Result) string {
	if query := strings.TrimSpace(item.Get("action.query").String()); query != "" {
		return query
	}
	if query := strings.TrimSpace(item.Get("action.queries.0").String()); query != "" {
		return query
	}
	return strings.TrimSpace(item.Get("action.url").String())
}

func responsesWebSearchResultsToClaude(results ojson.Result) []byte {
	if results.IsObject() {
		return []byte(results.Raw)
	}
	if !results.IsArray() {
		return nil
	}
	var blocks [][]byte
	results.ForEach(func(_, entry ojson.Result) bool {
		if entry.Get("type").String() == "web_search_tool_result_error" {
			blocks = append(blocks, []byte(entry.Raw))
			return true
		}
		// Anthropic validates encrypted_content and rejects the whole request when
		// it is missing or forged. An entry that lost it in transit is therefore
		// unusable; an empty result list is the only safe degradation, and it is
		// accepted alongside the mandatory server_tool_use block.
		if strings.TrimSpace(entry.Get("encrypted_content").String()) == "" {
			return true
		}
		block := []byte(entry.Raw)
		block, _ = ojson.SetBytes(block, "type", "web_search_result")
		blocks = append(blocks, block)
		return true
	})
	if len(blocks) == 0 {
		return nil
	}
	return translatorcommon.JoinRawArray(blocks)
}

// responsesAnnotationToClaudeCitation is the inverse of
// claudeCitationToResponsesAnnotation. Anthropic rejects a citation without a
// genuine encrypted_index, so an annotation lacking one - a native OpenAI
// url_citation, say - yields nil and is dropped. Histories recorded before the
// url_citation mapping carry the Claude citation verbatim and replay unchanged.
func responsesAnnotationToClaudeCitation(annotation ojson.Result) []byte {
	if strings.TrimSpace(annotation.Get("encrypted_index").String()) == "" {
		return nil
	}
	if annotation.Get("type").String() != "url_citation" {
		return []byte(annotation.Raw)
	}
	// Anthropic validates citations strictly, so the Responses-only fields
	// (start_index, end_index) must not ride along.
	citationDoc := ojson.NewDocString(`{"type":"web_search_result_location","cited_text":"","url":"","title":"","encrypted_index":""}`)
	_ = citationDoc.Set("cited_text", annotation.Get("cited_text").String())
	_ = citationDoc.Set("url", annotation.Get("url").String())
	_ = citationDoc.Set("title", annotation.Get("title").String())
	_ = citationDoc.Set("encrypted_index", annotation.Get("encrypted_index").String())
	return citationDoc.Bytes()
}

// attachClaudeCitations mirrors Responses `annotations` back onto a Claude text
// block as `citations`; see responsesAnnotationToClaudeCitation for which
// annotations survive.
func attachClaudeCitations(textBlock []byte, annotations ojson.Result) []byte {
	if !annotations.IsArray() {
		return textBlock
	}
	var citations [][]byte
	annotations.ForEach(func(_, annotation ojson.Result) bool {
		if citation := responsesAnnotationToClaudeCitation(annotation); citation != nil {
			citations = append(citations, citation)
		}
		return true
	})
	if len(citations) == 0 {
		return textBlock
	}
	updated, err := ojson.SetRawBytes(textBlock, "citations", translatorcommon.JoinRawArray(citations))
	if err != nil {
		return textBlock
	}
	return updated
}

func claudeBlocksHaveEmptyWebSearchResult(blocks [][]byte) bool {
	for _, raw := range blocks {
		block := util.ParseGJSONBytesNoCopy(raw)
		if block.Get("type").String() != "web_search_tool_result" {
			continue
		}
		content := block.Get("content")
		if !content.Exists() || content.Type == ojson.Null || (content.IsArray() && len(content.Array()) == 0) {
			return true
		}
	}
	return false
}

// stripClaudeWebSearchCitationsFromParts removes web_search_result_location
// citations from text blocks when a web_search_tool_result in the same
// assistant turn ended up without content. Anthropic intermittently rejects
// citations whose encrypted_index has no matching search result in the turn.
func stripClaudeWebSearchCitationsFromParts(parts [][]byte) ([][]byte, int) {
	droppedTotal := 0
	for i, raw := range parts {
		part := util.ParseGJSONBytesNoCopy(raw)
		if part.Get("type").String() != "text" {
			continue
		}
		updated, dropped := stripClaudeWebSearchCitationsFromTextBlock(raw, part.Get("citations"))
		if dropped > 0 {
			parts[i] = updated
			droppedTotal += dropped
		}
	}
	return parts, droppedTotal
}

func stripClaudeWebSearchCitationsFromTextBlock(textBlock []byte, citations ojson.Result) ([]byte, int) {
	if !citations.Exists() || !citations.IsArray() {
		return textBlock, 0
	}
	var kept [][]byte
	dropped := 0
	citations.ForEach(func(_, citation ojson.Result) bool {
		if citation.Get("type").String() == "web_search_result_location" {
			dropped++
			return true
		}
		kept = append(kept, []byte(citation.Raw))
		return true
	})
	if dropped == 0 {
		return textBlock, 0
	}
	if len(kept) == 0 {
		updated, errDelete := ojson.DeleteBytes(textBlock, "citations")
		if errDelete != nil {
			return textBlock, 0
		}
		return updated, dropped
	}
	updated, errSet := ojson.SetRawBytes(textBlock, "citations", translatorcommon.JoinRawArray(kept))
	if errSet != nil {
		return textBlock, 0
	}
	return updated, dropped
}
