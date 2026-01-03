# LASK-LM Roadmap

## Current Status

### Completed (Phases 1-5)

| Phase | Feature | Status |
|-------|---------|--------|
| 1-2 | Core types, decomposition prompts | Done |
| 3 | Single-file sequential graph | Done |
| 4 | Parallel execution with Send() API | Done |
| 4.5 | Grouped output, MODIFY manifest structure | Done |
| 5 | Contract registry validation | Done |

### Test Coverage

- 70 unit tests across 4 test modules
- E2E test demonstrating streaming and tree inspection

---

## Known Gaps

Issues identified during architecture review:

### 1. Duplicate Contract Implementations ✅ FIXED

**Problem:** Two nodes can provide the same contract with the same signature - not detected.

**Solution:** Added `provider_node_id` field to `Contract` model. Updated `validate_contract_registration` to detect when different nodes provide the same contract (ERROR severity). Same signature from same node is allowed.

### 2. Missing Obligation Tracking ✅ FIXED

**Problem:** Child nodes don't know what contracts they're obligated to provide.

**Solution:** Updated all decomposition prompts (FILE, CLASS, METHOD, BLOCK) to include CONTRACT OBLIGATIONS section. Prompts now explain that obligated contracts must be distributed to children and implemented exactly.

### 3. No Contract Fulfillment Validation

**Problem:** No validation that terminal prompt intent actually implements the contract signature.

**Current behavior:** System trusts that if a node has `contracts_provided`, the generated code will match.

**Fix:** Could add signature keywords check, or rely on LASK-level validation.

### 4. Intra-File Dependencies

**Problem:** File header (using statements) is processed early but needs to know what types are used in methods below.

**Current behavior:** LLM at FILE level must anticipate all needed imports.

**Fix:** Two-pass approach (decompose first, emit prompts second), or defer file header to last.

### 5. MODIFY Operations Are Blind

**Problem:** For MODIFY operations, file contents are not read or passed to the LLM.

**Current behavior:** LLM guesses file structure from the description alone.

**Fix:** Phase 6 - AST parsing and file reading.

---

## Planned: Phase 6 - Full MODIFY Support

From the original implementation plan:

- [ ] Parse existing files with AST
- [ ] Pass current structure to decomposition LLM
- [ ] Smart SKIP for unchanged nodes
- [ ] Accurate `line_range` and `ast_path` in LocationMetadata
- [ ] REPLACE/DELETE operations with precise targeting

---

## Proposed Enhancements

### Short-Term (Fixes to Current System)

1. **Pass contracts_provided to decomposition prompts** ✅ DONE
   - Modified `DECOMPOSE_FILE_PROMPT`, `DECOMPOSE_CLASS_PROMPT`, `DECOMPOSE_METHOD_PROMPT`, `TERMINAL_BLOCK_PROMPT`
   - Each prompt now has a CONTRACT OBLIGATIONS section explaining requirements
   - Effort: Small

2. **Detect duplicate providers** ✅ DONE
   - Added `provider_node_id` field to `Contract` model
   - Updated `validate_contract_registration` to detect different nodes providing same contract
   - Added 5 new unit tests for duplicate provider detection
   - Effort: Small

3. **Validate contract fulfillment**
   - Check that terminal prompts reference their contract signatures
   - Could be warning-level, not blocking
   - Effort: Medium

### Medium-Term (Phase 6)

4. **File reading for MODIFY**
   - Add file system access to decomposition
   - Parse files to extract structure
   - Pass structure summary to LLM
   - Effort: Medium-Large

5. **AST-based location targeting**
   - Use tree-sitter or similar for multi-language AST
   - Generate accurate `line_range` and `ast_path`
   - Effort: Medium

6. **Smart SKIP for unchanged code**
   - Mark sections that don't need modification
   - Reduce prompt generation for stable code
   - Effort: Medium

### Longer-Term (Enhancements)

7. **Two-pass decomposition**
   - Pass 1: Build full tree structure
   - Pass 2: Emit prompts with full tree context
   - Fixes ordering issues (imports, forward references)
   - Effort: Large (architectural change)

8. **Cross-file contract tracking**
   - Contracts span multiple files
   - Validate dependencies across file boundaries
   - Effort: Medium

9. **Incremental decomposition**
   - Cache previous decomposition
   - Only re-decompose changed portions
   - Effort: Large

---

## Open Design Questions

### 1. Import Handling

Should LASK-LM generate accurate imports, or should LASK aggregate them?

- **LASK-LM handles it:** Requires two-pass or deferred file header
- **LASK handles it:** Current design works, LASK infers imports from generated code

### 2. File Reading Location

For MODIFY operations, where should file reading happen?

- **LASK-LM:** More accurate decomposition, requires file system access
- **LASK:** Use @context directives, simpler decomposition layer

### 3. Validation Strictness

How strict should contract validation be?

- **Warnings only (current):** Flexible, trusts LLM consistency
- **Errors:** Strict enforcement, may block valid decompositions

---

## Architecture Notes

### LangGraph Patterns Used

- `StateGraph` with typed state
- `Send()` API for dynamic parallel fan-out
- Annotated reducers for parallel result merging
- Conditional edges for loop control

### Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `models/core.py` | ~400 | All data models, enums, reducers |
| `agents/implement/parallel_graph.py` | ~800 | LangGraph implementation |
| `agents/implement/validation.py` | ~180 | Contract validation |
| `agents/implement/prompts.py` | ~100 | LLM prompts |
| `agents/implement/schemas.py` | ~70 | Structured output schemas |
