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

- 80 unit tests across 4 test modules
- E2E test demonstrating streaming and tree inspection

---

## Completed Work

### Duplicate Contract Detection
- Added `provider_node_id` field to `Contract` model
- `validate_contract_registration` detects when different nodes provide the same contract (ERROR severity)
- 5 unit tests

### Obligation Tracking
- All decomposition prompts (FILE, CLASS, METHOD, BLOCK) include CONTRACT OBLIGATIONS section
- Prompts explain that obligated contracts must be distributed to children and implemented exactly

### Contract Fulfillment Validation
- `validate_contract_fulfillment()` checks terminal prompt `intent` references contract names
- Uses ERROR severity to block decomposition if contracts not referenced
- 7 unit tests

### Cross-File Contract Tracking
- `external_contracts` input for contracts from files outside decomposition
- Global `contract_registry` shared across all file nodes
- `validate_all_dependencies_satisfied` and `detect_circular_dependencies` work across file boundaries

### Validation Strictness
- Contract validation failures block decomposition
- Catches issues early rather than trusting LLM consistency

---

## Open Issues

### Intra-File Dependencies

**Problem:** File header (using statements) is processed early but needs to know what types are used in methods below.

**Current behavior:** LLM at FILE level must anticipate all needed imports.

**Design decision:** LASK-LM does not write any code, including imports.
- Inter-file dependencies: Use `@context` directive to reference other files
- Intra-file dependencies: Use `@layer` directive for ordering (layers processed ascending, starting with `@layer(0)` which is the default)

**Fix:** Two-pass decomposition (see Future Enhancements).

### MODIFY Operations Are Blind

**Problem:** For MODIFY operations, file contents are not read or passed to the LLM.

**Current behavior:** LLM guesses file structure from the description alone.

**Fix:** Phase 6 - AST parsing and file reading.

---

## Phase 6 - Full MODIFY Support

**Design decision:** LASK-LM reads files via configurable tool call.
- LASK-LM may not execute on the same machine as the target files
- File access happens through a configurable tool call interface (not direct disk reads)
- Enables remote execution scenarios and flexible deployment

### Tasks

- [x] **File reading for MODIFY** - Caller passes `existing_content` for MODIFY files; content included in LLM context
- [ ] **AST-based location targeting** - Use tree-sitter for multi-language AST, generate accurate `line_range` and `ast_path`
- [ ] **Smart SKIP for unchanged code** - Mark sections that don't need modification, reduce prompt generation for stable code
- [ ] **REPLACE/DELETE operations** - Precise targeting with location metadata

---

## Future Enhancements

### Two-Pass Decomposition
- Pass 1: Build full tree structure
- Pass 2: Emit prompts with full tree context
- Fixes intra-file dependency ordering (imports, forward references)
- Effort: Large (architectural change)

### Incremental Decomposition
- Cache previous decomposition
- Only re-decompose changed portions
- Effort: Large

### LLM Injection via MCP Sampling
- **Goal:** Allow callers (e.g., Claude Code) to provide LLM capabilities instead of LASK-LM requiring its own API key
- **Approach:** Use MCP sampling protocol - server requests completions from client
- **Blocker:** Claude Code does not currently support MCP sampling (client-side capability)
- **LangGraph pattern:** Use `Runtime[LaskContext]` with `context_schema` for dependency injection
- **When available:** Refactor `_get_llm()` to use injected `runtime.context.llm` instead of hardcoded `ChatOpenAI`
- Effort: Medium (protocol support pending)

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
