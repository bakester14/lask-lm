# Implement Agent - Implementation Plan

> **Note:** Delete this file once all phases are complete.

## Overview

The Implement agent recursively decomposes code generation tasks into LASK-compatible prompts. It takes a plan from the Plan subagent and breaks it down through FILE → CLASS → METHOD → BLOCK granularity levels, emitting LASK prompts at the leaf nodes (~10 lines of code per prompt).

## Architecture Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Termination** | Line count ≤10 | LLM decides if output is ≤10 lines as part of decomposition |
| **Context Passing** | Parent provides upfront | Children use `@context(file.cs)` for cross-file refs |
| **MODIFY mode** | Recursive with skip | Same tree as CREATE; skip unchanged subtrees |

## Phases

### Phase 1: Core Types ✅ COMPLETE
- [x] Pydantic models: `CodeNode`, `ImplementState`, `LaskPrompt`, `Contract`
- [x] Enums: `NodeType`, `NodeStatus`, `FileOperation`
- [x] LaskDirective model for `@context`, `@model`, etc.

### Phase 2: Decomposition Prompts ✅ COMPLETE
- [x] System prompts for FILE-level decomposition
- [x] System prompts for CLASS-level decomposition
- [x] System prompts for METHOD-level decomposition
- [x] System prompts for BLOCK-level (terminal) emission
- [x] Output schemas compatible with OpenAI strict structured output

### Phase 3: Single-File Graph ✅ COMPLETE
- [x] LangGraph with Router → Decomposer → Collector
- [x] Sequential processing of pending nodes
- [x] Conditional routing via `should_continue()`
- [x] Context file inheritance from parent to child nodes
- [x] End-to-end test demonstrating streaming execution
- [x] MCP server for Claude Code integration

### Phase 4: Parallel Spawning ✅ COMPLETE
- [x] Use LangGraph `Send()` API for parallel sibling processing
- [x] Fan-out at each decomposition level via `dispatch_to_parallel` routing
- [x] Aggregate results via `aggregator_node` that rebuilds pending from node statuses
- [x] State reducers (`merge_dicts`, `append_prompts`, `max_int`) for merging parallel results
- [x] Unit tests for all components (22 tests passing)
- [x] E2E test demonstrating parallel execution with multiple files
- [x] Consolidated to single implementation (removed sequential graph.py and nodes.py)
- [x] Compatibility tests ensuring backwards-compatible API (11 tests passing)

### Phase 4.5: Grouped Output & MODIFY Manifest ✅ COMPLETE
- [x] New models: `OperationType`, `LocationMetadata`, `ModifyOperation`, `ModifyManifest`, `OrderedFilePrompts`, `GroupedOutput`
- [x] Tree traversal to preserve prompt order (`_depth_first_collect_prompts`)
- [x] `collector_node` groups prompts by file with tree-traversal ordering
- [x] MODIFY manifest generation with location-based metadata
- [x] `LaskPromptOutput` schema extended with `insertion_point` and `replaces` fields
- [x] Unit tests for grouped output (16 tests)
- [x] All 48 tests passing

### Phase 5: Contract Registry ⬜ NOT STARTED
- [ ] Register contracts from each node as they're created
- [ ] Resolve contract dependencies before emitting LASK prompts
- [ ] Validate all required contracts are satisfied
- [ ] Include contract signatures in LASK prompt context

### Phase 6: MODIFY Support ⬜ NOT STARTED
- [ ] Parse existing file structure (AST or heuristic)
- [ ] Mark unchanged nodes as `SKIP`
- [ ] Support REPLACE/INSERT/DELETE operations
- [ ] Emit edit-style LASK prompts for modifications

## Future Considerations

- **LASK Emitter Agent**: Specialized agent with tools for creating well-formed LASK prompts
- **Research Subagent Integration**: Hook into Research agent for context lookup
- **Plan Subagent Integration**: Accept structured plan output as input
- **Validation**: Verify generated prompts against LASK syntax rules

## Files

```
src/lask_lm/
├── models/
│   └── core.py              # CodeNode, ImplementState, LaskPrompt, GroupedOutput, ModifyManifest, reducers
├── agents/
│   └── implement/
│       ├── parallel_graph.py # LangGraph with Send() API, tree traversal, grouped output
│       ├── prompts.py       # Decomposition prompts
│       └── schemas.py       # LLM output schemas (including MODIFY fields)
├── tools/                   # (placeholder for LASK emitter tools)
├── main.py                  # Entry point
└── mcp_server.py            # MCP server for Claude Code

tests/
├── test_parallel_graph.py       # Unit tests for parallel spawning (22 tests)
├── test_graph_compatibility.py  # Compatibility tests for API (11 tests)
└── test_grouped_output.py       # Tests for grouped output and MODIFY manifest (16 tests)
```

## Testing

```bash
# Run all tests (48 total)
source .venv/bin/activate
PYTHONPATH=src python -m pytest tests/ -v

# Run end-to-end test (includes parallel demo)
PYTHONPATH=src python test_e2e.py

# Test MCP server
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | PYTHONPATH=src python -m lask_lm.mcp_server
```
