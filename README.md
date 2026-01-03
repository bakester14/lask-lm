# lask-lm

An LLM-powered agent that recursively decomposes code generation tasks into LASK-compatible prompts. Takes a high-level plan and breaks it down through FILE → CLASS → METHOD → BLOCK granularity levels, emitting ~10-line code prompts at leaf nodes.

## What It Does

1. **Accepts a plan** - Description of what to implement + target files
2. **Recursively decomposes** - Breaks down each file into classes, methods, and blocks
3. **Emits LASK prompts** - Terminal nodes produce prompts like `// @ @context(Helper.cs) Add validation logic`
4. **Groups output by file** - Prompts are ordered by tree traversal, preserving code structure
5. **Generates MODIFY manifests** - For file modifications, produces JSON manifests with location metadata

## Output Structure

```python
GroupedOutput(
    plan_summary="Implement user authentication",
    files=[
        OrderedFilePrompts(
            file_path="UserService.cs",
            operation=CREATE,
            prompts=[...],  # In tree-traversal order
            modify_manifest=None,
        ),
        OrderedFilePrompts(
            file_path="AuthHelper.cs",
            operation=MODIFY,
            prompts=[...],
            modify_manifest=ModifyManifest(
                operations=[
                    ModifyOperation(
                        operation_type=INSERT,
                        location=LocationMetadata(insertion_point="after method Validate"),
                        intent="Add token refresh logic",
                    ),
                ],
            ),
        ),
    ],
)
```

## Usage

### As MCP Server (Claude Code)

```bash
# Add to claude code config
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | PYTHONPATH=src python -m lask_lm.mcp_server
```

### Programmatic

```python
from lask_lm.models import ImplementState, FileTarget, FileOperation
from lask_lm.agents.implement import compile_implement_graph

state = ImplementState(
    plan_summary="Create a calculator service",
    target_files=[
        FileTarget(
            path="Calculator.cs",
            operation=FileOperation.CREATE,
            description="Calculator with basic arithmetic operations",
        ),
    ],
)

app = compile_implement_graph()
result = app.invoke(state)

# Flat list (backwards compatible)
for prompt in result["lask_prompts"]:
    print(prompt.to_comment())

# Grouped output (recommended)
grouped = result["grouped_output"]
for file_output in grouped.files:
    print(f"\n=== {file_output.file_path} ===")
    for prompt in file_output.prompts:
        print(prompt.to_comment())
```

## Architecture

```
START → router → parallel_decomposer (x N via Send()) → aggregator → collector → END
                        ↑                                    │
                        └────────────── loop ────────────────┘
```

- **Router**: Creates FILE nodes for each target
- **Parallel Decomposer**: LLM decomposes nodes into children or emits terminal prompts
- **Aggregator**: Rebuilds pending queue from node statuses
- **Collector**: Groups prompts by file, preserves tree order, generates MODIFY manifests

## Key Models

| Model | Purpose |
|-------|---------|
| `LaskPrompt` | Terminal output with intent, directives, MODIFY metadata |
| `GroupedOutput` | Final output with files grouped and ordered |
| `ModifyManifest` | JSON manifest for MODIFY operations |
| `LocationMetadata` | Where to apply modifications (insertion_point, line_range, ast_path) |
| `Contract` | Interface contracts for cross-node dependencies |

## Development

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run tests (65 total)
PYTHONPATH=src python -m pytest tests/ -v

# Run E2E demo
PYTHONPATH=src python test_e2e.py
```

## Implementation Status

- [x] Phase 1-4: Core types, prompts, graph, parallel execution
- [x] Phase 4.5: Grouped output, MODIFY manifests
- [x] Phase 5: Contract registry validation
- [ ] Phase 6: Full MODIFY support (AST parsing, SKIP detection)

See `IMPLEMENTATION_PLAN.md` for details.
