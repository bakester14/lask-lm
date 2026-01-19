#!/usr/bin/env python3
"""
MCP Server that exposes the LangGraph implement agent to Claude Code.

Run with: python -m lask_lm.mcp_server
Or add to Claude Code: claude mcp add implement-agent -- python -m lask_lm.mcp_server
"""

import json
import sys
from typing import Any

from lask_lm.models import ImplementState, FileTarget, FileOperation, Contract
from lask_lm.agents.implement import compile_implement_graph


def list_tools() -> dict:
    """Return available tools."""
    contract_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Unique identifier for this contract (e.g., 'IUserService', 'UserRepository.GetById'). Used for dependency resolution between files."},
            "signature": {"type": "string", "description": "The type signature or interface definition. For methods: 'Task<User> GetById(int id)'. For interfaces: 'interface IUserService { ... }'. Included in generated prompts as [requires: ...] annotations."},
            "description": {"type": "string", "description": "Human-readable description of what this contract provides or expects. Helps the decomposer understand how to use the dependency."},
            "context_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files to include via @context(file) directive in generated prompts. LASK-CLI sends these files to the LLM for reference when generating code."
            }
        },
        "required": ["name", "signature", "description"]
    }

    return {
        "tools": [
            {
                "name": "decompose_to_lask",
                "description": """Decompose a code implementation task into LASK-compatible prompts.

LASK (Language as Source Kit) embeds LLM prompts in code comments using marker syntax. This tool generates those prompts from your implementation plan.

WORKFLOW:
1. Plan what files to create/modify and describe their purpose
2. Call this tool with file specifications
3. Write the returned 'comment' strings to your source files
4. Run 'lask <file>' CLI to generate code from those prompts

OUTPUT FORMAT:
Returns JSON with:
- total_prompts: Number of prompts generated
- prompts: Array of {file, comment, intent}
  - file: Target file path
  - comment: Ready-to-write LASK comment (e.g., '// @ @context(Helper.cs) Add validation')
  - intent: Human-readable description of what this prompt will generate
- tree_summary: Decomposition statistics

MARKER SYNTAX (written to source files):
- Basic: '// @ Add a method to validate email'
- With context: '// @ @context(Utils.cs) Use helper from Utils'
- With directives: '// @ @model(gpt-4) @temperature(0.3) Generate complex logic'
- Language-aware: Uses # for Python, <!-- --> for HTML, etc.

AFTER WRITING PROMPTS TO FILES:
- Generate code: 'lask MyFile.cs'
- Preview result: 'lask --preview MyFile.cs'
- Build from shadow: 'lask --build'
""",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "plan_summary": {
                            "type": "string",
                            "description": "High-level description of the feature or change being implemented. Include key requirements and constraints. This provides context for decomposition decisions. Example: 'Add user authentication with JWT tokens and refresh token support'"
                        },
                        "files": {
                            "type": "array",
                            "description": "List of files to create or modify. Each file is decomposed into LASK prompts independently. For related files, use contracts to define interfaces between them.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string", "description": "File path relative to project root (e.g., 'src/Services/UserService.cs'). The file extension determines comment syntax (// for .cs/.js, # for .py, etc.)"},
                                    "operation": {"type": "string", "enum": ["create", "modify"], "description": "CREATE: Generate a new file from scratch. MODIFY: Add to or change an existing file (requires existing_content field)."},
                                    "description": {"type": "string", "description": "Detailed description of the file's purpose and contents. Be specific about classes, methods, and behaviors needed. More detail = better decomposition. Example: 'Repository class implementing IUserRepository with GetById, Create, Update, Delete methods using Entity Framework'"},
                                    "language": {"type": "string", "description": "Programming language for comment syntax. Defaults to 'csharp'. Supported: python, javascript, typescript, go, rust, java, html, css, sql. Usually inferred from file extension."},
                                    "contracts_provided": {
                                        "type": "array",
                                        "items": contract_schema,
                                        "description": "Contracts (interfaces/APIs) this file must implement. Use when other files depend on this file's public interface. Each contract specifies a name, signature, and description. The decomposer ensures these contracts are satisfied."
                                    },
                                    "existing_content": {
                                        "type": "string",
                                        "description": "REQUIRED for MODIFY operations: The complete current file content. The decomposer analyzes this to identify what to add/change vs keep unchanged. Unchanged sections are marked with SMART SKIP. Without this, MODIFY behaves like CREATE."
                                    }
                                },
                                "required": ["path", "operation", "description"]
                            }
                        },
                        "external_contracts": {
                            "type": "array",
                            "items": contract_schema,
                            "description": "Contracts from files outside this decomposition that the new/modified files may depend on. Provide the interface signatures so generated code can correctly reference them. Include context_files to add @context directives pointing to those files."
                        }
                    },
                    "required": ["plan_summary", "files"]
                }
            }
        ]
    }


def call_tool(name: str, arguments: dict) -> dict:
    """Execute a tool call."""
    if name == "decompose_to_lask":
        return decompose_to_lask(arguments)
    return {"error": f"Unknown tool: {name}"}


def _parse_contract(c: dict) -> Contract:
    """Parse a contract dict into a Contract model."""
    return Contract(
        name=c["name"],
        signature=c["signature"],
        description=c["description"],
        context_files=c.get("context_files", []),
    )


def decompose_to_lask(args: dict) -> dict:
    """Run the implement agent to decompose a task into LASK prompts."""
    try:
        # Build file targets with their contract obligations
        files = [
            FileTarget(
                path=f["path"],
                operation=FileOperation(f.get("operation", "create")),
                description=f["description"],
                language=f.get("language", "csharp"),
                existing_content=f.get("existing_content"),
                contracts_provided=[
                    _parse_contract(c) for c in f.get("contracts_provided", [])
                ],
            )
            for f in args.get("files", [])
        ]

        # Parse external contracts
        external_contracts = [
            _parse_contract(c) for c in args.get("external_contracts", [])
        ]

        # Create state
        state = ImplementState(
            plan_summary=args["plan_summary"],
            target_files=files,
            external_contracts=external_contracts,
        )

        # Run the graph
        app = compile_implement_graph()
        result = app.invoke(state)

        # Format output
        prompts = []
        for p in result["lask_prompts"]:
            prompts.append({
                "file": p.file_path,
                "comment": p.to_comment(),
                "intent": p.intent,
            })

        output = {
            "total_prompts": len(prompts),
            "prompts": prompts,
            "tree_summary": f"{len(result['nodes'])} nodes decomposed",
        }

        return {
            "content": [{
                "type": "text",
                "text": json.dumps(output, indent=2)
            }]
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error: {str(e)}"
            }],
            "isError": True
        }


def handle_request(request: dict) -> dict:
    """Handle incoming MCP request."""
    method = request.get("method", "")
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "implement-agent", "version": "0.1.0"}
            }
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": list_tools()
        }

    elif method == "tools/call":
        params = request.get("params", {})
        result = call_tool(params.get("name"), params.get("arguments", {}))
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result
        }

    elif method == "notifications/initialized":
        return None  # No response needed for notifications

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"}
    }


def main():
    """Main loop - reads JSON-RPC from stdin, writes to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            response = handle_request(request)
            if response:  # Some methods don't need responses
                print(json.dumps(response), flush=True)
        except json.JSONDecodeError as e:
            error = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"}
            }
            print(json.dumps(error), flush=True)
        except Exception as e:
            error = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": f"Internal error: {e}"}
            }
            print(json.dumps(error), flush=True)


if __name__ == "__main__":
    main()
