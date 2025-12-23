#!/usr/bin/env python3
"""
MCP Server that exposes the LangGraph implement agent to Claude Code.

Run with: python -m lask_lm.mcp_server
Or add to Claude Code: claude mcp add implement-agent -- python -m lask_lm.mcp_server
"""

import json
import sys
from typing import Any

from lask_lm.models import ImplementState, FileTarget, FileOperation
from lask_lm.agents.implement.graph import compile_implement_graph


def list_tools() -> dict:
    """Return available tools."""
    return {
        "tools": [
            {
                "name": "decompose_to_lask",
                "description": "Decompose a code implementation task into LASK-compatible prompts. Use this when you need to break down a feature into smaller code generation tasks.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "plan_summary": {
                            "type": "string",
                            "description": "Description of what needs to be implemented"
                        },
                        "files": {
                            "type": "array",
                            "description": "List of files to create or modify",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string", "description": "File path"},
                                    "operation": {"type": "string", "enum": ["create", "modify"]},
                                    "description": {"type": "string", "description": "What the file should do"},
                                    "language": {"type": "string", "description": "Programming language (default: csharp)"}
                                },
                                "required": ["path", "operation", "description"]
                            }
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


def decompose_to_lask(args: dict) -> dict:
    """Run the implement agent to decompose a task into LASK prompts."""
    try:
        # Build file targets
        files = [
            FileTarget(
                path=f["path"],
                operation=FileOperation(f.get("operation", "create")),
                description=f["description"],
                language=f.get("language", "csharp"),
            )
            for f in args.get("files", [])
        ]

        # Create state
        state = ImplementState(
            plan_summary=args["plan_summary"],
            target_files=files,
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
