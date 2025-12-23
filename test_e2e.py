"""End-to-end test of the implement agent with debugging features."""

import json
from pprint import pprint

from lask_lm.models import ImplementState, FileTarget, FileOperation
from lask_lm.agents.implement.graph import create_implement_graph


def run_with_debugging():
    """Run the agent with step-by-step debugging."""

    # Create initial state
    initial_state = ImplementState(
        plan_summary="Create a simple calculator service",
        target_files=[
            FileTarget(
                path="Calculator.cs",
                operation=FileOperation.CREATE,
                description="A calculator class with add, subtract, multiply, divide methods",
                language="csharp",
            )
        ],
    )

    # Compile with debugging
    graph = create_implement_graph()
    app = graph.compile()

    print("=" * 60)
    print("LANGGRAPH BENEFITS DEMO")
    print("=" * 60)

    # 1. Stream execution step-by-step
    print("\n[1] STREAMING EXECUTION - see each node as it runs:\n")

    step_count = 0
    for event in app.stream(initial_state, stream_mode="updates"):
        step_count += 1
        for node_name, updates in event.items():
            print(f"Step {step_count}: {node_name}")
            if updates:  # Guard against None
                if "pending_node_ids" in updates:
                    print(f"  Pending nodes: {len(updates['pending_node_ids'])}")
                if "lask_prompts" in updates:
                    print(f"  LASK prompts generated: {len(updates['lask_prompts'])}")
                if "nodes" in updates:
                    print(f"  Nodes in tree: {len(updates['nodes'])}")
        print()

    # 2. Get final state
    print("\n[2] FINAL STATE INSPECTION:\n")
    final_state = app.invoke(initial_state)

    print(f"Total nodes created: {len(final_state['nodes'])}")
    print(f"LASK prompts generated: {len(final_state['lask_prompts'])}")

    # 3. Show the decomposition tree
    print("\n[3] DECOMPOSITION TREE:\n")
    for node_id, node in final_state['nodes'].items():
        indent = "  " * (0 if node.parent_id is None else 1)
        status_icon = "✓" if node.status.value == "complete" else "○"
        print(f"{indent}{status_icon} [{node.node_type.value.upper()}] {node.intent[:60]}...")
        if node.children_ids:
            print(f"{indent}  └─ children: {node.children_ids}")

    # 4. Show generated LASK prompts
    print("\n[4] GENERATED LASK PROMPTS:\n")
    for i, prompt in enumerate(final_state['lask_prompts'], 1):
        print(f"Prompt {i}:")
        print(f"  File: {prompt.file_path}")
        print(f"  Intent: {prompt.intent}")
        print(f"  As comment: {prompt.to_comment()}")
        print()

    return final_state


def show_graph_visualization():
    """Show the graph structure."""
    print("\n[5] GRAPH STRUCTURE (what LangGraph gives you):\n")

    graph = create_implement_graph()

    # Get the graph's edges
    print("Nodes:")
    for node in graph.nodes:
        print(f"  - {node}")

    print("\nEdges:")
    print("  START -> router")
    print("  router -> decomposer")
    print("  decomposer -> decomposer (if pending nodes)")
    print("  decomposer -> collector (if no pending nodes)")
    print("  collector -> END")

    # Try to generate mermaid diagram
    app = graph.compile()
    try:
        mermaid = app.get_graph().draw_mermaid()
        print("\nMermaid diagram (paste into mermaid.live):")
        print(mermaid)
    except Exception as e:
        print(f"\n(Mermaid generation not available: {e})")


if __name__ == "__main__":
    show_graph_visualization()
    print("\n" + "=" * 60 + "\n")

    final_state = run_with_debugging()

    print("\n" + "=" * 60)
    print("KEY LANGGRAPH BENEFITS:")
    print("=" * 60)
    print("""
1. STREAMING: See each step as it executes (app.stream())
2. STATE MANAGEMENT: Automatic state merging between nodes
3. CONDITIONAL ROUTING: should_continue() decides next node
4. GRAPH VISUALIZATION: Export to Mermaid for docs
5. CHECKPOINTING: Can save/restore state mid-execution (not shown)
6. DEBUGGING: Inspect state after each node
7. PARALLELISM: Send() API for parallel subgraph execution (Phase 3)
8. LANGSMITH: Full tracing with LangSmith integration (if configured)
""")
