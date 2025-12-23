"""LangGraph definition for the Implement agent."""

from langgraph.graph import StateGraph, START, END

from lask_lm.models import ImplementState
from .nodes import (
    router_node,
    decomposer_node,
    collector_node,
    should_continue,
)


def create_implement_graph() -> StateGraph:
    """
    Create the Implement agent graph.

    Graph structure:
        START -> router -> decomposer <-> (loop while pending)
                              |
                              v
                          collector -> END
    """
    graph = StateGraph(ImplementState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("decomposer", decomposer_node)
    graph.add_node("collector", collector_node)

    # Add edges
    graph.add_edge(START, "router")
    graph.add_edge("router", "decomposer")

    # Conditional edge: continue decomposing or finish
    graph.add_conditional_edges(
        "decomposer",
        should_continue,
        {
            "decomposer": "decomposer",
            "collector": "collector",
        },
    )

    graph.add_edge("collector", END)

    return graph


def compile_implement_graph():
    """Compile the graph for execution."""
    graph = create_implement_graph()
    return graph.compile()
