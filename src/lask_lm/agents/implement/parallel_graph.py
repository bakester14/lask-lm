"""LangGraph definition using Send() API for concurrent decomposition.

This is the primary implementation of the Implement agent graph.
It uses LangGraph's Send() API to process sibling nodes in parallel,
while maintaining backwards compatibility with the ImplementState API.
"""

import uuid
from typing import Literal, Sequence, Any

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from lask_lm.models import (
    ImplementState,
    ParallelImplementState,
    SingleNodeState,
    CodeNode,
    NodeType,
    NodeStatus,
    FileOperation,
    Contract,
    LaskPrompt,
    LaskDirective,
    FileTarget,
)
from .prompts import SYSTEM_PROMPT_BASE, DECOMPOSITION_PROMPTS
from .schemas import (
    DecomposeFileOutput,
    DecomposeClassOutput,
    DecomposeMethodOutput,
    LaskPromptOutput,
)


# =============================================================================
# Utility Functions
# =============================================================================

def _generate_node_id() -> str:
    """Generate a unique node ID."""
    return str(uuid.uuid4())[:8]


def _get_llm():
    """Get the LLM instance for decomposition."""
    return ChatOpenAI(model="gpt-4o", temperature=0.3)


def _structured_output(llm, schema):
    """Get structured output using OpenAI's strict mode."""
    return llm.with_structured_output(schema)


def _convert_to_parallel_state(state: ImplementState | dict) -> ParallelImplementState:
    """Convert ImplementState (Pydantic) to ParallelImplementState (TypedDict)."""
    if isinstance(state, ImplementState):
        return {
            "plan_summary": state.plan_summary,
            "target_files": state.target_files,
            "max_depth": state.max_depth,
            "nodes": dict(state.nodes),
            "root_node_ids": list(state.root_node_ids),
            "pending_node_ids": list(state.pending_node_ids),
            "contract_registry": dict(state.contract_registry),
            "lask_prompts": list(state.lask_prompts),
            "current_depth": state.current_depth,
        }
    return state  # Already a dict/TypedDict


def router_node(state: ParallelImplementState | ImplementState) -> dict:
    """
    Entry point: creates root nodes for each target file.

    Same as sequential version - creates FILE nodes and adds to pending.
    Accepts both ImplementState (Pydantic) and ParallelImplementState (dict).
    """
    nodes = {}
    root_ids = []
    pending = []

    # Handle both Pydantic and dict state
    if isinstance(state, ImplementState):
        target_files = state.target_files
    else:
        target_files = state.get("target_files", [])

    for file_target in target_files:
        node_id = _generate_node_id()
        node = CodeNode(
            node_id=node_id,
            node_type=NodeType.FILE,
            intent=file_target.description,
            status=NodeStatus.PENDING,
            context_files=[file_target.path],
        )
        nodes[node_id] = node
        root_ids.append(node_id)
        pending.append(node_id)

    return {
        "nodes": nodes,
        "root_node_ids": root_ids,
        "pending_node_ids": pending,
    }


def dispatch_to_parallel(state: ParallelImplementState) -> Sequence[Send] | Literal["collector"]:
    """
    Routing function that dispatches pending nodes to parallel decomposers.

    Uses LangGraph's Send() API to fan-out to parallel workers.
    Each pending node gets its own decomposer instance running concurrently.

    Returns:
        - List of Send objects for parallel execution
        - "collector" string when no more pending nodes
    """
    pending = state.get("pending_node_ids", [])
    nodes = state.get("nodes", {})

    if not pending:
        return "collector"

    sends = []
    for node_id in pending:
        node = nodes.get(node_id)
        if node:
            # Create a SingleNodeState for this node
            single_state: SingleNodeState = {
                "node_id": node_id,
                "node": node,
                "plan_summary": state.get("plan_summary", ""),
                "contract_registry": state.get("contract_registry", {}),
                "current_depth": state.get("current_depth", 0),
                "max_depth": state.get("max_depth", 10),
            }
            sends.append(Send("parallel_decomposer", single_state))

    return sends


def aggregator_node(state: ParallelImplementState) -> dict:
    """
    Aggregation node after parallel decomposition.

    This node runs after all parallel decomposers complete.
    It rebuilds the pending list from actual node statuses,
    ensuring processed nodes are not re-queued.
    """
    nodes = state.get("nodes", {})

    # Rebuild pending from nodes that are actually PENDING
    new_pending = [
        node_id
        for node_id, node in nodes.items()
        if node.status == NodeStatus.PENDING
    ]

    return {"pending_node_ids": new_pending}


def parallel_decomposer_node(state: SingleNodeState) -> dict:
    """
    Process a single node in parallel.

    This node receives state via Send() from the dispatcher.
    Returns partial state that will be merged with other parallel results.
    """
    node_id = state.get("node_id")
    node = state.get("node")

    if not node:
        return {}

    current_depth = state.get("current_depth", 0)
    max_depth = state.get("max_depth", 10)
    contract_registry = state.get("contract_registry", {})

    # Safety check - force terminal if too deep
    if current_depth >= max_depth:
        return _emit_terminal_parallel(node, contract_registry)

    llm = _get_llm()

    # Select prompt based on node type
    system_prompt = SYSTEM_PROMPT_BASE + "\n\n" + DECOMPOSITION_PROMPTS[node.node_type.value]

    # Build context message
    context_parts = [f"Intent: {node.intent}"]

    if node.contracts_required:
        required_contracts = [
            contract_registry.get(name)
            for name in node.contracts_required
            if name in contract_registry
        ]
        if required_contracts:
            context_parts.append("Required contracts:")
            for c in required_contracts:
                if c:
                    context_parts.append(f"  - {c.name}: {c.signature}")

    if node.context_files:
        context_parts.append(f"Context files: {', '.join(node.context_files)}")

    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="\n".join(context_parts)),
    ]

    if node.node_type == NodeType.FILE:
        response = _structured_output(llm, DecomposeFileOutput).invoke(messages)
        return _process_file_decomposition_parallel(node, response, current_depth)

    elif node.node_type == NodeType.CLASS:
        response = _structured_output(llm, DecomposeClassOutput).invoke(messages)
        return _process_class_decomposition_parallel(node, response, current_depth)

    elif node.node_type == NodeType.METHOD:
        response = _structured_output(llm, DecomposeMethodOutput).invoke(messages)
        return _process_method_decomposition_parallel(node, response, current_depth)

    elif node.node_type == NodeType.BLOCK:
        return _emit_terminal_parallel(node, contract_registry)

    return {}


def _process_file_decomposition_parallel(
    node: CodeNode,
    response: DecomposeFileOutput,
    current_depth: int,
) -> dict:
    """Process FILE decomposition response in parallel context."""
    new_nodes = {}
    new_pending = []
    new_contracts = {}

    # Mark current node as decomposing
    updated_node = node.model_copy()
    updated_node.status = NodeStatus.DECOMPOSING
    new_nodes[node.node_id] = updated_node

    # Create child nodes
    child_ids = []
    for comp in response.components:
        child_id = _generate_node_id()

        # Map component type to NodeType
        if comp.component_type in ("class", "struct", "interface", "enum"):
            child_type = NodeType.CLASS
        elif comp.component_type in ("method", "function"):
            child_type = NodeType.METHOD
        else:
            child_type = NodeType.BLOCK

        if comp.is_terminal:
            child_type = NodeType.BLOCK

        # Inherit parent's context_files if component doesn't specify any
        child_context_files = comp.context_files if comp.context_files else node.context_files

        child_node = CodeNode(
            node_id=child_id,
            node_type=child_type,
            intent=comp.intent,
            parent_id=node.node_id,
            contracts_provided=[
                Contract(
                    name=c.name,
                    signature=c.signature,
                    description=c.description,
                    context_files=child_context_files,
                )
                for c in comp.contracts_provided
            ],
            contracts_required=comp.contracts_required,
            context_files=child_context_files,
            status=NodeStatus.PENDING,
        )

        new_nodes[child_id] = child_node
        child_ids.append(child_id)
        new_pending.append(child_id)

        # Register contracts
        for contract in child_node.contracts_provided:
            new_contracts[contract.name] = contract

    # Update parent with children
    updated_node.children_ids = child_ids
    new_nodes[node.node_id] = updated_node

    # Don't return pending_node_ids - aggregator rebuilds from node statuses
    return {
        "nodes": new_nodes,
        "contract_registry": new_contracts,
        "current_depth": current_depth + 1,
    }


def _process_class_decomposition_parallel(
    node: CodeNode,
    response: DecomposeClassOutput,
    current_depth: int,
) -> dict:
    """Process CLASS decomposition response in parallel context."""
    new_nodes = {}
    new_pending = []
    new_contracts = {}

    updated_node = node.model_copy()
    updated_node.status = NodeStatus.DECOMPOSING

    child_ids = []
    for comp in response.components:
        child_id = _generate_node_id()

        if comp.component_type in ("method", "function", "constructor"):
            child_type = NodeType.METHOD
        else:
            child_type = NodeType.BLOCK

        if comp.is_terminal:
            child_type = NodeType.BLOCK

        child_context_files = comp.context_files if comp.context_files else node.context_files

        child_node = CodeNode(
            node_id=child_id,
            node_type=child_type,
            intent=comp.intent,
            parent_id=node.node_id,
            contracts_provided=[
                Contract(
                    name=c.name,
                    signature=c.signature,
                    description=c.description,
                    context_files=child_context_files,
                )
                for c in comp.contracts_provided
            ],
            contracts_required=comp.contracts_required,
            context_files=child_context_files,
            status=NodeStatus.PENDING,
        )

        new_nodes[child_id] = child_node
        child_ids.append(child_id)
        new_pending.append(child_id)

        for contract in child_node.contracts_provided:
            new_contracts[contract.name] = contract

    updated_node.children_ids = child_ids
    new_nodes[node.node_id] = updated_node

    # Don't return pending_node_ids - aggregator rebuilds from node statuses
    return {
        "nodes": new_nodes,
        "contract_registry": new_contracts,
        "current_depth": current_depth + 1,
    }


def _process_method_decomposition_parallel(
    node: CodeNode,
    response: DecomposeMethodOutput,
    current_depth: int,
) -> dict:
    """Process METHOD decomposition response in parallel context."""
    new_nodes = {}
    new_pending = []
    new_prompts = []

    if response.is_terminal:
        # This method is small enough - emit directly
        updated_node = node.model_copy()
        updated_node.status = NodeStatus.COMPLETE
        updated_node.lask_prompt = LaskPrompt(
            file_path=node.context_files[0] if node.context_files else "unknown",
            intent=response.terminal_intent or node.intent,
            directives=[
                LaskDirective(directive_type="context", value=f)
                for f in node.context_files
            ],
        )
        new_nodes[node.node_id] = updated_node
        new_prompts.append(updated_node.lask_prompt)

        # Don't return pending_node_ids - aggregator rebuilds from node statuses
        return {
            "nodes": new_nodes,
            "lask_prompts": new_prompts,
        }

    # Decompose into blocks
    updated_node = node.model_copy()
    updated_node.status = NodeStatus.DECOMPOSING

    child_ids = []
    for comp in response.blocks:
        child_id = _generate_node_id()
        child_node = CodeNode(
            node_id=child_id,
            node_type=NodeType.BLOCK,
            intent=comp.intent,
            parent_id=node.node_id,
            contracts_required=comp.contracts_required,
            context_files=comp.context_files or node.context_files,
            status=NodeStatus.PENDING,
        )
        new_nodes[child_id] = child_node
        child_ids.append(child_id)

    updated_node.children_ids = child_ids
    new_nodes[node.node_id] = updated_node

    # Don't return pending_node_ids - aggregator rebuilds from node statuses
    return {
        "nodes": new_nodes,
        "current_depth": current_depth + 1,
    }


def _emit_terminal_parallel(node: CodeNode, contract_registry: dict) -> dict:
    """Emit a LASK prompt for a terminal node in parallel context."""
    llm = _get_llm()

    system_prompt = SYSTEM_PROMPT_BASE + "\n\n" + DECOMPOSITION_PROMPTS["block"]

    context_parts = [f"Intent: {node.intent}"]
    if node.context_files:
        context_parts.append(f"Context files: {', '.join(node.context_files)}")
    if node.contracts_required:
        context_parts.append(f"Required contracts: {', '.join(node.contracts_required)}")

    from langchain_core.messages import SystemMessage, HumanMessage
    response = _structured_output(llm, LaskPromptOutput).invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="\n".join(context_parts)),
    ])

    # Build LASK prompt
    directives = [
        LaskDirective(directive_type="context", value=f)
        for f in response.context_files
    ]
    for d in response.additional_directives:
        directives.append(LaskDirective(directive_type=d.name, value=d.value))

    lask_prompt = LaskPrompt(
        file_path=node.context_files[0] if node.context_files else "unknown",
        intent=response.intent,
        directives=directives,
    )

    updated_node = node.model_copy()
    updated_node.status = NodeStatus.COMPLETE
    updated_node.lask_prompt = lask_prompt

    # Don't return pending_node_ids - aggregator rebuilds from node statuses
    return {
        "nodes": {node.node_id: updated_node},
        "lask_prompts": [lask_prompt],
    }


def collector_node(state: ParallelImplementState) -> dict:
    """
    Final node: collects all LASK prompts and prepares output.

    At this point, all parallel decomposers have finished and
    their results have been merged via reducers.
    """
    # Just pass through - the reducers have already done the work
    return {}


def create_parallel_implement_graph() -> StateGraph:
    """
    Create the parallel Implement agent graph using Send() API.

    Graph structure:
        START -> router --(conditional)--> parallel_decomposer (x N via Send)
                              |                      |
                              |                      v
                              |              aggregator (merge results)
                              |                      |
                              |<---(loop back)-------+
                              |
                              v (when no pending)
                          collector -> END

    The dispatch_to_parallel function fans out to N parallel decomposers.
    Results are automatically merged via state reducers.
    """
    graph = StateGraph(ParallelImplementState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("parallel_decomposer", parallel_decomposer_node)
    graph.add_node("aggregator", aggregator_node)
    graph.add_node("collector", collector_node)

    # Add edges
    graph.add_edge(START, "router")

    # Router dispatches to parallel decomposers via Send()
    graph.add_conditional_edges(
        "router",
        dispatch_to_parallel,
        ["parallel_decomposer", "collector"],
    )

    # Parallel decomposers feed into aggregator
    graph.add_edge("parallel_decomposer", "aggregator")

    # Aggregator decides: more decomposition or finish
    graph.add_conditional_edges(
        "aggregator",
        dispatch_to_parallel,
        ["parallel_decomposer", "collector"],
    )

    graph.add_edge("collector", END)

    return graph


def compile_parallel_implement_graph():
    """Compile the parallel graph for execution."""
    graph = create_parallel_implement_graph()
    return graph.compile()


# =============================================================================
# Backwards-compatible API (aliases for sequential graph API)
# =============================================================================

def create_implement_graph() -> StateGraph:
    """
    Create the Implement agent graph.

    This is the main entry point for creating the graph.
    Uses parallel execution via Send() API internally.

    For backwards compatibility, the graph accepts ImplementState
    and returns compatible output structure.
    """
    return create_parallel_implement_graph()


def compile_implement_graph():
    """
    Compile the graph for execution.

    Returns a compiled graph that can be invoked with either
    ImplementState (Pydantic) or ParallelImplementState (TypedDict).
    """
    return compile_parallel_implement_graph()


# Node aliases for compatibility with code that imports specific nodes
decomposer_node = parallel_decomposer_node  # Alias for tests that check node names
