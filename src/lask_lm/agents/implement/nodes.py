"""Node implementations for the Implement agent graph."""

import uuid
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Send

from lask_lm.models import (
    CodeNode,
    NodeType,
    NodeStatus,
    FileOperation,
    Contract,
    LaskPrompt,
    LaskDirective,
    ImplementState,
)
from .prompts import SYSTEM_PROMPT_BASE, DECOMPOSITION_PROMPTS
from .schemas import (
    DecomposeFileOutput,
    DecomposeClassOutput,
    DecomposeMethodOutput,
    LaskPromptOutput,
    ComponentOutput,
)


def _generate_node_id() -> str:
    """Generate a unique node ID."""
    return str(uuid.uuid4())[:8]


def _get_llm():
    """Get the LLM instance for decomposition."""
    return ChatOpenAI(model="gpt-4o", temperature=0.3)


def _structured_output(llm, schema):
    """Get structured output using OpenAI's strict mode."""
    return llm.with_structured_output(schema)


def router_node(state: ImplementState) -> dict:
    """
    Entry point: creates root nodes for each target file.

    Routes each file to either CREATE or MODIFY handling.
    """
    nodes = {}
    root_ids = []
    pending = []

    for file_target in state.target_files:
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


def decomposer_node(state: ImplementState) -> dict:
    """
    Main decomposition node: processes pending nodes.

    For each pending node:
    - If terminal (BLOCK + is_terminal): emit LASK prompt
    - Otherwise: decompose into children and add to pending
    """
    if not state.pending_node_ids:
        return {}

    # Process one node at a time (for now - parallel comes in Phase 3)
    node_id = state.pending_node_ids[0]
    node = state.nodes[node_id]

    # Safety check
    if state.current_depth >= state.max_depth:
        # Force terminal
        return _emit_terminal(state, node)

    llm = _get_llm()

    # Select prompt based on node type
    system_prompt = SYSTEM_PROMPT_BASE + "\n\n" + DECOMPOSITION_PROMPTS[node.node_type.value]

    # Build context message
    context_parts = [f"Intent: {node.intent}"]

    if node.contracts_required:
        required_contracts = [
            state.contract_registry.get(name)
            for name in node.contracts_required
            if name in state.contract_registry
        ]
        if required_contracts:
            context_parts.append("Required contracts:")
            for c in required_contracts:
                if c:
                    context_parts.append(f"  - {c.name}: {c.signature}")

    if node.context_files:
        context_parts.append(f"Context files: {', '.join(node.context_files)}")

    # Get structured output based on node type
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="\n".join(context_parts)),
    ]

    if node.node_type == NodeType.FILE:
        response = _structured_output(llm, DecomposeFileOutput).invoke(messages)
        return _process_file_decomposition(state, node, response)

    elif node.node_type == NodeType.CLASS:
        response = _structured_output(llm, DecomposeClassOutput).invoke(messages)
        return _process_class_decomposition(state, node, response)

    elif node.node_type == NodeType.METHOD:
        response = _structured_output(llm, DecomposeMethodOutput).invoke(messages)
        return _process_method_decomposition(state, node, response)

    elif node.node_type == NodeType.BLOCK:
        # Terminal - emit LASK prompt
        return _emit_terminal(state, node)

    return {}


def _process_file_decomposition(
    state: ImplementState,
    node: CodeNode,
    response: DecomposeFileOutput,
) -> dict:
    """Process FILE decomposition response."""
    new_nodes = dict(state.nodes)
    new_pending = list(state.pending_node_ids)
    new_contracts = dict(state.contract_registry)
    new_prompts = list(state.lask_prompts)

    # Mark current node as decomposing
    updated_node = node.model_copy()
    updated_node.status = NodeStatus.DECOMPOSING
    new_nodes[node.node_id] = updated_node

    # Remove from pending
    new_pending.remove(node.node_id)

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

        # If marked terminal, make it a BLOCK
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

    return {
        "nodes": new_nodes,
        "pending_node_ids": new_pending,
        "contract_registry": new_contracts,
        "lask_prompts": new_prompts,
        "current_depth": state.current_depth + 1,
    }


def _process_class_decomposition(
    state: ImplementState,
    node: CodeNode,
    response: DecomposeClassOutput,
) -> dict:
    """Process CLASS decomposition response."""
    new_nodes = dict(state.nodes)
    new_pending = list(state.pending_node_ids)
    new_contracts = dict(state.contract_registry)

    # Mark current node
    updated_node = node.model_copy()
    updated_node.status = NodeStatus.DECOMPOSING
    new_pending.remove(node.node_id)

    child_ids = []
    for comp in response.components:
        child_id = _generate_node_id()

        if comp.component_type in ("method", "function", "constructor"):
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

        for contract in child_node.contracts_provided:
            new_contracts[contract.name] = contract

    updated_node.children_ids = child_ids
    new_nodes[node.node_id] = updated_node

    return {
        "nodes": new_nodes,
        "pending_node_ids": new_pending,
        "contract_registry": new_contracts,
        "current_depth": state.current_depth + 1,
    }


def _process_method_decomposition(
    state: ImplementState,
    node: CodeNode,
    response: DecomposeMethodOutput,
) -> dict:
    """Process METHOD decomposition response."""
    new_nodes = dict(state.nodes)
    new_pending = list(state.pending_node_ids)
    new_prompts = list(state.lask_prompts)

    new_pending.remove(node.node_id)

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

        return {
            "nodes": new_nodes,
            "pending_node_ids": new_pending,
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
        new_pending.append(child_id)

    updated_node.children_ids = child_ids
    new_nodes[node.node_id] = updated_node

    return {
        "nodes": new_nodes,
        "pending_node_ids": new_pending,
        "current_depth": state.current_depth + 1,
    }


def _emit_terminal(state: ImplementState, node: CodeNode) -> dict:
    """Emit a LASK prompt for a terminal node."""
    llm = _get_llm()

    system_prompt = SYSTEM_PROMPT_BASE + "\n\n" + DECOMPOSITION_PROMPTS["block"]

    context_parts = [f"Intent: {node.intent}"]
    if node.context_files:
        context_parts.append(f"Context files: {', '.join(node.context_files)}")
    if node.contracts_required:
        context_parts.append(f"Required contracts: {', '.join(node.contracts_required)}")

    response = _structured_output(llm, LaskPromptOutput).invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="\n".join(context_parts)),
    ])

    new_nodes = dict(state.nodes)
    new_pending = list(state.pending_node_ids)
    new_prompts = list(state.lask_prompts)

    if node.node_id in new_pending:
        new_pending.remove(node.node_id)

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
    new_nodes[node.node_id] = updated_node

    new_prompts.append(lask_prompt)

    return {
        "nodes": new_nodes,
        "pending_node_ids": new_pending,
        "lask_prompts": new_prompts,
    }


def collector_node(state: ImplementState) -> dict:
    """
    Final node: collects all LASK prompts and prepares output.

    This is where we could do final validation, ordering, etc.
    """
    # For now, just pass through - prompts are already collected
    return {}


def should_continue(state: ImplementState) -> Literal["decomposer", "collector"]:
    """Routing function: continue decomposing or finish."""
    if state.pending_node_ids:
        return "decomposer"
    return "collector"
