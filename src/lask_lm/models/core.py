"""Core Pydantic models for the Implement agent's recursive decomposition."""

import operator
from enum import Enum
from typing import Annotated, Any
from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Granularity level in the decomposition tree."""
    FILE = "file"
    CLASS = "class"
    METHOD = "method"
    BLOCK = "block"


class NodeStatus(str, Enum):
    """Processing status of a CodeNode."""
    PENDING = "pending"
    DECOMPOSING = "decomposing"
    SKIP = "skip"  # For MODIFY mode: unchanged nodes
    COMPLETE = "complete"


class FileOperation(str, Enum):
    """Whether a file is being created or modified."""
    CREATE = "create"
    MODIFY = "modify"


class OperationType(str, Enum):
    """Type of modification operation for MODIFY mode."""
    INSERT = "insert"
    REPLACE = "replace"
    DELETE = "delete"


class ValidationSeverity(str, Enum):
    """Severity level for contract validation issues."""
    WARNING = "warning"
    ERROR = "error"


class ContractValidationIssue(BaseModel):
    """A single contract validation issue detected during decomposition."""
    severity: ValidationSeverity = Field(description="Issue severity level")
    code: str = Field(description="Error code (e.g., 'MISSING_CONTRACT', 'DUPLICATE_NAME', 'CIRCULAR_DEP')")
    message: str = Field(description="Human-readable description of the issue")
    node_id: str | None = Field(default=None, description="Related node ID if applicable")
    contract_name: str | None = Field(default=None, description="Related contract name if applicable")


class Contract(BaseModel):
    """
    Interface contract that a CodeNode exposes to its siblings/children.

    Contracts are how we pass only the necessary context between parallel
    subagents without duplicating the full tree state.
    """
    name: str = Field(description="Identifier for this contract (e.g., 'UserService.validate')")
    signature: str = Field(description="Type signature or interface definition")
    description: str = Field(description="What this contract provides/expects")
    context_files: list[str] = Field(
        default_factory=list,
        description="Files that should be referenced via @context directive"
    )
    provider_node_id: str | None = Field(
        default=None,
        description="ID of the node that provides this contract (for duplicate detection)"
    )


class LaskDirective(BaseModel):
    """A LASK directive to be included in a prompt (e.g., @context, @model)."""
    directive_type: str = Field(description="Directive name without @ (e.g., 'context', 'model')")
    value: str = Field(description="Directive value (e.g., 'UserRepository.cs', 'gpt-4')")


class LaskPrompt(BaseModel):
    """
    A fully-formed LASK prompt ready to be written to a source file.

    This is the terminal output of the decomposition tree - a single
    "thought" of 1-10 lines that LASK will expand into actual code.
    """
    file_path: str = Field(description="Target file path for this prompt")
    intent: str = Field(description="What this code block should accomplish")
    directives: list[LaskDirective] = Field(
        default_factory=list,
        description="LASK directives (@context, @model, etc.)"
    )
    insertion_point: str | None = Field(
        default=None,
        description="For MODIFY: where to insert (e.g., 'after method GetById')"
    )
    replaces: str | None = Field(
        default=None,
        description="For MODIFY: description of code being replaced"
    )
    resolved_contracts: list["Contract"] = Field(
        default_factory=list,
        description="Contracts required by this prompt, resolved from registry"
    )
    is_delete: bool = Field(
        default=False,
        description="True if this is a DELETE operation (removes code, no generation)"
    )

    def to_comment(self, comment_prefix: str = "//") -> str:
        """Render as a LASK-compatible comment."""
        # Handle DELETE operations specially
        if self.is_delete:
            return f"{comment_prefix} @delete {self.replaces or 'target code'}"

        parts = []

        # Add directives first
        for directive in self.directives:
            parts.append(f"@{directive.directive_type}({directive.value})")

        # Add contract context files as @context directives (deduplicated)
        existing_context = {d.value for d in self.directives if d.directive_type == "context"}
        for contract in self.resolved_contracts:
            for ctx_file in contract.context_files:
                if ctx_file not in existing_context:
                    parts.append(f"@context({ctx_file})")
                    existing_context.add(ctx_file)

        # Build intent with contract requirements appended
        intent = self.intent
        if self.resolved_contracts:
            contract_info = ", ".join(f"{c.name}: {c.signature}" for c in self.resolved_contracts)
            intent = f"{intent} [requires: {contract_info}]"

        parts.append(intent)

        return f"{comment_prefix} @ {' '.join(parts)}"


class CodeNode(BaseModel):
    """
    A node in the recursive decomposition tree.

    Each node represents a unit of code at some granularity level.
    Internal nodes decompose into children; leaf nodes (BLOCK) emit LASK prompts.
    """
    node_id: str = Field(description="Unique identifier for this node")
    node_type: NodeType = Field(description="Granularity level")
    intent: str = Field(description="What this node should accomplish")

    # Relationships
    parent_id: str | None = Field(default=None, description="Parent node ID")
    children_ids: list[str] = Field(default_factory=list, description="Child node IDs")

    # Context passing
    contracts_provided: list[Contract] = Field(
        default_factory=list,
        description="Contracts this node exposes to siblings"
    )
    contracts_required: list[str] = Field(
        default_factory=list,
        description="Contract names this node depends on"
    )
    context_files: list[str] = Field(
        default_factory=list,
        description="Files to reference via @context"
    )

    # State
    status: NodeStatus = Field(default=NodeStatus.PENDING)

    # Output (only set for terminal BLOCK nodes)
    lask_prompt: LaskPrompt | None = Field(
        default=None,
        description="The LASK prompt (only for terminal BLOCK nodes)"
    )

    # MODIFY mode: existing file content (only set for FILE nodes with MODIFY operation)
    existing_content: str | None = Field(
        default=None,
        description="For MODIFY FILE nodes: the current file content"
    )


class FileTarget(BaseModel):
    """A file that the Implement agent will create or modify."""
    path: str = Field(description="File path relative to project root")
    operation: FileOperation = Field(description="CREATE or MODIFY")
    description: str = Field(description="High-level description of the file's purpose")
    language: str = Field(default="csharp", description="Programming language for comment syntax")
    existing_content: str | None = Field(
        default=None,
        description="For MODIFY: the current file content"
    )
    contracts_provided: list[Contract] = Field(
        default_factory=list,
        description="Contracts this file is obligated to implement (from plan)"
    )


class LocationMetadata(BaseModel):
    """Location information for MODIFY operations."""
    insertion_point: str | None = Field(
        default=None,
        description="Human-readable insertion point (e.g., 'after method GetById')"
    )
    line_range: tuple[int, int] | None = Field(
        default=None,
        description="Start and end line numbers (1-indexed) - populated by Phase 6"
    )
    ast_path: str | None = Field(
        default=None,
        description="Structural path (e.g., 'class Foo > method Bar') - populated by Phase 6"
    )


class ModifyOperation(BaseModel):
    """A single modification operation in a MODIFY manifest."""
    operation_id: str = Field(description="Unique ID for this operation")
    operation_type: OperationType = Field(description="INSERT, REPLACE, or DELETE")
    location: LocationMetadata = Field(description="Where in the file to apply")
    replaces: str | None = Field(
        default=None,
        description="Description of code being replaced (for REPLACE ops)"
    )
    intent: str = Field(description="What this modification accomplishes")
    directives: list[LaskDirective] = Field(
        default_factory=list,
        description="LASK directives for this operation"
    )


class ModifyManifest(BaseModel):
    """Manifest for MODIFY operations on a single file."""
    manifest_version: str = Field(default="1.0", description="Schema version")
    target_file: str = Field(description="File path being modified")
    existing_content_hash: str | None = Field(
        default=None,
        description="Hash of original content for validation"
    )
    operations: list[ModifyOperation] = Field(
        default_factory=list,
        description="Ordered list of modifications"
    )


class OrderedFilePrompts(BaseModel):
    """Prompts for a single file in tree-traversal order."""
    file_path: str = Field(description="Target file path")
    operation: FileOperation = Field(description="CREATE or MODIFY")
    prompts: list[LaskPrompt] = Field(
        description="LASK prompts in depth-first tree order"
    )
    modify_manifest: ModifyManifest | None = Field(
        default=None,
        description="MODIFY manifest with location metadata (only for MODIFY operations)"
    )


class GroupedOutput(BaseModel):
    """Final grouped output from the implement agent."""
    plan_summary: str = Field(description="Original plan summary")
    files: list[OrderedFilePrompts] = Field(
        description="Outputs grouped by file, each with ordered prompts"
    )
    total_prompts: int = Field(description="Total number of prompts generated")
    has_modify_operations: bool = Field(
        default=False,
        description="True if any files use MODIFY mode"
    )
    validation_issues: list[ContractValidationIssue] = Field(
        default_factory=list,
        description="All contract validation issues detected during decomposition"
    )


# Reducer functions for parallel state aggregation
def merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dictionaries, with right taking precedence."""
    if not left:
        return right
    if not right:
        return left
    return {**left, **right}


def merge_lists(left: list, right: list) -> list:
    """Merge two lists, removing duplicates while preserving order."""
    if not left:
        return right
    if not right:
        return left
    seen = set()
    result = []
    for item in left + right:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def max_int(left: int, right: int) -> int:
    """Return the maximum of two integers."""
    return max(left, right)


def append_prompts(left: list, right: list) -> list:
    """Append prompts from right to left (allows duplicates for ordering)."""
    if not left:
        return right or []
    if not right:
        return left
    return left + right


class ImplementState(BaseModel):
    """
    State that flows through the LangGraph implement agent.

    This is the top-level state container that tracks the entire
    decomposition process across all files.
    """
    # Input from Plan agent
    plan_summary: str = Field(description="Summary of what we're implementing")
    target_files: list[FileTarget] = Field(description="Files to create/modify")
    external_contracts: list[Contract] = Field(
        default_factory=list,
        description="Contracts from external files not being processed by lask-lm"
    )

    # Decomposition tree
    nodes: dict[str, CodeNode] = Field(
        default_factory=dict,
        description="All nodes by ID"
    )
    root_node_ids: list[str] = Field(
        default_factory=list,
        description="Top-level file nodes"
    )

    # Processing queues
    pending_node_ids: list[str] = Field(
        default_factory=list,
        description="Nodes waiting to be processed"
    )

    # Contract registry (for cross-node dependencies)
    contract_registry: dict[str, Contract] = Field(
        default_factory=dict,
        description="All contracts by name, for dependency resolution"
    )

    # Final output
    lask_prompts: list[LaskPrompt] = Field(
        default_factory=list,
        description="Collected LASK prompts from all terminal nodes"
    )

    # Metadata
    current_depth: int = Field(default=0, description="Current recursion depth")
    max_depth: int = Field(default=10, description="Safety limit on recursion")


# TypedDict-based state for parallel LangGraph execution with reducers
from typing_extensions import TypedDict


class ParallelImplementState(TypedDict, total=False):
    """
    TypedDict-based state for parallel execution with LangGraph reducers.

    Uses Annotated types with reducer functions to automatically merge
    results from parallel node executions.
    """
    # Input (no reducer needed - set once at start)
    plan_summary: str
    target_files: list[FileTarget]
    external_contracts: list[Contract]
    max_depth: int

    # Decomposition tree - merge dicts from parallel workers
    nodes: Annotated[dict[str, CodeNode], merge_dicts]
    root_node_ids: list[str]  # Set once by router

    # Processing queue - NO reducer (last write wins from aggregator)
    # The aggregator rebuilds this from node statuses after each round
    pending_node_ids: list[str]

    # Contract registry - merge dicts from parallel workers
    contract_registry: Annotated[dict[str, Contract], merge_dicts]

    # Final output - append prompts in order
    lask_prompts: Annotated[list[LaskPrompt], append_prompts]

    # Validation issues - append from parallel workers
    validation_issues: Annotated[list[ContractValidationIssue], append_prompts]

    # Metadata - take the max depth reached
    current_depth: Annotated[int, max_int]

    # Grouped output (set by collector_node)
    grouped_output: GroupedOutput | None


class SingleNodeState(TypedDict, total=False):
    """
    State passed to a single parallel decomposer instance via Send().

    Contains just the node to process plus read-only context.
    """
    # The specific node to process
    node_id: str
    node: CodeNode

    # Read-only context from parent state
    plan_summary: str
    contract_registry: dict[str, Contract]
    current_depth: int
    max_depth: int
