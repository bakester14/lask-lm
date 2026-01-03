"""Contract registry validation logic for Phase 5.

This module provides validation functions that are called at different points
during the decomposition process:

1. Registration: When a contract is registered, check for duplicates
2. Lookup: When a node looks up required contracts, warn if missing
3. Final: After all decomposition, check all dependencies are satisfied
"""

from lask_lm.models import (
    Contract,
    CodeNode,
    NodeStatus,
    ValidationSeverity,
    ContractValidationIssue,
)


def validate_contract_registration(
    new_contract: Contract,
    existing_registry: dict[str, Contract],
) -> ContractValidationIssue | None:
    """
    Validate a single contract registration.

    Checks for two types of conflicts:
    1. Same name with different signature (indicates interface mismatch)
    2. Same name from different provider nodes (indicates duplicate implementation)

    Args:
        new_contract: The contract being registered
        existing_registry: Current contract registry

    Returns:
        A validation issue if there's a conflict, None otherwise
    """
    if new_contract.name in existing_registry:
        existing = existing_registry[new_contract.name]

        # Check for signature mismatch (most serious)
        if existing.signature != new_contract.signature:
            return ContractValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="DUPLICATE_CONTRACT_NAME",
                message=f"Contract '{new_contract.name}' registered with conflicting signatures: "
                        f"'{existing.signature}' vs '{new_contract.signature}'",
                contract_name=new_contract.name,
                node_id=new_contract.provider_node_id,
            )

        # Check for duplicate provider (same contract from different nodes)
        if (existing.provider_node_id is not None
            and new_contract.provider_node_id is not None
            and existing.provider_node_id != new_contract.provider_node_id):
            return ContractValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="DUPLICATE_PROVIDER",
                message=f"Contract '{new_contract.name}' already provided by node '{existing.provider_node_id}', "
                        f"cannot be provided again by node '{new_contract.provider_node_id}'",
                contract_name=new_contract.name,
                node_id=new_contract.provider_node_id,
            )

    return None


def validate_contract_lookup(
    required_names: list[str],
    registry: dict[str, Contract],
    node_id: str,
) -> list[ContractValidationIssue]:
    """
    Validate that required contracts exist in registry during lookup.

    This is called when a node is about to decompose and needs to look up
    its required contracts. Missing contracts are warnings because in
    parallel execution, the providing node may not have registered yet.

    Args:
        required_names: Contract names the node requires
        registry: Current contract registry
        node_id: ID of the node doing the lookup

    Returns:
        List of validation issues for missing contracts
    """
    issues = []
    for name in required_names:
        if name not in registry:
            issues.append(ContractValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="MISSING_CONTRACT_AT_LOOKUP",
                message=f"Required contract '{name}' not found in registry during lookup",
                node_id=node_id,
                contract_name=name,
            ))
    return issues


def detect_circular_dependencies(
    nodes: dict[str, CodeNode],
    contract_registry: dict[str, Contract],
) -> list[ContractValidationIssue]:
    """
    Detect circular dependencies in contract requirements using DFS.

    Builds a dependency graph where edges go from nodes that require contracts
    to nodes that provide them, then uses DFS to find cycles.

    Args:
        nodes: All nodes in the decomposition tree
        contract_registry: All registered contracts

    Returns:
        List of validation issues for any detected cycles
    """
    # Build contract -> providing node mapping
    contract_providers: dict[str, str] = {}
    for node_id, node in nodes.items():
        for contract in node.contracts_provided:
            contract_providers[contract.name] = node_id

    # Build node dependency graph: node_id -> set of node_ids it depends on
    dependencies: dict[str, set[str]] = {node_id: set() for node_id in nodes}

    for node_id, node in nodes.items():
        for req_name in node.contracts_required:
            provider_id = contract_providers.get(req_name)
            if provider_id and provider_id != node_id:
                dependencies[node_id].add(provider_id)

    # DFS for cycle detection
    issues = []
    visited: set[str] = set()
    rec_stack: set[str] = set()

    def dfs(node_id: str, path: list[str]) -> None:
        visited.add(node_id)
        rec_stack.add(node_id)
        path.append(node_id)

        for dep_id in dependencies.get(node_id, set()):
            if dep_id not in visited:
                dfs(dep_id, path)
            elif dep_id in rec_stack:
                # Found cycle - extract the cycle from path
                cycle_start = path.index(dep_id)
                cycle = path[cycle_start:] + [dep_id]
                issues.append(ContractValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="CIRCULAR_DEPENDENCY",
                    message=f"Circular contract dependency detected: {' -> '.join(cycle)}",
                ))

        path.pop()
        rec_stack.remove(node_id)

    for node_id in nodes:
        if node_id not in visited:
            dfs(node_id, [])

    return issues


def validate_all_dependencies_satisfied(
    nodes: dict[str, CodeNode],
    contract_registry: dict[str, Contract],
) -> list[ContractValidationIssue]:
    """
    Final validation: ensure all required contracts are satisfied.

    This runs after all decomposition is complete, so the contract registry
    should be fully populated. Any missing contracts at this point are errors.

    Args:
        nodes: All nodes in the decomposition tree
        contract_registry: All registered contracts

    Returns:
        List of validation issues for unsatisfied dependencies
    """
    issues = []

    for node_id, node in nodes.items():
        # Only check completed nodes (not skipped or still pending)
        if node.status != NodeStatus.COMPLETE:
            continue

        for req_name in node.contracts_required:
            if req_name not in contract_registry:
                issues.append(ContractValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="UNSATISFIED_DEPENDENCY",
                    message=f"Node '{node_id}' requires contract '{req_name}' which was never provided",
                    node_id=node_id,
                    contract_name=req_name,
                ))

    return issues


def validate_contract_fulfillment(
    prompt_intent: str,
    contracts_provided: list[Contract],
    node_id: str,
) -> list[ContractValidationIssue]:
    """
    Validate that a terminal prompt references the contracts it must implement.

    This checks that the prompt's intent text mentions the contract names
    that the node is obligated to provide. Uses simple name matching.

    Args:
        prompt_intent: The intent text from the LaskPrompt
        contracts_provided: Contracts this node is obligated to implement
        node_id: ID of the node for error reporting

    Returns:
        List of validation issues for unreferenced contracts
    """
    issues = []
    intent_lower = prompt_intent.lower()

    for contract in contracts_provided:
        # Check for full contract name
        if contract.name.lower() in intent_lower:
            continue

        # Check for method part (after last dot)
        method_part = contract.name.split(".")[-1]
        if method_part.lower() in intent_lower:
            continue

        # Contract not referenced in intent
        issues.append(ContractValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="CONTRACT_NOT_REFERENCED",
            message=f"Terminal prompt for node '{node_id}' does not reference "
                    f"contract '{contract.name}' which it is obligated to implement",
            node_id=node_id,
            contract_name=contract.name,
        ))

    return issues
