"""Implement agent: recursive decomposition to LASK prompts."""

# Primary implementation uses parallel execution via Send() API
from .parallel_graph import (
    # Main API (backwards compatible)
    create_implement_graph,
    compile_implement_graph,
    # Explicit parallel API
    create_parallel_implement_graph,
    compile_parallel_implement_graph,
    # Node functions (for direct testing)
    router_node,
    parallel_decomposer_node,
    aggregator_node,
    collector_node,
    dispatch_to_parallel,
)
from .prompts import DECOMPOSITION_PROMPTS
from .validation import (
    validate_contract_registration,
    validate_contract_lookup,
    validate_all_dependencies_satisfied,
    detect_circular_dependencies,
)

__all__ = [
    # Main API
    "create_implement_graph",
    "compile_implement_graph",
    # Explicit parallel API
    "create_parallel_implement_graph",
    "compile_parallel_implement_graph",
    # Node functions
    "router_node",
    "parallel_decomposer_node",
    "aggregator_node",
    "collector_node",
    "dispatch_to_parallel",
    # Prompts
    "DECOMPOSITION_PROMPTS",
    # Validation
    "validate_contract_registration",
    "validate_contract_lookup",
    "validate_all_dependencies_satisfied",
    "detect_circular_dependencies",
]
