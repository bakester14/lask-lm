"""Unit tests for the parallel implementation graph.

Tests the Phase 4 parallel spawning feature using LangGraph's Send() API.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from lask_lm.models import (
    ParallelImplementState,
    SingleNodeState,
    CodeNode,
    NodeType,
    NodeStatus,
    FileTarget,
    FileOperation,
    Contract,
    LaskPrompt,
    LaskDirective,
    # Reducers
    merge_dicts,
    merge_lists,
    max_int,
    append_prompts,
)
from lask_lm.agents.implement.parallel_graph import (
    router_node,
    dispatch_to_parallel,
    aggregator_node,
    parallel_decomposer_node,
    collector_node,
    create_parallel_implement_graph,
)
from lask_lm.agents.implement.schemas import (
    DecomposeFileOutput,
    DecomposeClassOutput,
    DecomposeMethodOutput,
    LaskPromptOutput,
    ComponentOutput,
    ContractOutput,
)


class TestReducers:
    """Test the reducer functions used for parallel state aggregation."""

    def test_merge_dicts_empty_left(self):
        """merge_dicts returns right when left is empty."""
        result = merge_dicts({}, {"a": 1})
        assert result == {"a": 1}

    def test_merge_dicts_empty_right(self):
        """merge_dicts returns left when right is empty."""
        result = merge_dicts({"a": 1}, {})
        assert result == {"a": 1}

    def test_merge_dicts_both_populated(self):
        """merge_dicts merges with right taking precedence."""
        result = merge_dicts({"a": 1, "b": 2}, {"b": 3, "c": 4})
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_lists_empty_left(self):
        """merge_lists returns right when left is empty."""
        result = merge_lists([], ["a", "b"])
        assert result == ["a", "b"]

    def test_merge_lists_empty_right(self):
        """merge_lists returns left when right is empty."""
        result = merge_lists(["a", "b"], [])
        assert result == ["a", "b"]

    def test_merge_lists_deduplicates(self):
        """merge_lists removes duplicates while preserving order."""
        result = merge_lists(["a", "b", "c"], ["b", "c", "d"])
        assert result == ["a", "b", "c", "d"]

    def test_max_int(self):
        """max_int returns the maximum of two integers."""
        assert max_int(5, 3) == 5
        assert max_int(3, 5) == 5
        assert max_int(5, 5) == 5

    def test_append_prompts_empty_left(self):
        """append_prompts returns right when left is empty."""
        prompts = [Mock()]
        result = append_prompts([], prompts)
        assert result == prompts

    def test_append_prompts_concatenates(self):
        """append_prompts concatenates lists."""
        p1 = Mock()
        p2 = Mock()
        result = append_prompts([p1], [p2])
        assert result == [p1, p2]


class TestRouterNode:
    """Test the router_node function."""

    def test_creates_file_nodes(self):
        """router_node creates a FILE node for each target file."""
        state: ParallelImplementState = {
            "plan_summary": "Test plan",
            "target_files": [
                FileTarget(
                    path="File1.cs",
                    operation=FileOperation.CREATE,
                    description="First file",
                ),
                FileTarget(
                    path="File2.cs",
                    operation=FileOperation.CREATE,
                    description="Second file",
                ),
            ],
        }

        result = router_node(state)

        assert len(result["nodes"]) == 2
        assert len(result["root_node_ids"]) == 2
        assert len(result["pending_node_ids"]) == 2

        # Verify all nodes are FILE type with PENDING status
        for node_id, node in result["nodes"].items():
            assert node.node_type == NodeType.FILE
            assert node.status == NodeStatus.PENDING
            assert node_id in result["pending_node_ids"]

    def test_empty_targets(self):
        """router_node handles empty target list."""
        state: ParallelImplementState = {
            "plan_summary": "Test plan",
            "target_files": [],
        }

        result = router_node(state)

        assert result["nodes"] == {}
        assert result["root_node_ids"] == []
        assert result["pending_node_ids"] == []


class TestDispatchToParallel:
    """Test the dispatch_to_parallel routing function."""

    def test_dispatches_pending_nodes(self):
        """dispatch_to_parallel creates Send objects for each pending node."""
        node1 = CodeNode(
            node_id="node1",
            node_type=NodeType.FILE,
            intent="First file",
            status=NodeStatus.PENDING,
        )
        node2 = CodeNode(
            node_id="node2",
            node_type=NodeType.FILE,
            intent="Second file",
            status=NodeStatus.PENDING,
        )

        state: ParallelImplementState = {
            "plan_summary": "Test plan",
            "nodes": {"node1": node1, "node2": node2},
            "pending_node_ids": ["node1", "node2"],
            "contract_registry": {},
            "current_depth": 0,
            "max_depth": 10,
        }

        result = dispatch_to_parallel(state)

        assert isinstance(result, list)
        assert len(result) == 2
        # Verify these are Send objects targeting parallel_decomposer
        for send in result:
            assert send.node == "parallel_decomposer"
            assert "node_id" in send.arg
            assert "node" in send.arg

    def test_returns_collector_when_empty(self):
        """dispatch_to_parallel returns 'collector' when no pending nodes."""
        state: ParallelImplementState = {
            "plan_summary": "Test plan",
            "nodes": {},
            "pending_node_ids": [],
        }

        result = dispatch_to_parallel(state)

        assert result == "collector"


class TestAggregatorNode:
    """Test the aggregator_node function."""

    def test_rebuilds_pending_from_node_statuses(self):
        """aggregator_node rebuilds pending from PENDING status nodes."""
        # Create nodes with different statuses
        pending_node = CodeNode(
            node_id="pending1",
            node_type=NodeType.CLASS,
            intent="Pending class",
            status=NodeStatus.PENDING,
        )
        complete_node = CodeNode(
            node_id="complete1",
            node_type=NodeType.BLOCK,
            intent="Complete block",
            status=NodeStatus.COMPLETE,
        )
        decomposing_node = CodeNode(
            node_id="decomposing1",
            node_type=NodeType.FILE,
            intent="Decomposing file",
            status=NodeStatus.DECOMPOSING,
        )

        state: ParallelImplementState = {
            "nodes": {
                "pending1": pending_node,
                "complete1": complete_node,
                "decomposing1": decomposing_node,
            },
            "pending_node_ids": ["old_pending"],  # Old list - should be ignored
        }

        result = aggregator_node(state)

        # Should only include PENDING nodes
        assert result["pending_node_ids"] == ["pending1"]

    def test_empty_nodes(self):
        """aggregator_node returns empty pending when no nodes."""
        state: ParallelImplementState = {
            "nodes": {},
            "pending_node_ids": [],
        }

        result = aggregator_node(state)

        assert result["pending_node_ids"] == []


class TestParallelDecomposerNode:
    """Test the parallel_decomposer_node function."""

    def test_force_terminal_at_max_depth(self):
        """parallel_decomposer_node forces terminal at max depth."""
        node = CodeNode(
            node_id="test_node",
            node_type=NodeType.METHOD,
            intent="Test method",
            status=NodeStatus.PENDING,
            context_files=["Test.cs"],
        )

        state: SingleNodeState = {
            "node_id": "test_node",
            "node": node,
            "plan_summary": "Test",
            "contract_registry": {},
            "current_depth": 10,
            "max_depth": 10,  # At max depth
        }

        # Mock the LLM call for terminal emission
        mock_response = LaskPromptOutput(
            intent="Generated intent",
            context_files=["Test.cs"],
            additional_directives=[],
            notes="",
        )

        with patch(
            "lask_lm.agents.implement.parallel_graph._get_llm"
        ) as mock_get_llm, patch(
            "lask_lm.agents.implement.parallel_graph._structured_output"
        ) as mock_structured:
            mock_chain = Mock()
            mock_chain.invoke.return_value = mock_response
            mock_structured.return_value = mock_chain

            result = parallel_decomposer_node(state)

            # Should have emitted a terminal prompt
            assert "lask_prompts" in result
            assert len(result["lask_prompts"]) == 1
            assert result["nodes"]["test_node"].status == NodeStatus.COMPLETE

    def test_handles_empty_node(self):
        """parallel_decomposer_node handles missing node gracefully."""
        state: SingleNodeState = {
            "node_id": "missing",
            "node": None,  # type: ignore
        }

        result = parallel_decomposer_node(state)

        assert result == {}


class TestCollectorNode:
    """Test the collector_node function."""

    def test_returns_grouped_output(self):
        """collector_node returns grouped_output with empty state."""
        state: ParallelImplementState = {
            "lask_prompts": [Mock(), Mock()],
            "nodes": {},
            "root_node_ids": [],
            "target_files": [],
            "plan_summary": "test plan",
        }

        result = collector_node(state)

        assert "grouped_output" in result
        assert result["grouped_output"].plan_summary == "test plan"
        assert result["grouped_output"].files == []
        assert result["grouped_output"].total_prompts == 0


class TestGraphCreation:
    """Test graph creation and compilation."""

    def test_create_graph(self):
        """create_parallel_implement_graph creates a valid graph."""
        graph = create_parallel_implement_graph()

        # Verify nodes exist
        assert "router" in graph.nodes
        assert "aggregator" in graph.nodes
        assert "parallel_decomposer" in graph.nodes
        assert "collector" in graph.nodes

    def test_compile_graph(self):
        """compile_parallel_implement_graph compiles without error."""
        from lask_lm.agents.implement.parallel_graph import (
            compile_parallel_implement_graph,
        )

        app = compile_parallel_implement_graph()

        # Should have a compiled graph
        assert app is not None


class TestIntegration:
    """Integration tests for the parallel graph."""

    @pytest.mark.integration
    def test_parallel_decomposition_flow(self):
        """Test the full parallel decomposition flow with mocked LLM."""
        from lask_lm.agents.implement.parallel_graph import (
            compile_parallel_implement_graph,
        )

        # Create initial state
        state: ParallelImplementState = {
            "plan_summary": "Create a simple service",
            "target_files": [
                FileTarget(
                    path="Service.cs",
                    operation=FileOperation.CREATE,
                    description="A simple service class",
                    language="csharp",
                )
            ],
            "nodes": {},
            "root_node_ids": [],
            "pending_node_ids": [],
            "contract_registry": {},
            "lask_prompts": [],
            "current_depth": 0,
            "max_depth": 3,  # Low max depth for testing
        }

        # Mock responses for the LLM based on schema type
        file_response = DecomposeFileOutput(
            components=[
                ComponentOutput(
                    name="ServiceClass",
                    component_type="class",
                    intent="Main service class",
                    contracts_provided=[
                        ContractOutput(
                            name="IService.DoWork",
                            signature="void DoWork()",
                            description="Main work method",
                        )
                    ],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=False,
                )
            ],
            file_header_intent="",
            notes="",
        )

        class_response = DecomposeClassOutput(
            class_declaration_intent="public class Service",
            components=[
                ComponentOutput(
                    name="DoWork",
                    component_type="method",
                    intent="Performs the main work",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=True,  # Terminal method - becomes BLOCK
                )
            ],
            notes="",
        )

        terminal_response = LaskPromptOutput(
            intent="Create DoWork method that performs the main work",
            context_files=["Service.cs"],
            additional_directives=[],
            notes="",
        )

        # Map schema types to responses
        schema_responses = {
            DecomposeFileOutput: file_response,
            DecomposeClassOutput: class_response,
            DecomposeMethodOutput: DecomposeMethodOutput(
                is_terminal=True,
                terminal_intent="Simple method",
                blocks=[],
                notes="",
            ),
            LaskPromptOutput: terminal_response,
        }

        def mock_structured_output(llm, schema):
            """Return a mock chain that returns the right response for the schema."""
            mock_chain = Mock()
            mock_chain.invoke.return_value = schema_responses.get(schema, terminal_response)
            return mock_chain

        with patch(
            "lask_lm.agents.implement.parallel_graph._get_llm"
        ) as mock_get_llm, patch(
            "lask_lm.agents.implement.parallel_graph._structured_output",
            side_effect=mock_structured_output,
        ):
            app = compile_parallel_implement_graph()
            result = app.invoke(state)

            # Verify we got LASK prompts
            assert len(result["lask_prompts"]) > 0

            # Verify nodes were created
            assert len(result["nodes"]) > 0

    @pytest.mark.integration
    def test_parallel_multiple_files(self):
        """Test parallel decomposition with multiple files (true parallel fan-out)."""
        from lask_lm.agents.implement.parallel_graph import (
            compile_parallel_implement_graph,
        )

        # Create initial state with 3 files - should process in parallel
        state: ParallelImplementState = {
            "plan_summary": "Create multiple services",
            "target_files": [
                FileTarget(
                    path="UserService.cs",
                    operation=FileOperation.CREATE,
                    description="User service",
                    language="csharp",
                ),
                FileTarget(
                    path="OrderService.cs",
                    operation=FileOperation.CREATE,
                    description="Order service",
                    language="csharp",
                ),
                FileTarget(
                    path="PaymentService.cs",
                    operation=FileOperation.CREATE,
                    description="Payment service",
                    language="csharp",
                ),
            ],
            "nodes": {},
            "root_node_ids": [],
            "pending_node_ids": [],
            "contract_registry": {},
            "lask_prompts": [],
            "current_depth": 0,
            "max_depth": 2,  # Limit depth for faster test
        }

        # Terminal response for all decompositions
        file_response = DecomposeFileOutput(
            components=[
                ComponentOutput(
                    name="MainClass",
                    component_type="class",
                    intent="Main class",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=True,  # Make terminal immediately
                )
            ],
            file_header_intent="",
            notes="",
        )

        terminal_response = LaskPromptOutput(
            intent="Create main class",
            context_files=[],
            additional_directives=[],
            notes="",
        )

        def mock_structured_output(llm, schema):
            mock_chain = Mock()
            if schema == DecomposeFileOutput:
                mock_chain.invoke.return_value = file_response
            else:
                mock_chain.invoke.return_value = terminal_response
            return mock_chain

        with patch(
            "lask_lm.agents.implement.parallel_graph._get_llm"
        ), patch(
            "lask_lm.agents.implement.parallel_graph._structured_output",
            side_effect=mock_structured_output,
        ):
            app = compile_parallel_implement_graph()
            result = app.invoke(state)

            # Should have 3 root nodes (one per file)
            assert len(result["root_node_ids"]) == 3

            # Should have LASK prompts for each file
            assert len(result["lask_prompts"]) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
