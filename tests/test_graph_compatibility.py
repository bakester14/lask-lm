"""Compatibility tests for the implement graph.

These tests define the expected behavior of the implement graph,
ensuring any implementation (sequential or parallel) produces
consistent results.
"""

import pytest
from unittest.mock import Mock, patch

from lask_lm.models import (
    ImplementState,
    FileTarget,
    FileOperation,
    NodeType,
    NodeStatus,
)
from lask_lm.agents.implement import (
    create_implement_graph,
    compile_implement_graph,
    router_node,
    parallel_decomposer_node,
)
from lask_lm.agents.implement.schemas import (
    DecomposeFileOutput,
    DecomposeClassOutput,
    DecomposeMethodOutput,
    LaskPromptOutput,
    ComponentOutput,
    ContractOutput,
)


class TestGraphStructure:
    """Test that the graph has expected structure."""

    def test_graph_has_required_nodes(self):
        """Graph must have router, parallel_decomposer, aggregator, and collector nodes."""
        graph = create_implement_graph()

        assert "router" in graph.nodes
        assert "parallel_decomposer" in graph.nodes
        assert "aggregator" in graph.nodes
        assert "collector" in graph.nodes

    def test_graph_compiles(self):
        """Graph must compile without error."""
        app = compile_implement_graph()
        assert app is not None


class TestRouterBehavior:
    """Test router node behavior."""

    def test_router_creates_file_nodes_for_each_target(self):
        """Router creates one FILE node per target file."""
        state = ImplementState(
            plan_summary="Test plan",
            target_files=[
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
        )

        result = router_node(state)

        # Should create 2 nodes
        assert len(result["nodes"]) == 2
        assert len(result["root_node_ids"]) == 2
        assert len(result["pending_node_ids"]) == 2

        # All nodes should be FILE type and PENDING
        for node in result["nodes"].values():
            assert node.node_type == NodeType.FILE
            assert node.status == NodeStatus.PENDING

    def test_router_inherits_context_files_from_path(self):
        """Router sets context_files to the file path."""
        state = ImplementState(
            plan_summary="Test",
            target_files=[
                FileTarget(
                    path="MyService.cs",
                    operation=FileOperation.CREATE,
                    description="A service",
                ),
            ],
        )

        result = router_node(state)
        node = list(result["nodes"].values())[0]

        assert node.context_files == ["MyService.cs"]


class TestDecompositionBehavior:
    """Test decomposition produces expected outputs."""

    def test_decomposition_marks_parent_as_decomposing(self):
        """When a node is decomposed, its status becomes DECOMPOSING."""
        initial_state = ImplementState(
            plan_summary="Test",
            target_files=[
                FileTarget(
                    path="Test.cs",
                    operation=FileOperation.CREATE,
                    description="Test file",
                ),
            ],
        )

        # Mock the LLM response - file with non-terminal class
        file_response = DecomposeFileOutput(
            components=[
                ComponentOutput(
                    name="TestClass",
                    component_type="class",
                    intent="A test class",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=False,  # Not terminal - will decompose
                )
            ],
            file_header_intent="",
            notes="",
        )

        class_response = DecomposeClassOutput(
            class_declaration_intent="public class Test",
            components=[
                ComponentOutput(
                    name="Method",
                    component_type="method",
                    intent="A method",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=True,
                )
            ],
            notes="",
        )

        terminal_response = LaskPromptOutput(
            intent="Create method",
            context_files=[],
            additional_directives=[],
            notes="",
        )

        def mock_structured_output(llm, schema):
            mock_chain = Mock()
            if schema == DecomposeFileOutput:
                mock_chain.invoke.return_value = file_response
            elif schema == DecomposeClassOutput:
                mock_chain.invoke.return_value = class_response
            else:
                mock_chain.invoke.return_value = terminal_response
            return mock_chain

        with patch(
            "lask_lm.agents.implement.parallel_graph._get_llm"
        ), patch(
            "lask_lm.agents.implement.parallel_graph._structured_output",
            side_effect=mock_structured_output,
        ):
            app = compile_implement_graph()
            result = app.invoke(initial_state)

        # The file node should be DECOMPOSING (has children)
        file_nodes = [n for n in result["nodes"].values() if n.node_type == NodeType.FILE]
        assert len(file_nodes) == 1
        assert file_nodes[0].status == NodeStatus.DECOMPOSING

    def test_terminal_nodes_get_lask_prompts(self):
        """Terminal nodes (BLOCK) get LASK prompts assigned."""
        initial_state = ImplementState(
            plan_summary="Test",
            target_files=[
                FileTarget(
                    path="Test.cs",
                    operation=FileOperation.CREATE,
                    description="Simple file",
                ),
            ],
        )

        # File decomposes to a terminal block directly
        file_response = DecomposeFileOutput(
            components=[
                ComponentOutput(
                    name="SimpleBlock",
                    component_type="block",
                    intent="A simple block",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=True,
                )
            ],
            file_header_intent="",
            notes="",
        )

        terminal_response = LaskPromptOutput(
            intent="Create the method",
            context_files=["Test.cs"],
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
            app = compile_implement_graph()
            result = app.invoke(initial_state)

        # Should have a LASK prompt
        assert len(result["lask_prompts"]) >= 1

        # Terminal node should be COMPLETE
        complete_nodes = [n for n in result["nodes"].values() if n.status == NodeStatus.COMPLETE]
        assert len(complete_nodes) >= 1


class TestEndToEndBehavior:
    """End-to-end tests for graph execution."""

    def test_single_file_produces_prompts(self):
        """A single file decomposition produces LASK prompts."""
        initial_state = ImplementState(
            plan_summary="Create a calculator",
            target_files=[
                FileTarget(
                    path="Calculator.cs",
                    operation=FileOperation.CREATE,
                    description="Calculator with add method",
                ),
            ],
        )

        # Mock responses
        file_response = DecomposeFileOutput(
            components=[
                ComponentOutput(
                    name="Calculator",
                    component_type="class",
                    intent="Calculator class",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=True,  # Make it terminal for simplicity
                )
            ],
            file_header_intent="",
            notes="",
        )

        terminal_response = LaskPromptOutput(
            intent="Create Calculator class with add method",
            context_files=["Calculator.cs"],
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
            app = compile_implement_graph()
            result = app.invoke(initial_state)

        # Should have produced LASK prompts
        assert len(result["lask_prompts"]) > 0

        # Should have nodes
        assert len(result["nodes"]) > 0

        # Should have root node IDs
        assert len(result["root_node_ids"]) == 1

    def test_multiple_files_all_get_processed(self):
        """Multiple target files all get processed."""
        initial_state = ImplementState(
            plan_summary="Create services",
            target_files=[
                FileTarget(
                    path="UserService.cs",
                    operation=FileOperation.CREATE,
                    description="User service",
                ),
                FileTarget(
                    path="OrderService.cs",
                    operation=FileOperation.CREATE,
                    description="Order service",
                ),
            ],
        )

        file_response = DecomposeFileOutput(
            components=[
                ComponentOutput(
                    name="ServiceClass",
                    component_type="class",
                    intent="Service class",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=True,
                )
            ],
            file_header_intent="",
            notes="",
        )

        terminal_response = LaskPromptOutput(
            intent="Create service",
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
            app = compile_implement_graph()
            result = app.invoke(initial_state)

        # Should have 2 root nodes (one per file)
        assert len(result["root_node_ids"]) == 2

        # Should have prompts for both files
        assert len(result["lask_prompts"]) >= 2

    def test_max_depth_forces_termination(self):
        """Hitting max_depth forces nodes to terminate."""
        initial_state = ImplementState(
            plan_summary="Deep decomposition",
            target_files=[
                FileTarget(
                    path="Deep.cs",
                    operation=FileOperation.CREATE,
                    description="Deep file",
                ),
            ],
            max_depth=1,  # Very shallow - should force terminal quickly
        )

        # Response that would normally recurse
        file_response = DecomposeFileOutput(
            components=[
                ComponentOutput(
                    name="DeepClass",
                    component_type="class",
                    intent="Class that needs decomposition",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=False,  # Would normally recurse
                )
            ],
            file_header_intent="",
            notes="",
        )

        terminal_response = LaskPromptOutput(
            intent="Forced terminal",
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
            app = compile_implement_graph()
            result = app.invoke(initial_state)

        # Should still complete without infinite recursion
        assert len(result["lask_prompts"]) > 0

        # No pending nodes should remain
        assert len(result["pending_node_ids"]) == 0


class TestOutputFormat:
    """Test the output format matches expected structure."""

    def test_lask_prompt_has_required_fields(self):
        """LASK prompts have file_path, intent, and directives."""
        from lask_lm.models import LaskPrompt, LaskDirective

        prompt = LaskPrompt(
            file_path="Test.cs",
            intent="Create a method",
            directives=[
                LaskDirective(directive_type="context", value="Other.cs")
            ],
        )

        # to_comment() should produce valid output
        comment = prompt.to_comment()
        assert "Create a method" in comment
        assert "@context(Other.cs)" in comment

    def test_output_state_has_required_keys(self):
        """Final state has all required keys."""
        initial_state = ImplementState(
            plan_summary="Test",
            target_files=[
                FileTarget(
                    path="Test.cs",
                    operation=FileOperation.CREATE,
                    description="Test",
                ),
            ],
        )

        file_response = DecomposeFileOutput(
            components=[
                ComponentOutput(
                    name="Test",
                    component_type="class",
                    intent="Test",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=True,
                )
            ],
            file_header_intent="",
            notes="",
        )

        terminal_response = LaskPromptOutput(
            intent="Test",
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
            app = compile_implement_graph()
            result = app.invoke(initial_state)

        # Check required keys exist
        assert "nodes" in result
        assert "root_node_ids" in result
        assert "pending_node_ids" in result
        assert "lask_prompts" in result
        assert "contract_registry" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
