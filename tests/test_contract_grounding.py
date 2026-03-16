"""Tests for contract signature grounding in terminal emission.

Verifies that _emit_terminal_parallel() includes full contract signatures
(not just names) in the LLM context, so the model can write accurate
intent prose referencing correct field names and types.

These tests are expected to FAIL against the current code, confirming
the bug identified in the friction report (FP-1).
"""

import pytest
from unittest.mock import Mock, patch

from lask_lm.models import (
    SingleNodeState,
    CodeNode,
    NodeType,
    NodeStatus,
    FileOperation,
    Contract,
)
from lask_lm.agents.implement.parallel_graph import _emit_terminal_parallel
from lask_lm.agents.implement.schemas import LaskPromptOutput


def _make_terminal_node(
    *,
    contracts_required: list[str] | None = None,
    context_files: list[str] | None = None,
    operation: FileOperation = FileOperation.CREATE,
) -> CodeNode:
    """Create a minimal BLOCK node ready for terminal emission."""
    return CodeNode(
        node_id="test_node",
        node_type=NodeType.BLOCK,
        intent="Test intent",
        status=NodeStatus.PENDING,
        contracts_required=contracts_required or [],
        context_files=context_files or ["TestFile.cs"],
        operation=operation,
    )


def _mock_terminal_emission(node, contract_registry):
    """Call _emit_terminal_parallel with mocked LLM, returning captured messages."""
    mock_response = LaskPromptOutput(
        intent="Generated intent",
        context_files=node.context_files,
        additional_directives=[],
        notes="",
    )

    captured_messages = []

    def capture_invoke(messages):
        captured_messages.extend(messages)
        return mock_response

    with patch(
        "lask_lm.agents.implement.parallel_graph._get_llm"
    ), patch(
        "lask_lm.agents.implement.parallel_graph._structured_output"
    ) as mock_structured:
        mock_chain = Mock()
        mock_chain.invoke.side_effect = capture_invoke
        mock_structured.return_value = mock_chain

        result = _emit_terminal_parallel(node, contract_registry)

    return result, captured_messages


class TestTerminalEmissionContractContext:
    """Test that _emit_terminal_parallel includes full contract details in LLM context."""

    def test_contract_signatures_included_in_llm_context(self):
        """Contract signatures must appear in the human message sent to the LLM.

        Currently only the contract name is passed (e.g. 'IUserService'),
        but the signature ('interface IUserService { User GetById(int id); }')
        is needed for the LLM to write accurate intent prose.
        """
        node = _make_terminal_node(contracts_required=["IUserService"])
        registry = {
            "IUserService": Contract(
                name="IUserService",
                signature="interface IUserService { User GetById(int id); }",
                description="Service for user operations",
            ),
        }

        result, captured = _mock_terminal_emission(node, registry)

        assert len(captured) == 2  # System + Human
        human_content = captured[1].content
        assert "interface IUserService { User GetById(int id); }" in human_content

    def test_contract_descriptions_included_in_llm_context(self):
        """Contract descriptions must appear in the human message sent to the LLM.

        The description provides semantic context that helps the LLM understand
        the purpose of the contract when writing intent prose.
        """
        node = _make_terminal_node(contracts_required=["IUserService"])
        registry = {
            "IUserService": Contract(
                name="IUserService",
                signature="interface IUserService { User GetById(int id); }",
                description="Service for user operations",
            ),
        }

        result, captured = _mock_terminal_emission(node, registry)

        human_content = captured[1].content
        assert "Service for user operations" in human_content

    def test_multiple_contracts_all_signatures_in_context(self):
        """When multiple contracts are required, all signatures must be present.

        Each contract's signature contains distinct field names that the LLM
        needs to reference accurately in the generated intent.
        """
        node = _make_terminal_node(contracts_required=["Location", "Player"])
        registry = {
            "Location": Contract(
                name="Location",
                signature="class Location { string name; dict[str, str] connections; list[Item] items; }",
                description="A location in the game world",
            ),
            "Player": Contract(
                name="Player",
                signature="class Player { string name; int health; Inventory inventory; }",
                description="The player character",
            ),
        }

        result, captured = _mock_terminal_emission(node, registry)

        human_content = captured[1].content
        assert "dict[str, str] connections" in human_content
        assert "list[Item] items" in human_content
        assert "int health" in human_content
        assert "Inventory inventory" in human_content

    def test_rpg_location_scenario_fields_in_context(self):
        """Reproduce FP-1: RPG Location contract fields must be in LLM context.

        Without seeing the actual signature, the LLM hallucinated
        'latitude, longitude' instead of the real fields:
        'id, name, description, connections, items'.
        """
        node = _make_terminal_node(contracts_required=["Location"])
        registry = {
            "Location": Contract(
                name="Location",
                signature="@dataclass Location { id: str; name: str; description: str; connections: dict[str, str]; items: list[Item] }",
                description="Represents a location in the RPG game world with connections to other locations and collectible items",
            ),
        }

        result, captured = _mock_terminal_emission(node, registry)

        human_content = captured[1].content
        # These are the real fields that must be visible to the LLM
        assert "connections: dict[str, str]" in human_content
        assert "items: list[Item]" in human_content
        assert "id: str" in human_content
        assert "name: str" in human_content
        assert "description: str" in human_content

    def test_no_contracts_required_still_works(self):
        """Terminal emission works correctly when no contracts are required.

        Sanity test: nodes without contracts should still emit prompts
        without any contract-related content in the message.
        """
        node = _make_terminal_node(contracts_required=[])
        registry = {}

        result, captured = _mock_terminal_emission(node, registry)

        # Should still produce a valid result
        assert "lask_prompts" in result
        assert len(result["lask_prompts"]) == 1
        assert result["nodes"]["test_node"].status == NodeStatus.COMPLETE

        # Human message should not mention contracts
        human_content = captured[1].content
        assert "contract" not in human_content.lower() or "Required contracts" not in human_content
