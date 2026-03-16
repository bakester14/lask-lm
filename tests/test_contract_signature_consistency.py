"""Tests for contract signature-intent consistency validation (FP-3)."""

import pytest
from lask_lm.models import (
    Contract,
    ValidationSeverity,
)
from lask_lm.agents.implement.validation import validate_signature_consistency


class TestValidateSignatureConsistency:
    """Tests for signature-intent mismatch detection."""

    def test_matching_intent_passes(self):
        """No issues when intent mentions all signature fields."""
        contracts = [
            Contract(
                name="Location",
                signature="id: int, name: str, description: str, connections: dict[str, str], items: list[Item]",
                description="A game location",
            ),
        ]
        issues = validate_signature_consistency(
            prompt_intent="Define Location with id, name, description, connections and items fields",
            contracts_provided=contracts,
            node_id="node1",
        )
        assert len(issues) == 0

    def test_hallucinated_fields_caught(self):
        """WARNING when intent describes fields not in the signature."""
        contracts = [
            Contract(
                name="Location",
                signature="id: int, name: str, description: str, connections: dict[str, str], items: list[Item]",
                description="A game location",
            ),
        ]
        issues = validate_signature_consistency(
            prompt_intent="Define Location with latitude and longitude coordinates",
            contracts_provided=contracts,
            node_id="node1",
        )
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.WARNING
        assert issues[0].code == "SIGNATURE_INTENT_MISMATCH"
        assert issues[0].contract_name == "Location"
        assert issues[0].node_id == "node1"

    def test_partial_match_below_threshold(self):
        """WARNING when fewer than 50% of identifiers are found."""
        contracts = [
            Contract(
                name="Player",
                signature="health: int, inventory: list[Item], position: Location, score: int, achievements: list[str]",
                description="Player state",
            ),
        ]
        # Only mentions "health" out of {health, inventory, position, score, achievements, Item, Location, Player}
        issues = validate_signature_consistency(
            prompt_intent="Define the Player class with health tracking",
            contracts_provided=contracts,
            node_id="node1",
        )
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.WARNING
        assert issues[0].code == "SIGNATURE_INTENT_MISMATCH"

    def test_partial_match_above_threshold(self):
        """No issues when 50% or more identifiers are found."""
        contracts = [
            Contract(
                name="Player",
                signature="health: int, inventory: list[Item], position: Location, score: int, achievements: list[str]",
                description="Player state",
            ),
        ]
        # Mentions health, inventory, position, score — 4 out of domain ids
        issues = validate_signature_consistency(
            prompt_intent="Define Player with health, inventory, position, and score fields",
            contracts_provided=contracts,
            node_id="node1",
        )
        assert len(issues) == 0

    def test_common_types_filtered(self):
        """No validation when signature contains only common type tokens."""
        contracts = [
            Contract(
                name="SimpleMethod",
                signature="def get(self) -> str",
                description="A simple getter",
            ),
        ]
        # All tokens (def, get, self, str) are noise — fewer than 2 domain ids remain
        issues = validate_signature_consistency(
            prompt_intent="Something completely unrelated to the signature",
            contracts_provided=contracts,
            node_id="node1",
        )
        assert len(issues) == 0

    def test_empty_contracts(self):
        """No issues when there are no contracts."""
        issues = validate_signature_consistency(
            prompt_intent="Some intent text",
            contracts_provided=[],
            node_id="node1",
        )
        assert len(issues) == 0

    def test_multiple_contracts_one_drifted(self):
        """Only the drifted contract gets a WARNING."""
        contracts = [
            Contract(
                name="GoodContract",
                signature="username: str, email: str, password_hash: str",
                description="User credentials",
            ),
            Contract(
                name="DriftedContract",
                signature="connection_pool: Pool, timeout: int, retry_count: int",
                description="DB config",
            ),
        ]
        issues = validate_signature_consistency(
            prompt_intent="Implement GoodContract with username, email, and password_hash fields. Also add DriftedContract.",
            contracts_provided=contracts,
            node_id="node1",
        )
        assert len(issues) == 1
        assert issues[0].contract_name == "DriftedContract"
        assert issues[0].code == "SIGNATURE_INTENT_MISMATCH"
