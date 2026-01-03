"""Tests for Phase 5: Contract Registry Validation."""

import pytest
from lask_lm.models import (
    Contract,
    CodeNode,
    NodeType,
    NodeStatus,
    LaskPrompt,
    LaskDirective,
    ValidationSeverity,
    ContractValidationIssue,
)
from lask_lm.agents.implement.validation import (
    validate_contract_registration,
    validate_contract_lookup,
    detect_circular_dependencies,
    validate_all_dependencies_satisfied,
)


class TestValidateContractRegistration:
    """Tests for duplicate contract detection at registration."""

    def test_no_issue_for_new_contract(self):
        """No issue when registering a new contract name."""
        contract = Contract(
            name="IService.Method",
            signature="void Method()",
            description="Test method",
        )
        issue = validate_contract_registration(contract, {})
        assert issue is None

    def test_no_issue_for_same_signature(self):
        """No issue when duplicate has the same signature without provider_node_id."""
        existing = Contract(
            name="IService.Method",
            signature="void Method()",
            description="First registration",
        )
        new = Contract(
            name="IService.Method",
            signature="void Method()",
            description="Second registration",
        )
        issue = validate_contract_registration(new, {"IService.Method": existing})
        assert issue is None

    def test_warning_for_conflicting_signature(self):
        """Warning when duplicate has a different signature."""
        existing = Contract(
            name="IService.Method",
            signature="void Method()",
            description="First registration",
        )
        new = Contract(
            name="IService.Method",
            signature="int Method(string)",
            description="Second registration",
        )
        issue = validate_contract_registration(new, {"IService.Method": existing})
        assert issue is not None
        assert issue.severity == ValidationSeverity.WARNING
        assert issue.code == "DUPLICATE_CONTRACT_NAME"
        assert issue.contract_name == "IService.Method"

    def test_error_for_duplicate_provider(self):
        """Error when different nodes provide the same contract."""
        existing = Contract(
            name="IService.Method",
            signature="void Method()",
            description="First registration",
            provider_node_id="node_a",
        )
        new = Contract(
            name="IService.Method",
            signature="void Method()",
            description="Second registration",
            provider_node_id="node_b",
        )
        issue = validate_contract_registration(new, {"IService.Method": existing})
        assert issue is not None
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.code == "DUPLICATE_PROVIDER"
        assert issue.contract_name == "IService.Method"
        assert issue.node_id == "node_b"
        assert "node_a" in issue.message
        assert "node_b" in issue.message

    def test_no_issue_for_same_provider(self):
        """No issue when the same node re-registers a contract."""
        existing = Contract(
            name="IService.Method",
            signature="void Method()",
            description="First registration",
            provider_node_id="node_a",
        )
        new = Contract(
            name="IService.Method",
            signature="void Method()",
            description="Updated registration",
            provider_node_id="node_a",
        )
        issue = validate_contract_registration(new, {"IService.Method": existing})
        assert issue is None

    def test_no_duplicate_check_when_existing_has_no_provider(self):
        """No duplicate provider error when existing contract has no provider_node_id."""
        existing = Contract(
            name="IService.Method",
            signature="void Method()",
            description="External contract",
            # No provider_node_id (e.g., external contract)
        )
        new = Contract(
            name="IService.Method",
            signature="void Method()",
            description="Internal implementation",
            provider_node_id="node_a",
        )
        issue = validate_contract_registration(new, {"IService.Method": existing})
        # Should be allowed - external contracts can be overridden
        assert issue is None

    def test_no_duplicate_check_when_new_has_no_provider(self):
        """No duplicate provider error when new contract has no provider_node_id."""
        existing = Contract(
            name="IService.Method",
            signature="void Method()",
            description="First implementation",
            provider_node_id="node_a",
        )
        new = Contract(
            name="IService.Method",
            signature="void Method()",
            description="Re-registration without provider",
            # No provider_node_id
        )
        issue = validate_contract_registration(new, {"IService.Method": existing})
        # Should be allowed when new contract doesn't claim a provider
        assert issue is None

    def test_signature_mismatch_takes_precedence(self):
        """Signature mismatch is reported even if providers differ."""
        existing = Contract(
            name="IService.Method",
            signature="void Method()",
            description="First registration",
            provider_node_id="node_a",
        )
        new = Contract(
            name="IService.Method",
            signature="int Method(string)",  # Different signature
            description="Second registration",
            provider_node_id="node_b",
        )
        issue = validate_contract_registration(new, {"IService.Method": existing})
        assert issue is not None
        # Signature mismatch is checked first
        assert issue.code == "DUPLICATE_CONTRACT_NAME"


class TestValidateContractLookup:
    """Tests for missing contract detection during lookup."""

    def test_no_issues_when_all_found(self):
        """No issues when all required contracts exist."""
        registry = {
            "Contract.A": Contract(name="Contract.A", signature="A", description=""),
            "Contract.B": Contract(name="Contract.B", signature="B", description=""),
        }
        issues = validate_contract_lookup(["Contract.A", "Contract.B"], registry, "node1")
        assert issues == []

    def test_warning_for_missing_contract(self):
        """Warning when a required contract is not found."""
        registry = {
            "Contract.A": Contract(name="Contract.A", signature="A", description=""),
        }
        issues = validate_contract_lookup(["Contract.A", "Contract.B"], registry, "node1")
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.WARNING
        assert issues[0].code == "MISSING_CONTRACT_AT_LOOKUP"
        assert issues[0].contract_name == "Contract.B"
        assert issues[0].node_id == "node1"

    def test_empty_required_list(self):
        """No issues for empty required list."""
        issues = validate_contract_lookup([], {}, "node1")
        assert issues == []


class TestDetectCircularDependencies:
    """Tests for circular dependency detection."""

    def test_no_cycle_in_linear_chain(self):
        """No cycle in A -> B -> C dependency chain."""
        nodes = {
            "a": CodeNode(
                node_id="a",
                node_type=NodeType.CLASS,
                intent="A",
                contracts_provided=[Contract(name="A", signature="", description="")],
                contracts_required=[],
                status=NodeStatus.COMPLETE,
            ),
            "b": CodeNode(
                node_id="b",
                node_type=NodeType.CLASS,
                intent="B",
                contracts_provided=[Contract(name="B", signature="", description="")],
                contracts_required=["A"],
                status=NodeStatus.COMPLETE,
            ),
            "c": CodeNode(
                node_id="c",
                node_type=NodeType.CLASS,
                intent="C",
                contracts_required=["B"],
                status=NodeStatus.COMPLETE,
            ),
        }
        registry = {
            "A": nodes["a"].contracts_provided[0],
            "B": nodes["b"].contracts_provided[0],
        }
        issues = detect_circular_dependencies(nodes, registry)
        assert issues == []

    def test_detects_simple_cycle(self):
        """Detects A -> B -> A cycle."""
        nodes = {
            "a": CodeNode(
                node_id="a",
                node_type=NodeType.CLASS,
                intent="A",
                contracts_provided=[Contract(name="A", signature="", description="")],
                contracts_required=["B"],
                status=NodeStatus.COMPLETE,
            ),
            "b": CodeNode(
                node_id="b",
                node_type=NodeType.CLASS,
                intent="B",
                contracts_provided=[Contract(name="B", signature="", description="")],
                contracts_required=["A"],
                status=NodeStatus.COMPLETE,
            ),
        }
        registry = {
            "A": nodes["a"].contracts_provided[0],
            "B": nodes["b"].contracts_provided[0],
        }
        issues = detect_circular_dependencies(nodes, registry)
        assert len(issues) >= 1
        assert any(i.severity == ValidationSeverity.ERROR for i in issues)
        assert any(i.code == "CIRCULAR_DEPENDENCY" for i in issues)

    def test_no_issues_for_empty_nodes(self):
        """No issues for empty node dict."""
        issues = detect_circular_dependencies({}, {})
        assert issues == []


class TestValidateAllDependenciesSatisfied:
    """Tests for final dependency satisfaction check."""

    def test_all_satisfied(self):
        """No errors when all dependencies are satisfied."""
        contract = Contract(
            name="Provider.Method",
            signature="void()",
            description="",
        )
        nodes = {
            "provider": CodeNode(
                node_id="provider",
                node_type=NodeType.CLASS,
                intent="Provider",
                contracts_provided=[contract],
                status=NodeStatus.COMPLETE,
            ),
            "consumer": CodeNode(
                node_id="consumer",
                node_type=NodeType.CLASS,
                intent="Consumer",
                contracts_required=["Provider.Method"],
                status=NodeStatus.COMPLETE,
            ),
        }
        registry = {"Provider.Method": contract}
        issues = validate_all_dependencies_satisfied(nodes, registry)
        assert issues == []

    def test_error_for_unsatisfied_dependency(self):
        """Error when a required contract is not in the registry."""
        nodes = {
            "consumer": CodeNode(
                node_id="consumer",
                node_type=NodeType.CLASS,
                intent="Consumer",
                contracts_required=["Missing.Contract"],
                status=NodeStatus.COMPLETE,
            ),
        }
        issues = validate_all_dependencies_satisfied(nodes, {})
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.ERROR
        assert issues[0].code == "UNSATISFIED_DEPENDENCY"
        assert issues[0].contract_name == "Missing.Contract"
        assert issues[0].node_id == "consumer"

    def test_skips_non_complete_nodes(self):
        """Skips nodes that aren't COMPLETE."""
        nodes = {
            "pending": CodeNode(
                node_id="pending",
                node_type=NodeType.CLASS,
                intent="Pending",
                contracts_required=["Missing.Contract"],
                status=NodeStatus.PENDING,
            ),
        }
        issues = validate_all_dependencies_satisfied(nodes, {})
        assert issues == []


class TestLaskPromptContractIntegration:
    """Tests for contract inclusion in LASK prompts."""

    def test_to_comment_includes_contract_requirements(self):
        """to_comment() includes contract info in intent."""
        prompt = LaskPrompt(
            file_path="test.cs",
            intent="Create user repository",
            resolved_contracts=[
                Contract(
                    name="IUserService.GetById",
                    signature="User GetById(int id)",
                    description="Gets user by ID",
                ),
            ],
        )
        comment = prompt.to_comment()
        assert "[requires: IUserService.GetById: User GetById(int id)]" in comment
        assert "Create user repository" in comment

    def test_to_comment_adds_contract_context_files(self):
        """to_comment() adds contract context files as @context directives."""
        prompt = LaskPrompt(
            file_path="test.cs",
            intent="Create repository",
            directives=[],
            resolved_contracts=[
                Contract(
                    name="IService.Method",
                    signature="void()",
                    description="",
                    context_files=["Service.cs", "Helper.cs"],
                ),
            ],
        )
        comment = prompt.to_comment()
        assert "@context(Service.cs)" in comment
        assert "@context(Helper.cs)" in comment

    def test_to_comment_deduplicates_context_files(self):
        """to_comment() doesn't add duplicate @context directives."""
        prompt = LaskPrompt(
            file_path="test.cs",
            intent="Create repository",
            directives=[
                LaskDirective(directive_type="context", value="Service.cs"),
            ],
            resolved_contracts=[
                Contract(
                    name="IService.Method",
                    signature="void()",
                    description="",
                    context_files=["Service.cs", "Helper.cs"],
                ),
            ],
        )
        comment = prompt.to_comment()
        # Service.cs should appear only once
        assert comment.count("@context(Service.cs)") == 1
        assert "@context(Helper.cs)" in comment

    def test_to_comment_without_contracts(self):
        """to_comment() works normally without contracts."""
        prompt = LaskPrompt(
            file_path="test.cs",
            intent="Simple intent",
            directives=[
                LaskDirective(directive_type="context", value="File.cs"),
            ],
        )
        comment = prompt.to_comment()
        assert comment == "// @ @context(File.cs) Simple intent"

    def test_multiple_contracts(self):
        """to_comment() handles multiple contracts."""
        prompt = LaskPrompt(
            file_path="test.cs",
            intent="Complex task",
            resolved_contracts=[
                Contract(name="A.Method", signature="void A()", description=""),
                Contract(name="B.Method", signature="int B()", description=""),
            ],
        )
        comment = prompt.to_comment()
        assert "[requires: A.Method: void A(), B.Method: int B()]" in comment
