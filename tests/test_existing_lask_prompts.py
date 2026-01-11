"""Tests for recognizing existing LASK prompts in file content during MODIFY operations.

When LASK-LM receives a file for modification that contains LASK prompt comments
(e.g., "// @ Add validation logic"), it should recognize these as prompts awaiting
expansion rather than code. This allows LASK-LM to:
- Update existing prompts if the modification intent affects them
- Skip unchanged prompts appropriately
- Not treat prompts as "code to preserve"
"""

import pytest
from unittest.mock import Mock, patch

from lask_lm.models import (
    ParallelImplementState,
    SingleNodeState,
    CodeNode,
    NodeType,
    NodeStatus,
    FileTarget,
    FileOperation,
    LaskPrompt,
)
from lask_lm.agents.implement.parallel_graph import (
    router_node,
    parallel_decomposer_node,
)
from lask_lm.agents.implement.schemas import (
    DecomposeFileOutput,
    DecomposeClassOutput,
    LaskPromptOutput,
    ComponentOutput,
)


class TestExistingLaskPromptsInContent:
    """Test that LASK-LM correctly handles files containing existing LASK prompts."""

    # Sample file content with LASK prompts (C# style)
    CSHARP_FILE_WITH_PROMPTS = """namespace MyApp.Services
{
    public class UserService
    {
        // @ Implement constructor that accepts IUserRepository dependency

        // @ Add method to validate user input with email and password checks

        // @ Create SaveUser method that persists to repository
    }
}"""

    # Sample file content with mixed code and LASK prompts
    CSHARP_MIXED_CONTENT = """namespace MyApp.Services
{
    public class OrderService
    {
        private readonly IOrderRepository _repository;

        public OrderService(IOrderRepository repository)
        {
            _repository = repository;
        }

        // @ Add validation logic for order items

        public void ProcessOrder(Order order)
        {
            // Existing implementation
            _repository.Save(order);
        }

        // @ Implement refund method with transaction support
    }
}"""

    # Python style LASK prompts
    PYTHON_FILE_WITH_PROMPTS = """class DataProcessor:
    # @ Initialize with configuration dictionary

    # @ Add method to validate input data schema

    # @ Implement batch processing with progress callback
"""

    # HTML style LASK prompts
    HTML_FILE_WITH_PROMPTS = """<!DOCTYPE html>
<html>
<head>
    <!-- @ Add meta tags for SEO optimization -->
</head>
<body>
    <header>
        <!-- @ Create responsive navigation menu -->
    </header>
    <main>
        <!-- @ Add hero section with call-to-action button -->
    </main>
</body>
</html>"""

    def test_router_passes_content_with_lask_prompts(self):
        """router_node passes file content containing LASK prompts to CodeNode."""
        state: ParallelImplementState = {
            "plan_summary": "Update UserService validation",
            "target_files": [
                FileTarget(
                    path="UserService.cs",
                    operation=FileOperation.MODIFY,
                    description="Update the validation prompt to include phone number",
                    existing_content=self.CSHARP_FILE_WITH_PROMPTS,
                ),
            ],
        }

        result = router_node(state)

        assert len(result["nodes"]) == 1
        node = list(result["nodes"].values())[0]
        assert node.existing_content == self.CSHARP_FILE_WITH_PROMPTS
        # Verify the LASK prompt comments are in the content
        assert "// @" in node.existing_content
        assert "Implement constructor" in node.existing_content

    def test_router_passes_mixed_content_with_code_and_prompts(self):
        """router_node passes mixed content (code + LASK prompts) correctly."""
        state: ParallelImplementState = {
            "plan_summary": "Add order cancellation feature",
            "target_files": [
                FileTarget(
                    path="OrderService.cs",
                    operation=FileOperation.MODIFY,
                    description="Add order cancellation method",
                    existing_content=self.CSHARP_MIXED_CONTENT,
                ),
            ],
        }

        result = router_node(state)

        node = list(result["nodes"].values())[0]
        # Should have both actual code and LASK prompts
        assert "public void ProcessOrder" in node.existing_content  # Real code
        assert "// @ Add validation logic" in node.existing_content  # LASK prompt
        assert "// @ Implement refund method" in node.existing_content  # LASK prompt

    @pytest.mark.integration
    def test_decomposer_receives_content_with_lask_prompts(self):
        """parallel_decomposer_node receives file content with LASK prompts in context."""
        node = CodeNode(
            node_id="test_node",
            node_type=NodeType.FILE,
            intent="Update validation prompt to include phone number check",
            status=NodeStatus.PENDING,
            context_files=["UserService.cs"],
            existing_content=self.CSHARP_FILE_WITH_PROMPTS,
        )

        state: SingleNodeState = {
            "node_id": "test_node",
            "node": node,
            "plan_summary": "Update validation logic",
            "contract_registry": {},
            "current_depth": 0,
            "max_depth": 10,
        }

        file_response = DecomposeFileOutput(
            components=[
                ComponentOutput(
                    name="ValidationPrompt",
                    component_type="block",
                    intent="Update validation prompt to include phone check",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=True,
                )
            ],
            file_header_intent="",
            notes="",
        )

        captured_messages = []

        def capture_invoke(messages):
            captured_messages.extend(messages)
            return file_response

        with patch(
            "lask_lm.agents.implement.parallel_graph._get_llm"
        ), patch(
            "lask_lm.agents.implement.parallel_graph._structured_output"
        ) as mock_structured:
            mock_chain = Mock()
            mock_chain.invoke.side_effect = capture_invoke
            mock_structured.return_value = mock_chain

            parallel_decomposer_node(state)

            # Verify the LASK prompts are in the context message
            assert len(captured_messages) >= 2
            human_message = captured_messages[1].content
            assert "EXISTING FILE CONTENT" in human_message
            assert "// @ Implement constructor" in human_message
            assert "// @ Add method to validate" in human_message
            assert "// @ Create SaveUser" in human_message

    @pytest.mark.integration
    def test_decomposer_receives_mixed_content(self):
        """parallel_decomposer_node correctly handles mixed code and LASK prompts."""
        node = CodeNode(
            node_id="test_node",
            node_type=NodeType.FILE,
            intent="Add order cancellation method",
            status=NodeStatus.PENDING,
            context_files=["OrderService.cs"],
            existing_content=self.CSHARP_MIXED_CONTENT,
        )

        state: SingleNodeState = {
            "node_id": "test_node",
            "node": node,
            "plan_summary": "Add cancellation feature",
            "contract_registry": {},
            "current_depth": 0,
            "max_depth": 10,
        }

        file_response = DecomposeFileOutput(
            components=[
                # Existing code should potentially be marked unchanged
                ComponentOutput(
                    name="ProcessOrder",
                    component_type="method",
                    intent="Existing ProcessOrder method",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=True,
                    is_unchanged=True,
                ),
                # LASK prompt that's unaffected - could be skipped
                ComponentOutput(
                    name="ValidationPrompt",
                    component_type="block",
                    intent="Existing validation prompt",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=True,
                    is_unchanged=True,
                ),
                # New component being added
                ComponentOutput(
                    name="CancelOrder",
                    component_type="method",
                    intent="Add order cancellation method",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=True,
                ),
            ],
            file_header_intent="",
            notes="",
        )

        captured_messages = []

        def capture_invoke(messages):
            captured_messages.extend(messages)
            return file_response

        with patch(
            "lask_lm.agents.implement.parallel_graph._get_llm"
        ), patch(
            "lask_lm.agents.implement.parallel_graph._structured_output"
        ) as mock_structured:
            mock_chain = Mock()
            mock_chain.invoke.side_effect = capture_invoke
            mock_structured.return_value = mock_chain

            parallel_decomposer_node(state)

            # Verify both code and prompts are in context
            human_message = captured_messages[1].content
            assert "public void ProcessOrder" in human_message  # Real code
            assert "// @ Add validation logic" in human_message  # LASK prompt


class TestLaskPromptRecognitionInDifferentLanguages:
    """Test LASK prompt recognition across different language comment styles."""

    @pytest.mark.parametrize("file_path,content,expected_prompt_marker", [
        (
            "Service.cs",
            "public class Service {\n    // @ Add constructor\n}",
            "// @"
        ),
        (
            "service.py",
            "class Service:\n    # @ Add constructor\n    pass",
            "# @"
        ),
        (
            "index.html",
            "<body>\n    <!-- @ Add header section -->\n</body>",
            "<!-- @"
        ),
        (
            "Service.java",
            "public class Service {\n    // @ Add constructor\n}",
            "// @"
        ),
        (
            "service.rb",
            "class Service\n  # @ Add constructor\nend",
            "# @"
        ),
    ])
    def test_router_passes_language_specific_prompts(
        self, file_path: str, content: str, expected_prompt_marker: str
    ):
        """router_node correctly passes content with language-specific LASK prompts."""
        state: ParallelImplementState = {
            "plan_summary": "Modify service",
            "target_files": [
                FileTarget(
                    path=file_path,
                    operation=FileOperation.MODIFY,
                    description="Update service",
                    existing_content=content,
                ),
            ],
        }

        result = router_node(state)

        node = list(result["nodes"].values())[0]
        assert expected_prompt_marker in node.existing_content


class TestUpdateExistingPromptScenario:
    """Test scenario where modification intent should update an existing LASK prompt."""

    ORIGINAL_CONTENT = """namespace MyApp
{
    public class Validator
    {
        // @ Implement email validation using regex
    }
}"""

    @pytest.mark.integration
    def test_modification_affects_existing_prompt(self):
        """When modification intent affects an existing prompt, decomposer should update it."""
        node = CodeNode(
            node_id="test_node",
            node_type=NodeType.FILE,
            intent="Change email validation to also check domain blacklist",
            status=NodeStatus.PENDING,
            context_files=["Validator.cs"],
            existing_content=self.ORIGINAL_CONTENT,
        )

        state: SingleNodeState = {
            "node_id": "test_node",
            "node": node,
            "plan_summary": "Update email validation to include domain blacklist",
            "contract_registry": {},
            "current_depth": 0,
            "max_depth": 10,
        }

        # Response should be an updated prompt, not preserving the old one unchanged
        file_response = DecomposeFileOutput(
            components=[
                ComponentOutput(
                    name="EmailValidation",
                    component_type="block",
                    intent="Implement email validation with regex AND domain blacklist check",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=True,
                    is_unchanged=False,  # NOT unchanged - this prompt needs updating
                )
            ],
            file_header_intent="",
            notes="",
        )

        captured_messages = []

        def capture_invoke(messages):
            captured_messages.extend(messages)
            return file_response

        with patch(
            "lask_lm.agents.implement.parallel_graph._get_llm"
        ), patch(
            "lask_lm.agents.implement.parallel_graph._structured_output"
        ) as mock_structured:
            mock_chain = Mock()
            mock_chain.invoke.side_effect = capture_invoke
            mock_structured.return_value = mock_chain

            result = parallel_decomposer_node(state)

            # The decomposer should have seen the existing prompt
            human_message = captured_messages[1].content
            assert "// @ Implement email validation" in human_message

            # Result should contain a node (not skipped) since prompt needs updating
            assert "nodes" in result
            # The component is marked is_unchanged=False, so it should create a node


class TestSkipUnaffectedPromptScenario:
    """Test scenario where existing LASK prompts are unaffected by modification."""

    ORIGINAL_CONTENT = """namespace MyApp
{
    public class UserService
    {
        // @ Implement user authentication with JWT

        // @ Add logging for all operations

        public void GetUser(int id) { }
    }
}"""

    @pytest.mark.integration
    def test_unaffected_prompts_can_be_skipped(self):
        """When modification doesn't affect existing prompts, they can be marked unchanged."""
        node = CodeNode(
            node_id="test_node",
            node_type=NodeType.FILE,
            intent="Add caching to GetUser method",
            status=NodeStatus.PENDING,
            context_files=["UserService.cs"],
            existing_content=self.ORIGINAL_CONTENT,
        )

        state: SingleNodeState = {
            "node_id": "test_node",
            "node": node,
            "plan_summary": "Add caching to GetUser",
            "contract_registry": {},
            "current_depth": 0,
            "max_depth": 10,
        }

        # Response should skip unrelated prompts
        file_response = DecomposeFileOutput(
            components=[
                # These existing prompts are unaffected
                ComponentOutput(
                    name="AuthPrompt",
                    component_type="block",
                    intent="JWT authentication prompt",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=True,
                    is_unchanged=True,  # Unaffected - skip
                ),
                ComponentOutput(
                    name="LoggingPrompt",
                    component_type="block",
                    intent="Logging prompt",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=True,
                    is_unchanged=True,  # Unaffected - skip
                ),
                # Only this one is actually being modified
                ComponentOutput(
                    name="GetUserCaching",
                    component_type="method",
                    intent="Add caching wrapper to GetUser method",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=True,
                    is_unchanged=False,  # This one needs work
                ),
            ],
            file_header_intent="",
            notes="",
        )

        with patch(
            "lask_lm.agents.implement.parallel_graph._get_llm"
        ), patch(
            "lask_lm.agents.implement.parallel_graph._structured_output"
        ) as mock_structured:
            mock_chain = Mock()
            mock_chain.invoke.return_value = file_response
            mock_structured.return_value = mock_chain

            result = parallel_decomposer_node(state)

            # Should have created nodes, some marked as SKIP
            assert "nodes" in result
            nodes = result["nodes"]

            # Count statuses - unchanged components should be SKIP
            skip_count = sum(1 for n in nodes.values() if n.status == NodeStatus.SKIP)
            non_skip_count = sum(1 for n in nodes.values() if n.status != NodeStatus.SKIP)

            # We expect 2 skipped (the unchanged prompts) and 1 pending/other (the actual work)
            assert skip_count == 2
            assert non_skip_count >= 1


class TestPromptContentInSystemPrompt:
    """Verify the system prompt contains guidance about recognizing LASK prompts."""

    def test_system_prompt_base_contains_lask_recognition_guidance(self):
        """SYSTEM_PROMPT_BASE should contain guidance about recognizing LASK prompts."""
        from lask_lm.agents.implement.prompts import SYSTEM_PROMPT_BASE

        # Should mention LASK prompt comments
        assert "LASK prompt" in SYSTEM_PROMPT_BASE.lower() or "// @" in SYSTEM_PROMPT_BASE
        # Should explain these are prompts, not code
        assert "not code" in SYSTEM_PROMPT_BASE.lower() or "prompts" in SYSTEM_PROMPT_BASE.lower()
        # Should mention they represent terminal nodes
        assert "terminal" in SYSTEM_PROMPT_BASE.lower()

    def test_decompose_file_prompt_mentions_lask_prompts(self):
        """DECOMPOSE_FILE_PROMPT should mention how to handle existing LASK prompts."""
        from lask_lm.agents.implement.prompts import DECOMPOSE_FILE_PROMPT

        # Should mention LASK prompt comments in SMART SKIP section
        assert "// @" in DECOMPOSE_FILE_PROMPT or "LASK prompt" in DECOMPOSE_FILE_PROMPT

    def test_decompose_class_prompt_mentions_lask_prompts(self):
        """DECOMPOSE_CLASS_PROMPT should mention how to handle existing LASK prompts."""
        from lask_lm.agents.implement.prompts import DECOMPOSE_CLASS_PROMPT

        assert "// @" in DECOMPOSE_CLASS_PROMPT or "LASK prompt" in DECOMPOSE_CLASS_PROMPT

    def test_decompose_method_prompt_mentions_lask_prompts(self):
        """DECOMPOSE_METHOD_PROMPT should mention how to handle existing LASK prompts."""
        from lask_lm.agents.implement.prompts import DECOMPOSE_METHOD_PROMPT

        assert "// @" in DECOMPOSE_METHOD_PROMPT or "LASK prompt" in DECOMPOSE_METHOD_PROMPT


class TestOperationPropagation:
    """Test that operation type is propagated through the decomposition tree."""

    def test_router_sets_operation_on_file_nodes(self):
        """router_node should set operation from FileTarget on FILE nodes."""
        from lask_lm.models import FileOperation

        # Test CREATE operation
        state_create: ParallelImplementState = {
            "plan_summary": "Create new service",
            "target_files": [
                FileTarget(
                    path="NewService.cs",
                    operation=FileOperation.CREATE,
                    description="Create new service",
                ),
            ],
        }

        result = router_node(state_create)
        node = list(result["nodes"].values())[0]
        assert node.operation == FileOperation.CREATE

        # Test MODIFY operation
        state_modify: ParallelImplementState = {
            "plan_summary": "Modify existing service",
            "target_files": [
                FileTarget(
                    path="ExistingService.cs",
                    operation=FileOperation.MODIFY,
                    description="Modify existing service",
                    existing_content="// existing code",
                ),
            ],
        }

        result = router_node(state_modify)
        node = list(result["nodes"].values())[0]
        assert node.operation == FileOperation.MODIFY

    def test_operation_propagates_to_child_nodes(self):
        """Operation should propagate from parent to child nodes during decomposition."""
        from lask_lm.models import FileOperation
        from lask_lm.agents.implement.parallel_graph import _process_file_decomposition_parallel

        parent_node = CodeNode(
            node_id="parent",
            node_type=NodeType.FILE,
            intent="Test file",
            status=NodeStatus.PENDING,
            context_files=["Test.cs"],
            operation=FileOperation.MODIFY,
        )

        response = DecomposeFileOutput(
            components=[
                ComponentOutput(
                    name="TestClass",
                    component_type="class",
                    intent="Test class",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=False,
                ),
            ],
            file_header_intent="",
            notes="",
        )

        result = _process_file_decomposition_parallel(
            node=parent_node,
            response=response,
            current_depth=0,
            contract_registry={},
        )

        # Find the child node (not the parent)
        child_nodes = [n for n in result["nodes"].values() if n.node_id != "parent"]
        assert len(child_nodes) == 1
        assert child_nodes[0].operation == FileOperation.MODIFY

    def test_operation_propagates_through_skip_nodes(self):
        """Operation should propagate even to SKIP nodes."""
        from lask_lm.models import FileOperation
        from lask_lm.agents.implement.parallel_graph import _process_file_decomposition_parallel

        parent_node = CodeNode(
            node_id="parent",
            node_type=NodeType.FILE,
            intent="Test file",
            status=NodeStatus.PENDING,
            context_files=["Test.cs"],
            operation=FileOperation.MODIFY,
        )

        response = DecomposeFileOutput(
            components=[
                ComponentOutput(
                    name="UnchangedClass",
                    component_type="class",
                    intent="Unchanged class",
                    contracts_provided=[],
                    contracts_required=[],
                    context_files=[],
                    is_terminal=True,
                    is_unchanged=True,  # SKIP node
                ),
            ],
            file_header_intent="",
            notes="",
        )

        result = _process_file_decomposition_parallel(
            node=parent_node,
            response=response,
            current_depth=0,
            contract_registry={},
        )

        # Find the SKIP child node
        child_nodes = [n for n in result["nodes"].values() if n.node_id != "parent"]
        assert len(child_nodes) == 1
        assert child_nodes[0].status == NodeStatus.SKIP
        assert child_nodes[0].operation == FileOperation.MODIFY


class TestTerminalPromptSelection:
    """Test that the correct terminal prompt is selected based on operation."""

    def test_terminal_prompts_exist(self):
        """Both CREATE and MODIFY terminal prompts should exist."""
        from lask_lm.agents.implement.prompts import DECOMPOSITION_PROMPTS

        assert "block_create" in DECOMPOSITION_PROMPTS
        assert "block_modify" in DECOMPOSITION_PROMPTS

    def test_create_prompt_does_not_mention_replaces(self):
        """CREATE prompt should focus on new code generation."""
        from lask_lm.agents.implement.prompts import TERMINAL_BLOCK_CREATE_PROMPT

        # CREATE prompt should not have MODIFY-specific fields
        assert "replaces" not in TERMINAL_BLOCK_CREATE_PROMPT.lower()
        assert "insertion_point" not in TERMINAL_BLOCK_CREATE_PROMPT.lower()

    def test_modify_prompt_explains_operations(self):
        """MODIFY prompt should explain INSERT, REPLACE, DELETE operations."""
        from lask_lm.agents.implement.prompts import TERMINAL_BLOCK_MODIFY_PROMPT

        assert "INSERT" in TERMINAL_BLOCK_MODIFY_PROMPT
        assert "REPLACE" in TERMINAL_BLOCK_MODIFY_PROMPT
        assert "DELETE" in TERMINAL_BLOCK_MODIFY_PROMPT
        assert "replaces" in TERMINAL_BLOCK_MODIFY_PROMPT.lower()
        assert "insertion_point" in TERMINAL_BLOCK_MODIFY_PROMPT.lower()

    def test_modify_prompt_mentions_lask_comments(self):
        """MODIFY prompt should explain how to handle existing LASK prompt comments."""
        from lask_lm.agents.implement.prompts import TERMINAL_BLOCK_MODIFY_PROMPT

        assert "// @" in TERMINAL_BLOCK_MODIFY_PROMPT
        assert "LASK prompt" in TERMINAL_BLOCK_MODIFY_PROMPT or "prompts" in TERMINAL_BLOCK_MODIFY_PROMPT.lower()

    @pytest.mark.integration
    def test_emit_terminal_selects_create_prompt_for_create_operation(self):
        """_emit_terminal_parallel should use CREATE prompt for CREATE operations."""
        from lask_lm.models import FileOperation
        from lask_lm.agents.implement.parallel_graph import _emit_terminal_parallel

        node = CodeNode(
            node_id="test",
            node_type=NodeType.BLOCK,
            intent="Create validation method",
            status=NodeStatus.PENDING,
            context_files=["Test.cs"],
            operation=FileOperation.CREATE,
        )

        captured_messages = []

        def capture_invoke(messages):
            captured_messages.extend(messages)
            return LaskPromptOutput(
                intent="Create validation",
                context_files=[],
                additional_directives=[],
                notes="",
            )

        with patch(
            "lask_lm.agents.implement.parallel_graph._get_llm"
        ), patch(
            "lask_lm.agents.implement.parallel_graph._structured_output"
        ) as mock_structured:
            mock_chain = Mock()
            mock_chain.invoke.side_effect = capture_invoke
            mock_structured.return_value = mock_chain

            _emit_terminal_parallel(node, {})

            # Verify CREATE prompt was used (no MODIFY-specific content)
            system_message = captured_messages[0].content
            assert "INSERT" not in system_message
            assert "REPLACE" not in system_message

    @pytest.mark.integration
    def test_emit_terminal_selects_modify_prompt_for_modify_operation(self):
        """_emit_terminal_parallel should use MODIFY prompt for MODIFY operations."""
        from lask_lm.models import FileOperation
        from lask_lm.agents.implement.parallel_graph import _emit_terminal_parallel

        node = CodeNode(
            node_id="test",
            node_type=NodeType.BLOCK,
            intent="Update validation method",
            status=NodeStatus.PENDING,
            context_files=["Test.cs"],
            operation=FileOperation.MODIFY,
        )

        captured_messages = []

        def capture_invoke(messages):
            captured_messages.extend(messages)
            return LaskPromptOutput(
                intent="Update validation",
                context_files=[],
                additional_directives=[],
                notes="",
                replaces="existing validation",
            )

        with patch(
            "lask_lm.agents.implement.parallel_graph._get_llm"
        ), patch(
            "lask_lm.agents.implement.parallel_graph._structured_output"
        ) as mock_structured:
            mock_chain = Mock()
            mock_chain.invoke.side_effect = capture_invoke
            mock_structured.return_value = mock_chain

            _emit_terminal_parallel(node, {})

            # Verify MODIFY prompt was used
            system_message = captured_messages[0].content
            assert "INSERT" in system_message
            assert "REPLACE" in system_message
            assert "DELETE" in system_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
