"""Tests for grouped output and MODIFY manifest generation."""

import pytest

from lask_lm.models import (
    CodeNode,
    NodeType,
    NodeStatus,
    FileOperation,
    OperationType,
    LaskPrompt,
    LaskDirective,
    FileTarget,
    LocationMetadata,
    ModifyOperation,
    ModifyManifest,
    OrderedFilePrompts,
    GroupedOutput,
    ParallelImplementState,
)
from lask_lm.agents.implement.parallel_graph import (
    collector_node,
    _depth_first_collect_prompts,
    _build_file_to_root_mapping,
    _build_modify_manifest,
)


class TestDepthFirstCollectPrompts:
    """Test the _depth_first_collect_prompts function."""

    def test_collects_single_terminal_node(self):
        """Collects prompt from a single terminal node."""
        prompt = LaskPrompt(file_path="test.cs", intent="Test intent")
        node = CodeNode(
            node_id="n1",
            node_type=NodeType.BLOCK,
            intent="Test",
            status=NodeStatus.COMPLETE,
            lask_prompt=prompt,
        )
        nodes = {"n1": node}
        collected = []

        _depth_first_collect_prompts("n1", nodes, collected)

        assert len(collected) == 1
        assert collected[0] == prompt

    def test_collects_in_depth_first_order(self):
        """Collects prompts in depth-first order following children_ids."""
        # Tree structure:
        #       root
        #      /    \
        #   child1  child2
        #    /  \
        #  gc1  gc2
        prompts = [
            LaskPrompt(file_path="test.cs", intent=f"Intent {i}")
            for i in range(4)
        ]

        root = CodeNode(
            node_id="root",
            node_type=NodeType.FILE,
            intent="Root",
            children_ids=["child1", "child2"],
            status=NodeStatus.DECOMPOSING,
        )
        child1 = CodeNode(
            node_id="child1",
            node_type=NodeType.CLASS,
            intent="Child 1",
            parent_id="root",
            children_ids=["gc1", "gc2"],
            status=NodeStatus.DECOMPOSING,
        )
        child2 = CodeNode(
            node_id="child2",
            node_type=NodeType.BLOCK,
            intent="Child 2",
            parent_id="root",
            status=NodeStatus.COMPLETE,
            lask_prompt=prompts[2],
        )
        gc1 = CodeNode(
            node_id="gc1",
            node_type=NodeType.BLOCK,
            intent="Grandchild 1",
            parent_id="child1",
            status=NodeStatus.COMPLETE,
            lask_prompt=prompts[0],
        )
        gc2 = CodeNode(
            node_id="gc2",
            node_type=NodeType.BLOCK,
            intent="Grandchild 2",
            parent_id="child1",
            status=NodeStatus.COMPLETE,
            lask_prompt=prompts[1],
        )

        nodes = {
            "root": root,
            "child1": child1,
            "child2": child2,
            "gc1": gc1,
            "gc2": gc2,
        }
        collected = []

        _depth_first_collect_prompts("root", nodes, collected)

        # Should be: gc1, gc2, child2 (depth-first order)
        assert len(collected) == 3
        assert collected[0].intent == "Intent 0"  # gc1
        assert collected[1].intent == "Intent 1"  # gc2
        assert collected[2].intent == "Intent 2"  # child2

    def test_handles_missing_node(self):
        """Handles missing node ID gracefully."""
        collected = []
        _depth_first_collect_prompts("nonexistent", {}, collected)
        assert collected == []

    def test_skips_nodes_without_prompts(self):
        """Non-terminal nodes without prompts are skipped."""
        root = CodeNode(
            node_id="root",
            node_type=NodeType.FILE,
            intent="Root",
            status=NodeStatus.DECOMPOSING,
            children_ids=[],
        )
        nodes = {"root": root}
        collected = []

        _depth_first_collect_prompts("root", nodes, collected)

        assert collected == []


class TestBuildFileToRootMapping:
    """Test the _build_file_to_root_mapping function."""

    def test_builds_mapping_from_root_nodes(self):
        """Builds correct mapping from root nodes to file targets."""
        root1 = CodeNode(
            node_id="r1",
            node_type=NodeType.FILE,
            intent="File 1",
            context_files=["File1.cs"],
        )
        root2 = CodeNode(
            node_id="r2",
            node_type=NodeType.FILE,
            intent="File 2",
            context_files=["File2.cs"],
        )

        nodes = {"r1": root1, "r2": root2}
        root_node_ids = ["r1", "r2"]
        target_files = [
            FileTarget(path="File1.cs", operation=FileOperation.CREATE, description="First"),
            FileTarget(path="File2.cs", operation=FileOperation.MODIFY, description="Second"),
        ]

        mapping = _build_file_to_root_mapping(root_node_ids, nodes, target_files)

        assert len(mapping) == 2
        assert "File1.cs" in mapping
        assert "File2.cs" in mapping
        assert mapping["File1.cs"][0] == "r1"
        assert mapping["File1.cs"][1].operation == FileOperation.CREATE
        assert mapping["File2.cs"][0] == "r2"
        assert mapping["File2.cs"][1].operation == FileOperation.MODIFY


class TestBuildModifyManifest:
    """Test the _build_modify_manifest function."""

    def test_creates_manifest_with_operations(self):
        """Creates manifest from prompts with correct operations."""
        prompts = [
            LaskPrompt(
                file_path="test.cs",
                intent="Add validation",
                insertion_point="after method GetById",
            ),
            LaskPrompt(
                file_path="test.cs",
                intent="Replace error handling",
                replaces="old error handling logic",
            ),
        ]
        file_target = FileTarget(
            path="test.cs",
            operation=FileOperation.MODIFY,
            description="Test file",
            existing_content="public class Test {}",
        )

        manifest = _build_modify_manifest("test.cs", prompts, file_target)

        assert manifest.manifest_version == "1.0"
        assert manifest.target_file == "test.cs"
        assert manifest.existing_content_hash is not None
        assert len(manifest.existing_content_hash) == 16
        assert len(manifest.operations) == 2

        # First operation - INSERT
        op1 = manifest.operations[0]
        assert op1.operation_id == "op_000"
        assert op1.operation_type == OperationType.INSERT
        assert op1.location.insertion_point == "after method GetById"
        assert op1.replaces is None

        # Second operation - REPLACE
        op2 = manifest.operations[1]
        assert op2.operation_id == "op_001"
        assert op2.operation_type == OperationType.REPLACE
        assert op2.replaces == "old error handling logic"

    def test_handles_no_existing_content(self):
        """Handles file target without existing content."""
        prompts = [LaskPrompt(file_path="test.cs", intent="Test")]
        file_target = FileTarget(
            path="test.cs",
            operation=FileOperation.MODIFY,
            description="Test file",
        )

        manifest = _build_modify_manifest("test.cs", prompts, file_target)

        assert manifest.existing_content_hash is None


class TestCollectorNodeGroupedOutput:
    """Test collector_node grouped output generation."""

    def test_groups_prompts_by_file(self):
        """Groups prompts by file path."""
        prompt1 = LaskPrompt(file_path="File1.cs", intent="Intent 1")
        prompt2 = LaskPrompt(file_path="File2.cs", intent="Intent 2")

        root1 = CodeNode(
            node_id="r1",
            node_type=NodeType.FILE,
            intent="File 1",
            context_files=["File1.cs"],
            children_ids=["n1"],
            status=NodeStatus.DECOMPOSING,
        )
        root2 = CodeNode(
            node_id="r2",
            node_type=NodeType.FILE,
            intent="File 2",
            context_files=["File2.cs"],
            children_ids=["n2"],
            status=NodeStatus.DECOMPOSING,
        )
        node1 = CodeNode(
            node_id="n1",
            node_type=NodeType.BLOCK,
            intent="Block 1",
            parent_id="r1",
            context_files=["File1.cs"],
            status=NodeStatus.COMPLETE,
            lask_prompt=prompt1,
        )
        node2 = CodeNode(
            node_id="n2",
            node_type=NodeType.BLOCK,
            intent="Block 2",
            parent_id="r2",
            context_files=["File2.cs"],
            status=NodeStatus.COMPLETE,
            lask_prompt=prompt2,
        )

        state: ParallelImplementState = {
            "plan_summary": "Test plan",
            "nodes": {"r1": root1, "r2": root2, "n1": node1, "n2": node2},
            "root_node_ids": ["r1", "r2"],
            "target_files": [
                FileTarget(path="File1.cs", operation=FileOperation.CREATE, description="F1"),
                FileTarget(path="File2.cs", operation=FileOperation.CREATE, description="F2"),
            ],
            "lask_prompts": [prompt1, prompt2],
        }

        result = collector_node(state)

        output = result["grouped_output"]
        assert isinstance(output, GroupedOutput)
        assert output.plan_summary == "Test plan"
        assert len(output.files) == 2
        assert output.total_prompts == 2

    def test_preserves_tree_order_within_file(self):
        """Preserves depth-first order within each file."""
        prompts = [
            LaskPrompt(file_path="test.cs", intent=f"Intent {i}")
            for i in range(3)
        ]

        root = CodeNode(
            node_id="root",
            node_type=NodeType.FILE,
            intent="File",
            context_files=["test.cs"],
            children_ids=["c1", "c2"],
            status=NodeStatus.DECOMPOSING,
        )
        c1 = CodeNode(
            node_id="c1",
            node_type=NodeType.CLASS,
            intent="Class",
            parent_id="root",
            context_files=["test.cs"],
            children_ids=["m1"],
            status=NodeStatus.DECOMPOSING,
        )
        m1 = CodeNode(
            node_id="m1",
            node_type=NodeType.BLOCK,
            intent="Method 1",
            parent_id="c1",
            context_files=["test.cs"],
            status=NodeStatus.COMPLETE,
            lask_prompt=prompts[0],
        )
        c2 = CodeNode(
            node_id="c2",
            node_type=NodeType.BLOCK,
            intent="Block 2",
            parent_id="root",
            context_files=["test.cs"],
            status=NodeStatus.COMPLETE,
            lask_prompt=prompts[1],
        )

        state: ParallelImplementState = {
            "plan_summary": "Test",
            "nodes": {"root": root, "c1": c1, "c2": c2, "m1": m1},
            "root_node_ids": ["root"],
            "target_files": [
                FileTarget(path="test.cs", operation=FileOperation.CREATE, description="Test"),
            ],
            "lask_prompts": prompts,
        }

        result = collector_node(state)
        output = result["grouped_output"]

        assert len(output.files) == 1
        file_output = output.files[0]
        assert len(file_output.prompts) == 2
        # Should be in depth-first order: m1, c2
        assert file_output.prompts[0].intent == "Intent 0"
        assert file_output.prompts[1].intent == "Intent 1"

    def test_generates_modify_manifest_for_modify_operations(self):
        """Generates MODIFY manifest for files with MODIFY operation."""
        prompt = LaskPrompt(
            file_path="test.cs",
            intent="Add method",
            insertion_point="after class declaration",
        )

        root = CodeNode(
            node_id="root",
            node_type=NodeType.FILE,
            intent="File",
            context_files=["test.cs"],
            children_ids=["n1"],
            status=NodeStatus.DECOMPOSING,
        )
        n1 = CodeNode(
            node_id="n1",
            node_type=NodeType.BLOCK,
            intent="Block",
            parent_id="root",
            context_files=["test.cs"],
            status=NodeStatus.COMPLETE,
            lask_prompt=prompt,
        )

        state: ParallelImplementState = {
            "plan_summary": "Modify test",
            "nodes": {"root": root, "n1": n1},
            "root_node_ids": ["root"],
            "target_files": [
                FileTarget(
                    path="test.cs",
                    operation=FileOperation.MODIFY,
                    description="Test",
                    existing_content="public class Test {}",
                ),
            ],
            "lask_prompts": [prompt],
        }

        result = collector_node(state)
        output = result["grouped_output"]

        assert output.has_modify_operations is True
        assert len(output.files) == 1

        file_output = output.files[0]
        assert file_output.operation == FileOperation.MODIFY
        assert file_output.modify_manifest is not None
        assert len(file_output.modify_manifest.operations) == 1

    def test_no_manifest_for_create_operations(self):
        """No MODIFY manifest for CREATE operations."""
        prompt = LaskPrompt(file_path="test.cs", intent="Create class")

        root = CodeNode(
            node_id="root",
            node_type=NodeType.FILE,
            intent="File",
            context_files=["test.cs"],
            children_ids=["n1"],
            status=NodeStatus.DECOMPOSING,
        )
        n1 = CodeNode(
            node_id="n1",
            node_type=NodeType.BLOCK,
            intent="Block",
            parent_id="root",
            context_files=["test.cs"],
            status=NodeStatus.COMPLETE,
            lask_prompt=prompt,
        )

        state: ParallelImplementState = {
            "plan_summary": "Create test",
            "nodes": {"root": root, "n1": n1},
            "root_node_ids": ["root"],
            "target_files": [
                FileTarget(path="test.cs", operation=FileOperation.CREATE, description="Test"),
            ],
            "lask_prompts": [prompt],
        }

        result = collector_node(state)
        output = result["grouped_output"]

        assert output.has_modify_operations is False
        assert output.files[0].modify_manifest is None


class TestModels:
    """Test the new Pydantic models."""

    def test_location_metadata(self):
        """LocationMetadata model works correctly."""
        loc = LocationMetadata(
            insertion_point="after method GetById",
            line_range=(10, 20),
            ast_path="class Foo > method Bar",
        )
        assert loc.insertion_point == "after method GetById"
        assert loc.line_range == (10, 20)
        assert loc.ast_path == "class Foo > method Bar"

    def test_modify_operation(self):
        """ModifyOperation model works correctly."""
        op = ModifyOperation(
            operation_id="op_001",
            operation_type=OperationType.REPLACE,
            location=LocationMetadata(insertion_point="at line 10"),
            replaces="old code",
            intent="Replace validation logic",
            directives=[LaskDirective(directive_type="context", value="Helper.cs")],
        )
        assert op.operation_type == OperationType.REPLACE
        assert op.replaces == "old code"

    def test_modify_manifest_serialization(self):
        """ModifyManifest can be serialized to JSON."""
        manifest = ModifyManifest(
            target_file="test.cs",
            existing_content_hash="abc123",
            operations=[
                ModifyOperation(
                    operation_id="op_000",
                    operation_type=OperationType.INSERT,
                    location=LocationMetadata(insertion_point="after imports"),
                    intent="Add using statement",
                ),
            ],
        )

        json_str = manifest.model_dump_json()
        assert "test.cs" in json_str
        assert "op_000" in json_str
        assert "insert" in json_str

    def test_grouped_output(self):
        """GroupedOutput model works correctly."""
        output = GroupedOutput(
            plan_summary="Test plan",
            files=[
                OrderedFilePrompts(
                    file_path="test.cs",
                    operation=FileOperation.CREATE,
                    prompts=[LaskPrompt(file_path="test.cs", intent="Test")],
                ),
            ],
            total_prompts=1,
            has_modify_operations=False,
        )
        assert output.plan_summary == "Test plan"
        assert len(output.files) == 1
        assert output.total_prompts == 1


class TestSkipUnchangedComponents:
    """Tests for is_unchanged component handling (Smart SKIP)."""

    def test_skip_nodes_excluded_from_prompt_collection(self):
        """SKIP nodes don't contribute to collected prompts."""
        prompt = LaskPrompt(file_path="test.cs", intent="Modified method")

        root = CodeNode(
            node_id="root",
            node_type=NodeType.FILE,
            intent="File",
            context_files=["test.cs"],
            children_ids=["unchanged", "modified"],
            status=NodeStatus.DECOMPOSING,
        )
        # SKIP node - no prompt
        unchanged = CodeNode(
            node_id="unchanged",
            node_type=NodeType.BLOCK,
            intent="Unchanged class",
            parent_id="root",
            context_files=["test.cs"],
            status=NodeStatus.SKIP,  # Marked as SKIP
        )
        # COMPLETE node - has prompt
        modified = CodeNode(
            node_id="modified",
            node_type=NodeType.BLOCK,
            intent="Modified method",
            parent_id="root",
            context_files=["test.cs"],
            status=NodeStatus.COMPLETE,
            lask_prompt=prompt,
        )

        nodes = {"root": root, "unchanged": unchanged, "modified": modified}
        collected = []

        _depth_first_collect_prompts("root", nodes, collected)

        # Only the modified node's prompt should be collected
        assert len(collected) == 1
        assert collected[0].intent == "Modified method"

    def test_collector_excludes_skip_nodes_from_output(self):
        """Collector node excludes SKIP nodes from grouped output."""
        prompt = LaskPrompt(file_path="test.cs", intent="New feature")

        root = CodeNode(
            node_id="root",
            node_type=NodeType.FILE,
            intent="File",
            context_files=["test.cs"],
            children_ids=["skip1", "complete1"],
            status=NodeStatus.DECOMPOSING,
        )
        skip1 = CodeNode(
            node_id="skip1",
            node_type=NodeType.BLOCK,
            intent="Unchanged",
            parent_id="root",
            context_files=["test.cs"],
            status=NodeStatus.SKIP,
        )
        complete1 = CodeNode(
            node_id="complete1",
            node_type=NodeType.BLOCK,
            intent="New feature",
            parent_id="root",
            context_files=["test.cs"],
            status=NodeStatus.COMPLETE,
            lask_prompt=prompt,
        )

        state: ParallelImplementState = {
            "plan_summary": "MODIFY with SKIP",
            "nodes": {"root": root, "skip1": skip1, "complete1": complete1},
            "root_node_ids": ["root"],
            "target_files": [
                FileTarget(
                    path="test.cs",
                    operation=FileOperation.MODIFY,
                    description="Test",
                    existing_content="public class Test {}",
                ),
            ],
            "lask_prompts": [prompt],
        }

        result = collector_node(state)
        output = result["grouped_output"]

        # Only 1 prompt (from complete1), skip1 is excluded
        assert output.total_prompts == 1
        assert len(output.files[0].prompts) == 1
        assert output.files[0].prompts[0].intent == "New feature"


class TestDeleteOperations:
    """Tests for DELETE operation support."""

    def test_delete_prompt_creates_delete_operation(self):
        """Prompt with is_delete=True creates DELETE operation in manifest."""
        prompts = [
            LaskPrompt(
                file_path="test.cs",
                intent="Remove deprecated validation",
                replaces="LegacyValidate method",
                is_delete=True,
            ),
            LaskPrompt(
                file_path="test.cs",
                intent="Add new validation",
                insertion_point="after constructor",
            ),
        ]
        file_target = FileTarget(
            path="test.cs",
            operation=FileOperation.MODIFY,
            description="Test file",
            existing_content="public class Test { void LegacyValidate() {} }",
        )

        manifest = _build_modify_manifest("test.cs", prompts, file_target)

        assert len(manifest.operations) == 2

        # First operation - DELETE
        op1 = manifest.operations[0]
        assert op1.operation_type == OperationType.DELETE
        assert op1.replaces == "LegacyValidate method"

        # Second operation - INSERT
        op2 = manifest.operations[1]
        assert op2.operation_type == OperationType.INSERT
        assert op2.location.insertion_point == "after constructor"

    def test_delete_to_comment_format(self):
        """DELETE prompts render as @delete comments."""
        prompt = LaskPrompt(
            file_path="test.cs",
            intent="Remove old code",
            replaces="the deprecated method",
            is_delete=True,
        )

        comment = prompt.to_comment()

        assert "@delete" in comment
        assert "the deprecated method" in comment

    def test_delete_to_comment_without_replaces(self):
        """DELETE prompts without replaces use default text."""
        prompt = LaskPrompt(
            file_path="test.cs",
            intent="Remove old code",
            is_delete=True,
        )

        comment = prompt.to_comment()

        assert "@delete" in comment
        assert "target code" in comment

    def test_delete_operation_in_collector_output(self):
        """DELETE operations appear in collector grouped output."""
        prompt = LaskPrompt(
            file_path="test.cs",
            intent="Remove deprecated code",
            replaces="old method",
            is_delete=True,
        )

        root = CodeNode(
            node_id="root",
            node_type=NodeType.FILE,
            intent="File",
            context_files=["test.cs"],
            children_ids=["n1"],
            status=NodeStatus.DECOMPOSING,
        )
        n1 = CodeNode(
            node_id="n1",
            node_type=NodeType.BLOCK,
            intent="Delete block",
            parent_id="root",
            context_files=["test.cs"],
            status=NodeStatus.COMPLETE,
            lask_prompt=prompt,
        )

        state: ParallelImplementState = {
            "plan_summary": "Delete operation test",
            "nodes": {"root": root, "n1": n1},
            "root_node_ids": ["root"],
            "target_files": [
                FileTarget(
                    path="test.cs",
                    operation=FileOperation.MODIFY,
                    description="Test",
                    existing_content="public class Test { void OldMethod() {} }",
                ),
            ],
            "lask_prompts": [prompt],
        }

        result = collector_node(state)
        output = result["grouped_output"]

        assert output.has_modify_operations is True
        manifest = output.files[0].modify_manifest
        assert manifest is not None
        assert len(manifest.operations) == 1
        assert manifest.operations[0].operation_type == OperationType.DELETE
