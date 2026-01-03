"""Entry point for the LASK-LM implement agent."""

import sys

from lask_lm.models import ImplementState, FileTarget, FileOperation, GroupedOutput
from lask_lm.agents.implement import compile_implement_graph


def run_implement_agent(
    plan_summary: str,
    target_files: list[dict],
) -> GroupedOutput:
    """
    Run the implement agent on a plan.

    Args:
        plan_summary: Description of what we're implementing
        target_files: List of file targets, each with:
            - path: File path
            - operation: "create" or "modify"
            - description: What the file should do
            - language: Programming language (default: "csharp")
            - existing_content: For modify, the current content

    Returns:
        GroupedOutput containing prompts grouped by file and validation issues
    """
    # Convert to FileTarget objects
    files = [
        FileTarget(
            path=f["path"],
            operation=FileOperation(f.get("operation", "create")),
            description=f["description"],
            language=f.get("language", "csharp"),
            existing_content=f.get("existing_content"),
        )
        for f in target_files
    ]

    # Create initial state
    initial_state = ImplementState(
        plan_summary=plan_summary,
        target_files=files,
    )

    # Run the graph
    app = compile_implement_graph()
    final_state = app.invoke(initial_state)

    return final_state["grouped_output"]


def main() -> int:
    """
    CLI entry point. Returns exit code based on validation issues.

    Exit codes:
        0: Success, no validation issues
        1: Validation issues detected (warnings or errors)
    """
    # Example usage - in practice this would parse CLI args
    result = run_implement_agent(
        plan_summary="Create a simple user service with CRUD operations",
        target_files=[
            {
                "path": "UserService.cs",
                "operation": "create",
                "description": "A service class that handles user CRUD operations with repository pattern",
                "language": "csharp",
            }
        ],
    )

    # Print generated prompts
    print("Generated LASK prompts:")
    for file_output in result.files:
        print(f"\n=== {file_output.file_path} ({file_output.operation.value}) ===")
        for prompt in file_output.prompts:
            print(f"  {prompt.to_comment()}")

    # Print validation issues if any
    if result.validation_issues:
        print(f"\n{'='*60}")
        print(f"Validation issues ({len(result.validation_issues)}):")
        for issue in result.validation_issues:
            print(f"  [{issue.severity.value.upper()}] {issue.code}: {issue.message}")
            if issue.node_id:
                print(f"           node: {issue.node_id}")
            if issue.contract_name:
                print(f"           contract: {issue.contract_name}")
        return 1

    print(f"\nSuccess: {result.total_prompts} prompts generated, no validation issues.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
