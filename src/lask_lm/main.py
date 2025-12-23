"""Entry point for the LASK-LM implement agent."""

from lask_lm.models import ImplementState, FileTarget, FileOperation
from lask_lm.agents.implement import create_implement_graph, compile_implement_graph


def run_implement_agent(
    plan_summary: str,
    target_files: list[dict],
) -> list[dict]:
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
        List of LASK prompts ready to write to files
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

    # Extract prompts
    return [
        {
            "file_path": p.file_path,
            "comment": p.to_comment(),
            "intent": p.intent,
            "directives": [
                {"type": d.directive_type, "value": d.value}
                for d in p.directives
            ],
        }
        for p in final_state["lask_prompts"]
    ]


if __name__ == "__main__":
    # Example usage
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

    print("Generated LASK prompts:")
    for prompt in result:
        print(f"\n{prompt['file_path']}:")
        print(f"  {prompt['comment']}")
