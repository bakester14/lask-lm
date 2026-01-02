"""Output schemas for LLM decomposition responses.

All schemas are designed to be compatible with OpenAI's strict structured output mode:
- All fields are required (no defaults)
- No dict types (use list of typed objects instead)
- No union types with None (use empty string instead)
"""

from pydantic import BaseModel, Field


class ContractOutput(BaseModel):
    """Contract definition from decomposition."""
    name: str = Field(description="Contract identifier (e.g., 'IUserService.GetById')")
    signature: str = Field(description="Type signature")
    description: str = Field(description="What this contract provides")


class ComponentOutput(BaseModel):
    """A decomposed component (child node)."""
    name: str = Field(description="Component name")
    component_type: str = Field(description="Type: class, method, property, field, block, etc.")
    intent: str = Field(description="What this component should accomplish")
    contracts_provided: list[ContractOutput] = Field(description="Contracts this component exposes (empty list if none)")
    contracts_required: list[str] = Field(description="Contract names this component depends on (empty list if none)")
    context_files: list[str] = Field(description="Files to reference via @context (empty list if none)")
    is_terminal: bool = Field(description="True if this is ≤10 lines and should become a LASK prompt")


class DecomposeFileOutput(BaseModel):
    """LLM output for FILE-level decomposition."""
    components: list[ComponentOutput] = Field(description="Structural components of the file")
    file_header_intent: str = Field(description="Intent for file-level imports/header, or empty string if not needed")
    notes: str = Field(description="Any notes about the decomposition strategy, or empty string if none")


class DecomposeClassOutput(BaseModel):
    """LLM output for CLASS-level decomposition."""
    class_declaration_intent: str = Field(description="Intent for the class declaration line (inheritance, attributes, etc.)")
    components: list[ComponentOutput] = Field(description="Class members (methods, properties, fields)")
    notes: str = Field(description="Any notes, or empty string if none")


class DecomposeMethodOutput(BaseModel):
    """LLM output for METHOD-level decomposition."""
    is_terminal: bool = Field(description="True if this method is ≤10 lines and needs no further decomposition")
    terminal_intent: str = Field(description="If terminal, the intent for the LASK prompt. Empty string if not terminal.")
    blocks: list[ComponentOutput] = Field(description="If not terminal, the logical blocks. Empty list if terminal.")
    notes: str = Field(description="Any notes, or empty string if none")


class DirectiveOutput(BaseModel):
    """A single LASK directive."""
    name: str = Field(description="Directive name (e.g., 'model', 'temperature')")
    value: str = Field(description="Directive value (e.g., 'gpt-4', '0.3')")


class LaskPromptOutput(BaseModel):
    """LLM output for creating a terminal LASK prompt."""
    intent: str = Field(description="The core intent for the LASK prompt")
    context_files: list[str] = Field(description="Files to include via @context directive (empty list if none)")
    additional_directives: list[DirectiveOutput] = Field(description="Other directives like model, temperature (empty list if none)")
    notes: str = Field(description="Any notes about what code should be generated, or empty string if none")
    # MODIFY-specific fields (empty string if not applicable)
    # Note: defaults added for backwards compatibility with tests
    insertion_point: str = Field(default="", description="For MODIFY: where to insert (e.g., 'after method GetById'). Empty string if not applicable.")
    replaces: str = Field(default="", description="For MODIFY: description of code being replaced. Empty string if not applicable.")
