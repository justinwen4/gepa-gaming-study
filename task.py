"""The task being optimized: extracting contact information from clinical notes.

This module defines:
- ContactInfo: alias for ClinicalContactInfo (structured output schema)
- TaskInput: alias for ClinicalTaskInput
- extract_contact_info: The task function that runs the agent
- contact_agent: The pydantic-ai agent performing the extraction
"""

from __future__ import annotations

from pydantic_ai import Agent

from dataset import ClinicalContactInfo, ClinicalTaskInput

# Backward-compatible aliases used by evaluators, adapter, and gating.
ContactInfo = ClinicalContactInfo
TaskInput = ClinicalTaskInput

# The base agent with minimal instructions.
# The actual instructions will be overridden during optimization.
contact_agent = Agent(
    'openai:gpt-4o-mini',
    output_type=ClinicalContactInfo,
    instructions='Extract contact information from the provided text.',
    instrument=True,
    defer_model_check=True,  # Defer model validation to runtime
)


async def extract_contact_info(input: ClinicalTaskInput) -> ClinicalContactInfo:
    """Run the contact extraction agent on the input text.

    This is the task function that will be evaluated and optimized.
    The agent's instructions can be overridden via agent.override() to test
    different prompts during optimization.
    """
    result = await contact_agent.run(input.text)
    return result.output
