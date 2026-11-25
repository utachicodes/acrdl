"""Prompt templates for the patch generator."""

from __future__ import annotations

from acrdl.types import BuggyProgram

REPAIR_TEMPLATE = """You are an expert Python developer. The function below is buggy.

### Specification
{prompt}

### Buggy implementation
```python
{buggy}
```

### Failing tests
{failing}

Return only the complete fixed function in a single ```python code block."""


def repair_prompt(bug: BuggyProgram, failing: list[str]) -> str:
    return REPAIR_TEMPLATE.format(
        prompt=bug.problem.prompt.strip(),
        buggy=bug.source.strip(),
        failing="\n".join(failing) or "(none provided)",
    )


def extract_code(completion: str) -> str:
    """Pull the first fenced python block out of an LLM completion."""
    for fence in ("```python", "```"):
        if fence in completion:
            body = completion.split(fence, 1)[1]
            return body.split("```", 1)[0].strip() + "\n"
    return completion.strip() + "\n"
