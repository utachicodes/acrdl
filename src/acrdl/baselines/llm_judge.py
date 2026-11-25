"""LLM-as-judge baseline: ask an instruction-tuned LLM for a 3-way verdict."""

from __future__ import annotations

from typing import Callable

from acrdl.types import Label, Patch

JUDGE_TEMPLATE = """You review a bug fix.

### Specification
{spec}

### Buggy code
```python
{buggy}
```

### Patched code
```python
{patch}
```

The patch passes the visible tests. Answer with exactly one word:
CORRECT (matches the specification), OVERFITTING (passes tests but is wrong),
or AMBIGUOUS (the specification does not determine the right behaviour)."""


def judge_prompt(patch: Patch) -> str:
    return JUDGE_TEMPLATE.format(spec=patch.bug.problem.prompt.strip(),
                                 buggy=patch.bug.source.strip(), patch=patch.source.strip())


def parse_verdict(text: str) -> Label:
    t = text.upper()
    for lab in (Label.OVERFITTING, Label.AMBIGUOUS, Label.CORRECT):
        if lab.name in t:
            return lab
    return Label.OVERFITTING  # unparseable answers are treated as rejections


class LLMJudge:
    """``complete`` is any callable prompt -> completion (HF pipeline, API client...)."""

    def __init__(self, complete: Callable[[str], str]):
        self.complete = complete

    def judge(self, patch: Patch) -> Label:
        return parse_verdict(self.complete(judge_prompt(patch)))
