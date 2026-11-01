---
name: python-function-standards
description: Conventions for writing or editing any Python function in this project. Use whenever you define or modify a Python function.
---

For every Python function you write or change:

1. Type-declare every parameter and the return value in the `def` line
   (e.g. `def f(x: float, names: list[str]) -> dict[str, float]:`). Use `-> None` when nothing is returned.
2. Give every function a short, concise docstring (1-3 lines) that states what
   the function does, plus the non-obvious input meaning if needed.
   Example: `"""Return {strike: mid} for the given strikes at timestamp."""`
3. Keep it brief: no long essays in docstrings.
