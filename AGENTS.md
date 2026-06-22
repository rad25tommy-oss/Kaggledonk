# Repository Guidance

This repository uses a fixed analysis format for behavior investigations.

When asked to analyze a play, bug, or decision:

1. Explain the cause first.
2. Then explain the code interpretation.
3. Then give the improvement policy.

Rules for analysis:

- Verify against the actual logs and the code before drawing conclusions.
- Prefer concrete, traceable evidence over inference.
- If a detail is uncertain, say so explicitly instead of guessing.
- Do not collapse distinct events into one explanation if the log shows separate actions.
- Keep the explanation aligned with the selected evidence and the current code path.

Repository-wide editing expectations:

- Keep changes narrow and consistent with the existing style.
- Use the existing code patterns before introducing new abstractions.
- Avoid unrelated refactors when a targeted fix is enough.
