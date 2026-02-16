def _format_reviewer_prompt(template: str, notes: str, review_input: str) -> str:
    """Substitute {notes} and {review_input} in the reviewer prompt template."""
    return template.replace("{notes}", notes).replace("{review_input}", review_input)


def _extract_json(text: str) -> str:
    """Return the first JSON object found in *text* (strips markdown fences)."""
    text = text.strip()
    # Strip ```json ... ``` fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text
