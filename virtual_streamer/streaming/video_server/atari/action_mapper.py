"""
Maps LLM-output action names to ALE action integers.
"""


def action_name_to_int(name: str, meanings: list) -> int:
    """Case-insensitive lookup; returns 0 (NOOP) on no match."""
    name_upper = name.strip().upper()
    for i, m in enumerate(meanings):
        if m.upper() == name_upper:
            return i
    return 0
