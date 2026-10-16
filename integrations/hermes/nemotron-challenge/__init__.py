"""Native Hermes registration. The team implements the comparison tool."""
from typing import Any


TOOL_SCHEMA = {
    "name": "challenge_compare",
    "description": "Comparison tool placeholder. Requires a team implementation before use.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The comparison request to pass to the team's implementation.",
                "minLength": 1,
                "maxLength": 2000,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def handle_compare(params: dict[str, Any], **kwargs: Any) -> str:
    """TODO: define tool behavior, result format, evidence, and error handling."""
    del params, kwargs
    raise NotImplementedError(
        "TODO: implement the team's Hermes comparison tool. Read docs/hermes.md. "
        "The starter has no comparison result."
    )


def register(ctx: Any) -> None:
    """Register the unfinished tool through the native Hermes interface."""
    ctx.register_tool(
        name="challenge_compare",
        toolset="nemotron_challenge",
        schema=TOOL_SCHEMA,
        handler=handle_compare,
    )
