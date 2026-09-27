"""jev_grep: semantic code search (a jevgrep reimplementation) on TypeSafe's Jev.

Registers a ``semantic_grep`` tool alongside the regular ``grep``. It is only
advertised to agents once ``TYPESAFE_API_KEY`` is configured, so agents never
see a tool that can only fail.
"""

from code_puppy.callbacks import register_callback

from .config import get_typesafe_api_key
from .tool import register_semantic_grep

TOOL_NAME = "semantic_grep"


def _register_tools():
    return [{"name": TOOL_NAME, "register_func": register_semantic_grep}]


def _advertise_when_configured(agent_name=None):
    return [TOOL_NAME] if get_typesafe_api_key() else []


register_callback("register_tools", _register_tools)
register_callback("register_agent_tools", _advertise_when_configured)
