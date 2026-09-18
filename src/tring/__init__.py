"""Tring — open-source voice agent stack.

One agent contract, any runtime (cascade / speech-to-speech / hybrid),
local-first providers, honest cost accounting.
"""

from tring.agent import AgentSpec, RuntimeMode
from tring.session import CallSession

__version__ = "0.2.0"
__all__ = ["AgentSpec", "CallSession", "RuntimeMode", "__version__"]
