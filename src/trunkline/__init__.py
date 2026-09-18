"""Trunkline — open-source voice agent stack.

One agent contract, any runtime (cascade / speech-to-speech / hybrid),
local-first providers, honest cost accounting.
"""

from trunkline.agent import AgentSpec, RuntimeMode
from trunkline.session import CallSession

__version__ = "0.1.0"
__all__ = ["AgentSpec", "CallSession", "RuntimeMode", "__version__"]
