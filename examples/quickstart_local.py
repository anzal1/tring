#!/usr/bin/env python3
"""Real-audio WebSocket server for local voice conversations.

An example showing the full trunkline stack: faster-whisper for speech-to-text,
Ollama for the LLM, and Kokoro for text-to-speech synthesis. All running
locally on your machine, with zero cloud dependencies and zero per-minute cost.

Connect with a browser or mobile WebSocket client and speak into your
microphone. The audio is transcribed, processed by the LLM, and synthesized
back as speech in real time.

Setup:
    1. Install trunkline with full local support:
       pip install -e ".[local,transports]"
    2. Download and start Ollama (if not already running):
       ollama serve &
    3. Pull a language model:
       ollama pull mistral  # or llama2, neural-chat, etc.
    4. Run this server:
       python examples/quickstart_local.py
    5. Connect with a WebSocket client (e.g., a browser with a
       microphone, or a mobile app). The server prints the connection info.

Note:
    - First startup may be slow as models are loaded into VRAM.
    - faster-whisper requires GPU or CPU; startup may take 10-30 seconds.
    - For cloud deployment, see the docs on hosted runtimes.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

# Add the source directory to the path so we can import trunkline.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trunkline import AgentSpec, CallSession
from trunkline.cost.rates import DEFAULT_RATES


async def main() -> None:
    """Load agent, create a runtime factory, and serve WebSocket connections."""
    # Configure logging so we see what's happening.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load the agent specification from YAML.
    agent_path = Path(__file__).parent / "agent.yaml"
    if not agent_path.exists():
        logger.error(f"Agent file not found: {agent_path}")
        sys.exit(1)

    agent = AgentSpec.from_yaml(agent_path)
    logger.info(f"Loaded agent: {agent.name}")

    # Import the cascade runtime and cost meter.
    try:
        from trunkline.runtimes.cascade import CascadeRuntime
    except ImportError as e:
        logger.error(f"CascadeRuntime import failed: {e}")
        sys.exit(1)

    try:
        from trunkline.cost.meter import CostMeter
    except ImportError:
        logger.warning("CostMeter not available; cost tracking disabled.")
        CostMeter = None

    # Import the WebSocket server.
    try:
        from trunkline.transports.websocket import serve
    except ImportError as e:
        logger.error(
            f"WebSocket transport not available: {e}. "
            f"Install trunkline[transports] to enable."
        )
        sys.exit(1)

    # Create a factory function that the WebSocket server will call for each
    # new connection. Each connection gets its own session and runtime.
    def create_runtime(session: CallSession) -> CascadeRuntime:
        """Factory function to create a CascadeRuntime for each WebSocket connection.

        Args:
            session: A CallSession instance for this connection.

        Returns:
            A configured CascadeRuntime ready to handle audio.
        """
        # Create the runtime with the agent and session.
        runtime = CascadeRuntime(session)

        # Attach a cost meter if available.
        if CostMeter:
            CostMeter(session, DEFAULT_RATES)
            logger.debug(
                f"Cost tracking enabled for session {session.session_id[:8]}..."
            )

        # Bind tool handlers. In this demo, we fake a check_order_status tool.
        async def check_order_status(order_id: str) -> str:
            """Fake order status lookup."""
            orders = {
                "ORD-001": "Shipped on Sep 15, tracking TRACK123",
                "ORD-002": "Delivered on Sep 10",
                "ORD-003": "Processing, ships within 24 hours",
            }
            return orders.get(order_id, f"Order {order_id} not found.")

        # Store handlers in the session.
        session.handlers = {"check_order_status": check_order_status}

        return runtime

    # Start the WebSocket server.
    logger.info("Starting WebSocket server on ws://0.0.0.0:8765")
    logger.info(
        "Connect with a WebSocket client and send 16kHz mono 16-bit PCM audio."
    )

    try:
        await serve(create_runtime, host="0.0.0.0", port=8765)
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
