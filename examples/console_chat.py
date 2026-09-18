#!/usr/bin/env python3
"""Text-only console chat with a local LLM agent.

A simple example showing how to run an tring agent entirely locally via the
console transport. This uses:
- Ollama for the LLM (no API keys, runs on your machine)
- Kokoro for text-to-speech synthesis (local)
- text_input for "STT" (console input becomes transcripts)

This is a great starting point for local development and testing. Zero cloud
dependencies beyond what Ollama provides.

Setup:
    1. Install tring with local dependencies:
       pip install -e ".[local]"
    2. Start Ollama and pull a model:
       ollama serve &
       ollama pull mistral  # or your preferred model
    3. Run this example:
       python examples/console_chat.py

The agent will listen for your input on stdin. Type your messages, and the
bot will respond. Type Ctrl+D to exit.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add the source directory to the path so we can import tring.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tring import AgentSpec, CallSession
from tring.cost.rates import DEFAULT_RATES
from tring.transports.console import ConsoleTransport


async def main() -> None:
    """Load agent, create runtime, and run the console transport."""
    # Load the agent specification from YAML.
    agent_path = Path(__file__).parent / "agent.yaml"
    if not agent_path.exists():
        print(f"Error: {agent_path} not found.", file=sys.stderr)
        sys.exit(1)

    agent = AgentSpec.from_yaml(agent_path)
    print(f"Loaded agent: {agent.name}", file=sys.stderr)

    # Create a session for this conversation.
    session = CallSession(agent)

    # Import the runtime here so the dependency is lazy.
    # If CascadeRuntime is not available, the error message will guide the user.
    try:
        from tring.runtimes.cascade import CascadeRuntime
    except ImportError as e:
        print(
            f"Error: CascadeRuntime not found. "
            f"Make sure tring is installed with local support. "
            f"({e})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create the runtime (handles STT -> LLM -> TTS pipeline).
    runtime = CascadeRuntime(session)

    # Optional: attach a cost meter to track usage.
    # CostMeter will subscribe to session events and record costs.
    try:
        from tring.cost.meter import CostMeter

        meter = CostMeter(session, DEFAULT_RATES)
        print("Cost tracking enabled (rates as of 2025-06-01)", file=sys.stderr)
    except ImportError:
        print(
            "Note: CostMeter not available; cost tracking disabled.",
            file=sys.stderr,
        )
        meter = None

    # Bind tool handlers. In this demo, we fake a check_order_status tool.
    async def check_order_status(order_id: str) -> str:
        """Fake order status lookup."""
        # In a real app, this would query a database.
        orders = {
            "ORD-001": "Shipped on Sep 15, tracking: TRACK123",
            "ORD-002": "Delivered on Sep 10",
            "ORD-003": "Processing, ships within 24 hours",
        }
        return orders.get(order_id, f"Order {order_id} not found.")

    # Store tool handlers in the session for the runtime to discover.
    session.handlers = {"check_order_status": check_order_status}

    # Create the transport and run the conversation.
    print(f"\n{agent.greeting or 'Ready for input.'}\n", file=sys.stderr)
    transport = ConsoleTransport(runtime)

    try:
        await transport.run()
    except KeyboardInterrupt:
        print("\nSession interrupted.", file=sys.stderr)
        await runtime.stop(reason="interrupted")

    # Print final stats.
    if meter:
        total_cost = sum(
            e.amount
            for e in session.history
            if hasattr(e, "type") and e.type == "cost_recorded"
        )
        print(f"\nTotal estimated cost: ${total_cost:.6f}", file=sys.stderr)

    print(f"Session {session.session_id[:8]}... ended.", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
