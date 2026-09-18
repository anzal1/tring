# Trunkline Examples

Two quick-start examples showing how to run an trunkline voice agent locally,
with zero cloud dependencies.

## Example 1: Text Console Chat

The simplest way to test an agent: read text input from the console, process
it through an LLM, and print the response.

### Requirements

- Ollama (for the LLM): https://ollama.ai
- Python 3.11+ with trunkline installed

### Setup

```bash
# Install trunkline with local LLM support
pip install -e ".[local]"

# Start Ollama in a separate terminal
ollama serve

# In another terminal, pull a model
ollama pull mistral  # or llama2, neural-chat, etc.
```

### Run

```bash
python examples/console_chat.py
```

You will see:

```
Loaded agent: customer-support

Hello! Welcome to customer support. How can I help you today?

User> what's the status of order ORD-001?
Bot> I'll check that for you right away.
[Tool] check_order_status(order_id='ORD-001')
[Tool Done] check_order_status: OK
Bot> Your order ORD-001 was shipped on Sep 15 and has tracking number TRACK123.
User> thanks!
Bot> You're welcome! Is there anything else I can help you with?
[Cost] llm/ollama: 45 tokens_out = $0.000000
User> ^D
[Session ended: completed] Duration: 12.3s
Total estimated cost: $0.000000
```

Type Ctrl+D to exit.

### What's happening

1. **Input** → UTF-8 text from your keyboard becomes a "transcript"
2. **LLM** → Ollama processes the text through the agent's persona and tools
3. **Output** → Bot responses are printed to the console
4. **Cost** → Every API call (in this case: free, local) is logged with its cost

No audio files. No cloud. No API keys. Just text and the LLM on your machine.

## Example 2: Real-Audio WebSocket Server

A production-grade local WebSocket server that handles real speech audio:
speech-to-text (faster-whisper), language understanding (Ollama LLM), and
text-to-speech (Kokoro). All local, all real-time.

### Requirements

- Ollama: https://ollama.ai
- PyAudio or PortAudio (for microphone access)
- Python 3.11+ with trunkline installed

### Setup

```bash
# Install trunkline with full local + transport support
pip install -e ".[local,transports]"

# Start Ollama
ollama serve &

# Pull a model (if you haven't already)
ollama pull mistral
```

### Run

```bash
python examples/quickstart_local.py
```

The server starts on `ws://0.0.0.0:8765` and waits for connections:

```
2025-09-18 16:42:15,123 [INFO] __main__: Loaded agent: customer-support
2025-09-18 16:42:15,124 [INFO] __main__: Starting WebSocket server on ws://0.0.0.0:8765
2025-09-18 16:42:15,125 [INFO] __main__: Connect with a WebSocket client...
```

#### Connecting from a Browser or Mobile Client

The client must send binary WebSocket messages containing 16 kHz mono 16-bit
PCM audio frames (typical for speech). You can write a simple client in
JavaScript:

```javascript
const ws = new WebSocket("ws://localhost:8765");
const audioContext = new (window.AudioContext || window.webkitAudioContext)();

ws.binaryType = "arraybuffer";

ws.onopen = () => {
  // Start capturing microphone input and send as binary frames
  navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(processor);
    processor.connect(audioContext.destination);

    processor.onaudioprocess = (e) => {
      const pcm = new Int16Array(e.inputBuffer.getChannelData(0));
      ws.send(pcm.buffer);  // Send as binary
    };
  });
};

ws.onmessage = (e) => {
  // Received bot audio (also PCM)
  const pcm = new Float32Array(e.data);
  // Play the audio...
};

ws.onclose = () => console.log("Connection closed");
```

To end the session, send a JSON message:

```javascript
ws.send(JSON.stringify({ type: "end" }));
```

### What's happening

1. **Microphone** → 16 kHz PCM audio frames are sent to the server
2. **STT** → Faster-whisper transcribes them in real time
3. **LLM** → Ollama generates a response based on the transcript
4. **TTS** → Kokoro synthesizes the response back to audio
5. **Audio** → Returned to the client over WebSocket as PCM frames

All of this happens on your local machine, with zero API calls and zero cost.

## Cost Tracking

Both examples can track the estimated cost of each agent call. In the console
example, you'll see lines like:

```
[Cost] stt/deepgram: 3.2 audio_seconds = $0.013760
[Cost] llm/openai: 42 tokens_in = $0.000006
[Cost] llm/openai: 18 tokens_out = $0.000010
```

The `*` symbol indicates an estimated value (e.g., character counts estimated
from token counts). Exact usage is always preferred, and you should verify
your actual charges against your vendor's billing dashboard.

## Agent Customization

Edit `agent.yaml` to change the agent's behavior:

- **persona**: The system prompt (defines the agent's role and instructions)
- **greeting**: What the agent says when the session starts
- **tools**: Custom tools the agent can call (with async handlers)
- **runtime.routing**: Which provider to use for each language

For example, to use a cloud LLM instead of Ollama:

```yaml
runtime:
  routing:
    default:
      stt: faster_whisper
      llm: openai  # requires OPENAI_API_KEY env var
      tts: kokoro
```

Then set the `OPENAI_API_KEY` environment variable before running the example.

## Troubleshooting

### "Ollama connection refused"

Make sure Ollama is running:

```bash
ollama serve
```

In another terminal, test it:

```bash
curl http://localhost:11434/api/tags
```

### "ImportError: No module named websockets"

Install the transports extra:

```bash
pip install -e ".[transports]"
```

### "ImportError: No module named faster_whisper"

Install the local audio dependencies:

```bash
pip install -e ".[local]"
```

Faster-whisper requires a few heavy dependencies; the import is lazy so it
only happens if you actually use that provider.

### Microphone not working

Make sure you have PyAudio installed:

```bash
pip install pyaudio
```

On macOS:

```bash
brew install portaudio
pip install pyaudio
```

On Linux:

```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

## Next Steps

- **Edit the agent**: Modify `agent.yaml` to change the agent's behavior
- **Add more tools**: Define new tool definitions and handlers
- **Change providers**: Swap providers to use different STT/LLM/TTS engines
- **Deploy**: Deploy the WebSocket server to a server for real-world use
- **Monitor**: Use the cost meter data to optimize your agent's efficiency
