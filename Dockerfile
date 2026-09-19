# Tring Studio in a container. Build:  docker build -t tring .
# Run:    docker run -p 8900:8900 -v tring-sessions:/data/sessions tring
# Keys:   pass provider keys as env vars, e.g. -e GEMINI_API_KEY=...
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[transports]"

# Session recordings and the agent spec live on a volume, so upgrades keep
# your history and your agent.
VOLUME ["/data"]
EXPOSE 8900

CMD ["tring", "studio", "--host", "0.0.0.0", "--port", "8900", \
     "--agent", "/data/agent.yaml", "--sessions-dir", "/data/sessions"]
