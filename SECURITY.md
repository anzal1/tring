# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities privately via GitHub's security advisory process:

1. Go to the [Awaaz repository](https://github.com/anzalabidi/awaaz).
2. Click the "Security" tab.
3. Select "Report a vulnerability" and fill in the details.

We will acknowledge your report within 48 hours and work with you to develop and release a fix.

## Vulnerability Disclosure

- Do not open public issues for security vulnerabilities.
- Do not publish exploits or proof-of-concept code before a fix is released.
- We will credit you in the security advisory unless you request anonymity.

## Secure Practices

### API Keys and Secrets

- Never commit API keys, credentials, or tokens to the repository.
- Use environment variables to pass secrets to the application at runtime.
- Example: `export OPENAI_API_KEY="sk-..."` before running the agent.

### Dependencies

- We keep dependencies minimal and regularly audit them for known vulnerabilities.
- Report any dependency vulnerabilities in the same manner as code vulnerabilities.

### Local Deployment

- When running providers locally (faster-whisper, Ollama, Kokoro), ensure services are properly isolated and firewalled.
- The WebSocket transport is unencrypted by default; use TLS in production.
