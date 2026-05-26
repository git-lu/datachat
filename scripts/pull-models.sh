#!/bin/sh
set -eu

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
AGENT_MODEL="${AGENT_MODEL:-granite3-dense:2b}"
FORMATTER_MODEL="${FORMATTER_MODEL:-granite3-moe:1b}"

echo "Waiting for Ollama at ${OLLAMA_HOST}..."
until ollama list >/dev/null 2>&1; do
  sleep 2
done

echo "Pulling agent model: ${AGENT_MODEL}"
ollama pull "${AGENT_MODEL}"

echo "Pulling formatter model: ${FORMATTER_MODEL}"
ollama pull "${FORMATTER_MODEL}"

echo "Models ready:"
ollama list
