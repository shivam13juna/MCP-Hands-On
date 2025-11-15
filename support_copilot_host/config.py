"""
Configuration for the Support Copilot Host.
"""
import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parents[1]

# LLM Configuration
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))

# MCP Server Configuration
MCP_SERVER_COMMAND = ["python", "-m", "support_mcp_server.server"]

# Trace Configuration
TRACE_DIR = BASE_DIR / "traces"
TRACE_DIR.mkdir(exist_ok=True)
TRACE_LOG_PATH = TRACE_DIR / "trace.jsonl"

# Data paths
SAMPLE_TICKETS_PATH = BASE_DIR / "data" / "tickets" / "samples.json"
