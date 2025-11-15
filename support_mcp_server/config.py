"""
Configuration for the Support MCP Server.
"""
from pathlib import Path

# Base project directory (parent of support_mcp_server/)
BASE_DIR = Path(__file__).resolve().parents[1]

# Data directories
RUNBOOKS_DIR = BASE_DIR / "data" / "docs" / "runbooks"
INCIDENTS_PATH = BASE_DIR / "data" / "incidents" / "incidents.json"
STATUS_PATH = BASE_DIR / "data" / "status" / "status.json"

# Default limits
DEFAULT_MAX_RESULTS = 3
