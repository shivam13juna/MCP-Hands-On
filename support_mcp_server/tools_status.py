"""
Tool implementation for status.check
"""
import json
from typing import Any
from .config import STATUS_PATH


class StatusDatabase:
    """Simple in-memory database of service status."""

    def __init__(self):
        self.services = {}
        self._load_status()

    def _load_status(self):
        """Load service status from JSON file."""
        if not STATUS_PATH.exists():
            return

        with open(STATUS_PATH, "r", encoding="utf-8") as f:
            services_list = json.load(f)

        # Convert list to dict for easy lookup
        for service in services_list:
            service_name = service.get("service_name", "").lower()
            self.services[service_name] = service

    def check(self, service_name: str) -> dict[str, Any]:
        """
        Check the status of a specific service.

        Args:
            service_name: Name of the service to check

        Returns:
            Dictionary with service_name, status, and notes
        """
        service_name_lower = service_name.lower()

        if service_name_lower in self.services:
            return self.services[service_name_lower]
        else:
            return {
                "service_name": service_name,
                "status": "Unknown",
                "notes": f"Service '{service_name}' not found in status map."
            }


# Global instance
_db = StatusDatabase()


def check_status(service_name: str) -> dict[str, Any]:
    """
    Handler function for status.check MCP tool.

    Args:
        service_name: Name of the service to check

    Returns:
        Dictionary with service status information
    """
    return _db.check(service_name)
