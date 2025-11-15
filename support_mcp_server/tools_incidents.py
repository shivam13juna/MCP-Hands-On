"""
Tool implementation for incidents.search
"""
import json
from typing import Any, Optional
from .config import INCIDENTS_PATH, DEFAULT_MAX_RESULTS


class IncidentDatabase:
    """Simple in-memory database of incidents."""

    def __init__(self):
        self.incidents = []
        self._load_incidents()

    def _load_incidents(self):
        """Load incidents from JSON file."""
        if not INCIDENTS_PATH.exists():
            return

        with open(INCIDENTS_PATH, "r", encoding="utf-8") as f:
            self.incidents = json.load(f)

    def search(
        self,
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        status_filter: Optional[list[str]] = None
    ) -> list[dict[str, Any]]:
        """
        Search incidents using simple keyword matching.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            status_filter: Optional list of status values to filter by

        Returns:
            List of incident result dictionaries
        """
        query_tokens = set(query.lower().split())
        results = []

        for incident in self.incidents:
            # Apply status filter if provided
            if status_filter and incident.get("status") not in status_filter:
                continue

            # Create searchable text from title, summary, and tags
            searchable = " ".join([
                incident.get("title", "").lower(),
                incident.get("summary", "").lower(),
                " ".join(incident.get("tags", []))
            ])

            # Count token matches
            score = sum(1 for token in query_tokens if token in searchable)

            if score > 0:
                # Find matched tags
                incident_tags = [tag.lower() for tag in incident.get("tags", [])]
                matched_tags = [
                    tag for tag in incident.get("tags", [])
                    if tag.lower() in query_tokens
                ]

                results.append({
                    "incident_id": incident.get("incident_id"),
                    "title": incident.get("title"),
                    "status": incident.get("status"),
                    "summary": incident.get("summary"),
                    "matched_tags": matched_tags,
                    "score": score
                })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:max_results]


# Global instance
_db = IncidentDatabase()


def search_incidents(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    status_filter: Optional[list[str]] = None
) -> dict[str, Any]:
    """
    Handler function for incidents.search MCP tool.

    Args:
        query: Search query string
        max_results: Maximum number of results
        status_filter: Optional list of status values to filter by

    Returns:
        Dictionary with results and query_used
    """
    results = _db.search(query, max_results, status_filter)

    return {
        "results": results,
        "query_used": query
    }
