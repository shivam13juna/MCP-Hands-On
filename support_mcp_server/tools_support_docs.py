"""
Tool implementation for support_docs.search
"""
from pathlib import Path
from typing import Any
from .config import RUNBOOKS_DIR, DEFAULT_MAX_RESULTS


class RunbookDatabase:
    """Simple in-memory database of runbooks."""

    def __init__(self):
        self.runbooks = []
        self._load_runbooks()

    def _load_runbooks(self):
        """Load all markdown files from the runbooks directory."""
        if not RUNBOOKS_DIR.exists():
            return

        for md_file in RUNBOOKS_DIR.glob("*.md"):
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract title (first line starting with #)
            lines = content.split("\n")
            title = md_file.stem.replace("_", " ").title()
            for line in lines:
                if line.startswith("#"):
                    title = line.lstrip("#").strip()
                    break

            self.runbooks.append({
                "title": title,
                "body": content,
                "path": str(md_file.relative_to(RUNBOOKS_DIR.parent.parent))
            })

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, Any]]:
        """
        Search runbooks using simple keyword matching.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of result dictionaries with title, snippet, path, and score
        """
        query_tokens = query.lower().split()
        results = []

        for runbook in self.runbooks:
            # Count occurrences of query tokens in title and body
            searchable = f"{runbook['title']} {runbook['body']}".lower()
            score = sum(searchable.count(token) for token in query_tokens)

            if score > 0:
                # Extract snippet (first 300 chars or find first match context)
                body_lower = runbook['body'].lower()
                snippet_start = 0
                for token in query_tokens:
                    idx = body_lower.find(token)
                    if idx >= 0:
                        snippet_start = max(0, idx - 50)
                        break

                snippet = runbook['body'][snippet_start:snippet_start + 300].strip()
                if snippet_start > 0:
                    snippet = "..." + snippet
                if len(runbook['body']) > snippet_start + 300:
                    snippet = snippet + "..."

                results.append({
                    "title": runbook["title"],
                    "snippet": snippet,
                    "path": runbook["path"],
                    "score": score
                })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:max_results]


# Global instance
_db = RunbookDatabase()


def search_support_docs(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, Any]:
    """
    Handler function for support_docs.search MCP tool.

    Args:
        query: Search query string
        max_results: Maximum number of results

    Returns:
        Dictionary with results and query_used
    """
    results = _db.search(query, max_results)

    return {
        "results": results,
        "query_used": query
    }
