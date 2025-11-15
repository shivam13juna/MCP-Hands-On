"""
MCP client wrapper for calling Support MCP Server tools.
"""
import json
import asyncio
from typing import Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import MCP_SERVER_COMMAND


class SupportMCPClient:
    """Wrapper for MCP client to call support tools."""

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self._context = None
        self._read_stream = None
        self._write_stream = None

    async def connect(self):
        """Connect to the MCP server."""
        server_params = StdioServerParameters(
            command=MCP_SERVER_COMMAND[0],
            args=MCP_SERVER_COMMAND[1:],
            env=None
        )

        # Create stdio client context
        self._context = stdio_client(server_params)
        self._read_stream, self._write_stream = await self._context.__aenter__()

        # Create session
        self.session = ClientSession(self._read_stream, self._write_stream)
        await self.session.__aenter__()

        # Initialize
        await self.session.initialize()

    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.session:
            await self.session.__aexit__(None, None, None)

        if self._context:
            await self._context.__aexit__(None, None, None)

    async def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Internal method to call a tool via MCP.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Parsed result from the tool
        """
        if not self.session:
            raise RuntimeError("MCP client not connected. Call connect() first.")

        result = await self.session.call_tool(tool_name, arguments)

        # Parse the text content as JSON
        if result.content and len(result.content) > 0:
            text_content = result.content[0].text
            return json.loads(text_content)

        return {}

    async def call_support_docs_search(
        self,
        query: str,
        max_results: int = 3
    ) -> list[dict]:
        """
        Search support documentation.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of document results
        """
        result = await self._call_tool(
            "support_docs.search",
            {"query": query, "max_results": max_results}
        )
        return result.get("results", [])

    async def call_incidents_search(
        self,
        query: str,
        max_results: int = 3,
        status_filter: Optional[list[str]] = None
    ) -> list[dict]:
        """
        Search incidents.

        Args:
            query: Search query
            max_results: Maximum results to return
            status_filter: Optional filter by status

        Returns:
            List of incident results
        """
        arguments = {"query": query, "max_results": max_results}
        if status_filter:
            arguments["status_filter"] = status_filter

        result = await self._call_tool("incidents.search", arguments)
        return result.get("results", [])

    async def call_status_check(self, service_name: str) -> dict:
        """
        Check service status.

        Args:
            service_name: Name of the service to check

        Returns:
            Status information dict
        """
        return await self._call_tool(
            "status.check",
            {"service_name": service_name}
        )


# Synchronous wrapper functions for convenience
def create_client() -> SupportMCPClient:
    """Create a new MCP client instance."""
    return SupportMCPClient()


async def with_client(func):
    """
    Decorator/context helper to run an async function with an MCP client.

    Usage:
        async def my_func(client):
            results = await client.call_support_docs_search("export")
            return results

        results = await with_client(my_func)
    """
    client = create_client()
    try:
        await client.connect()
        return await func(client)
    finally:
        await client.disconnect()
