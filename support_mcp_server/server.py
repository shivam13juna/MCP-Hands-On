"""
MCP Server for Support Tools

Exposes tools for searching support documentation, incidents, and checking service status.
"""
import asyncio
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .tools_support_docs import search_support_docs
from .tools_incidents import search_incidents
from .tools_status import check_status

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create server instance
app = Server("support-mcp-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    List all available tools.
    """
    return [
        Tool(
            name="support_docs.search",
            description="Search internal support documentation and runbooks for relevant solutions", # have more detail over here
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for finding relevant documentation"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 3)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="incidents.search",
            description="Search for similar or relevant incidents",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for finding relevant incidents"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 3)",
                        "default": 3
                    },
                    "status_filter": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by incident status (e.g., ['Investigating', 'Mitigating'])"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="status.check",
            description="Check the current health status of a service",
            inputSchema={
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "Name of the service to check (e.g., 'export_service', 'auth_service')"
                    }
                },
                "required": ["service_name"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    Handle tool calls.
    """
    logger.info(f"Tool called: {name} with arguments: {arguments}")

    try:
        if name == "support_docs.search":
            query = arguments.get("query")
            max_results = arguments.get("max_results", 3)

            result = search_support_docs(query, max_results)

            import json
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

        elif name == "incidents.search":
            query = arguments.get("query")
            max_results = arguments.get("max_results", 3)
            status_filter = arguments.get("status_filter")

            result = search_incidents(query, max_results, status_filter)

            import json
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

        elif name == "status.check":
            service_name = arguments.get("service_name")

            result = check_status(service_name)

            import json
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}", exc_info=True)
        import json
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]


async def main():
    """
    Main entry point for the MCP server.
    """
    logger.info("Starting Support MCP Server...")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
