"""
Orchestrator for the multi-agent ticket processing flow.
"""
import uuid
from typing import Tuple

from .models import Ticket, Trace, SupervisorOutput, ResearchReport
from .agents import TriageAgent, ResearchAgent, ActionAgent, SupervisorAgent
from .mcp_client import SupportMCPClient


async def run_ticket_flow(ticket: Ticket) -> Tuple[SupervisorOutput, Trace]:
    """
    Run the complete multi-agent flow for a ticket.

    Args:
        ticket: The support ticket to process

    Returns:
        Tuple of (SupervisorOutput, Trace)
    """
    # Generate trace ID
    trace_id = str(uuid.uuid4()) # a unique identifier for this flow run 16 characters. 

    # Create trace
    trace = Trace(trace_id=trace_id, ticket_id=ticket.id)

    # Instantiate MCP client
    mcp_client = SupportMCPClient()

    try:
        # Connect to MCP server
        await mcp_client.connect()

        # Instantiate agents
        triage_agent = TriageAgent()
        research_agent = ResearchAgent()
        action_agent = ActionAgent()
        supervisor_agent = SupervisorAgent()

        # Step 1: Triage
        plan = await triage_agent.run(ticket, trace)

        # Step 2: Research (if tools specified)
        if plan.tools_to_call:
            report = await research_agent.run(ticket, plan, mcp_client, trace)
        else:
            # Empty report if no tools needed
            report = ResearchReport(
                docs_results=[],
                incident_results=[],
                status_results=[],
                summary="No research performed (no tools specified by triage)."
            )

        # Step 3: Action
        action = await action_agent.run(ticket, plan, report, trace)

        # Step 4: Supervisor
        supervisor = await supervisor_agent.run(ticket, plan, report, action, trace)

        return supervisor, trace

    finally:
        # Disconnect MCP client
        await mcp_client.disconnect()
