"""
Agent implementations for the multi-agent orchestration.
"""
import json
from typing import Optional

from .models import (
    Ticket, TriagePlan, ResearchReport, ActionOutput, SupervisorOutput, Trace
)
from .llm_client import call_llm_for_json
from .mcp_client import SupportMCPClient
from .observability import log_llm_call, log_tool_call, log_tool_result, log_agent_output


class TriageAgent:
    """
    Triage agent: analyzes the ticket and creates a plan.
    """

    async def run(self, ticket: Ticket, trace: Trace) -> TriagePlan:
        """
        Analyze ticket and create triage plan.

        Args:
            ticket: The support ticket to analyze
            trace: Trace object for logging

        Returns:
            TriagePlan with issue summary, type, and tools to call
        """
        system_prompt = """You are a support ticket triage specialist.

Your task is to:
1. Summarize the issue described in the ticket
2. Classify the issue type
3. Decide which tools to use for research

Available tools:
- support_docs.search: Search internal runbooks and documentation
- incidents.search: Search for similar incidents
- status.check: Check service health status

Issue types:
- KNOWN_ISSUE: Matches a known problem with documented solution
- POSSIBLE_BUG: Likely a software bug
- CONFIG_ERROR: User configuration mistake
- OUTAGE_SUSPECTED: Possible service outage
- OTHER: Doesn't fit other categories

Output your analysis as JSON with this structure:
{
  "issue_summary": "brief summary",
  "issue_type": "one of the types above",
  "tools_to_call": ["tool names to use"],
  "notes": "additional context"
}"""

        # Build user message with ticket details
        user_content = f"""Please triage this support ticket:

Description: {ticket.description}

{f'Log snippet: {ticket.log_snippet}' if ticket.log_snippet else 'No logs provided.'}

{f'Screenshot: Available at {ticket.screenshot_path}' if ticket.screenshot_path else 'No screenshot provided.'}"""

        messages = [{"role": "user", "content": user_content}]

        # Log LLM call
        log_llm_call(
            trace,
            "TriageAgent",
            "Analyzing ticket and creating triage plan",
            {"ticket_id": ticket.id}
        )

        # Call LLM
        result = call_llm_for_json(
            system_prompt=system_prompt,
            messages=messages,
            image_path=ticket.screenshot_path
        )

        # Create TriagePlan
        plan = TriagePlan(**result)

        # Log output
        log_agent_output(
            trace,
            "TriageAgent",
            f"Classified as {plan.issue_type}, will use tools: {', '.join(plan.tools_to_call)}"
        )

        return plan


class ResearchAgent:
    """
    Research agent: calls MCP tools to gather information.
    """

    async def run(
        self,
        ticket: Ticket,
        plan: TriagePlan,
        mcp_client: SupportMCPClient,
        trace: Trace
    ) -> ResearchReport:
        """
        Execute research using MCP tools.

        Args:
            ticket: The support ticket
            plan: Triage plan with tools to call
            mcp_client: MCP client for tool calls
            trace: Trace object for logging

        Returns:
            ResearchReport with gathered information
        """
        docs_results = []
        incident_results = []
        status_results = []

        # Build query from ticket description
        query = ticket.description

        # Call each tool specified in the plan
        for tool_name in plan.tools_to_call:
            if tool_name == "support_docs.search":
                log_tool_call(trace, tool_name, {"query": query})

                results = await mcp_client.call_support_docs_search(query, max_results=3)
                docs_results = results

                log_tool_result(
                    trace,
                    tool_name,
                    {"count": len(results), "top_title": results[0]["title"] if results else None}
                )

            elif tool_name == "incidents.search":
                log_tool_call(trace, tool_name, {"query": query})

                # Filter for active incidents if it's an outage
                status_filter = ["Investigating", "Mitigating"] if plan.issue_type == "OUTAGE_SUSPECTED" else None

                results = await mcp_client.call_incidents_search(
                    query,
                    max_results=3,
                    status_filter=status_filter
                )
                incident_results = results

                log_tool_result(
                    trace,
                    tool_name,
                    {"count": len(results), "top_id": results[0]["incident_id"] if results else None}
                )

            elif tool_name == "status.check":
                # Infer service names from logs or description
                service_names = self._infer_service_names(ticket)

                for service_name in service_names:
                    log_tool_call(trace, tool_name, {"service_name": service_name})

                    result = await mcp_client.call_status_check(service_name)
                    status_results.append(result)

                    log_tool_result(
                        trace,
                        tool_name,
                        {"service": service_name, "status": result.get("status")}
                    )

        # Create summary
        summary = self._create_summary(docs_results, incident_results, status_results)

        report = ResearchReport(
            docs_results=docs_results,
            incident_results=incident_results,
            status_results=status_results,
            summary=summary
        )

        log_agent_output(
            trace,
            "ResearchAgent",
            f"Gathered {len(docs_results)} docs, {len(incident_results)} incidents, {len(status_results)} status checks"
        )

        return report

    def _infer_service_names(self, ticket: Ticket) -> list[str]:
        """Infer service names from ticket content."""
        services = set()

        # Look for service keywords in description and logs
        content = f"{ticket.description} {ticket.log_snippet or ''}".lower()

        if "export" in content or "csv" in content:
            services.add("export_service")
        if "auth" in content or "login" in content or "sso" in content or "saml" in content:
            services.add("auth_service")
        if "workspace" in content or "permission" in content:
            services.add("workspace_service")

        return list(services) if services else ["export_service"]  # Default to export_service

    def _create_summary(
        self,
        docs_results: list[dict],
        incident_results: list[dict],
        status_results: list[dict]
    ) -> str:
        """Create a text summary of research results."""
        parts = []

        if docs_results:
            parts.append(f"Found {len(docs_results)} relevant documentation entries:")
            for doc in docs_results[:2]:
                parts.append(f"  - {doc['title']}")

        if incident_results:
            parts.append(f"\nFound {len(incident_results)} related incidents:")
            for inc in incident_results[:2]:
                parts.append(f"  - {inc['incident_id']}: {inc['title']} ({inc['status']})")

        if status_results:
            parts.append(f"\nService status:")
            for status in status_results:
                parts.append(f"  - {status['service_name']}: {status['status']}")

        return "\n".join(parts) if parts else "No research results found."


class ActionAgent:
    """
    Action agent: drafts customer reply and internal note.
    """

    async def run(
        self,
        ticket: Ticket,
        plan: TriagePlan,
        report: ResearchReport,
        trace: Trace
    ) -> ActionOutput:
        """
        Draft customer-facing reply and internal note.

        Args:
            ticket: The support ticket
            plan: Triage plan
            report: Research report
            trace: Trace object for logging

        Returns:
            ActionOutput with customer reply and internal note
        """
        system_prompt = """You are a support engineer drafting responses to customer tickets.

Your task is to create:
1. A customer-facing reply: empathetic, clear, actionable but extremely sarcastic and patronizing.
2. An internal note: technical details, context, next steps

Guidelines for customer reply:
- Be empathetic and professional
- Explain what's happening in simple terms
- Provide actionable next steps
- Do NOT expose internal incident IDs or technical implementation details
- If there's an ongoing incident, acknowledge it without alarm
- Mention relevant documentation if helpful

Guidelines for internal note:
- Include technical details and error codes
- Reference specific incidents by ID
- Note which runbooks apply
- Suggest escalation if needed

Output as JSON:
{
  "customer_reply": "...",
  "internal_note": "..."
}"""

        user_content = f"""Ticket: {ticket.description}

Triage Analysis:
- Type: {plan.issue_type}
- Summary: {plan.issue_summary}

Research Results:
{report.summary}

Full research data:
- Docs: {json.dumps([{'title': d['title'], 'path': d['path']} for d in report.docs_results], indent=2)}
- Incidents: {json.dumps([{'id': i['incident_id'], 'title': i['title'], 'status': i['status']} for i in report.incident_results], indent=2)}
- Status: {json.dumps(report.status_results, indent=2)}

Please draft the customer reply and internal note."""

        messages = [{"role": "user", "content": user_content}]

        log_llm_call(
            trace,
            "ActionAgent",
            "Drafting customer reply and internal note",
            {"issue_type": plan.issue_type}
        )

        result = call_llm_for_json(
            system_prompt=system_prompt,
            messages=messages
        )

        output = ActionOutput(**result)

        log_agent_output(
            trace,
            "ActionAgent",
            "Drafted customer reply and internal note"
        )

        return output


class SupervisorAgent:
    """
    Supervisor agent: reviews and approves/edits the action output.
    """

    async def run(
        self,
        ticket: Ticket,
        plan: TriagePlan,
        report: ResearchReport,
        action: ActionOutput,
        trace: Trace
    ) -> SupervisorOutput:
        """
        Review action output for quality and safety.

        Args:
            ticket: The support ticket
            plan: Triage plan
            report: Research report
            action: Drafted action output
            trace: Trace object for logging

        Returns:
            SupervisorOutput with final approved messages
        """
        system_prompt = """You are a senior support lead reviewing ticket responses.

Your task is to review the customer reply and internal note for:

1. Hallucinations: Does the response claim features/behaviors not supported by the research data?
2. Information leakage: Does the customer reply expose internal incident IDs, service names, or implementation details?
3. Missing context: Are there relevant incidents or runbooks that weren't mentioned?
4. Tone and clarity: Is the customer reply professional and clear?

If issues are found:
- Set approved = false
- Edit the messages to fix the issues
- Explain what was changed in review_notes

If everything looks good:
- Set approved = true
- Copy the messages as-is to final_customer_reply and final_internal_note
- Note "No issues found" in review_notes

Output as JSON:
{
  "approved": true/false,
  "final_customer_reply": "...",
  "final_internal_note": "...",
  "review_notes": "..."
}"""

        user_content = f"""Original ticket: {ticket.description}

Research summary:
{report.summary}

Drafted customer reply:
{action.customer_reply}

Drafted internal note:
{action.internal_note}

Please review and approve or edit."""

        messages = [{"role": "user", "content": user_content}]

        log_llm_call(
            trace,
            "SupervisorAgent",
            "Reviewing action output for quality and safety"
        )

        result = call_llm_for_json(
            system_prompt=system_prompt,
            messages=messages
        )

        output = SupervisorOutput(**result)

        log_agent_output(
            trace,
            "SupervisorAgent",
            f"Review complete: {'Approved' if output.approved else 'Edited'}"
        )

        return output
