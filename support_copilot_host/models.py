"""
Data models for the Support Copilot Host.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Ticket(BaseModel):
    """Support ticket submitted by user."""
    id: str
    description: str
    screenshot_path: Optional[str] = None
    log_snippet: Optional[str] = None


class TriagePlan(BaseModel):
    """Output from TriageAgent."""
    issue_summary: str
    issue_type: str  # KNOWN_ISSUE, POSSIBLE_BUG, CONFIG_ERROR, OUTAGE_SUSPECTED, OTHER
    tools_to_call: list[str]
    notes: str


class ResearchReport(BaseModel):
    """Output from ResearchAgent."""
    docs_results: list[dict] = Field(default_factory=list)
    incident_results: list[dict] = Field(default_factory=list)
    status_results: list[dict] = Field(default_factory=list)
    summary: str = ""


class ActionOutput(BaseModel):
    """Output from ActionAgent."""
    customer_reply: str
    internal_note: str


class SupervisorOutput(BaseModel):
    """Output from SupervisorAgent."""
    approved: bool
    final_customer_reply: str
    final_internal_note: str
    review_notes: str


class TraceEvent(BaseModel):
    """A single event in a trace."""
    timestamp: str
    agent_name: Optional[str] = None
    step_type: str  # LLM_CALL, TOOL_CALL, TOOL_RESULT, AGENT_OUTPUT
    detail: str
    payload: Optional[dict] = None

    @staticmethod
    def create(
        agent_name: Optional[str],
        step_type: str,
        detail: str,
        payload: Optional[dict] = None
    ) -> "TraceEvent":
        """Create a new TraceEvent with current timestamp."""
        return TraceEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            agent_name=agent_name,
            step_type=step_type,
            detail=detail,
            payload=payload
        )


class Trace(BaseModel):
    """Complete trace of a ticket processing flow."""
    trace_id: str
    ticket_id: str
    events: list[TraceEvent] = Field(default_factory=list)

    def add_event(
        self,
        agent_name: Optional[str],
        step_type: str,
        detail: str,
        payload: Optional[dict] = None
    ):
        """Add an event to this trace."""
        event = TraceEvent.create(agent_name, step_type, detail, payload)
        self.events.append(event)
