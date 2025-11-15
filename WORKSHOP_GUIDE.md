# AI Support Copilot - Complete Workshop Guide

**Workshop Duration**: 3 hours
**Level**: Intermediate to Advanced
**Prerequisites**: Python, basic LLM knowledge, async/await concepts

---

## Table of Contents

1. [Learning Objectives](#learning-objectives)
2. [Core Concepts](#core-concepts)
3. [Architecture Deep Dive](#architecture-deep-dive)
4. [Project Structure Explained](#project-structure-explained)
5. [Component Walkthroughs](#component-walkthroughs)
6. [Data Flow Analysis](#data-flow-analysis)
7. [Workshop Execution Plan](#workshop-execution-plan)
8. [Common Issues & Solutions](#common-issues--solutions)
9. [Extension Ideas](#extension-ideas)

---

## Learning Objectives

By the end of this workshop, we will understand:

### Conceptual Understanding
- **Model Context Protocol (MCP)**: Why it exists, how it standardizes LLM-tool integration
- **Multi-Agent Systems**: When and why to break AI systems into multiple specialized agents
- **Observability**: How to trace and debug complex AI workflows
- **Multimodal AI**: Integrating text and vision in LLM applications

### Technical Skills
- Implementing an MCP server from scratch
- Building MCP clients that consume tools
- Designing agent-based architectures without frameworks
- Creating observable, traceable AI systems
- Structuring prompts for specialized agents
- Handling LLM JSON outputs reliably

### Practical Application
- Real-world use case: Support ticket automation
- Production considerations: error handling, logging, extensibility
- Synthetic data creation for testing AI systems

---

## Core Concepts

### 1. Model Context Protocol (MCP)

#### What Problem Does MCP Solve?

**Before MCP:**
```
LLM Application A → Custom Tool Integration → Database
LLM Application B → Different Tool Integration → Same Database
LLM Application C → Yet Another Integration → Same Database
```

Every application reinvents tool integration. No standardization.

**With MCP:**
```
LLM Applications → MCP Client → MCP Server → Tools
                                    ├── Database
                                    ├── APIs
                                    └── File Systems
```

MCP provides a **standard protocol** for:
- Tool discovery (what tools are available?)
- Tool invocation (how to call them?)
- Result formatting (what comes back?)

#### MCP Architecture Components

```
┌─────────────────────────────────────────────────┐
│              MCP Host (Client)                  │
│  - Discovers available tools                    │
│  - Sends tool call requests                     │
│  - Receives structured responses                │
└────────────┬────────────────────────────────────┘
             │ MCP Protocol (stdio, HTTP, etc.)
┌────────────┴────────────────────────────────────┐
│              MCP Server                         │
│  - Registers tools with schemas                 │
│  - Handles tool invocations                     │
│  - Returns structured results                   │
└────────────┬────────────────────────────────────┘
             │
    ┌────────┴────────┬──────────┐
    ▼                 ▼          ▼
  Tool A           Tool B     Tool C
```

**Key Insight**: The LLM doesn't directly call tools. The host application:
1. Gets tool descriptions from MCP server
2. Asks LLM to decide which tools to use
3. Invokes tools via MCP
4. Feeds results back to LLM

### 2. Multi-Agent Orchestration

#### Why Multiple Agents?

**Single-Agent Problem:**
```python
# One prompt tries to do everything
prompt = """
1. Analyze this ticket
2. Search for solutions
3. Write customer reply
4. Write internal note
5. Check for safety issues
6. Approve or reject
"""
```

**Issues:**
- Cognitive overload for the LLM
- Hard to debug which step failed
- Can't iterate on individual steps
- Mixing concerns (analysis vs. safety)

**Multi-Agent Solution:**
```python
TriageAgent → ResearchAgent → ActionAgent → SupervisorAgent
   (What?)      (Context?)      (Draft)       (Review)
```

**Benefits:**
- **Separation of concerns**: Each agent has one job
- **Specialized prompts**: Optimized for specific tasks
- **Iterative improvement**: Fix one agent without touching others
- **Observable**: See exactly where things go wrong
- **Composable**: Add/remove agents as needed

#### Agent Design Principles

Each agent should:
1. **Have a single responsibility**
   - Triage: Classify and plan
   - Research: Gather information
   - Action: Create outputs
   - Supervisor: Quality control

2. **Have clear inputs and outputs**
   ```python
   def run(self, inputs: TypedInput) -> TypedOutput:
       # Well-defined contract
   ```

3. **Be stateless**
   - No hidden state between calls
   - All context passed explicitly

4. **Be testable**
   - Mock inputs, verify outputs
   - No side effects during core logic

### 3. Observability in AI Systems

#### Why Observability Matters

AI systems are **non-deterministic**:
- Same input ≠ same output
- Hard to reproduce bugs
- Complex multi-step flows

**Without observability:**
```
Input → [BLACK BOX] → Wrong Output
         (What happened??)
```

**With observability:**
```
Input → [Triage] → [Research] → [Action] → [Supervisor] → Output
         ✓ 0.3s    ✓ 1.2s      ✗ ERROR     ⊗ SKIPPED

Trace ID: abc-123
Step 3 failed: JSON parse error in ActionAgent
Prompt: "Draft customer reply..."
Response: "Sure, I'll help with that..." (non-JSON!)
```

#### Trace Structure

```python
Trace:
  trace_id: "uuid"
  ticket_id: "TICKET-001"
  events: [
    {
      timestamp: "2025-11-14T10:00:00Z",
      agent_name: "TriageAgent",
      step_type: "LLM_CALL",
      detail: "Analyzing ticket",
      payload: {prompt_tokens: 150, ...}
    },
    {
      timestamp: "2025-11-14T10:00:02Z",
      agent_name: "TriageAgent",
      step_type: "AGENT_OUTPUT",
      detail: "Classified as KNOWN_ISSUE",
      payload: {issue_type: "KNOWN_ISSUE", ...}
    },
    ...
  ]
```

**Benefits:**
- **Debugging**: Find exactly where things failed
- **Performance**: Measure time per step
- **Cost tracking**: Count LLM tokens used
- **Quality**: Replay and analyze edge cases

---

## Architecture Deep Dive

### System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    SUPPORT COPILOT HOST                      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               ORCHESTRATOR                            │  │
│  │  - Manages agent lifecycle                            │  │
│  │  - Coordinates data flow                              │  │
│  │  - Handles MCP client connection                      │  │
│  └───┬──────────────────────────────────────────────────┘  │
│      │                                                      │
│      ├─▶ TriageAgent                                       │
│      │    Input: Ticket                                    │
│      │    Output: TriagePlan                               │
│      │    LLM Call: Classify & plan                        │
│      │                                                      │
│      ├─▶ ResearchAgent                                     │
│      │    Input: Ticket, TriagePlan                        │
│      │    Output: ResearchReport                           │
│      │    MCP Calls: search docs, incidents, status        │
│      │                                                      │
│      ├─▶ ActionAgent                                       │
│      │    Input: Ticket, Plan, Report                      │
│      │    Output: ActionOutput                             │
│      │    LLM Call: Draft replies                          │
│      │                                                      │
│      └─▶ SupervisorAgent                                   │
│           Input: All previous outputs                       │
│           Output: SupervisorOutput (final)                  │
│           LLM Call: Review & approve                        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │             MCP CLIENT                                │  │
│  │  - Connects to MCP server                             │  │
│  │  - Exposes simple Python API                          │  │
│  └─────────────────┬────────────────────────────────────┘  │
└────────────────────┼─────────────────────────────────────────┘
                     │
                     │ stdio transport
                     │
┌────────────────────┼─────────────────────────────────────────┐
│                    ▼      MCP SERVER                        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           TOOL REGISTRY                               │  │
│  │                                                        │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ support_docs.search                            │  │  │
│  │  │  - Loads markdown runbooks                     │  │  │
│  │  │  - Keyword search with scoring                 │  │  │
│  │  │  - Returns top N matches                       │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  │                                                      │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ incidents.search                               │  │  │
│  │  │  - Loads incident JSON                         │  │  │
│  │  │  - Filters by status                           │  │  │
│  │  │  - Matches tags and text                       │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  │                                                      │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ status.check                                   │  │  │
│  │  │  - Loads status JSON                           │  │  │
│  │  │  - Lookup by service name                      │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────┘ │
│                                                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           DATA LAYER                                  │  │
│  │  - data/docs/runbooks/*.md                            │  │
│  │  - data/incidents/incidents.json                      │  │
│  │  - data/status/status.json                            │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Component Interactions

#### Sequence Diagram: Processing a Ticket

```
User    CLI         Orchestrator    Triage    Research    MCP      Action    Supervisor
 │       │               │            │          │        Server     │          │
 │──1───▶│               │            │          │          │        │          │
 │       │──2──────────▶ │            │          │          │        │          │
 │       │               │──3───────▶ │          │          │        │          │
 │       │               │            │─▶4-LLM   │          │        │          │
 │       │               │◀──Plan────│           │          │        │          │
 │       │               │──5──────────────────▶ │          │        │          │
 │       │               │                       │──6──────▶│        │          │
 │       │               │                       │◀─Results─│        │          │
 │       │               │                       │──6──────▶│        │          │
 │       │               │                       │◀─Results─│        │          │
 │       │               │◀──Report───────────── │          │        │          │
 │       │               │──7─────────────────────────────▶ │        │          │
 │       │               │                                  │──LLM───│          │
 │       │               │◀──ActionOutput─────────────────  │        │          │
 │       │               │──8─────────────────────────────────────────────────▶ │          
 │       │               │                                           │──LLM─────│
 │       │               │◀──SupervisorOutput──────────────────────  │          │
 │       │◀─Result───────│                                           │          │
 │◀──9──│                │                                           │          │

Steps:
1. User submits ticket via CLI
2. CLI calls orchestrator.run_ticket_flow()
3. Orchestrator invokes TriageAgent
4. Triage calls LLM to classify ticket
5. Orchestrator invokes ResearchAgent with plan
6. Research calls MCP tools (multiple times)
7. Orchestrator invokes ActionAgent
8. Orchestrator invokes SupervisorAgent
9. CLI displays final output to user
```

---

## Project Structure Explained

### Directory Layout

```
support-copilot/
│
├── data/                           # All synthetic data
│   ├── docs/runbooks/             # Markdown documentation
│   │   ├── export_to_csv_errors.md
│   │   ├── authentication_failures.md
│   │   ├── workspace_permissions_issues.md
│   │   └── logs_inspection_guide.md
│   │
│   ├── incidents/                  # Incident database
│   │   └── incidents.json         # 4 sample incidents
│   │
│   ├── status/                     # Service status
│   │   └── status.json            # 4 service statuses
│   │
│   └── tickets/                    # Sample tickets
│       ├── samples.json           # 4 predefined tickets
│       └── screenshots/           # (Optional) ticket screenshots
│
├── support_mcp_server/            # MCP Server implementation
│   ├── __init__.py
│   ├── __main__.py                # Entry point
│   ├── config.py                  # Paths and constants
│   ├── server.py                  # Main MCP server
│   ├── tools_support_docs.py      # Doc search tool
│   ├── tools_incidents.py         # Incident search tool
│   └── tools_status.py            # Status check tool
│
├── support_copilot_host/          # Host application
│   ├── __init__.py
│   ├── __main__.py                # CLI entry point
│   ├── config.py                  # LLM and MCP settings
│   ├── models.py                  # Pydantic data models
│   ├── llm_client.py              # OpenAI wrapper
│   ├── mcp_client.py              # MCP client wrapper
│   ├── observability.py           # Tracing utilities
│   ├── agents.py                  # 4 agent classes
│   ├── orchestrator.py            # Main flow coordination
│   ├── cli.py                     # Typer CLI
│   └── examples.py                # Workshop demos
│
├── tests/                          # (Optional) Test files
│
├── traces/                         # Generated trace logs
│   └── trace.jsonl                # JSONL trace output
│
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
├── README.md                       # Quick start guide
├── prd.md                          # Product requirements
├── tasks.md                        # Implementation tasks
└── verify_setup.py                 # Setup validation
```

### Why This Structure?

**Separation of Concerns:**
- `data/` - Pure data, no code
- `support_mcp_server/` - Server logic, isolated
- `support_copilot_host/` - Client logic, isolated
- Easy to test each component independently

**Discoverability:**
- Clear naming (`tools_*.py` for tools)
- Logical grouping (all agents in `agents.py`)
- Documentation alongside code

**Extensibility:**
- Add new tool? Create `tools_newfeature.py`
- Add new agent? Extend `agents.py`
- Add new data? Drop files in `data/`

---

## Component Walkthroughs

### 1. MCP Server (`support_mcp_server/`)

#### 1.1 Configuration (`config.py`)

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RUNBOOKS_DIR = BASE_DIR / "data" / "docs" / "runbooks"
INCIDENTS_PATH = BASE_DIR / "data" / "incidents" / "incidents.json"
STATUS_PATH = BASE_DIR / "data" / "status" / "status.json"
DEFAULT_MAX_RESULTS = 3
```

**Key Points:**
- Uses `pathlib.Path` for cross-platform compatibility
- Relative paths from server location
- Single source of truth for data locations


---

#### 1.2 Support Docs Tool (`tools_support_docs.py`)

**Architecture:**

```python
class RunbookDatabase:
    def __init__(self):
        self.runbooks = []    # In-memory cache
        self._load_runbooks()  # Load at startup

    def _load_runbooks(self):
        # Load all .md files from runbooks directory
        for md_file in RUNBOOKS_DIR.glob("*.md"):
            # Parse and store

    def search(self, query: str, max_results: int) -> list[dict]:
        # Tokenize query
        # Score each runbook
        # Return top N
```

**Search Algorithm:**

```python
def search(self, query: str, max_results: int):
    query_tokens = query.lower().split()
    results = []

    for runbook in self.runbooks:
        # Create searchable text
        searchable = f"{runbook['title']} {runbook['body']}".lower()

        # Count occurrences of each token
        score = sum(searchable.count(token) for token in query_tokens)

        if score > 0:
            # Extract snippet around first match
            snippet = self._extract_snippet(runbook, query_tokens)

            results.append({
                "title": runbook["title"],
                "snippet": snippet,
                "path": runbook["path"],
                "score": score
            })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:max_results]
```

**Why This Approach?**
- **Simple**: No external dependencies (no vector DB needed)
- **Fast**: Everything in memory
- **Good enough**: For small datasets, keyword search works well
- **Educational**: we can understand and modify easily

**Production Considerations:**
In real systems, you'd use:
- Embedding-based search (e.g., with sentence-transformers)
- Vector database (e.g., Pinecone, Weaviate)
- BM25 for better keyword ranking


---

#### 1.3 Incidents Tool (`tools_incidents.py`)

**Key Difference from Docs Tool:**

```python
def search(self, query: str, max_results: int, status_filter: list[str]):
    query_tokens = set(query.lower().split())
    results = []

    for incident in self.incidents:
        # Status filtering
        if status_filter and incident["status"] not in status_filter:
            continue  # Skip this incident

        # Search across title, summary, AND tags
        searchable = " ".join([
            incident["title"].lower(),
            incident["summary"].lower(),
            " ".join(incident["tags"])
        ])

        # Token matching
        score = sum(1 for token in query_tokens if token in searchable)

        if score > 0:
            # Find which tags matched
            matched_tags = [
                tag for tag in incident["tags"]
                if tag.lower() in query_tokens
            ]

            results.append({
                "incident_id": incident["incident_id"],
                "title": incident["title"],
                "status": incident["status"],
                "summary": incident["summary"],
                "matched_tags": matched_tags,  # Extra metadata
                "score": score
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:max_results]
```

**Features:**
1. **Filtering**: Can filter by status (e.g., only active incidents)
2. **Tag matching**: Highlights which tags matched the query
3. **Structured data**: Returns full incident objects

**Use Case:**
```python
# Find active export-related incidents
incidents.search(
    query="export timeout csv",
    status_filter=["Investigating", "Mitigating"]
)
```

---

#### 1.4 Status Tool (`tools_status.py`)

**Simplest Tool:**

```python
class StatusDatabase:
    def __init__(self):
        self.services = {}  # Dict for O(1) lookup
        self._load_status()

    def _load_status(self):
        with open(STATUS_PATH) as f:
            services_list = json.load(f)

        # Convert list to dict for fast lookup
        for service in services_list:
            service_name = service["service_name"].lower()
            self.services[service_name] = service

    def check(self, service_name: str) -> dict:
        service_name_lower = service_name.lower()

        if service_name_lower in self.services:
            return self.services[service_name_lower]
        else:
            return {
                "service_name": service_name,
                "status": "Unknown",
                "notes": f"Service '{service_name}' not found."
            }
```

**Design Decisions:**
- **Case-insensitive**: `export_service` == `Export_Service`
- **Graceful degradation**: Returns "Unknown" instead of error
- **Dict storage**: O(1) lookup vs O(n) list search

**Teaching Moment:**
Discuss data structure choice: Why dict over list? When would you use a list?

---

#### 1.5 MCP Server Main (`server.py`)

**Tool Registration:**

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

app = Server("support-mcp-server")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """Tell clients what tools are available"""
    return [
        Tool(
            name="support_docs.search",
            description="Search internal support documentation",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        ),
        # ... other tools
    ]
```

**Tool Invocation:**

```python
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls from clients"""

    if name == "support_docs.search":
        query = arguments.get("query")
        max_results = arguments.get("max_results", 3)

        # Call our search function
        result = search_support_docs(query, max_results)

        # Return as JSON text
        import json
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    elif name == "incidents.search":
        # Similar handling
        pass

    # ... other tools
```

**Key Concepts:**

1. **Async/Await**: MCP uses asyncio
   - `async def` for all handlers
   - Allows concurrent tool calls

2. **JSON Schema**: Tool inputs are validated
   - `inputSchema` describes expected arguments
   - MCP validates before calling handler

3. **Text Content**: Results are JSON strings
   - Could be other types (images, etc.)
   - For this project, JSON is sufficient

**Server Startup:**

```python
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

**Transport: stdio**
- Server reads from stdin, writes to stdout
- Client launches server as subprocess
- Bidirectional communication via pipes

**Alternative transports:**
- HTTP/SSE (for remote servers)
- WebSockets (for browser clients)

**Teaching Exercise:**
"What are pros/cons of stdio vs HTTP for tool servers?"

---

### 2. Host Application (`support_copilot_host/`)

#### 2.1 Data Models (`models.py`)

**Why Pydantic?**

```python
from pydantic import BaseModel

class Ticket(BaseModel):
    id: str
    description: str
    screenshot_path: Optional[str] = None
    log_snippet: Optional[str] = None
```

**Benefits:**
- **Type validation**: Auto-validates at runtime
- **IDE support**: Autocomplete and type checking
- **Serialization**: `.model_dump()` for JSON
- **Documentation**: Self-documenting code

**Agent Data Flow:**

```python
Ticket
  ↓
TriagePlan
  ├── issue_summary: str
  ├── issue_type: str
  ├── tools_to_call: list[str]
  └── notes: str
  ↓
ResearchReport
  ├── docs_results: list[dict]
  ├── incident_results: list[dict]
  ├── status_results: list[dict]
  └── summary: str
  ↓
ActionOutput
  ├── customer_reply: str
  └── internal_note: str
  ↓
SupervisorOutput
  ├── approved: bool
  ├── final_customer_reply: str
  ├── final_internal_note: str
  └── review_notes: str
```

**Trace Models:**

```python
class TraceEvent(BaseModel):
    timestamp: str
    agent_name: Optional[str]
    step_type: str  # LLM_CALL, TOOL_CALL, TOOL_RESULT, AGENT_OUTPUT
    detail: str
    payload: Optional[dict] = None

    @staticmethod
    def create(agent_name, step_type, detail, payload=None):
        return TraceEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            agent_name=agent_name,
            step_type=step_type,
            detail=detail,
            payload=payload
        )

class Trace(BaseModel):
    trace_id: str
    ticket_id: str
    events: list[TraceEvent] = []

    def add_event(self, agent_name, step_type, detail, payload=None):
        event = TraceEvent.create(agent_name, step_type, detail, payload)
        self.events.append(event)
```

**Design Pattern: Builder**
- `Trace` is mutable (collects events over time)
- `TraceEvent.create()` factory method ensures consistent timestamps
- `Trace.add_event()` convenience method

---

#### 2.2 LLM Client (`llm_client.py`)

**Vision Support:**

```python
def chat_with_vision(
    system_prompt: str,
    messages: list[dict],
    image_path: Optional[str] = None
) -> str:
    # Build messages array
    api_messages = [{"role": "system", "content": system_prompt}]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # If user message + image provided, attach image
        if role == "user" and image_path and Path(image_path).exists():
            encoded_image = _encode_image(image_path)
            api_messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": content},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    }
                ]
            })
            image_path = None  # Only attach once
        else:
            api_messages.append({"role": role, "content": content})

    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=api_messages
    )

    return response.choices[0].message.content
```

**Key Points:**
1. **Content array**: For multimodal, content is list of parts
2. **Base64 encoding**: Images sent as data URLs
3. **Single attachment**: Image only attached to first user message

**JSON Parsing with Retry:**

```python
def call_llm_for_json(system_prompt, messages, image_path=None) -> dict:
    # Add instruction to output JSON
    enhanced_prompt = f"""{system_prompt}

IMPORTANT: Respond with valid JSON only. No markdown, no extra text."""

    text_response = chat_with_vision(enhanced_prompt, messages, image_path)

    try:
        # Try to parse
        cleaned = text_response.strip()
        if cleaned.startswith("```"):
            # Remove markdown code blocks
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])

        return json.loads(cleaned)

    except json.JSONDecodeError:
        # Retry with explicit instruction
        retry_messages = messages + [
            {"role": "assistant", "content": text_response},
            {"role": "user", "content": "Reformat as valid JSON only."}
        ]

        retry_response = chat_with_vision(
            enhanced_prompt,
            retry_messages,
            temperature=0.0  # Deterministic retry
        )

        # Parse retry
        cleaned = retry_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])

        return json.loads(cleaned)
```

**Why This Approach?**
- **LLMs are unpredictable**: Sometimes output markdown wrappers
- **Retry helps**: Second attempt usually succeeds
- **Lower temperature on retry**: More deterministic
- **User-friendly**: Handles common LLM quirks automatically

**Production Alternative:**
Use OpenAI's JSON mode:
```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    response_format={"type": "json_object"}
)
```

---

#### 2.3 MCP Client (`mcp_client.py`)

**Async Context Management:**

```python
class SupportMCPClient:
    def __init__(self):
        self.session = None
        self._context = None

    async def connect(self):
        # Create server parameters
        server_params = StdioServerParameters(
            command="python",
            args=["-m", "support_mcp_server.server"]
        )

        # Create stdio client context
        self._context = stdio_client(server_params)
        read_stream, write_stream = await self._context.__aenter__()

        # Create session
        self.session = ClientSession(read_stream, write_stream)
        await self.session.__aenter__()

        # Initialize connection
        await self.session.initialize()

    async def disconnect(self):
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self._context:
            await self._context.__aexit__(None, None, None)
```

**Tool Call Abstraction:**

```python
async def call_support_docs_search(self, query: str, max_results: int = 3):
    result = await self.session.call_tool(
        "support_docs.search",
        {"query": query, "max_results": max_results}
    )

    # Parse text content as JSON
    text_content = result.content[0].text
    parsed = json.loads(text_content)

    return parsed.get("results", [])
```

**Why Wrap MCP Calls?**
- **Type safety**: Returns Python dicts, not MCP types
- **Simplicity**: Hides MCP protocol details
- **Testability**: Can mock these methods easily

**Teaching Moment:**
Show raw MCP call vs wrapped call. Discuss abstraction layers.

---

#### 2.4 Observability (`observability.py`)

**Logging Helpers:**

```python
def log_llm_call(trace: Trace, agent_name: str, detail: str, payload: dict):
    trace.add_event(
        agent_name=agent_name,
        step_type="LLM_CALL",
        detail=detail,
        payload=_truncate_payload(payload)
    )

def log_tool_call(trace: Trace, tool_name: str, input_summary: dict):
    trace.add_event(
        agent_name=None,
        step_type="TOOL_CALL",
        detail=f"Calling tool: {tool_name}",
        payload=_truncate_payload(input_summary)
    )
```

**Payload Truncation:**

```python
def _truncate_payload(payload: dict, max_length: int = 500) -> dict:
    if not payload:
        return None

    serialized = json.dumps(payload)
    if len(serialized) > max_length:
        return {
            "_truncated": True,
            "preview": serialized[:max_length] + "..."
        }

    return payload
```

**Why Truncate?**
- Payloads can be huge (full prompts, responses)
- JSONL files get massive
- Truncation keeps logs manageable
- Still shows enough for debugging

**Rendering Traces:**

```python
def render_trace_to_stdout(trace: Trace):
    table = Table(title=f"Trace: {trace.trace_id}")

    table.add_column("Timestamp", style="cyan")
    table.add_column("Agent", style="magenta")
    table.add_column("Step Type", style="green")
    table.add_column("Detail", style="white")

    for event in trace.events:
        time_part = event.timestamp.split("T")[1][:12]
        table.add_row(
            time_part,
            event.agent_name or "-",
            event.step_type,
            event.detail[:80] + ("..." if len(event.detail) > 80 else "")
        )

    console.print(table)
```

**Output Example:**
```
╭──────────────────────────────────────────────────────────────╮
│                     Trace: abc-123-def                       │
├─────────────┬────────────┬────────────┬──────────────────────┤
│ Timestamp   │ Agent      │ Step Type  │ Detail               │
├─────────────┼────────────┼────────────┼──────────────────────┤
│ 10:00:01.23 │ Triage     │ LLM_CALL   │ Analyzing ticket     │
│ 10:00:02.45 │ Triage     │ AGENT_OUT  │ Classified as KNOWN  │
│ 10:00:03.12 │ -          │ TOOL_CALL  │ Calling: docs.search │
│ 10:00:04.56 │ -          │ TOOL_RES   │ Tool result: 3 docs  │
╰─────────────┴────────────┴────────────┴──────────────────────╯
```

---

#### 2.5 Agents (`agents.py`)

##### TriageAgent

**Purpose**: Analyze ticket and create execution plan

**Prompt Design:**

```python
system_prompt = """You are a support ticket triage specialist.

Your task is to:
1. Summarize the issue
2. Classify the issue type
3. Decide which tools to use

Available tools:
- support_docs.search
- incidents.search
- status.check

Issue types:
- KNOWN_ISSUE
- POSSIBLE_BUG
- CONFIG_ERROR
- OUTAGE_SUSPECTED
- OTHER

Output JSON:
{
  "issue_summary": "...",
  "issue_type": "...",
  "tools_to_call": ["..."],
  "notes": "..."
}"""
```

**Why This Works:**
- **Clear instructions**: Numbered steps
- **Constrained output**: Specific types, not free-form
- **Examples implicit**: Tool names match exactly what MCP provides
- **Structured**: JSON forces structure

**Implementation:**

```python
async def run(self, ticket: Ticket, trace: Trace) -> TriagePlan:
    user_content = f"""Triage this ticket:

Description: {ticket.description}

{f'Logs: {ticket.log_snippet}' if ticket.log_snippet else 'No logs.'}

{f'Screenshot: {ticket.screenshot_path}' if ticket.screenshot_path else 'No screenshot.'}"""

    messages = [{"role": "user", "content": user_content}]

    log_llm_call(trace, "TriageAgent", "Analyzing ticket")

    result = call_llm_for_json(
        system_prompt=system_prompt,
        messages=messages,
        image_path=ticket.screenshot_path  # Vision support!
    )

    plan = TriagePlan(**result)

    log_agent_output(trace, "TriageAgent", f"Classified as {plan.issue_type}")

    return plan
```

**Teaching Points:**
1. Screenshot automatically included if present
2. Logging before and after LLM call
3. Pydantic validation of LLM output
4. Return type is strongly typed

---

##### ResearchAgent

**Purpose**: Execute tools and gather context

**Dynamic Service Inference:**

```python
def _infer_service_names(self, ticket: Ticket) -> list[str]:
    """Infer which services to check based on ticket content"""
    services = set()
    content = f"{ticket.description} {ticket.log_snippet or ''}".lower()

    if "export" in content or "csv" in content:
        services.add("export_service")
    if "auth" in content or "login" in content:
        services.add("auth_service")
    if "workspace" in content or "permission" in content:
        services.add("workspace_service")

    return list(services) if services else ["export_service"]
```

**Tool Execution:**

```python
async def run(self, ticket, plan, mcp_client, trace):
    docs_results = []
    incident_results = []
    status_results = []

    query = ticket.description

    for tool_name in plan.tools_to_call:
        if tool_name == "support_docs.search":
            log_tool_call(trace, tool_name, {"query": query})

            results = await mcp_client.call_support_docs_search(query)
            docs_results = results

            log_tool_result(trace, tool_name, {"count": len(results)})

        elif tool_name == "incidents.search":
            log_tool_call(trace, tool_name, {"query": query})

            status_filter = ["Investigating", "Mitigating"] \
                if plan.issue_type == "OUTAGE_SUSPECTED" else None

            results = await mcp_client.call_incidents_search(
                query, status_filter=status_filter
            )
            incident_results = results

            log_tool_result(trace, tool_name, {"count": len(results)})

        elif tool_name == "status.check":
            service_names = self._infer_service_names(ticket)

            for service_name in service_names:
                log_tool_call(trace, tool_name, {"service": service_name})

                result = await mcp_client.call_status_check(service_name)
                status_results.append(result)

                log_tool_result(trace, tool_name, {"status": result["status"]})

    summary = self._create_summary(docs_results, incident_results, status_results)

    return ResearchReport(
        docs_results=docs_results,
        incident_results=incident_results,
        status_results=status_results,
        summary=summary
    )
```

**Key Design Decisions:**

1. **Conditional filtering**: Outage tickets only search active incidents
2. **Multiple status checks**: Checks multiple services based on heuristics
3. **Summary creation**: Deterministic text summary (no LLM needed here)
4. **Comprehensive logging**: Every tool call logged

**Alternative Approach:**
Could use LLM to decide which services to check:
```python
service_decision = await llm_client.call_llm_for_json(
    "Which services should we check for this ticket?",
    [{"role": "user", "content": ticket.description}]
)
```

**Trade-off**: Simple heuristics vs LLM call (cost, latency)

---

##### ActionAgent

**Purpose**: Draft customer-facing and internal responses

**Prompt Engineering:**

```python
system_prompt = """You are a support engineer drafting responses.

Create:
1. Customer-facing reply: empathetic, clear, actionable
2. Internal note: technical details, next steps

Customer reply guidelines:
- Be empathetic and professional
- Explain in simple terms
- Provide actionable steps
- DO NOT expose internal IDs or implementation details
- If ongoing incident, acknowledge without alarm

Internal note guidelines:
- Include error codes and incident IDs
- Reference runbooks by name
- Suggest escalation if needed

Output JSON:
{
  "customer_reply": "...",
  "internal_note": "..."
}"""
```

**Context Assembly:**

```python
user_content = f"""Ticket: {ticket.description}

Triage:
- Type: {plan.issue_type}
- Summary: {plan.issue_summary}

Research:
{report.summary}

Full data:
- Docs: {json.dumps([{'title': d['title']} for d in report.docs_results])}
- Incidents: {json.dumps([{'id': i['incident_id'], 'status': i['status']}
                          for i in report.incident_results])}
- Status: {json.dumps(report.status_results)}

Draft customer reply and internal note."""
```

**Why This Format?**
- **Hierarchical**: Important info first (triage), details later
- **Summarized**: Summary + full data (LLM can drill down if needed)
- **Structured**: JSON for consistency

**Teaching Moment:**
Discuss prompt engineering:
- Few-shot vs zero-shot
- Instruction clarity
- Context ordering (recency bias)

---

##### SupervisorAgent

**Purpose**: Quality control and safety check

**Prompt Focus:**

```python
system_prompt = """You are a senior support lead reviewing responses.

Check for:
1. Hallucinations: Claims not supported by research
2. Information leakage: Internal IDs in customer message
3. Missing context: Relevant incidents/docs not mentioned
4. Tone: Professional and clear

If issues found:
- Set approved = false
- Edit the messages
- Explain in review_notes

If good:
- Set approved = true
- Copy messages as-is
- Note "No issues found"

Output JSON:
{
  "approved": true/false,
  "final_customer_reply": "...",
  "final_internal_note": "...",
  "review_notes": "..."
}"""
```

**Implementation:**

```python
async def run(self, ticket, plan, report, action, trace):
    user_content = f"""Original ticket: {ticket.description}

Research: {report.summary}

Drafted customer reply:
{action.customer_reply}

Drafted internal note:
{action.internal_note}

Review and approve or edit."""

    log_llm_call(trace, "SupervisorAgent", "Reviewing outputs")

    result = call_llm_for_json(system_prompt, [{"role": "user", "content": user_content}])

    output = SupervisorOutput(**result)

    log_agent_output(
        trace,
        "SupervisorAgent",
        f"Review: {'Approved' if output.approved else 'Edited'}"
    )

    return output
```

**Why Supervisor Pattern?**
- **Safety**: Catches LLM mistakes before user sees them
- **Consistency**: Enforces quality standards
- **Auditability**: Review notes explain decisions
- **Flexibility**: Can add more checks (toxicity, etc.)

**Alternative Patterns:**
- **Constitutional AI**: Bake safety into system prompt
- **Rule-based**: Regex/keyword checks (faster, cheaper)
- **Human-in-loop**: Require human approval (slower)

---

#### 2.6 Orchestrator (`orchestrator.py`)

**Main Flow:**

```python
async def run_ticket_flow(ticket: Ticket) -> Tuple[SupervisorOutput, Trace]:
    # 1. Setup
    trace_id = str(uuid.uuid4())
    trace = Trace(trace_id=trace_id, ticket_id=ticket.id)

    # 2. Connect to MCP
    mcp_client = SupportMCPClient()
    await mcp_client.connect()

    try:
        # 3. Instantiate agents
        triage = TriageAgent()
        research = ResearchAgent()
        action = ActionAgent()
        supervisor = SupervisorAgent()

        # 4. Execute pipeline
        plan = await triage.run(ticket, trace)

        if plan.tools_to_call:
            report = await research.run(ticket, plan, mcp_client, trace)
        else:
            report = ResearchReport(summary="No research needed.")

        action_out = await action.run(ticket, plan, report, trace)

        supervisor_out = await supervisor.run(ticket, plan, report, action_out, trace)

        # 5. Return
        return supervisor_out, trace

    finally:
        # 6. Cleanup
        await mcp_client.disconnect()
```

**Design Principles:**

1. **Single Responsibility**: Just orchestration, no business logic
2. **Error Handling**: `try/finally` ensures cleanup
3. **Traceability**: Trace object passed to all agents
4. **Typed**: Clear input/output contracts

**Error Recovery:**

```python
try:
    await mcp_client.connect()
except Exception as e:
    # Log error to trace
    trace.add_event(None, "ERROR", f"MCP connection failed: {e}")
    # Create empty report and continue
    report = ResearchReport(summary="MCP unavailable")
```

**Why This Matters:**
- Graceful degradation
- System works even if MCP server down
- Trace shows what happened

---

#### 2.7 CLI (`cli.py`)

**Command Structure:**

```python
app = typer.Typer(help="Support Copilot Host")

@app.command()
def run_ticket(description: str, screenshot: str = None, log: str = None):
    """Process a custom ticket"""
    # Build ticket
    # Run flow
    # Display results

@app.command()
def run_sample(sample_id: str):
    """Process a predefined sample"""
    # Load from samples.json
    # Run flow
    # Display results

@app.command()
def list_samples():
    """Show available samples"""
    # Load samples.json
    # Print list
```

**Async Execution from Sync CLI:**

```python
def run_ticket(...):
    ticket = Ticket(...)

    # Typer is sync, but orchestrator is async
    supervisor, trace = asyncio.run(run_ticket_flow(ticket))

    # Display results
    console.print(Panel(supervisor.final_customer_reply, title="Customer Reply"))
```

**Why Typer?**
- **Intuitive**: Function arguments → CLI flags
- **Type hints**: Automatic validation
- **Help generation**: Auto-generates `--help`
- **Rich integration**: Works well with Rich for formatting

**Alternative:**
`argparse` (stdlib, more verbose) or `click` (similar to Typer)

---

## Data Flow Analysis

### End-to-End Flow Example

**Input: TICKET-001**

```json
{
  "id": "TICKET-001",
  "description": "Customer cannot export dashboard to CSV. Sees 'Export failed: timeout' after 30s.",
  "log_snippet": "2025-11-01T09:21:10Z ERR_EXPORT_TIMEOUT workspace_id=ws_1234 service=export_service"
}
```

**Step 1: TriageAgent**

Input: Ticket object

LLM Prompt (simplified):
```
Analyze this ticket:
Description: Customer cannot export dashboard to CSV...
Logs: ERR_EXPORT_TIMEOUT...

Output JSON with classification.
```

LLM Output:
```json
{
  "issue_summary": "User experiencing CSV export timeouts",
  "issue_type": "KNOWN_ISSUE",
  "tools_to_call": ["support_docs.search", "incidents.search", "status.check"],
  "notes": "Log shows ERR_EXPORT_TIMEOUT, likely related to export_service"
}
```

Trace Event:
```json
{
  "timestamp": "2025-11-14T10:00:01Z",
  "agent_name": "TriageAgent",
  "step_type": "AGENT_OUTPUT",
  "detail": "Classified as KNOWN_ISSUE, will use 3 tools"
}
```

---

**Step 2: ResearchAgent**

Input: Ticket, TriagePlan

Tool Call 1: `support_docs.search`
```python
query = "Customer cannot export dashboard to CSV timeout"
results = [
  {
    "title": "Export to CSV Errors",
    "snippet": "When users see 'Export failed: timeout'...",
    "path": "data/docs/runbooks/export_to_csv_errors.md",
    "score": 3
  }
]
```

Trace Event:
```json
{
  "timestamp": "2025-11-14T10:00:02Z",
  "agent_name": null,
  "step_type": "TOOL_CALL",
  "detail": "Calling tool: support_docs.search",
  "payload": {"query": "..."}
}
```

Tool Call 2: `incidents.search`
```python
query = "export timeout csv"
results = [
  {
    "incident_id": "INC-1001",
    "title": "Export to CSV timeouts for large workspaces",
    "status": "Mitigating",
    "summary": "Some customers experiencing timeouts...",
    "matched_tags": ["export", "csv", "timeout"]
  }
]
```

Tool Call 3: `status.check`
```python
# Inferred service: export_service
result = {
  "service_name": "export_service",
  "status": "Degraded",
  "notes": "Exports may time out for large workspaces..."
}
```

Output: ResearchReport
```python
ResearchReport(
  docs_results=[...],  # 1 doc found
  incident_results=[...],  # 1 incident found
  status_results=[...],  # 1 service checked
  summary="Found 1 doc, 1 incident (Mitigating), export_service is Degraded"
)
```

---

**Step 3: ActionAgent**

Input: Ticket, TriagePlan, ResearchReport

LLM Prompt (assembled):
```
Draft customer reply and internal note.

Ticket: Customer cannot export...
Triage: KNOWN_ISSUE
Research: Found doc "Export to CSV Errors", incident INC-1001 (Mitigating), export_service Degraded

Guidelines:
- Don't expose INC-1001 to customer
- Acknowledge issue
- Provide workaround
```

LLM Output:
```json
{
  "customer_reply": "Thank you for reporting this. We're aware of an issue causing export timeouts for large workspaces. Our team is actively working on a fix. In the meantime, please try exporting a smaller date range. We apologize for the inconvenience.",

  "internal_note": "Related to INC-1001 (Export to CSV timeouts). export_service is Degraded. Follows runbook 'Export to CSV Errors'. Customer has workspace ws_1234. If smaller exports don't work, escalate to engineering with correlation ID from logs."
}
```

---

**Step 4: SupervisorAgent**

Input: All previous outputs

LLM Prompt:
```
Review these responses:

Customer reply: "Thank you for reporting this..."
Internal note: "Related to INC-1001..."

Check: hallucinations, info leakage, missing context, tone
```

LLM Output:
```json
{
  "approved": true,
  "final_customer_reply": "Thank you for reporting this. We're aware of an issue...",
  "final_internal_note": "Related to INC-1001...",
  "review_notes": "No issues found. Customer reply doesn't expose incident ID. Tone is empathetic. Workaround provided."
}
```

---

**Final Output to User:**

```
╭─────────────────────────────────────────────────────╮
│                  Customer Reply                     │
├─────────────────────────────────────────────────────┤
│ Thank you for reporting this. We're aware of an     │
│ issue causing export timeouts for large             │
│ workspaces. Our team is actively working on a fix.  │
│ In the meantime, please try exporting a smaller     │
│ date range. We apologize for the inconvenience.     │
╰─────────────────────────────────────────────────────╯

╭─────────────────────────────────────────────────────╮
│                  Internal Note                      │
├─────────────────────────────────────────────────────┤
│ Related to INC-1001 (Export to CSV timeouts).       │
│ export_service is Degraded. Follows runbook        │
│ 'Export to CSV Errors'. Customer has workspace     │
│ ws_1234. If smaller exports don't work, escalate.  │
╰─────────────────────────────────────────────────────╯

Trace Summary:
  Trace ID: abc-123-def-456
  Total Events: 12
  Agents: TriageAgent, ResearchAgent, ActionAgent, SupervisorAgent
  LLM Calls: 4
  Tool Calls: 3
```

---

## Workshop Execution Plan

### Phase 0: Setup (Before Workshop)

**Preparation (1 day before):**

1. **Test Everything**
   ```bash
   python verify_setup.py
   python -m support_mcp_server.server &
   python -m support_copilot_host.cli run-sample --id TICKET-001
   ```

2. **Prepare Slides**
   - Architecture diagram
   - MCP concepts
   - Agent flow visualization
   - Code snippets (key functions)

3. **Environment Setup**
   - Share `requirements.txt` 
   - Provide OpenAI API keys 
   - Test on workshop network/machines

4. **Backup Plan**
   - Pre-recorded demo video
   - Screenshots of expected outputs
   - Sample trace files

---

### Phase 1: Introduction (20 minutes)

**Objective**: Set context and learning goals

**Agenda:**

1. **Problem Statement (5 min)**
   - Show example support ticket
   - Discuss manual process pain points
   - Introduce automation vision

2. **AI Landscape (5 min)**
   - Where does this fit? (not chatbot, not full automation)
   - Role of LLMs in workflows
   - Human + AI collaboration

3. **Today's Agenda (5 min)**
   - What we'll build
   - What concepts we'll learn
   - What you'll be able to do after

4. **Quick Demo (5 min)**
   - Run TICKET-001 end-to-end
   - Show customer reply, internal note, trace
   - "By end of workshop, you'll understand every line of code that made this happen"

---

### Phase 2: MCP Deep Dive (30 minutes)

**Objective**: Understand Model Context Protocol

**Activities:**

1. **MCP Concepts (10 min)**
   - Whiteboard: Before/After MCP
   - Protocol components: Server, Client, Tools
   - Transport: stdio, HTTP, WebSocket
   - Show MCP Python SDK docs

2. **Build a Simple Tool (15 min)**

   **Live Coding:**
   ```python
   # tools_hello.py
   def hello_tool(name: str) -> dict:
       return {"message": f"Hello, {name}!"}

   # Add to server.py
   Tool(
       name="hello",
       description="Say hello",
       inputSchema={"properties": {"name": {"type": "string"}}}
   )

   @app.call_tool()
   async def call_tool(name, arguments):
       if name == "hello":
           return [TextContent(text=json.dumps(hello_tool(arguments["name"])))]
   ```

   **Test:**
   ```bash
   # Terminal 1
   python -m support_mcp_server.server

   # Terminal 2
   python -c "
   from support_copilot_host.mcp_client import create_client
   import asyncio

   async def test():
       client = create_client()
       await client.connect()
       result = await client._call_tool('hello', {'name': 'Workshop'})
       print(result)
       await client.disconnect()

   asyncio.run(test())
   "
   ```

3. **Code Walkthrough (5 min)**
   - Open `tools_support_docs.py`
   - Explain `RunbookDatabase` class
   - Explain `search()` method
   - Discuss: "How would you improve this?"

---

### Phase 3: LLM Integration (30 minutes)

**Objective**: Understand LLM client patterns

**Activities:**

1. **OpenAI API Basics (10 min)**
   - Chat completions API
   - Messages format
   - System vs user vs assistant roles
   - Parameters: temperature, max_tokens

2. **Vision Support (10 min)**
   - Multimodal content arrays
   - Base64 image encoding
   - Use cases: screenshots, diagrams, receipts

   **Demo:**
   ```python
   from support_copilot_host.llm_client import chat_with_vision

   response = chat_with_vision(
       system_prompt="Describe what you see in the image.",
       messages=[{"role": "user", "content": "What's in this screenshot?"}],
       image_path="data/tickets/screenshots/export_timeout.png"
   )
   print(response)
   ```

3. **JSON Output Handling (10 min)**
   - Why JSON? (structured, parseable, type-safe)
   - Common LLM quirks (markdown wrappers, extra text)
   - Retry strategies

   **Live Coding:**
   Show `call_llm_for_json()` with intentional failure, then retry


---

### Phase 4: Single Agent Baseline (20 minutes)

**Objective**: See why multi-agent helps

**Activities:**

1. **Build Monolithic Version (15 min)**

   **Live Coding:**
   ```python
   # monolithic_agent.py
   async def process_ticket_single_agent(ticket: Ticket) -> str:
       prompt = f"""
       Analyze this ticket: {ticket.description}

       1. What's the issue?
       2. Search for solutions
       3. Draft a customer reply
       4. Check for safety issues

       Return everything as one response.
       """

       response = chat_with_vision(prompt, [])
       return response
   ```

   Run it on TICKET-001.

2. **Discuss Limitations (5 min)**
   - All-or-nothing: Can't rerun just one step
   - Hard to debug: Which part failed?
   - No observability: What did it search for?
   - Cognitive overload: Too many instructions


---

### Phase 5: Multi-Agent Refactor (40 minutes)

**Objective**: Build and understand the agent pipeline

**Activities:**

1. **Agent Design (10 min)**
   - Separation of concerns
   - Input/output contracts
   - Stateless design

   **Diagram on whiteboard:**
   ```
   Ticket → [Triage] → TriagePlan
                ↓
         [Research] → ResearchReport
                ↓
          [Action] → ActionOutput
                ↓
        [Supervisor] → SupervisorOutput
   ```

2. **Code Walkthrough: TriageAgent (10 min)**
   - Open `agents.py`
   - Explain `TriageAgent.run()`
   - Discuss prompt design
   - Show TriagePlan model

   **Run isolated:**
   ```python
   from support_copilot_host.agents import TriageAgent
   from support_copilot_host.models import Ticket, Trace
   import asyncio

   async def test_triage():
       ticket = Ticket(
           id="TEST",
           description="Export fails with timeout",
           log_snippet="ERR_EXPORT_TIMEOUT"
       )
       trace = Trace(trace_id="test", ticket_id="TEST")

       agent = TriageAgent()
       plan = await agent.run(ticket, trace)

       print(plan)
       print(trace.events)

   asyncio.run(test_triage())
   ```

3. **Code Walkthrough: ResearchAgent (10 min)**
   - Explain tool calling loop
   - Discuss service inference
   - Show MCP integration

   **Run isolated:** (same pattern as above)

4. **Code Walkthrough: ActionAgent & SupervisorAgent (10 min)**
   - Explain prompt assembly
   - Discuss safety checks in Supervisor
   - Show final output

---

### Phase 6: Observability & Debugging (30 minutes)

**Objective**: Learn to trace and debug AI systems

**Activities:**

1. **Trace Structure (10 min)**
   - Explain `Trace` and `TraceEvent` models
   - Show logging helpers
   - Discuss payload truncation

2. **Reading Traces (10 min)**

   **Demo:**
   ```bash
   # Run a ticket
   python -m support_copilot_host.cli run-sample --id TICKET-001

   # Show JSONL output
   cat traces/trace.jsonl | jq .

   # Show table output (already visible in CLI)
   ```

   **Walkthrough:**
   - Point out each step
   - Show timestamps (identify slow steps)
   - Show payloads (what was sent to LLM?)

3. **Debugging Exercise (10 min)**

   **Scenario:**
   "Supervisor is rejecting all responses. Why?"

   **Steps:**
   1. Run TICKET-003
   2. Check trace
   3. Find SupervisorAgent LLM_CALL event
   4. Check payload (prompt)
   5. Identify issue in prompt
   6. Fix and rerun


---

### Phase 7: Extensions & Production (30 minutes)

**Objective**: Discuss real-world deployment

**Activities:**

1. **Current Limitations (10 min)**
   - Keyword search (not semantic)
   - Sequential execution (not parallel)
   - No caching (repeated LLM calls)
   - No feedback loop (can't improve)

2. **Extension Ideas (15 min)**

   **Brainstorm:**
   - Vector search with embeddings
   - Parallel research (docs + incidents concurrently)
   - Caching layer (Redis for common queries)
   - Feedback collection (thumbs up/down)
   - Human-in-the-loop approval
   - Fine-tuning on historical tickets

   **Pick one and sketch design:**
   E.g., "How would you add vector search?"

   ```python
   from sentence_transformers import SentenceTransformer
   import faiss

   class VectorRunbookDatabase:
       def __init__(self):
           self.model = SentenceTransformer('all-MiniLM-L6-v2')
           self.index = faiss.IndexFlatL2(384)
           self._load_and_embed()

       def search(self, query, k=3):
           query_embedding = self.model.encode([query])
           distances, indices = self.index.search(query_embedding, k)
           return [self.runbooks[i] for i in indices[0]]
   ```

3. **Production Checklist (5 min)**
   - Error handling (retries, fallbacks)
   - Rate limiting (LLM API limits)
   - Monitoring (track costs, latency)
   - Security (API key management)
   - Scaling (async, queues)
   - Testing (unit, integration, E2E)


---

## Common Issues & Solutions

### Issue 1: MCP Server Won't Start

**Symptom:**
```
Error: ModuleNotFoundError: No module named 'mcp'
```

**Solution:**
```bash
pip install -r requirements.txt
```

**Prevention:**
Run `verify_setup.py` before workshop

---

### Issue 2: JSON Parse Errors from LLM

**Symptom:**
```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Cause:**
LLM output markdown wrapper or non-JSON text

**Solution:**
Already handled in `call_llm_for_json()` with retry logic

**If still failing:**
- Check prompt clarity
- Lower temperature (more deterministic)
- Use OpenAI JSON mode:
  ```python
  response_format={"type": "json_object"}
  ```

---

### Issue 3: Tool Returns Empty Results

**Symptom:**
ResearchAgent finds no docs/incidents

**Diagnosis:**
```python
# Add debug logging to tools_support_docs.py
def search(self, query, max_results):
    print(f"Searching for: {query}")
    print(f"Loaded {len(self.runbooks)} runbooks")

    query_tokens = query.lower().split()
    print(f"Query tokens: {query_tokens}")

    # ... rest of search

    print(f"Found {len(results)} results")
    return results
```

**Common causes:**
- Query too specific (no matches)
- Runbooks not loaded (check paths)
- Tokenization mismatch

---

### Issue 4: Slow Performance

**Symptom:**
Each ticket takes 30+ seconds

**Diagnosis:**
Check trace timestamps to find bottleneck

**Solutions:**
- **Parallel LLM calls**: Run Action + Supervisor in parallel (careful: Supervisor needs Action output)
- **Smaller models**: Use `gpt-4o-mini` instead of `gpt-4o`
- **Reduce max_tokens**: Lower from 2000 to 500
- **Cache MCP connection**: Don't reconnect each time

---

### Issue 5: OpenAI Rate Limits

**Symptom:**
```
RateLimitError: You exceeded your current quota
```

**Solutions:**
- **Tier upgrade**: Request higher quota from OpenAI
- **Retry with backoff**:
  ```python
  import time

  for attempt in range(3):
      try:
          response = client.chat.completions.create(...)
          break
      except RateLimitError:
          time.sleep(2 ** attempt)  # Exponential backoff
  ```
- **Local LLM**: Use Ollama for development
  ```python
  from ollama import Client
  client = Client(host='http://localhost:11434')
  ```

---

## Extension Ideas

### 1. Vector Search for Runbooks

**Goal**: Semantic search instead of keyword

**Implementation:**

```python
# Install: pip install sentence-transformers faiss-cpu

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorRunbookDatabase:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.runbooks = []
        self.embeddings = None
        self.index = None
        self._load_and_embed()

    def _load_and_embed(self):
        # Load runbooks
        for md_file in RUNBOOKS_DIR.glob("*.md"):
            with open(md_file) as f:
                content = f.read()
            self.runbooks.append({"title": ..., "body": content, "path": ...})

        # Generate embeddings
        texts = [r["body"] for r in self.runbooks]
        self.embeddings = self.model.encode(texts)

        # Build FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings.astype('float32'))

    def search(self, query: str, k: int = 3):
        # Embed query
        query_embedding = self.model.encode([query])

        # Search
        distances, indices = self.index.search(
            query_embedding.astype('float32'), k
        )

        # Return runbooks
        return [self.runbooks[i] for i in indices[0]]
```

**Benefits:**
- Understands synonyms ("export" ~ "download")
- Handles typos better
- Finds semantically similar content

**Trade-offs:**
- Requires ML dependencies
- Slower startup (embedding generation)
- More complex to debug

---

### 2. Parallel Research Agents

**Goal**: Speed up research by running tools concurrently

**Implementation:**

```python
import asyncio

async def run(self, ticket, plan, mcp_client, trace):
    # Create tasks for each tool
    tasks = []

    if "support_docs.search" in plan.tools_to_call:
        tasks.append(self._search_docs(ticket, mcp_client, trace))

    if "incidents.search" in plan.tools_to_call:
        tasks.append(self._search_incidents(ticket, plan, mcp_client, trace))

    if "status.check" in plan.tools_to_call:
        tasks.append(self._check_status(ticket, mcp_client, trace))

    # Run all in parallel
    results = await asyncio.gather(*tasks)

    # Unpack results
    docs_results = results[0] if len(results) > 0 else []
    incident_results = results[1] if len(results) > 1 else []
    status_results = results[2] if len(results) > 2 else []

    # ... create report
```

**Speed improvement:**
- Before: 3 seconds (1s per tool)
- After: 1 second (all parallel)

---

### 3. Feedback Loop

**Goal**: Learn from human corrections

**Implementation:**

```python
# models.py
class Feedback(BaseModel):
    trace_id: str
    rating: int  # 1-5
    corrected_customer_reply: Optional[str] = None
    corrected_internal_note: Optional[str] = None
    comments: str = ""

# feedback_db.py
class FeedbackDatabase:
    def save(self, feedback: Feedback):
        with open("data/feedback/feedback.jsonl", "a") as f:
            f.write(feedback.model_dump_json() + "\n")

    def load_all(self) -> list[Feedback]:
        # Load and parse all feedback

# cli.py
@app.command()
def give_feedback(trace_id: str, rating: int):
    """Provide feedback on a processed ticket"""
    feedback = Feedback(trace_id=trace_id, rating=rating)
    FeedbackDatabase().save(feedback)
    console.print("[green]Feedback saved![/green]")
```

**Usage:**
```bash
# After running a ticket
python -m support_copilot_host.cli give-feedback \
  --trace-id abc-123 \
  --rating 5
```

**Future:** Use feedback to fine-tune models or improve prompts

---

### 4. Local LLM Support

**Goal**: Run without OpenAI API (privacy, cost)

**Implementation:**

```python
# llm_client.py

def _get_client():
    if os.getenv("USE_LOCAL_LLM") == "true":
        from ollama import Client
        return Client(host='http://localhost:11434')
    else:
        return openai.OpenAI(api_key=OPENAI_API_KEY)

def chat_with_vision(...):
    client = _get_client()

    if isinstance(client, openai.OpenAI):
        # OpenAI path (existing code)
        pass
    else:
        # Ollama path
        response = client.chat(
            model="llama3",
            messages=messages
        )
        return response["message"]["content"]
```

**Setup:**
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull model
ollama pull llama3

# Set env var
export USE_LOCAL_LLM=true
```

---

### 5. Web UI

**Goal**: User-friendly interface for non-technical users

**Tech Stack:**
- Backend: FastAPI (reuse orchestrator)
- Frontend: React or Streamlit

**FastAPI Example:**

```python
# web_app.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TicketRequest(BaseModel):
    description: str
    log_snippet: str = None

@app.post("/process")
async def process_ticket(request: TicketRequest):
    ticket = Ticket(
        id=str(uuid.uuid4()),
        description=request.description,
        log_snippet=request.log_snippet
    )

    supervisor, trace = await run_ticket_flow(ticket)

    return {
        "customer_reply": supervisor.final_customer_reply,
        "internal_note": supervisor.final_internal_note,
        "trace_id": trace.trace_id
    }

# Run: uvicorn web_app:app --reload
```

**Streamlit Example:**

```python
# streamlit_app.py
import streamlit as st
import asyncio

st.title("AI Support Copilot")

description = st.text_area("Ticket Description")
log_snippet = st.text_area("Log Snippet (optional)")

if st.button("Process Ticket"):
    ticket = Ticket(
        id=str(uuid.uuid4()),
        description=description,
        log_snippet=log_snippet
    )

    supervisor, trace = asyncio.run(run_ticket_flow(ticket))

    st.success("Processing complete!")
    st.subheader("Customer Reply")
    st.write(supervisor.final_customer_reply)

    st.subheader("Internal Note")
    st.write(supervisor.final_internal_note)

# Run: streamlit run streamlit_app.py
```

---

## Conclusion

This workshop provides a complete, hands-on introduction to building production-quality AI systems with:

- **Modern protocols** (MCP for tool integration)
- **Clean architecture** (multi-agent, separation of concerns)
- **Observability** (traces, debugging)
- **Practical patterns** (error handling, retries, structured outputs)
