# AI Support Copilot — LLM + MCP + Multimodality + Observability + Multi-Agent Orchestration

A complete AI-powered support ticket handling system demonstrating:

- **Model Context Protocol (MCP)** for tool integration
- **Multi-agent orchestration** (Triage → Research → Action → Supervisor)
- **Multimodal LLM** support (text + vision)
- **Observability** with structured tracing
- **No frameworks** - hand-rolled architecture for educational clarity

This project is designed for workshops and learning, with synthetic data and clear implementation patterns.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Support Copilot Host                       │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │  Triage  │──▶│ Research │──▶│  Action  │──▶│Supervisor│   │
│  │  Agent   │   │  Agent   │   │  Agent   │   │  Agent   │   │
│  └──────────┘   └────┬─────┘   └──────────┘   └──────────┘   │
│                       │                                         │
│                       │ MCP Client                              │
│                       ▼                                         │
│              ┌────────────────┐                                 │
│              │ MCP Tools API  │                                 │
│              └────────┬───────┘                                 │
└───────────────────────┼─────────────────────────────────────────┘
                        │
┌───────────────────────┼─────────────────────────────────────────┐
│                       │   Support MCP Server                    │
│                       ▼                                         │
│       ┌────────────────────────────────────┐                   │
│       │  support_docs.search               │                   │
│       │  incidents.search                  │                   │
│       │  status.check                      │                   │
│       └────────────────────────────────────┘                   │
│                       │                                         │
│                       ▼                                         │
│           ┌──────────────────────┐                             │
│           │  Synthetic Data      │                             │
│           │  - Runbooks          │                             │
│           │  - Incidents         │                             │
│           │  - Service Status    │                             │
│           └──────────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
.
├── README.md                      # This file
├── prd.md                         # Product Requirements Document
├── requirements.txt               # Python dependencies
├── data/                          # Synthetic data
│   ├── docs/runbooks/            # Support documentation
│   ├── incidents/                # Incident database
│   ├── status/                   # Service status
│   └── tickets/                  # Sample tickets
├── support_mcp_server/           # MCP Server implementation
│   ├── config.py
│   ├── tools_support_docs.py
│   ├── tools_incidents.py
│   ├── tools_status.py
│   └── server.py
└── support_copilot_host/         # Host application
    ├── config.py
    ├── models.py                 # Data models
    ├── llm_client.py             # LLM wrapper
    ├── mcp_client.py             # MCP client
    ├── agents.py                 # Four agents
    ├── orchestrator.py           # Main flow
    ├── observability.py          # Tracing
    ├── cli.py                    # CLI interface
    └── examples.py               # Workshop demos
```

## Setup

### 1. Prerequisites

- Python 3.11+
- OpenAI API key (or compatible LLM provider)

### 2. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file (you can use the existing `openai_key.env` as reference):

```bash
OPENAI_API_KEY=your-api-key-here
LLM_MODEL_NAME=gpt-4o-mini  # Optional, defaults to gpt-4o-mini
```

## Interactive Jupyter Notebooks

**Perfect for workshops and interactive learning!**

We provide **3 Jupyter notebooks** for hands-on exploration:

### 1. Quick Start Demo (`demo_quickstart.ipynb`)
**⏱️ 5 minutes** - Perfect for workshop introduction

- Process 2 sample tickets end-to-end
- See complete output (customer reply + internal note)
- Compare different ticket types
- **Best for:** First-time demos, testing setup

### 2. MCP Tools Deep Dive (`demo_mcp_tools.ipynb`)
**⏱️ 15 minutes** - Understanding the tool layer

- Connect to MCP server
- Test each tool individually (`support_docs.search`, `incidents.search`, `status.check`)
- Explore search algorithms and scoring
- Performance analysis
- Real-world ticket triage scenario
- **Best for:** Teaching MCP concepts, tool customization

### 3. Complete Flow Exploration (`demo_complete_flow.ipynb`)
**⏱️ 20-30 minutes** - Full system deep dive

- Complete ticket processing flow
- Test each agent individually (Triage, Research, Action, Supervisor)
- Detailed trace exploration
- Process all 4 sample tickets
- Compare results and behaviors
- **Best for:** Understanding multi-agent orchestration, debugging

### Running Notebooks

**Prerequisites:**
```bash
# Install Jupyter
pip install jupyter

# Start MCP server (in separate terminal)
python -m support_mcp_server.server

# Start Jupyter
jupyter notebook
```

Then open any of the demo notebooks and run cells sequentially!

**Workshop Tip:** Start with `demo_quickstart.ipynb` to wow the audience, then dive into `demo_mcp_tools.ipynb` or `demo_complete_flow.ipynb` based on interest.

---

## Usage

### Quick Start: Run a Sample Ticket

This requires **two terminals**:

**Terminal 1 - Start the MCP Server:**

```bash
python -m support_mcp_server.server
```

**Terminal 2 - Run a sample ticket:**

```bash
python -m support_copilot_host.cli run-sample --id TICKET-001
```

### Available Commands

#### List available sample tickets

```bash
python -m support_copilot_host.cli list-samples
```

#### Run a specific sample ticket

```bash
python -m support_copilot_host.cli run-sample --id TICKET-001
python -m support_copilot_host.cli run-sample --id TICKET-002
python -m support_copilot_host.cli run-sample --id TICKET-003
python -m support_copilot_host.cli run-sample --id TICKET-004
```

#### Run a custom ticket

```bash
python -m support_copilot_host.cli run-ticket \
  --description "User cannot export dashboard to CSV" \
  --log "2025-11-01T10:00:00Z ERR_EXPORT_TIMEOUT service=export_service"
```

#### Run workshop demos

```bash
python -m support_copilot_host.examples
```

## Sample Tickets

The project includes four synthetic tickets:

1. **TICKET-001**: Export to CSV timeout (known issue + degraded service)
2. **TICKET-002**: SSO authentication failures (investigating incident)
3. **TICKET-003**: Workspace permissions issue (config error)
4. **TICKET-004**: API rate limiting (resolved incident)

## Multi-Agent Flow

Each ticket goes through four agents:

1. **TriageAgent**: Analyzes the ticket, classifies issue type, decides which tools to use
2. **ResearchAgent**: Calls MCP tools (docs, incidents, status) to gather context
3. **ActionAgent**: Drafts customer-facing reply and internal note
4. **SupervisorAgent**: Reviews output for quality, safety, and hallucinations

## MCP Tools

The MCP server exposes three tools:

- `support_docs.search`: Search internal runbooks
- `incidents.search`: Find related incidents
- `status.check`: Check service health status

## Observability

Every ticket execution generates a trace with:

- Unique trace ID
- Timestamp for each event
- Agent steps (LLM calls, tool calls, outputs)
- Full payload data (truncated if large)

Traces are:
- Displayed as pretty tables in the terminal
- Saved to `traces/trace.jsonl` as JSONL
- Used for debugging and workshop demonstrations

## Customization

### Add a New Runbook

Create a markdown file in `data/docs/runbooks/`:

```markdown
# My New Runbook

Description of the issue...

Resolution steps:
1. Step one
2. Step two
```

### Add a New Incident

Edit `data/incidents/incidents.json`:

```json
{
  "incident_id": "INC-1005",
  "title": "New incident",
  "status": "Investigating",
  "summary": "...",
  "tags": ["tag1", "tag2"]
}
```

### Change LLM Model

Set environment variable:

```bash
export LLM_MODEL_NAME="gpt-4o"
```

## Workshop Usage

This project is designed for a 3-hour workshop:

1. **Phase 1**: Explain architecture and MCP concepts
2. **Phase 2**: Demonstrate MCP server and tools
3. **Phase 3**: Show single-agent baseline (no tools)
4. **Phase 4**: Add MCP tool integration
5. **Phase 5**: Refactor to multi-agent pattern
6. **Phase 6**: Explore observability and traces
7. **Phase 7**: Discuss extensions (RAG, more agents, etc.)

See `prd.md` for detailed workshop plan.

## Development

### Run Tests

```bash
# Start MCP server first
python -m support_mcp_server.server

# In another terminal, run a test ticket
python -m support_copilot_host.cli run-sample --id TICKET-001
```

### View Traces

```bash
# Traces are saved to traces/trace.jsonl
cat traces/trace.jsonl | jq .
```

## Limitations

This is a **workshop/educational project**, not production-ready:

- Simple keyword search (no embeddings/vector search)
- No database persistence
- No authentication
- Synchronous agent execution (no parallelization)
- Limited error handling

## Future Extensions

See `prd.md` section 13 for ideas:

- Vector search for docs/incidents
- Parallel research agents
- Feedback collection
- Web UI
- Advanced evaluation metrics

## License

Educational use. See `prd.md` for full details.

## Questions?

Refer to `prd.md` for complete implementation details and architecture decisions.
