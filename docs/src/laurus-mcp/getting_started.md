# Getting Started with laurus-mcp

## Prerequisites

- The `laurus` CLI binary installed (`cargo install laurus-cli`)
- A running `laurus-server` instance (see [laurus-server getting started](../laurus-server/getting_started.md))
- An AI client that supports MCP (Claude Desktop, Claude Code, etc.)

## Configuration

### Step 1: Start laurus-server

```bash
laurus serve --port 50051
```

### Step 2: Configure the MCP client

#### Claude Code

Use the CLI command (recommended):

```bash
claude mcp add laurus -- laurus mcp --endpoint http://localhost:50051
```

Or edit `~/.claude/settings.json` directly:

```json
{
  "mcpServers": {
    "laurus": {
      "command": "laurus",
      "args": ["mcp", "--endpoint", "http://localhost:50051"]
    }
  }
}
```

#### Claude Desktop

Edit the configuration file for your platform:

- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "laurus": {
      "command": "laurus",
      "args": ["mcp", "--endpoint", "http://localhost:50051"]
    }
  }
}
```

## Usage Workflows

### Workflow 1: Pre-created index

Create the index using the CLI first, then use the MCP server to query it:

```bash
# Step 1: Create a schema file
cat > schema.toml << 'EOF'
[fields.title]
Text = { indexed = true, stored = true }

[fields.body]
Text = { indexed = true, stored = true }
EOF

# Step 2: Start the server and create the index
laurus serve --port 50051 &
laurus create index --schema schema.toml

# Step 3: Register the MCP server with Claude Code
claude mcp add laurus -- laurus mcp --endpoint http://localhost:50051
```

### Workflow 2: AI-driven index creation

Start laurus-server first, then register the MCP server and let the AI create the index:

```bash
# Step 1: Start laurus-server (no index required)
laurus serve --port 50051

# Step 2: Register the MCP server with Claude Code
claude mcp add laurus -- laurus mcp --endpoint http://localhost:50051
```

Then ask Claude:

> "Create a search index for blog posts. I need to search by title and body text,
> and I want to store the author and publication date."

Claude will design the schema and call `create_index` automatically.

### Workflow 3: Connect at runtime

Register the MCP server without specifying an endpoint:

```bash
claude mcp add laurus -- laurus mcp
```

Or edit the settings file directly:

```json
{
  "mcpServers": {
    "laurus": {
      "command": "laurus",
      "args": ["mcp"]
    }
  }
}
```

Then ask Claude to connect:

> "Connect to the laurus server at `http://localhost:50051`"

Claude will call `connect(endpoint: "http://localhost:50051")` before using other tools.

## Environment Variables

The `--endpoint` flag can be omitted and supplied through `LAURUS_ENDPOINT` instead — useful when the endpoint is fixed per machine and you do not want to hard-code it into every MCP client config:

```bash
export LAURUS_ENDPOINT=http://localhost:50051
claude mcp add laurus -- laurus mcp
```

Or inside the client config:

```json
{
  "mcpServers": {
    "laurus": {
      "command": "laurus",
      "args": ["mcp"],
      "env": {
        "LAURUS_ENDPOINT": "http://localhost:50051"
      }
    }
  }
}
```

When both are present, the explicit `--endpoint` flag wins (standard clap behaviour for `#[arg(long, env = "...")]`).

## Removing the MCP Server

To remove the registered MCP server from Claude Code:

```bash
claude mcp remove laurus
```

For Claude Desktop, remove the `laurus` entry from the configuration file and restart the application.

## Lifecycle

```text
laurus-server starts (separate process)
  └─ listens on gRPC port 50051

Claude starts
  └─ spawns: laurus mcp --endpoint `http://localhost:50051`
       └─ enters stdio event loop
            ├─ receives tool calls via stdin
            ├─ proxies calls to laurus-server via gRPC
            └─ sends results via stdout
Claude exits
  └─ laurus-mcp process terminates
  └─ laurus-server continues running
```
