# Hyperion Dashboard Backend

FastAPI server that provides the API for the Hyperion dashboard. It reads experiment data from the SQLite database and serves it via REST endpoints and WebSocket connections.

## Running

From the project root:

```bash
mise run ui-backend
```

Or directly:

```bash
python -m ui.backend.app
```

## Configuration

The server uses these environment variables:

- `HYPERION_DB_URL` - Database URL (default: `sqlite:///hyperion.db`)
- `HYPERION_CORS_ORIGIN` - CORS origin for frontend (default: `http://localhost:5173`)
- `HYPERION_WS_POLL_MS` - WebSocket polling interval in ms (default: 300)

## API Endpoints

- `/api/experiments` - List all experiments
- `/api/experiments/{id}` - Get experiment details
- `/api/experiments/{id}/trials` - Get trials for an experiment
- `/api/experiments/{id}/lineage` - Get lineage graph data
- `/api/events` - Get events (supports filtering)
- `/ws/events` - WebSocket for real-time event streaming

The server runs on port 8000 by default.
