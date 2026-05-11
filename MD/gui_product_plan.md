# GUI Productization Plan

This document is the roadmap for turning the current PySide lab GUI into an API-backed product surface for QickworkspaceV2.

## Product Goal

The GUI should become a thin client over a reliable experiment service. Notebooks, scripts, PySide GUI, and any future web UI should all use the same backend contract:

```text
GUI / Web UI / Notebook client
    -> QickworkspaceV2 API service
    -> Experiment job manager
    -> BaseExperiment / QICK hardware
    -> ExperimentData + CalibrationStore
```

## Principles

- The GUI should not own hardware lifecycle or calibration rules long term.
- Every run should be represented as a job with a stable id, config snapshot, status, result, and audit metadata.
- All experiment metadata should be schema-driven, not hardcoded separately in every UI.
- Fit results should update config through an explicit preview/apply flow.
- Persistent calibration state should go through `CalibrationStore`.
- Live data should stream from the service instead of being pulled from GUI-owned acquisition loops.

## Target API Surface

Core experiment endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service status |
| `GET` | `/experiments/schema` | GUI/client experiment catalog and parameter metadata |
| `POST` | `/experiments/run` | Submit experiment job |
| `GET` | `/experiments/{job_id}/status` | Poll job status |
| `GET` | `/experiments/{job_id}/result` | Fetch `ExperimentData` JSON |
| `POST` | `/experiments/{job_id}/stop` | Request graceful stop |
| `GET` | `/experiments/{job_id}/stream` | Live data/log stream via SSE or WebSocket |

Calibration/config endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/config/{qubit}` | Active config snapshot |
| `PATCH` | `/config/{qubit}` | Update runtime config with preview/audit |
| `GET` | `/calibrations/{qubit}/params` | Persistent calibration params |
| `POST` | `/calibrations/{qubit}/set` | Store a calibration value |
| `GET` | `/calibrations/{qubit}/stale` | Stale calibration report |

## Implementation Phases

### Phase 1 - Shared Contract

Status: started.

- Add a shared experiment registry in `QickworkspaceV2/core/experiment_registry.py`.
- Centralize class path resolution.
- Centralize fit-result to config-update mapping.
- Expose schema from the FastAPI service.
- Update GUI to consume shared mapping instead of duplicating rules.

### Phase 2 - Job Model

- Replace ad hoc in-memory job dict with a `JobRecord` model.
- Track `queued`, `running`, `stopping`, `completed`, `failed`.
- Store job metadata: user, qubit, experiment id, config snapshot, code version, timestamps.
- Add graceful stop requests to the API.
- Persist completed job summaries for audit/recovery.

### Phase 3 - Streaming

- Add server-side live data events:
  - `data`
  - `progress`
  - `log`
  - `fit`
  - `status`
- Use Server-Sent Events first for simplicity; WebSocket later if bidirectional control is needed.
- GUI subscribes to events instead of running acquisition loops locally.

### Phase 4 - Config Update Workflow

- Fit update should become a preview:
  - source job id
  - old value
  - suggested value
  - uncertainty
  - target config key
- User applies/rejects suggestions.
- Applied updates write to runtime config and optionally `CalibrationStore`.
- Config viewer shows active values plus pending suggestions.

### Phase 5 - Commercial UX

- Add run history panel.
- Add result review panel with quality flags, fit errors, and update suggestions.
- Add hardware health panel: QICK connection, Pyro server, clocks, data path, last heartbeat.
- Add experiment presets/templates.
- Add role model: viewer, operator, admin.

## Current First Step

The first implementation step is intentionally modest:

1. Keep the existing PySide GUI usable.
2. Add a shared experiment registry and fit-update mapping.
3. Let the FastAPI service expose that schema.
4. Let the GUI reuse the shared fit-update logic.

This creates the foundation for service-first operation without breaking current lab workflows.
