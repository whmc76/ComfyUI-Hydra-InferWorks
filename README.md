# ComfyUI-Hydra-HeyGem

Hydra-owned ComfyUI nodes for long-form, exact-audio HeyGem avatar generation.

This project keeps ComfyUI as the workflow and GPU scheduling entry point while
allowing HeyGem to remain in an isolated service or Docker container. It does
not bundle HeyGem, its models, or a container image.

## Why this node exists

Existing HeyGem nodes commonly start a fixed container on a fixed port and
expand the completed video into an `IMAGE` tensor. That is unsafe for long-form
production. `HydraHeyGemLongformAvatar` instead:

- accepts ComfyUI's native file-backed `VIDEO` plus the exact `AUDIO` waveform;
- resolves a fully configurable service URL, host, and port;
- can use an externally managed service or start/stop an existing Docker container;
- unloads resident ComfyUI models before handing the GPU to HeyGem;
- stages inputs through an explicitly configured shared host/container mount;
- returns a file-backed `VIDEO`, the artifact path, and a durable JSON receipt;
- rejects invalid endpoints, path traversal, empty artifacts, and provider failure.

## Requirements

- ComfyUI 0.30.0 or later.
- A working HeyGem-compatible service exposing submit and query routes.
- A host directory mounted into the service container, when container paths are required.
- Docker CLI only when `lifecycle_mode=docker_existing_container` is selected.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/whmc76/ComfyUI-Hydra-HeyGem.git
```

Restart ComfyUI with custom nodes enabled. The node appears under
`HydraMatrix/avatar` as **Hydra HeyGem Long-form Avatar**.

## Endpoint and port resolution

No port is part of the node's fixed contract. Resolution precedence is:

1. `service_url` supplied on the node, such as `http://127.0.0.1:59202`.
2. Explicit `service_host` and/or `service_port` supplied on the node.
3. `HYDRA_HEYGEM_SERVICE_URL`, `HYDRA_AVATAR_SERVICE_URL`, `AVATAR_SERVICE_URL`, or `HEYGEM_SERVICE_URL`.
4. `HYDRA_HEYGEM_HOST`, `HYDRA_HEYGEM_PORT`, and `HYDRA_HEYGEM_SCHEME`.
5. Compatibility defaults only when nothing is configured.

`submit_path`, `query_path`, and `health_path` are also editable. With
`health_path=auto`, the node probes `query_path?code=healthcheck`.

Example:

```powershell
$env:HYDRA_HEYGEM_SERVICE_URL = 'http://127.0.0.1:49202'
$env:HYDRA_HEYGEM_SHARED_HOST_ROOT = 'D:\duix_avatar_data\face2face'
$env:HYDRA_HEYGEM_CONTAINER_DATA_ROOT = '/code/data'
$env:HYDRA_HEYGEM_CONTAINER_NAME = 'hm-heygem'
```

## Lifecycle modes

- `external`: never runs Docker commands. The service lifecycle is owned elsewhere.
- `docker_existing_container`: starts the configured existing container when stopped.

`stop_container_after=true` stops that configured container in a `finally` block,
including when generation fails. Enable this only when the ComfyUI queue is the
authoritative owner of that service.

## Shared filesystem contract

HeyGem commonly accepts paths visible inside its container rather than file
uploads. Configure:

- `shared_host_root`: host side of the mounted data directory.
- `container_data_root`: the same mount inside HeyGem, commonly `/code/data`.

The node writes audio under `inputs/audio/`, the reference video under
`inputs/video/`, and resolves provider results back to the host root. Relative
path traversal is rejected.

## Outputs

1. `video`: native file-backed ComfyUI `VIDEO`; connect it to `Save Video` or downstream video nodes.
2. `artifact_path`: absolute path to the generated artifact.
3. `receipt_json`: endpoint resolution, lifecycle, hashes, paths, provider response, and timing.

The generated output is never decoded into a full in-memory frame tensor by
this node.

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The core endpoint, client, lifecycle, and path contracts are testable without
installing ComfyUI. Node import is additionally validated against a real
ComfyUI runtime before release.

## Scope and licensing

This repository contains only the ComfyUI integration. HeyGem and any container
image or model weights retain their own licenses and usage restrictions.

Licensed under Apache-2.0.

