# Changelog

## 0.1.0 - 2026-08-03

- Add the `HydraHeyGemLongformAvatar` ComfyUI node.
- Support configurable service URL, host, port, submit/query/health paths.
- Support external and existing-Docker-container lifecycle modes.
- Keep long video inputs and outputs file-backed.
- Add shared mount path validation, artifact hashes, and durable receipts.
- Persist each generation receipt atomically and verify Docker release state after stop.
- Mark generation as an uncached ComfyUI output node so API workflows execute it directly.
