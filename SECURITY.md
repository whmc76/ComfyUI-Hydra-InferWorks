# Security Policy

Please report vulnerabilities privately through GitHub's security advisory
feature for this repository.

The HeyGem node can contact a user-configured HTTP(S) endpoint and can optionally run
`docker start` / `docker stop` for one explicitly configured container name.
Do not expose an unauthenticated HeyGem service to untrusted networks. Use the
`external` lifecycle mode unless the ComfyUI queue is the service owner.

IndexTTS 2.5 and Qwen3-ASR process caller-provided speech and reference media
locally. Install models only from their declared upstream sources, verify model
identity before production use, and use voices or likenesses only with proper
authorization. Hydra InferWorks does not grant rights to third-party models,
voices, reference media, or HeyGem deployments.

