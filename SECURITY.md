# Security Policy

Please report vulnerabilities privately through GitHub's security advisory
feature for this repository.

The node can contact a user-configured HTTP(S) endpoint and can optionally run
`docker start` / `docker stop` for one explicitly configured container name.
Do not expose an unauthenticated HeyGem service to untrusted networks. Use the
`external` lifecycle mode unless the ComfyUI queue is the service owner.

