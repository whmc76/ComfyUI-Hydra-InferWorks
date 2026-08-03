from __future__ import annotations

from dataclasses import dataclass
from os import environ as process_environ
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit


DEFAULT_SCHEME = "http"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8383
SERVICE_URL_ENV_KEYS = (
    "HYDRA_HEYGEM_SERVICE_URL",
    "HYDRA_AVATAR_SERVICE_URL",
    "AVATAR_SERVICE_URL",
    "HEYGEM_SERVICE_URL",
)


class EndpointConfigError(ValueError):
    pass


@dataclass(frozen=True)
class EndpointConfig:
    base_url: str
    scheme: str
    host: str
    port: int
    source: str


def _text(value: object) -> str:
    return str(value or "").strip()


def _is_auto(value: object) -> bool:
    return _text(value).lower() in {"", "auto", "default", "environment", "env"}


def _first_environment_url(environ: Mapping[str, str]) -> str:
    for key in SERVICE_URL_ENV_KEYS:
        value = _text(environ.get(key))
        if value:
            return value
    return ""


def _parse_port(value: object, *, allow_auto: bool) -> int | None:
    if allow_auto and (_is_auto(value) or value == 0 or _text(value) == "0"):
        return None
    try:
        port = int(value)
    except (TypeError, ValueError) as error:
        raise EndpointConfigError(f"invalid_service_port:{value}") from error
    if port < 1 or port > 65535:
        raise EndpointConfigError(f"invalid_service_port:{port}")
    return port


def _parse_url(raw_url: str, source: str) -> EndpointConfig:
    parsed = urlsplit(_text(raw_url))
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise EndpointConfigError(f"unsupported_service_url_scheme:{scheme or 'missing'}")
    if not parsed.hostname:
        raise EndpointConfigError("service_url_host_required")
    try:
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError as error:
        raise EndpointConfigError("invalid_service_url_port") from error
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise EndpointConfigError("service_url_credentials_query_or_fragment_forbidden")
    path = parsed.path.rstrip("/")
    netloc_host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    default_port = 443 if scheme == "https" else 80
    netloc = netloc_host if port == default_port and parsed.port is None else f"{netloc_host}:{port}"
    return EndpointConfig(
        base_url=urlunsplit((scheme, netloc, path, "", "")),
        scheme=scheme,
        host=parsed.hostname,
        port=port,
        source=source,
    )


def resolve_endpoint_config(
    *,
    service_url: object = "auto",
    service_host: object = "auto",
    service_port: object = 0,
    environ: Mapping[str, str] | None = None,
) -> EndpointConfig:
    """Resolve a HeyGem endpoint without making any port part of the node contract.

    Precedence is explicit full URL, explicit host/port components, environment
    full URL, environment components, then compatibility defaults.
    """

    active_environ = process_environ if environ is None else environ
    if not _is_auto(service_url):
        return _parse_url(_text(service_url), "node_service_url")

    explicit_host = None if _is_auto(service_host) else _text(service_host)
    explicit_port = _parse_port(service_port, allow_auto=True)
    environment_url = _first_environment_url(active_environ)

    if explicit_host is not None or explicit_port is not None:
        environment_config = (
            _parse_url(environment_url, "environment_service_url") if environment_url else None
        )
        scheme = _text(active_environ.get("HYDRA_HEYGEM_SCHEME")).lower()
        if not scheme:
            scheme = environment_config.scheme if environment_config else DEFAULT_SCHEME
        if scheme not in {"http", "https"}:
            raise EndpointConfigError(f"unsupported_service_url_scheme:{scheme}")
        host = (
            explicit_host
            or (environment_config.host if environment_config else "")
            or _text(active_environ.get("HYDRA_HEYGEM_HOST"))
            or DEFAULT_HOST
        )
        port = (
            explicit_port
            or (environment_config.port if environment_config else None)
            or _parse_port(active_environ.get("HYDRA_HEYGEM_PORT"), allow_auto=True)
            or DEFAULT_PORT
        )
        netloc_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
        return _parse_url(f"{scheme}://{netloc_host}:{port}", "node_host_or_port")

    if environment_url:
        return _parse_url(environment_url, "environment_service_url")

    scheme = _text(active_environ.get("HYDRA_HEYGEM_SCHEME")).lower() or DEFAULT_SCHEME
    host = _text(active_environ.get("HYDRA_HEYGEM_HOST")) or DEFAULT_HOST
    port = _parse_port(active_environ.get("HYDRA_HEYGEM_PORT"), allow_auto=True) or DEFAULT_PORT
    return _parse_url(f"{scheme}://{host}:{port}", "environment_components_or_default")

