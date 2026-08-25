from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping
from urllib.parse import urlsplit


class ArtifactPathError(ValueError):
    pass


_AUTOMATIC_VALUES = {"", "auto", "default", "environment", "env"}
_SHARED_ROOT_ENV_KEYS = (
    "INFERWORKS_HEYGEM_SHARED_LOCAL_ROOT",
    "INFERWORKS_HEYGEM_SHARED_ROOT",
    "HEYGEM_SHARED_LOCAL_ROOT",
    "HEYGEM_SHARED_ROOT",
    "EXTERNAL_HEYGEM_STAGE_DIR",
    "AVATAR_STAGE_DIR",
    "HEYGEM_DATA_DIR",
    "HYDRA_HEYGEM_SHARED_HOST_ROOT",
    "HYDRA_AVATAR_STAGE_DIR",
)
_CROSS_NAMESPACE_LOCAL_ENV_KEYS = (
    "INFERWORKS_HEYGEM_SHARED_LOCAL_ROOT",
    "HEYGEM_SHARED_LOCAL_ROOT",
    "HYDRA_HEYGEM_SHARED_CONTAINER_ROOT",
)
_SERVICE_SHARED_ROOT_ENV_KEYS = (
    "INFERWORKS_HEYGEM_SERVICE_SHARED_ROOT",
    "HEYGEM_SERVICE_SHARED_ROOT",
    "AVATAR_CONTAINER_DATA_DIR",
    "HEYGEM_CONTAINER_DATA_DIR",
    "HYDRA_HEYGEM_SHARED_CONTAINER_ROOT",
    "HYDRA_HEYGEM_CONTAINER_DATA_ROOT",
    "HYDRA_AVATAR_CONTAINER_DATA_DIR",
)


def _text(value: object) -> str:
    return str(value or "").strip()


def _is_auto(value: object) -> bool:
    return _text(value).lower() in _AUTOMATIC_VALUES


def _first_environment_value(
    environ: Mapping[str, str],
    keys: tuple[str, ...],
) -> str:
    for key in keys:
        value = _text(environ.get(key))
        if value:
            return value
    return ""


def _is_windows_absolute(value: str) -> bool:
    return PureWindowsPath(value).is_absolute()


def resolve_shared_local_root(
    value: object = "auto",
    *,
    environ: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> Path:
    """Resolve the shared root as visible to the running ComfyUI process."""

    active_environ = os.environ if environ is None else environ
    requested = "" if _is_auto(value) else _text(value)
    requested = requested or _first_environment_value(
        active_environ,
        _SHARED_ROOT_ENV_KEYS,
    )
    if not requested:
        raise ArtifactPathError("heygem_shared_local_root_required")

    active_platform = platform_name or os.name
    if active_platform != "nt" and _is_windows_absolute(requested):
        local_mapping = _first_environment_value(
            active_environ,
            _CROSS_NAMESPACE_LOCAL_ENV_KEYS,
        )
        if not local_mapping or _is_windows_absolute(local_mapping):
            raise ArtifactPathError("shared_root_namespace_mismatch")
        requested = local_mapping

    root = Path(requested).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_service_shared_root(
    value: object = "auto",
    *,
    local_root: str | Path,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve the same shared storage as addressed by the HeyGem service."""

    active_environ = os.environ if environ is None else environ
    configured = "" if _is_auto(value) else _text(value)
    configured = configured or _first_environment_value(
        active_environ,
        _SERVICE_SHARED_ROOT_ENV_KEYS,
    )
    normalized = (configured or str(Path(local_root).resolve(strict=False))).replace(
        "\\",
        "/",
    ).rstrip("/")
    if not normalized:
        raise ArtifactPathError("heygem_service_shared_root_required")
    return normalized


def _require_under_root(candidate: Path, root: Path) -> Path:
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ArtifactPathError("result_path_escapes_shared_root") from error
    return resolved


def map_result_to_host(
    result: object,
    *,
    shared_host_root: str | Path,
    container_data_root: str | None = None,
    require_exists: bool = False,
) -> Path | str:
    raw = str(result or "").strip()
    if not raw:
        raise ArtifactPathError("result_path_required")
    if urlsplit(raw).scheme in {"http", "https"}:
        return raw

    root = Path(shared_host_root).expanduser().resolve(strict=False)
    container_root = str(container_data_root or root).replace("\\", "/").rstrip("/")
    normalized = raw.replace("\\", "/")
    candidate: Path

    if normalized == container_root or normalized.startswith(f"{container_root}/"):
        relative = normalized[len(container_root) :].lstrip("/")
        relative_path = PurePosixPath(relative)
        if ".." in relative_path.parts:
            raise ArtifactPathError("result_path_traversal_forbidden")
        candidate = _require_under_root(root.joinpath(*relative_path.parts), root)
    elif normalized.lstrip("/").count("/") == 0 and normalized.lower().endswith(
        (".mp4", ".avi", ".mkv", ".mov")
    ):
        # HeyGem's easy API commonly reports the final muxed artifact as
        # ``/<job>-r.mp4`` even though it lives under the shared temp folder.
        candidate = _require_under_root(root / "temp" / normalized.lstrip("/"), root)
    else:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            relative_path = PurePosixPath(normalized)
            if ".." in relative_path.parts:
                raise ArtifactPathError("result_path_traversal_forbidden")
            candidate = _require_under_root(root.joinpath(*relative_path.parts), root)
        else:
            candidate = path.resolve(strict=False)

    if require_exists and (not candidate.is_file() or candidate.stat().st_size <= 0):
        raise ArtifactPathError(f"result_artifact_missing_or_empty:{candidate}")
    return candidate


def prefer_final_muxed_artifact(
    mapped_result: Path | str,
    *,
    code: str,
    shared_host_root: str | Path,
) -> Path | str:
    """Replace HeyGem's transient per-job result.avi with its final muxed MP4."""

    if isinstance(mapped_result, str):
        return mapped_result
    root = Path(shared_host_root).expanduser().resolve(strict=False)
    resolved = mapped_result.resolve(strict=False)
    if resolved.name.lower() == "result.avi" and resolved.parent.name == code:
        return _require_under_root(root / "temp" / f"{code}-r.mp4", root)
    return resolved
