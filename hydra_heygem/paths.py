from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit


class ArtifactPathError(ValueError):
    pass


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
    container_data_root: str = "/code/data",
    require_exists: bool = False,
) -> Path | str:
    raw = str(result or "").strip()
    if not raw:
        raise ArtifactPathError("result_path_required")
    if urlsplit(raw).scheme in {"http", "https"}:
        return raw

    root = Path(shared_host_root).expanduser().resolve(strict=False)
    container_root = str(container_data_root or "/code/data").replace("\\", "/").rstrip("/")
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
