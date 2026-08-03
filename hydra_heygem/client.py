from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import EndpointConfig


class HeyGemClientError(RuntimeError):
    pass


class JsonTransport(Protocol):
    def request_json(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, object] | None,
        timeout_seconds: float,
    ) -> Mapping[str, object]: ...


class UrllibJsonTransport:
    def request_json(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, object] | None = None,
        timeout_seconds: float,
    ) -> Mapping[str, object]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            url,
            data=body,
            method=method.upper(),
            headers={"accept": "application/json", "content-type": "application/json"},
        )
        try:
            with urlopen(request, timeout=max(float(timeout_seconds), 0.1)) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise HeyGemClientError(f"heygem_http_error:{error.code}:{details[:500]}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise HeyGemClientError(f"heygem_request_failed:{error}") from error
        try:
            decoded = json.loads(raw or "{}")
        except json.JSONDecodeError as error:
            raise HeyGemClientError("heygem_response_not_json") from error
        if not isinstance(decoded, dict):
            raise HeyGemClientError("heygem_response_object_required")
        return decoded


@dataclass(frozen=True)
class HeyGemGenerationReceipt:
    code: str
    result: str
    poll_count: int
    elapsed_seconds: float
    submit_response: Mapping[str, object]
    final_response: Mapping[str, object]


def _join_url(base_url: str, path: str) -> str:
    normalized_path = str(path or "").strip()
    if normalized_path.startswith(("http://", "https://")):
        return normalized_path
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    return f"{base_url.rstrip('/')}{normalized_path}"


def _read_code(response: Mapping[str, object]) -> int | None:
    try:
        return int(response.get("code"))
    except (TypeError, ValueError):
        return None


def _read_data(response: Mapping[str, object]) -> Mapping[str, object]:
    data = response.get("data")
    return data if isinstance(data, dict) else {}


def _read_result(data: Mapping[str, object]) -> str:
    for key in (
        "result",
        "video_url",
        "videoUrl",
        "result_url",
        "resultUrl",
        "local_path",
        "localPath",
        "path",
    ):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_success(data: Mapping[str, object]) -> bool:
    status = str(data.get("status", "")).strip().lower()
    if status in {"2", "success", "succeeded", "complete", "completed"}:
        return bool(_read_result(data))
    try:
        progress = int(data.get("progress"))
    except (TypeError, ValueError):
        progress = 0
    return progress >= 100 and bool(_read_result(data))


def _is_failure(data: Mapping[str, object]) -> bool:
    status = str(data.get("status", "")).strip().lower()
    message = str(data.get("msg") or data.get("message") or "").lower()
    return status in {"-1", "3", "error", "failed", "failure"} or any(
        marker in message for marker in ("failed", "failure", "error", "失败")
    )


class HeyGemClient:
    def __init__(
        self,
        endpoint: EndpointConfig,
        *,
        submit_path: str = "/easy/submit",
        query_path: str = "/easy/query",
        transport: JsonTransport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        interrupt_check: Callable[[], None] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.submit_path = submit_path
        self.query_path = query_path
        self.transport = transport or UrllibJsonTransport()
        self.sleeper = sleeper
        self.monotonic = monotonic
        self.interrupt_check = interrupt_check

    def wait_until_ready(
        self,
        *,
        health_path: str,
        timeout_seconds: float,
        retry_interval_seconds: float = 1.0,
    ) -> Mapping[str, object]:
        normalized_path = str(health_path or "").strip()
        if normalized_path.lower() in {"", "off", "none", "disabled"}:
            return {"status": "health_probe_skipped"}
        started_at = self.monotonic()
        last_error = ""
        while self.monotonic() - started_at <= max(float(timeout_seconds), 0.1):
            if self.interrupt_check:
                self.interrupt_check()
            try:
                return self.transport.request_json(
                    "GET",
                    _join_url(self.endpoint.base_url, normalized_path),
                    payload=None,
                    timeout_seconds=min(max(float(timeout_seconds), 0.1), 10.0),
                )
            except HeyGemClientError as error:
                last_error = str(error)
                self.sleeper(max(float(retry_interval_seconds), 0.05))
        raise HeyGemClientError(f"heygem_service_not_ready:{last_error or 'timeout'}")

    def generate(
        self,
        *,
        code: str,
        audio_container_path: str,
        video_container_path: str,
        timeout_seconds: float,
        poll_interval_seconds: float,
        extra_payload: Mapping[str, object] | None = None,
    ) -> HeyGemGenerationReceipt:
        normalized_code = str(code or "").strip()
        if not normalized_code:
            raise HeyGemClientError("heygem_job_code_required")
        started_at = self.monotonic()
        timeout_seconds = max(float(timeout_seconds), 0.1)
        payload = {
            "audio_url": str(audio_container_path),
            "video_url": str(video_container_path),
            "code": normalized_code,
            **dict(extra_payload or {}),
        }
        submit_response = self.transport.request_json(
            "POST",
            _join_url(self.endpoint.base_url, self.submit_path),
            payload=payload,
            timeout_seconds=min(timeout_seconds, 120.0),
        )
        submit_code = _read_code(submit_response)
        if submit_code != 10000:
            message = submit_response.get("msg") or submit_response.get("message") or "unknown"
            raise HeyGemClientError(f"heygem_submit_rejected:{submit_code}:{message}")

        poll_count = 0
        while True:
            if self.interrupt_check:
                self.interrupt_check()
            elapsed = self.monotonic() - started_at
            if elapsed > timeout_seconds:
                raise HeyGemClientError(f"heygem_generation_timeout:{normalized_code}")
            poll_count += 1
            query_url = _join_url(self.endpoint.base_url, self.query_path)
            separator = "&" if "?" in query_url else "?"
            query_response = self.transport.request_json(
                "GET",
                f"{query_url}{separator}code={quote(normalized_code, safe='')}",
                payload=None,
                timeout_seconds=min(max(timeout_seconds - elapsed, 0.1), 120.0),
            )
            query_code = _read_code(query_response)
            if query_code != 10000:
                message = query_response.get("msg") or query_response.get("message") or "unknown"
                raise HeyGemClientError(f"heygem_query_rejected:{query_code}:{message}")
            data = _read_data(query_response)
            if _is_success(data):
                return HeyGemGenerationReceipt(
                    code=normalized_code,
                    result=_read_result(data),
                    poll_count=poll_count,
                    elapsed_seconds=round(self.monotonic() - started_at, 3),
                    submit_response=submit_response,
                    final_response=query_response,
                )
            if _is_failure(data):
                message = data.get("msg") or data.get("message") or data.get("status") or "unknown"
                raise HeyGemClientError(f"heygem_generation_failed:{message}")
            self.sleeper(max(float(poll_interval_seconds), 0.01))
