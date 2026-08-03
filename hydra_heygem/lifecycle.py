from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable, Sequence


class ContainerLifecycleError(RuntimeError):
    pass


CommandRunner = Callable[[Sequence[str]], tuple[int, str, str]]


def _default_runner(command: Sequence[str], *, timeout_seconds: float) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=False,
            timeout=max(float(timeout_seconds), 0.1),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ContainerLifecycleError(f"container_command_failed:{command[0]}:{error}") from error
    return result.returncode, result.stdout, result.stderr


@dataclass(frozen=True)
class ContainerLifecycleReceipt:
    mode: str
    container_name: str | None
    was_running: bool
    started: bool
    stopped: bool


class DockerContainerLifecycle:
    MODES = {"external", "docker_existing_container"}

    def __init__(
        self,
        mode: str,
        container_name: str,
        *,
        runner: Callable[..., tuple[int, str, str]] = _default_runner,
        command_timeout_seconds: float = 60.0,
    ) -> None:
        normalized_mode = str(mode or "external").strip().lower()
        if normalized_mode not in self.MODES:
            raise ContainerLifecycleError(f"unsupported_lifecycle_mode:{normalized_mode}")
        normalized_name = str(container_name or "").strip()
        if normalized_mode != "external" and not normalized_name:
            raise ContainerLifecycleError("container_name_required")
        self.mode = normalized_mode
        self.container_name = normalized_name
        self.runner = runner
        self.command_timeout_seconds = max(float(command_timeout_seconds), 0.1)
        self.was_running = False
        self.started = False

    def _run(self, command: Sequence[str]) -> tuple[int, str, str]:
        return self.runner(command, timeout_seconds=self.command_timeout_seconds)

    def _is_running(self) -> bool:
        code, stdout, _ = self._run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.container_name]
        )
        if code != 0:
            raise ContainerLifecycleError(f"configured_container_not_found:{self.container_name}")
        value = stdout.strip().lower()
        if value not in {"true", "false"}:
            raise ContainerLifecycleError(f"container_running_state_invalid:{self.container_name}")
        return value == "true"

    def prepare(self) -> ContainerLifecycleReceipt:
        if self.mode == "external":
            return ContainerLifecycleReceipt("external", None, False, False, False)
        self.was_running = self._is_running()
        if not self.was_running:
            code, _, stderr = self._run(["docker", "start", self.container_name])
            if code != 0:
                raise ContainerLifecycleError(
                    f"container_start_failed:{self.container_name}:{stderr.strip()[:300]}"
                )
            self.started = True
        return ContainerLifecycleReceipt(
            self.mode,
            self.container_name,
            self.was_running,
            self.started,
            False,
        )

    def release(self, *, stop_after_job: bool) -> ContainerLifecycleReceipt:
        stopped = False
        if self.mode != "external" and stop_after_job:
            code, _, stderr = self._run(["docker", "stop", self.container_name])
            if code != 0:
                raise ContainerLifecycleError(
                    f"container_stop_failed:{self.container_name}:{stderr.strip()[:300]}"
                )
            stopped = True
        return ContainerLifecycleReceipt(
            self.mode,
            self.container_name if self.mode != "external" else None,
            self.was_running,
            self.started,
            stopped,
        )

