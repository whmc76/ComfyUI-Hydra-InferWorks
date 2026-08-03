import pytest

from hydra_heygem.lifecycle import ContainerLifecycleError, DockerContainerLifecycle


class ScriptedRunner:
    def __init__(self, running=False):
        self.running = running
        self.calls = []

    def __call__(self, command, *, timeout_seconds):
        self.calls.append((list(command), timeout_seconds))
        if command[:2] == ["docker", "inspect"]:
            return 0, "true\n" if self.running else "false\n", ""
        if command[:2] == ["docker", "start"]:
            self.running = True
            return 0, "container\n", ""
        if command[:2] == ["docker", "stop"]:
            self.running = False
            return 0, "container\n", ""
        return 1, "", "unexpected"


def test_external_mode_never_touches_docker():
    runner = ScriptedRunner()
    manager = DockerContainerLifecycle("external", "hm-heygem", runner=runner)

    prepared = manager.prepare()
    released = manager.release(stop_after_job=True)

    assert prepared.mode == "external"
    assert prepared.started is False
    assert released.stopped is False
    assert runner.calls == []


def test_managed_mode_starts_configured_container_and_can_release_it():
    runner = ScriptedRunner(running=False)
    manager = DockerContainerLifecycle("docker_existing_container", "custom-heygem", runner=runner)

    prepared = manager.prepare()
    released = manager.release(stop_after_job=True)

    assert prepared.started is True
    assert prepared.container_name == "custom-heygem"
    assert released.stopped is True
    assert [call[0] for call in runner.calls] == [
        ["docker", "inspect", "-f", "{{.State.Running}}", "custom-heygem"],
        ["docker", "start", "custom-heygem"],
        ["docker", "stop", "custom-heygem"],
        ["docker", "inspect", "-f", "{{.State.Running}}", "custom-heygem"],
    ]


def test_managed_mode_keeps_warm_when_stop_is_not_requested():
    runner = ScriptedRunner(running=True)
    manager = DockerContainerLifecycle("docker_existing_container", "hm-heygem", runner=runner)

    prepared = manager.prepare()
    released = manager.release(stop_after_job=False)

    assert prepared.was_running is True
    assert prepared.started is False
    assert released.stopped is False


def test_managed_mode_fails_closed_when_stop_does_not_release_container():
    runner = ScriptedRunner(running=True)

    def ineffective_stop(command, *, timeout_seconds):
        if command[:2] == ["docker", "stop"]:
            runner.calls.append((list(command), timeout_seconds))
            return 0, "container\n", ""
        return runner(command, timeout_seconds=timeout_seconds)

    manager = DockerContainerLifecycle(
        "docker_existing_container",
        "hm-heygem",
        runner=ineffective_stop,
    )
    manager.prepare()

    with pytest.raises(ContainerLifecycleError, match="container_release_verification_failed"):
        manager.release(stop_after_job=True)
