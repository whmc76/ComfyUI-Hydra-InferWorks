import pytest

from hydra_heygem.client import HeyGemClient, HeyGemClientError
from hydra_heygem.config import EndpointConfig


class ScriptedTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request_json(self, method, url, *, payload=None, timeout_seconds):
        self.calls.append((method, url, payload, timeout_seconds))
        return self.responses.pop(0)


def endpoint(port=58123):
    return EndpointConfig(
        base_url=f"http://127.0.0.1:{port}",
        scheme="http",
        host="127.0.0.1",
        port=port,
        source="test",
    )


def test_submit_and_poll_use_the_resolved_custom_port():
    transport = ScriptedTransport(
        [
            {"code": 10000, "msg": "accepted"},
            {"code": 10000, "data": {"status": 1, "progress": 20}},
            {
                "code": 10000,
                "data": {"status": 2, "progress": 100, "result": "/code/data/temp/job-r.mp4"},
            },
        ]
    )
    client = HeyGemClient(endpoint(), transport=transport, sleeper=lambda _: None)

    receipt = client.generate(
        code="job",
        audio_container_path="/code/data/inputs/audio/job.wav",
        video_container_path="/code/data/inputs/video/job.mp4",
        timeout_seconds=10,
        poll_interval_seconds=0.01,
    )

    assert receipt.result == "/code/data/temp/job-r.mp4"
    assert [call[1] for call in transport.calls] == [
        "http://127.0.0.1:58123/easy/submit",
        "http://127.0.0.1:58123/easy/query?code=job",
        "http://127.0.0.1:58123/easy/query?code=job",
    ]


def test_paths_are_configurable_without_changing_the_service_port():
    transport = ScriptedTransport(
        [
            {"code": 10000},
            {"code": 10000, "data": {"status": "success", "video_url": "/code/data/temp/x.mp4"}},
        ]
    )
    client = HeyGemClient(
        endpoint(62000),
        submit_path="/avatar/submit",
        query_path="/avatar/query",
        transport=transport,
        sleeper=lambda _: None,
    )

    client.generate(
        code="x",
        audio_container_path="/code/data/inputs/audio/x.wav",
        video_container_path="/code/data/inputs/video/x.mp4",
        timeout_seconds=10,
        poll_interval_seconds=0.01,
    )

    assert transport.calls[0][1] == "http://127.0.0.1:62000/avatar/submit"
    assert transport.calls[1][1] == "http://127.0.0.1:62000/avatar/query?code=x"


def test_external_service_gpu_release_is_explicit_and_receipted():
    response = {
        "code": 10000,
        "success": True,
        "msg": "released",
        "data": {
            "error": "",
            "cuda_available": True,
            "cuda_empty_cache": True,
            "cuda_ipc_collect": True,
        },
    }
    transport = ScriptedTransport([response])
    client = HeyGemClient(endpoint(), transport=transport)

    receipt = client.release_gpu(release_path="/v1/system/gpu/release")

    assert receipt.accepted is True
    assert receipt.response == response
    assert transport.calls == [
        (
            "POST",
            "http://127.0.0.1:58123/v1/system/gpu/release",
            {},
            60.0,
        )
    ]


def test_external_service_gpu_release_fails_closed_without_provider_acceptance():
    transport = ScriptedTransport(
        [{"code": 10004, "success": False, "msg": "not released"}]
    )
    client = HeyGemClient(endpoint(), transport=transport)

    with pytest.raises(HeyGemClientError, match="heygem_gpu_release_rejected"):
        client.release_gpu(release_path="/v1/system/gpu/release")


def test_external_service_gpu_release_fails_closed_on_provider_cleanup_error():
    transport = ScriptedTransport(
        [
            {
                "code": 10000,
                "success": True,
                "msg": "released",
                "data": {
                    "error": "torch_cleanup_failed:CUDA error",
                    "cuda_available": True,
                    "cuda_empty_cache": False,
                    "cuda_ipc_collect": False,
                },
            }
        ]
    )
    client = HeyGemClient(endpoint(), transport=transport)

    with pytest.raises(HeyGemClientError, match="torch_cleanup_failed:CUDA error"):
        client.release_gpu(release_path="/v1/system/gpu/release")


@pytest.mark.parametrize(
    ("cuda_empty_cache", "cuda_ipc_collect"),
    [(False, True), (True, False), (False, False)],
)
def test_external_service_gpu_release_requires_complete_cuda_cleanup(
    cuda_empty_cache,
    cuda_ipc_collect,
):
    transport = ScriptedTransport(
        [
            {
                "code": 10000,
                "success": True,
                "msg": "released",
                "data": {
                    "error": "",
                    "cuda_available": True,
                    "cuda_empty_cache": cuda_empty_cache,
                    "cuda_ipc_collect": cuda_ipc_collect,
                },
            }
        ]
    )
    client = HeyGemClient(endpoint(), transport=transport)

    with pytest.raises(HeyGemClientError, match="cuda_cleanup_incomplete"):
        client.release_gpu(release_path="/v1/system/gpu/release")


def test_external_service_gpu_release_does_not_require_cuda_flags_without_cuda():
    response = {
        "code": 10000,
        "success": True,
        "msg": "released",
        "data": {
            "error": "",
            "cuda_available": False,
            "cuda_empty_cache": False,
            "cuda_ipc_collect": False,
        },
    }
    client = HeyGemClient(endpoint(), transport=ScriptedTransport([response]))

    receipt = client.release_gpu(release_path="/v1/system/gpu/release")

    assert receipt.accepted is True
    assert receipt.response == response


def test_external_service_gpu_release_requires_an_explicit_route():
    client = HeyGemClient(endpoint(), transport=ScriptedTransport([]))

    with pytest.raises(HeyGemClientError, match="heygem_gpu_release_path_required"):
        client.release_gpu()
