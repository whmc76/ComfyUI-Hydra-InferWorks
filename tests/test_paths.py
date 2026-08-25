import os
from pathlib import Path

import pytest

from hydra_heygem.paths import (
    ArtifactPathError,
    map_result_to_host,
    prefer_final_muxed_artifact,
    resolve_service_shared_root,
    resolve_shared_local_root,
)


def test_shared_local_root_uses_generic_inferworks_configuration(tmp_path):
    actual = resolve_shared_local_root(
        "auto",
        environ={"INFERWORKS_HEYGEM_SHARED_ROOT": str(tmp_path)},
    )

    assert actual == tmp_path.resolve()


@pytest.mark.skipif(os.name == "nt", reason="requires a POSIX filesystem namespace")
def test_posix_process_maps_windows_host_value_only_through_explicit_local_root(tmp_path):
    actual = resolve_shared_local_root(
        "D:/shared/heygem",
        environ={"INFERWORKS_HEYGEM_SHARED_LOCAL_ROOT": str(tmp_path)},
        platform_name="posix",
    )

    assert actual == tmp_path.resolve()


def test_posix_process_rejects_unmapped_windows_host_value():
    with pytest.raises(ArtifactPathError, match="shared_root_namespace_mismatch"):
        resolve_shared_local_root(
            "D:/shared/heygem",
            environ={},
            platform_name="posix",
        )


def test_service_shared_root_defaults_to_the_comfyui_visible_root(tmp_path):
    assert resolve_service_shared_root(
        "auto",
        local_root=tmp_path,
        environ={},
    ) == str(tmp_path.resolve()).replace("\\", "/")


def test_service_shared_root_supports_a_generic_remote_namespace(tmp_path):
    assert resolve_service_shared_root(
        "auto",
        local_root=tmp_path,
        environ={"INFERWORKS_HEYGEM_SERVICE_SHARED_ROOT": "/srv/heygem-data"},
    ) == "/srv/heygem-data"


def test_container_result_is_mapped_into_the_configured_shared_root(tmp_path):
    expected = tmp_path / "temp" / "job-r.mp4"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"video")

    actual = map_result_to_host(
        "/code/data/temp/job-r.mp4",
        shared_host_root=tmp_path,
        container_data_root="/code/data",
        require_exists=True,
    )

    assert actual == expected.resolve()


def test_bare_result_filename_uses_shared_temp_directory(tmp_path):
    expected = tmp_path / "temp" / "job-r.mp4"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"video")

    actual = map_result_to_host(
        "job-r.mp4",
        shared_host_root=tmp_path,
        container_data_root="/code/data",
        require_exists=True,
    )

    assert actual == expected.resolve()


def test_root_relative_final_result_uses_shared_temp_directory(tmp_path):
    expected = tmp_path / "temp" / "job-r.mp4"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"video")

    actual = map_result_to_host(
        "/job-r.mp4",
        shared_host_root=tmp_path,
        container_data_root="/code/data",
        require_exists=True,
    )

    assert actual == expected.resolve()


def test_transient_result_avi_is_replaced_by_final_muxed_mp4(tmp_path):
    transient = tmp_path / "temp" / "job" / "result.avi"

    actual = prefer_final_muxed_artifact(
        transient,
        code="job",
        shared_host_root=tmp_path,
    )

    assert actual == (tmp_path / "temp" / "job-r.mp4").resolve()


def test_relative_traversal_is_rejected(tmp_path):
    with pytest.raises(ArtifactPathError):
        map_result_to_host(
            "../../outside.mp4",
            shared_host_root=tmp_path,
            container_data_root="/code/data",
        )


def test_http_result_is_preserved_for_download_handling(tmp_path):
    assert (
        map_result_to_host(
            "https://cdn.example.test/result.mp4",
            shared_host_root=tmp_path,
            container_data_root="/code/data",
        )
        == "https://cdn.example.test/result.mp4"
    )
