from pathlib import Path

import pytest

from hydra_heygem.paths import (
    ArtifactPathError,
    map_result_to_host,
    prefer_final_muxed_artifact,
)


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
