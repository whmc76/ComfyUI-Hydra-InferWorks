import pytest

from hydra_heygem.config import EndpointConfigError, resolve_endpoint_config


def test_explicit_full_url_accepts_any_valid_custom_port():
    config = resolve_endpoint_config(
        service_url="http://10.20.30.40:58123/base/",
        service_host="auto",
        service_port=0,
        environ={"HYDRA_HEYGEM_SERVICE_URL": "http://127.0.0.1:49202"},
    )

    assert config.base_url == "http://10.20.30.40:58123/base"
    assert config.host == "10.20.30.40"
    assert config.port == 58123
    assert config.source == "node_service_url"


def test_auto_endpoint_requires_explicit_or_environment_configuration():
    with pytest.raises(EndpointConfigError, match="service_endpoint_required"):
        resolve_endpoint_config(
            service_url="auto",
            service_host="auto",
            service_port=0,
            environ={},
        )


def test_inferworks_environment_url_precedes_hydra_compatibility_alias():
    config = resolve_endpoint_config(
        service_url="auto",
        service_host="auto",
        service_port=0,
        environ={
            "INFERWORKS_HEYGEM_SERVICE_URL": "https://portable.example.test:7443",
            "HYDRA_HEYGEM_SERVICE_URL": "http://hydra.internal:49202",
        },
    )

    assert config.base_url == "https://portable.example.test:7443"
    assert config.source == "environment_service_url"


def test_generic_environment_components_accept_any_configured_port():
    config = resolve_endpoint_config(
        service_url="auto",
        service_host="auto",
        service_port=0,
        environ={
            "INFERWORKS_HEYGEM_SCHEME": "http",
            "INFERWORKS_HEYGEM_HOST": "heygem.example.test",
            "INFERWORKS_HEYGEM_PORT": "61007",
        },
    )

    assert config.base_url == "http://heygem.example.test:61007"
    assert config.source == "environment_components"


def test_explicit_port_overrides_environment_url_without_a_fixed_port_contract():
    config = resolve_endpoint_config(
        service_url="auto",
        service_host="auto",
        service_port=61007,
        environ={"HYDRA_HEYGEM_SERVICE_URL": "http://heygem.internal:49202"},
    )

    assert config.base_url == "http://heygem.internal:61007"
    assert config.port == 61007
    assert config.source == "node_host_or_port"


def test_environment_url_remains_the_zero_configuration_path():
    config = resolve_endpoint_config(
        service_url="auto",
        service_host="auto",
        service_port=0,
        environ={"HYDRA_AVATAR_SERVICE_URL": "https://avatar.example.test:7443"},
    )

    assert config.base_url == "https://avatar.example.test:7443"
    assert config.port == 7443
    assert config.source == "environment_service_url"


def test_hydra_environment_url_remains_a_compatibility_alias():
    config = resolve_endpoint_config(
        service_url="auto",
        service_host="auto",
        service_port=0,
        environ={"HYDRA_HEYGEM_SERVICE_URL": "http://hydra.internal:49202"},
    )

    assert config.base_url == "http://hydra.internal:49202"


@pytest.mark.parametrize("port", [-1, 65536])
def test_invalid_explicit_port_fails_closed(port):
    with pytest.raises(EndpointConfigError):
        resolve_endpoint_config(
            service_url="auto",
            service_host="127.0.0.1",
            service_port=port,
            environ={},
        )


def test_non_http_service_url_fails_closed():
    with pytest.raises(EndpointConfigError):
        resolve_endpoint_config(
            service_url="file:///tmp/heygem.sock",
            service_host="auto",
            service_port=0,
            environ={},
        )
