import pytest

from hydra_heygem.job_identity import JobIdentityError, resolve_job_code


def test_explicit_hydra_job_code_is_preserved_for_receipt_correlation():
    assert (
        resolve_job_code("hydra-heygem-task-123")
        == "hydra-heygem-task-123"
    )


@pytest.mark.parametrize(
    "value",
    ["../escape", "contains spaces", "", "auto", "hydra/heygem"],
)
def test_invalid_or_automatic_job_code_uses_a_generated_hydra_identity(value):
    generated = resolve_job_code(value, uuid_factory=lambda: "fixed-uuid")

    assert generated == "hydra-heygem-fixed-uuid"


def test_unsafe_explicit_job_code_fails_closed_when_generation_is_disabled():
    with pytest.raises(JobIdentityError):
        resolve_job_code("../escape", allow_generated=False)
