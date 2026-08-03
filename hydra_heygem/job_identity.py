from __future__ import annotations

import re
import uuid
from collections.abc import Callable


_SAFE_JOB_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_AUTOMATIC_VALUES = {"", "auto", "default", "generated", "uuid"}


class JobIdentityError(ValueError):
    pass


def resolve_job_code(
    value: object,
    *,
    allow_generated: bool = True,
    uuid_factory: Callable[[], object] = uuid.uuid4,
) -> str:
    requested = str(value or "").strip()
    if requested.lower() in _AUTOMATIC_VALUES:
        if not allow_generated:
            raise JobIdentityError("hydra_heygem_explicit_job_code_required")
        return f"hydra-heygem-{uuid_factory()}"
    if not _SAFE_JOB_CODE.fullmatch(requested):
        if allow_generated:
            return f"hydra-heygem-{uuid_factory()}"
        raise JobIdentityError("hydra_heygem_job_code_invalid")
    return requested
