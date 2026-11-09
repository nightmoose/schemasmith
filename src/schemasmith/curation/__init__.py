"""Contract gate, exact dedup, and the optional CuratorKIT pass."""

from schemasmith.curation.curatorkit import (
    CuratorKitUnavailable,
    curate_with_curatorkit,
)
from schemasmith.curation.local import CurationResult, curate_local

__all__ = [
    "CurationResult",
    "CuratorKitUnavailable",
    "curate_local",
    "curate_with_curatorkit",
]
