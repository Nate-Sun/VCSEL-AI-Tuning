from pathlib import Path

from vcsel_ai_tuning.v1_1.config import V11StorageConfig
from vcsel_ai_tuning.v1_1.constants import PRODUCT_VERSION, SCHEMA_VERSION


def test_v11_product_and_schema_versions_are_explicit():
    assert PRODUCT_VERSION == "1.2.0"
    assert SCHEMA_VERSION == 3


def test_v11_storage_defaults_are_isolated_from_v1():
    config = V11StorageConfig()

    assert config.sqlite_path == Path("state/v1.2/campaigns.sqlite3")
    assert config.artifact_root == Path("artifacts/v1.2")
    assert config.sqlite_path != Path("state/ai_tuning/campaigns.sqlite3")
    assert config.artifact_root != Path("artifacts/ai_tuning")


def test_v11_storage_resolves_relative_paths_without_mutating_defaults(tmp_path):
    defaults = V11StorageConfig()

    resolved = defaults.resolved(tmp_path)

    assert resolved.sqlite_path == tmp_path / "state/v1.2/campaigns.sqlite3"
    assert resolved.artifact_root == tmp_path / "artifacts/v1.2"
    assert defaults.sqlite_path == Path("state/v1.2/campaigns.sqlite3")
