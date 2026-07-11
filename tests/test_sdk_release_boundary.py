from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import zipfile
from pathlib import Path

from base.challenge_sdk.app_factory import create_challenge_app
from base.challenge_sdk.auth import build_internal_auth_dependency
from base.challenge_sdk.config import ChallengeSettings
from base.challenge_sdk.executor import DockerExecutor
from base.challenge_sdk.proof import ExecutionProof
from base.challenge_sdk.roles import public_route
from base.challenge_sdk.schemas import HealthResponse, VersionResponse, WeightsResponse
from base.challenge_sdk.version import ARTIFACT_VERSION, RELEASE_MANIFEST


def test_prism_uses_only_the_canonical_base_sdk() -> None:
    assert importlib.util.find_spec("prism_challenge.sdk") is None
    assert importlib.metadata.version("base") == ARTIFACT_VERSION
    assert RELEASE_MANIFEST.artifact_version == ARTIFACT_VERSION

    shared_symbols = (
        create_challenge_app,
        build_internal_auth_dependency,
        ChallengeSettings,
        DockerExecutor,
        ExecutionProof,
        public_route,
        HealthResponse,
        VersionResponse,
        WeightsResponse,
    )
    assert all(symbol.__module__.startswith("base.") for symbol in shared_symbols)


def test_prism_wheel_contains_no_vendored_sdk(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()

    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheelhouse)],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheelhouse.glob("prism_challenge-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()

    assert not any(name.startswith("prism_challenge/sdk/") for name in names)


def test_clean_artifacts_resolve_one_base_sdk(tmp_path: Path) -> None:
    prism_repository = Path(__file__).resolve().parents[1]
    base_repository = prism_repository.parent / "platform"
    wheelhouse = tmp_path / "wheelhouse"
    environment = tmp_path / "environment"
    wheelhouse.mkdir()
    clean_environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    for repository in (base_repository, prism_repository):
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(wheelhouse)],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            env=clean_environment,
        )
    base_wheel = next(wheelhouse.glob("base-*.whl"))
    prism_wheel = next(wheelhouse.glob("prism_challenge-*.whl"))

    subprocess.run(
        ["uv", "venv", "--python", "3.12", str(environment)],
        check=True,
        capture_output=True,
        text=True,
        env=clean_environment,
    )
    python = environment / "bin" / "python"
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), str(base_wheel)],
        check=True,
        capture_output=True,
        text=True,
        env=clean_environment,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            str(prism_wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=clean_environment,
    )

    probe = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            (
                "import importlib.metadata,importlib.util,json;"
                "from base.challenge_sdk.version import RELEASE_MANIFEST;"
                "from prism_challenge.config import PrismSettings;"
                "print(json.dumps({"
                "'base_version':importlib.metadata.version('base'),"
                "'base_requirement':importlib.metadata.requires('prism-challenge'),"
                "'manifest':RELEASE_MANIFEST.model_dump(mode='json'),"
                "'prism_sdk':importlib.util.find_spec('prism_challenge.sdk'),"
                "'sdk_version':PrismSettings().sdk_version"
                "},sort_keys=True,default=str))"
            ),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env=clean_environment,
    )
    evidence = json.loads(probe.stdout)

    assert evidence["base_version"] == evidence["manifest"]["artifact_version"]
    assert evidence["sdk_version"] == evidence["manifest"]["sdk_contract_version"]
    assert evidence["prism_sdk"] is None
    assert any(
        requirement.startswith(
            "base @ https://github.com/BaseIntelligence/base/releases/download/v3.1.0/"
        )
        for requirement in evidence["base_requirement"]
    )
