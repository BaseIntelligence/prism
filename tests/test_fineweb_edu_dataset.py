from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from prism_challenge.evaluator.dataset import (
    FINEWEB_EDU_DATASET,
    FINEWEB_EDU_FROZEN_REVISION,
    FINEWEB_EDU_SUBSETS,
    FineWebEduConfig,
    FineWebEduDatasetError,
    fineweb_edu_samples,
    load_fineweb_edu_contract,
)
from prism_challenge.evaluator.schemas import ExecutionMode

SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
FIXTURE = Path("tests/fixtures/tiny_fineweb_fixture.jsonl")


def test_local_fixture_loads_with_stable_fingerprints() -> None:
    contract = load_fineweb_edu_contract(
        FineWebEduConfig(mode=ExecutionMode.LOCAL_CPU_SMOKE, dataset_path=FIXTURE)
    )

    assert contract.config.dataset_name == FINEWEB_EDU_DATASET
    assert contract.texts("train") == [
        "A prism separates light into measurable bands for a science lesson.",
        "Students compare two algorithms by recording loss after each step.",
    ]
    assert contract.split_token_counts() == {"train": 23, "validation": 21, "test": 21}
    assert contract.split_fingerprints == {
        "train": "8bed91db84b66d4125a645a2b38c2f46c7b6caa979bcf416a63c18f46674e960",
        "validation": "77dd16f47bd67ef09e9870f7cc6685ce8ec24c7fc5a837ae2fda5eaddc190f93",
        "test": "0813a7f773eced77ea7c930ec58c81d13b9af2e5f69eb4c66f3790117f5b07b0",
    }
    assert SHA256_RE.match(contract.tokenizer_fingerprint)
    assert SHA256_RE.match(contract.evaluator_fingerprint)
    loss_metadata = contract.byte_normalized_loss_metadata()
    assert loss_metadata["raw_final_loss_cross_architecture_signal"] is False
    assert contract.contamination_report["required"] is False


def test_local_cpu_smoke_samples_use_tiny_fixture() -> None:
    samples = fineweb_edu_samples(5)

    assert samples == [
        "A prism separates light into measurable bands for a science lesson.",
        "Students compare two algorithms by recording loss after each step.",
        "A prism separates light into measurable bands for a science lesson.",
        "Students compare two algorithms by recording loss after each step.",
        "A prism separates light into measurable bands for a science lesson.",
    ]


@pytest.mark.parametrize(
    "mode,subset",
    [
        (ExecutionMode.GPU_PROXY_EVAL, "sample-10BT"),
        (ExecutionMode.FULL_SCALE_EVAL, "sample-100BT"),
    ],
)
def test_official_requires_dataset_path_and_does_not_fallback(
    mode: ExecutionMode, subset: str
) -> None:
    assert FINEWEB_EDU_SUBSETS[subset]["token_count"] > 0

    with pytest.raises(
        FineWebEduDatasetError, match="requires configured FineWeb-Edu dataset_path"
    ):
        load_fineweb_edu_contract(FineWebEduConfig(mode=mode))


def test_official_requires_benchmark_contamination_report(tmp_path: Path) -> None:
    dataset_path = tmp_path / "sample-10BT.jsonl"
    dataset_path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(FineWebEduDatasetError, match="benchmark-contamination report"):
        load_fineweb_edu_contract(
            FineWebEduConfig(mode=ExecutionMode.GPU_PROXY_EVAL, dataset_path=dataset_path)
        )


def test_official_contract_uses_frozen_subset_revision_and_report(tmp_path: Path) -> None:
    dataset_path = tmp_path / "sample-10BT.jsonl"
    contamination_path = tmp_path / "contamination.json"
    dataset_path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    contamination_path.write_text(
        json.dumps({"benchmarks_checked": ["mmlu"], "overlaps_found": 0}, sort_keys=True),
        encoding="utf-8",
    )

    contract = load_fineweb_edu_contract(
        FineWebEduConfig(
            mode=ExecutionMode.GPU_PROXY_EVAL,
            dataset_path=dataset_path,
            contamination_report_path=contamination_path,
            benchmark_fingerprints={"mmlu": "mmlu-frozen-v1"},
        )
    )

    manifest_fields = contract.manifest_fields()
    assert contract.config.revision == FINEWEB_EDU_FROZEN_REVISION
    assert contract.config.subset == "sample-10BT"
    assert manifest_fields["contamination_report_path"] == str(contamination_path)
    assert manifest_fields["benchmark_fingerprints"] == {"mmlu": "mmlu-frozen-v1"}
    assert contract.metadata()["official_subsets"]["sample-350BT"]["token_count"] == 350_000_000_000
    assert SHA256_RE.match(contract.contamination_report["fingerprint"])


def test_fixture_must_keep_train_validation_test_separate(tmp_path: Path) -> None:
    train_only = tmp_path / "train-only.jsonl"
    train_only.write_text(
        '{"split":"train","text":"only train data","token_count":3}\n', encoding="utf-8"
    )

    with pytest.raises(FineWebEduDatasetError, match="requires train/validation/test splits"):
        load_fineweb_edu_contract(
            FineWebEduConfig(mode=ExecutionMode.LOCAL_CPU_SMOKE, dataset_path=train_only)
        )
