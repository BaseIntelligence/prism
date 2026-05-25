from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .schemas import ExecutionMode

FINEWEB_EDU_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_FROZEN_REVISION = "fineweb-edu-contract-2026-05-25"
FINEWEB_EDU_SUBSETS: dict[str, dict[str, int | str]] = {
    "sample-10BT": {"token_count": 10_000_000_000, "official_mode": "gpu_proxy_eval"},
    "sample-100BT": {"token_count": 100_000_000_000, "official_mode": "full_scale_eval"},
    "sample-350BT": {"token_count": 350_000_000_000, "official_mode": "phase_2_scale"},
}
LOCAL_FIXTURE_REVISION = "tiny-fineweb-fixture-v1"
LOCAL_FIXTURE_SUBSET = "local_cpu_smoke_fixture"
LOCAL_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "tiny_fineweb_fixture.jsonl"
)
SPLITS: tuple[Literal["train", "validation", "test"], ...] = ("train", "validation", "test")


class FineWebEduDatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class FineWebEduRecord:
    split: Literal["train", "validation", "test"]
    text: str
    token_count: int
    source: str = "fineweb-edu"

    @classmethod
    def from_json(
        cls, payload: dict[str, Any], *, path: Path, line_number: int
    ) -> FineWebEduRecord:
        split = payload.get("split")
        text = payload.get("text")
        token_count = payload.get("token_count")
        if split not in SPLITS:
            raise FineWebEduDatasetError(f"{path}:{line_number} has invalid split {split!r}")
        if not isinstance(text, str) or not text:
            raise FineWebEduDatasetError(f"{path}:{line_number} has missing text")
        if not isinstance(token_count, int) or token_count < 0:
            raise FineWebEduDatasetError(f"{path}:{line_number} has invalid token_count")
        source = payload.get("source", "fineweb-edu")
        if not isinstance(source, str) or not source:
            raise FineWebEduDatasetError(f"{path}:{line_number} has invalid source")
        return cls(split=split, text=text, token_count=token_count, source=source)

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "split": self.split,
            "text": self.text,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class FineWebEduConfig:
    mode: ExecutionMode = ExecutionMode.LOCAL_CPU_SMOKE
    dataset_path: Path | None = None
    contamination_report_path: Path | None = None
    dataset_name: str = FINEWEB_EDU_DATASET
    revision: str | None = None
    subset: str | None = None
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    tokenizer_kind: str = "fixed_prism_hash_tokenizer"
    tokenizer_vocab_size: int = 4096
    evaluator_name: str = "prism_byte_normalized_lm_eval"
    evaluator_version: str = "v1"
    benchmark_fingerprints: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FineWebEduContract:
    config: FineWebEduConfig
    records: tuple[FineWebEduRecord, ...]
    split_fingerprints: dict[str, str]
    tokenizer_fingerprint: str
    evaluator_fingerprint: str
    contamination_report: dict[str, Any]

    def texts(self, split: Literal["train", "validation", "test"] = "train") -> list[str]:
        return [record.text for record in self.records if record.split == split]

    def split_token_counts(self) -> dict[str, int]:
        return {
            split: sum(record.token_count for record in self.records if record.split == split)
            for split in SPLITS
        }

    def byte_normalized_loss_metadata(self) -> dict[str, Any]:
        return {
            "supported": True,
            "metric": "bits_per_byte",
            "normalization_scope": "byte_normalized",
            "text_bytes_counted": True,
            "raw_final_loss_cross_architecture_signal": False,
        }

    def manifest_fields(self) -> dict[str, Any]:
        return {
            "name": self.config.dataset_name,
            "revision": self.config.revision or _default_revision(self.config.mode),
            "train_split_fingerprint": self.split_fingerprints["train"],
            "validation_split_fingerprint": self.split_fingerprints["validation"],
            "test_split_fingerprint": self.split_fingerprints["test"],
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "evaluator_fingerprint": self.evaluator_fingerprint,
            "benchmark_fingerprints": dict(sorted(self.config.benchmark_fingerprints.items())),
            "contamination_report_path": _stringify_path(self.config.contamination_report_path),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "dataset": self.manifest_fields(),
            "subset": self.config.subset or _default_subset(self.config.mode),
            "split_names": {
                "train": self.config.train_split,
                "validation": self.config.validation_split,
                "test": self.config.test_split,
            },
            "split_token_counts": self.split_token_counts(),
            "official_subsets": FINEWEB_EDU_SUBSETS,
            "byte_normalized_loss": self.byte_normalized_loss_metadata(),
            "benchmark_contamination": self.contamination_report,
        }


def fineweb_edu_samples(sample_count: int) -> list[str]:
    contract = load_fineweb_edu_contract(FineWebEduConfig(mode=ExecutionMode.LOCAL_CPU_SMOKE))
    samples = contract.texts("train")
    if not samples:
        raise FineWebEduDatasetError("local_cpu_smoke fixture has no train records")
    repeated: list[str] = []
    while len(repeated) < sample_count:
        repeated.extend(samples)
    return repeated[:sample_count]


def load_fineweb_edu_contract(config: FineWebEduConfig) -> FineWebEduContract:
    resolved = _resolve_config(config)
    records = tuple(_load_jsonl_records(_required_dataset_path(resolved)))
    _require_split_separation(records)
    split_fingerprints: dict[str, str] = {
        split: _fingerprint(
            {
                "dataset": resolved.dataset_name,
                "revision": resolved.revision,
                "subset": resolved.subset,
                "split": split,
                "records": [
                    record.fingerprint_payload() for record in records if record.split == split
                ],
            }
        )
        for split in SPLITS
    }
    tokenizer_fingerprint = _fingerprint(
        {
            "kind": resolved.tokenizer_kind,
            "vocab_size": resolved.tokenizer_vocab_size,
            "normalization": "utf-8-byte-hash",
        }
    )
    evaluator_fingerprint = _fingerprint(
        {
            "name": resolved.evaluator_name,
            "version": resolved.evaluator_version,
            "byte_normalized_loss": {
                "metric": "bits_per_byte",
                "raw_final_loss_cross_architecture_signal": False,
            },
        }
    )
    contamination_report = _contamination_report(resolved)
    return FineWebEduContract(
        config=resolved,
        records=records,
        split_fingerprints=split_fingerprints,
        tokenizer_fingerprint=tokenizer_fingerprint,
        evaluator_fingerprint=evaluator_fingerprint,
        contamination_report=contamination_report,
    )


def fineweb_edu_manifest_fields_for_mode(
    mode: ExecutionMode,
    *,
    benchmark_fingerprints: dict[str, str] | None = None,
    contamination_report_path: Path | None = None,
) -> dict[str, Any]:
    if mode is ExecutionMode.LOCAL_CPU_SMOKE:
        return load_fineweb_edu_contract(
            FineWebEduConfig(
                mode=mode,
                benchmark_fingerprints=benchmark_fingerprints or {},
                contamination_report_path=contamination_report_path,
            )
        ).manifest_fields()
    subset = _default_subset(mode)
    token_count = int(FINEWEB_EDU_SUBSETS[subset]["token_count"])
    split_fingerprints = {
        split: _fingerprint(
            {
                "dataset": FINEWEB_EDU_DATASET,
                "revision": FINEWEB_EDU_FROZEN_REVISION,
                "subset": subset,
                "split": split,
                "token_count": token_count,
                "contract": "offline-frozen-fineweb-edu-spec",
            }
        )
        for split in SPLITS
    }
    tokenizer_fingerprint = _fingerprint(
        {
            "kind": "fixed_prism_hash_tokenizer",
            "vocab_size": 4096,
            "normalization": "utf-8-byte-hash",
        }
    )
    evaluator_fingerprint = _fingerprint(
        {
            "name": "prism_byte_normalized_lm_eval",
            "version": "v1",
            "subset": subset,
            "byte_normalized_loss": True,
        }
    )
    return {
        "name": FINEWEB_EDU_DATASET,
        "revision": FINEWEB_EDU_FROZEN_REVISION,
        "train_split_fingerprint": split_fingerprints["train"],
        "validation_split_fingerprint": split_fingerprints["validation"],
        "test_split_fingerprint": split_fingerprints["test"],
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "evaluator_fingerprint": evaluator_fingerprint,
        "benchmark_fingerprints": dict(sorted((benchmark_fingerprints or {}).items())),
        "contamination_report_path": _stringify_path(contamination_report_path),
    }


def _resolve_config(config: FineWebEduConfig) -> FineWebEduConfig:
    if config.mode is ExecutionMode.LOCAL_CPU_SMOKE:
        return FineWebEduConfig(
            mode=config.mode,
            dataset_path=config.dataset_path or LOCAL_FIXTURE_PATH,
            contamination_report_path=config.contamination_report_path,
            dataset_name=config.dataset_name,
            revision=config.revision or LOCAL_FIXTURE_REVISION,
            subset=config.subset or LOCAL_FIXTURE_SUBSET,
            train_split=config.train_split,
            validation_split=config.validation_split,
            test_split=config.test_split,
            tokenizer_kind=config.tokenizer_kind,
            tokenizer_vocab_size=config.tokenizer_vocab_size,
            evaluator_name=config.evaluator_name,
            evaluator_version=config.evaluator_version,
            benchmark_fingerprints=config.benchmark_fingerprints,
        )
    return FineWebEduConfig(
        mode=config.mode,
        dataset_path=config.dataset_path,
        contamination_report_path=config.contamination_report_path,
        dataset_name=config.dataset_name,
        revision=config.revision or FINEWEB_EDU_FROZEN_REVISION,
        subset=config.subset or _default_subset(config.mode),
        train_split=config.train_split,
        validation_split=config.validation_split,
        test_split=config.test_split,
        tokenizer_kind=config.tokenizer_kind,
        tokenizer_vocab_size=config.tokenizer_vocab_size,
        evaluator_name=config.evaluator_name,
        evaluator_version=config.evaluator_version,
        benchmark_fingerprints=config.benchmark_fingerprints,
    )


def _required_dataset_path(config: FineWebEduConfig) -> Path:
    if config.dataset_path is None:
        raise FineWebEduDatasetError(
            f"{config.mode.value} requires configured FineWeb-Edu dataset_path; official modes "
            "must not fall back to the local CPU fixture"
        )
    path = config.dataset_path
    if not path.exists():
        raise FineWebEduDatasetError(f"FineWeb-Edu dataset path does not exist: {path}")
    if path.is_dir():
        split_file = path / f"{config.subset}.jsonl"
        if split_file.exists():
            return split_file
        raise FineWebEduDatasetError(f"FineWeb-Edu shard file is missing: {split_file}")
    return path


def _load_jsonl_records(path: Path) -> list[FineWebEduRecord]:
    records: list[FineWebEduRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise FineWebEduDatasetError(f"{path}:{line_number} must be a JSON object")
        records.append(FineWebEduRecord.from_json(payload, path=path, line_number=line_number))
    if not records:
        raise FineWebEduDatasetError(f"FineWeb-Edu dataset path has no records: {path}")
    return records


def _require_split_separation(records: tuple[FineWebEduRecord, ...]) -> None:
    missing = [
        split for split in SPLITS if not any(record.split == split for record in records)
    ]
    if missing:
        raise FineWebEduDatasetError(
            f"FineWeb-Edu contract requires train/validation/test splits; missing {missing}"
        )


def _contamination_report(config: FineWebEduConfig) -> dict[str, Any]:
    if config.mode is ExecutionMode.LOCAL_CPU_SMOKE:
        return {
            "required": False,
            "summary": (
                "local_cpu_smoke fixture validates wiring only; benchmark scoring is not official"
            ),
            "path": _stringify_path(config.contamination_report_path),
        }
    if config.contamination_report_path is None:
        raise FineWebEduDatasetError(
            f"{config.mode.value} requires benchmark-contamination report metadata"
        )
    if not config.contamination_report_path.exists():
        raise FineWebEduDatasetError(
            f"benchmark-contamination report is missing: {config.contamination_report_path}"
        )
    return {
        "required": True,
        "summary": "official mode includes benchmark-contamination filtering/disclosure metadata",
        "path": str(config.contamination_report_path),
        "fingerprint": _fingerprint(
            {
                "path": str(config.contamination_report_path),
                "bytes": config.contamination_report_path.stat().st_size,
                "sha256": hashlib.sha256(
                    config.contamination_report_path.read_bytes()
                ).hexdigest(),
            }
        ),
    }


def _default_revision(mode: ExecutionMode) -> str:
    if mode is ExecutionMode.LOCAL_CPU_SMOKE:
        return LOCAL_FIXTURE_REVISION
    return FINEWEB_EDU_FROZEN_REVISION


def _default_subset(mode: ExecutionMode) -> str:
    if mode is ExecutionMode.LOCAL_CPU_SMOKE:
        return LOCAL_FIXTURE_SUBSET
    if mode is ExecutionMode.GPU_PROXY_EVAL:
        return "sample-10BT"
    if mode is ExecutionMode.FULL_SCALE_EVAL:
        return "sample-100BT"
    raise FineWebEduDatasetError(f"unsupported execution mode: {mode}")


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stringify_path(path: Path | None) -> str | None:
    return None if path is None else str(path)
