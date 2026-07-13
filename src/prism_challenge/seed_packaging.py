"""Shared packaging harness for Prism lab seed families (two-script submit zips).

Produces explorer/lab submission zips with a stable outer contract:

- required entry scripts ``architecture.py`` + ``training.py``
- optional ``prism.yaml`` when present in the seed tree
- no miner secrets, wallets, or private keys
- a stable fingerprint surface: sorted path list + per-file SHA-256 + zip content digest

Registers both Transformer ``tiny-1m`` and pure-PyTorch Mamba/SSM ``mamba-tiny`` families
under one outer packaging contract (entry names, fingerprint, dual zip output).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"

REQUIRED_ENTRY_SCRIPTS = ("architecture.py", "training.py")
OPTIONAL_MANIFEST_NAMES = ("prism.yaml", "prism.yml")
# Text / code conjugations ranked for submission mining (ZIP safety already re-applies on ingest).
_ALLOWED_SEED_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".md", ".txt", ".json"})
_SKIP_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"})


@dataclass(frozen=True)
class SeedFamily:
    """One registered lab seed family packing entry."""

    family_id: str
    display_name: str
    architecture_family: str
    source_dir: Path
    description: str
    # Lab knobs relevant when interpreting family scores (documented, not enforced here).
    knobs: Mapping[str, str]


# Registry is the single packaging harness surface. Dual-family zips share this table.
SEED_FAMILIES: dict[str, SeedFamily] = {
    "transformer-tiny-1m": SeedFamily(
        family_id="transformer-tiny-1m",
        display_name="Transformer tiny-1m",
        architecture_family="transformer",
        source_dir=EXAMPLES_ROOT / "tiny-1m",
        description=(
            "Weight-tied ~1M decoder transformer (dim=128, heads=4, 2 layers, SwiGLU) "
            "under the two-script Prism contract and 150M param cap."
        ),
        knobs={
            "param_counting": (
                "Forced-seed build_model(ctx) counts realized nn.Module parameters "
                "(weight tying reduces emb/lm_head double-count)."
            ),
            "step_throughput": (
                "LOCAL_BATCH=4, AdamW lr=0.005, GRAD_CLIP_NORM=1.0; scores are "
                "compute-normalized (tokens), not wall-clock."
            ),
            "stability": (
                "Single-node multi-GPU (≤8) via init_process_group/DDP/"
                "DistributedSampler marker/rank-0 save; works at world_size=1."
            ),
            "tokenizer": "prism.yaml tokenizer=gpt2 (pre-staged offline reference).",
        },
    ),
    "mamba-tiny-1m": SeedFamily(
        family_id="mamba-tiny-1m",
        display_name="Mamba/SSM tiny pure-torch",
        architecture_family="mamba",
        source_dir=EXAMPLES_ROOT / "mamba-tiny",
        description=(
            "Weight-tied ~1M pure-PyTorch selective SSM (Mamba-style) language model "
            "(dim=128, 2 layers, d_state=16) under the two-script Prism contract and "
            "150M param cap. No blocked mamba_ssm C++/CUDA extension is required."
        ),
        knobs={
            "param_counting": (
                "Same architecture-agnostic forced-seed tensor count as Transformer; "
                "SSM params include A_log/D/conv/dt projections and weight-tied emb/head."
            ),
            "step_throughput": (
                "LOCAL_BATCH=4, AdamW lr=0.003 (slightly lower for sequential scan "
                "stability), GRAD_CLIP_NORM=1.0; pure-torch scan is slower than fused "
                "CUDA kernels; scores remain compute-normalized (tokens)."
            ),
            "stability": (
                "Single-node multi-GPU (≤8) via the same distributed primitives as "
                "transformer-tiny-1m; pure PyTorch only (no mamba_ssm / cpp_extension)."
            ),
            "tokenizer": "prism.yaml tokenizer=gpt2 (pre-staged offline reference).",
            "pure_torch_caveat": (
                "Selective scan is implemented in Python/Torch for AST-lab portability; "
                "do not import mamba_ssm for the static lab path."
            ),
        },
    ),
}


@dataclass(frozen=True)
class PackedSeed:
    family_id: str
    zip_path: Path
    entry_names: tuple[str, ...]
    content_sha256: str
    file_digests: Mapping[str, str]
    size_bytes: int


def list_families() -> tuple[str, ...]:
    return tuple(sorted(SEED_FAMILIES))


def get_family(family_id: str) -> SeedFamily:
    try:
        return SEED_FAMILIES[family_id]
    except KeyError as exc:
        known = ", ".join(list_families()) or "(none)"
        raise KeyError(f"unknown seed family {family_id!r}; known: {known}") from exc


def _is_packable(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in _SKIP_DIR_NAMES for part in rel.parts):
        return False
    if path.name.startswith("."):
        return False
    if path.suffix.lower() not in _ALLOWED_SEED_SUFFIXES:
        if path.name not in OPTIONAL_MANIFEST_NAMES:
            return False
    return path.is_file()


def collect_seed_files(source_dir: Path) -> dict[str, bytes]:
    """Collect allowed seed tree files keyed lab path (posix relative)."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"seed source directory missing: {source_dir}")
    files: dict[str, bytes] = {}
    for path in sorted(source_dir.rglob("*")):
        if not _is_packable(path, source_dir):
            continue
        rel = path.relative_to(source_dir).as_posix()
        files[rel] = path.read_bytes()
    for required in REQUIRED_ENTRY_SCRIPTS:
        if required not in files:
            raise FileNotFoundError(f"{source_dir} is missing required entry script {required}")
    return files


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def package_seed_zip(
    family_id: str,
    output_dir: Path | str,
    *,
    zip_name: str | None = None,
) -> PackedSeed:
    """Pack one registered family into a two-script submission zip under ``output_dir``."""
    family = get_family(family_id)
    files = collect_seed_files(family.source_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    name = zip_name or f"{family.family_id}.zip"
    zip_path = out_root / name

    # Deterministic archive: fixed order of names, fixed compression.
    ordered_names = tuple(sorted(files))
    file_digests = {name: _sha256_bytes(files[name]) for name in ordered_names}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ordered_names:
            archive.writestr(name, files[name])

    raw = zip_path.read_bytes()
    return PackedSeed(
        family_id=family.family_id,
        zip_path=zip_path,
        entry_names=ordered_names,
        content_sha256=_sha256_bytes(raw),
        file_digests=file_digests,
        size_bytes=len(raw),
    )


def package_all_families(output_dir: Path | str) -> list[PackedSeed]:
    """Pack every registered family with the same outer two-script contract."""
    return [package_seed_zip(family_id, output_dir) for family_id in list_families()]


def package_report(packed: Iterable[PackedSeed]) -> dict[str, Any]:
    rows = []
    for item in packed:
        rows.append(
            {
                "family_id": item.family_id,
                "zip_path": str(item.zip_path),
                "size_bytes": item.size_bytes,
                "content_sha256": item.content_sha256,
                "entry_names": list(item.entry_names),
                "file_digests": dict(item.file_digests),
            }
        )
    return {"families": rows, "required_entry_scripts": list(REQUIRED_ENTRY_SCRIPTS)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pack Prism lab seed families into submit zips.")
    parser.add_argument(
        "--family",
        action="append",
        dest="families",
        help="Family id to pack (repeatable). Default: all registered families.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist/seed-packages"),
        help="Directory that receives the zip files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable packaging report on stdout.",
    )
    args = parser.parse_args(argv)

    family_ids = args.families or list(list_families())
    packed: list[PackedSeed] = []
    for family_id in family_ids:
        item = package_seed_zip(family_id, args.output_dir)
        packed.append(item)
        if not args.json:
            print(
                f"packed {item.family_id}: {item.zip_path} "
                f"bytes={item.size_bytes} sha256={item.content_sha256}"
            )
            print(f"  entries: {', '.join(item.entry_names)}")

    if args.json:
        print(json.dumps(package_report(packed), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
