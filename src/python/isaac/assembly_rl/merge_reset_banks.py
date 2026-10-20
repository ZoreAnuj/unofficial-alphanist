"""Merge validated reset shards without mixing models or evaluation states."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import shutil

import numpy as np

from isaac.assembly_rl.reset_bank_builder import FAMILIES, validate_bank_entries


def _sha256(path: pathlib.Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load(path: pathlib.Path) -> tuple[dict, dict, pathlib.Path]:
    metadata = json.loads((path / "bank.json").read_text())
    if not metadata.get("physics_validated") or not metadata.get("physics_identity"):
        raise ValueError(f"Missing validated physics identity: {path}")
    with np.load(path / "bank.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    ids = validate_bank_entries(arrays, metadata)
    if metadata.get("sample_count") != len(ids) or any(
        value.ndim == 0 or len(value) != len(ids) for value in arrays.values()
    ):
        raise ValueError(f"Inconsistent bank array lengths: {path}")
    if len(metadata.get("settled_orientations", [])) != len(ids):
        raise ValueError(f"Missing settled orientation evidence: {path}")
    if (
        "active_part" not in arrays
        or arrays["active_part"].dtype.kind not in "iu"
        or arrays["active_part"].shape != (len(ids),)
    ):
        raise ValueError(
            f"Bank must record integer active_part for every state: {path}"
        )
    log_name = metadata.get("validation_log")
    if not log_name or pathlib.Path(log_name).name != log_name:
        raise ValueError(f"Invalid validation log path: {path}")
    evidence = path / log_name
    if _sha256(evidence) != metadata.get("validation_log_sha256"):
        raise ValueError(f"Validation evidence hash mismatch: {path}")
    return arrays, metadata, evidence


def merge(  # noqa: PLR0912 -- explicit identity checks and balanced quota validation
    inputs: list[pathlib.Path],
    output: pathlib.Path,
    assets: pathlib.Path,
    samples_per_task: int,
    *,
    exclude: list[pathlib.Path] = (),
    compatible_generators: set[str] | None = None,
) -> dict:
    """Select family quotas balanced by movable part, including valid partial shards."""
    output, assets = pathlib.Path(output), pathlib.Path(assets).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new immutable output directory: {output}")
    if (
        not inputs
        or samples_per_task < len(FAMILIES)
        or samples_per_task % len(FAMILIES)
    ):
        raise ValueError("Request a positive multiple of five states per task")
    inputs = list(dict.fromkeys(pathlib.Path(path).resolve() for path in inputs))
    banks = [_load(path) for path in inputs]
    excluded = [_load(pathlib.Path(path)) for path in exclude]
    reference = banks[0][1]
    identity_fields = (
        "schema_version",
        "content_id_schema",
        "env_schema_version",
        "physics_identity",
        "task_names",
        "task_part_counts",
        "source_hashes",
        "validation",
        "grasp_challenge_protocol",
    )

    def schema(arrays: dict) -> dict:
        return {
            key: (
                value.dtype.kind if value.dtype.kind in "US" else value.dtype,
                value.shape[1:],
            )
            for key, value in arrays.items()
        }

    array_schema = schema(banks[0][0])
    generators = set()
    for arrays, metadata, _ in banks + excluded:
        for field in identity_fields:
            if field not in reference or metadata.get(field) != reference[field]:
                raise ValueError(f"Incompatible bank {field}")
        hashes = metadata.get("reset_generator_sha256s") or [
            metadata.get("reset_generator_sha256")
        ]
        if not hashes or any(not isinstance(value, str) or not value for value in hashes):
            raise ValueError("Missing reset generator provenance")
        if metadata.get("reset_generator_sha256") not in (None, *hashes):
            raise ValueError("Conflicting reset generator provenance")
        generators.update(hashes)
        if schema(arrays) != array_schema:
            raise ValueError("Incompatible bank array schema")
    if compatible_generators is None:
        if len(generators) != 1:
            raise ValueError("Incompatible bank reset_generator_sha256")
    elif not generators <= compatible_generators:
        raise ValueError("Unreviewed bank reset_generator_sha256")
    if not reference["source_hashes"]:
        raise ValueError("Missing source hashes")
    for name, digest in reference["source_hashes"].items():
        path = (assets / name).resolve()
        if not path.is_relative_to(assets) or _sha256(path) != digest:
            raise ValueError(f"Source asset hash mismatch: {name}")
    movable = {}
    for index, name in enumerate(reference["task_names"]):
        if f"{name}.json" not in reference["source_hashes"]:
            raise ValueError(f"Missing task asset hash: {name}")
        task = json.loads((assets / f"{name}.json").read_text())
        parts = task["parts"]
        if (
            not task.get("collision_geometry_ready")
            or len(parts) != reference["task_part_counts"][index]
        ):
            raise ValueError(f"Task geometry is not ready: {name}")
        movable[index] = {part for part, spec in enumerate(parts) if not spec["fixed"]}
        if not movable[index]:
            raise ValueError(f"Task has no movable parts: {name}")
        if samples_per_task // len(FAMILIES) < len(movable[index]):
            raise ValueError(f"Family quota cannot cover all movable parts: {name}")
    seen = {item for _, metadata, _ in excluded for item in metadata["entry_ids"]}
    candidates = collections.defaultdict(list)
    discarded = 0
    for bank_index, (arrays, metadata, _) in enumerate(banks):
        for row, content_id in enumerate(metadata["entry_ids"]):
            task, family = int(arrays["task_ids"][row]), str(arrays["family"][row])
            active = int(arrays["active_part"][row])
            if family not in FAMILIES or active not in movable[task]:
                raise ValueError("Invalid family or active movable part")
            if content_id in seen:
                discarded += 1
                continue
            seen.add(content_id)
            candidates[task, family].append((bank_index, row, active))
    quota = samples_per_task // len(FAMILIES)
    selected, deficits = [], {}
    for task, name in enumerate(reference["task_names"]):
        for family in FAMILIES:
            rows = candidates[task, family]
            by_part = {part: [] for part in sorted(movable[task])}
            for row in rows:
                by_part[row[2]].append(row)
            minimum, remainder = divmod(quota, len(by_part))
            extra_parts = [
                part for part, values in by_part.items() if len(values) > minimum
            ][:remainder]
            if (
                any(len(values) < minimum for values in by_part.values())
                or len(extra_parts) < remainder
            ):
                deficits[f"{name}/{family}"] = {
                    "available": len(rows),
                    "required": quota,
                    "missing_active_parts": sorted(
                        part for part, values in by_part.items() if not values
                    ),
                    "available_by_active_part": {
                        str(part): len(values) for part, values in by_part.items()
                    },
                    "minimum_required_per_active_part": minimum,
                    "parts_requiring_one_extra": remainder,
                }
                continue
            for part, values in by_part.items():
                selected.extend(values[: minimum + (part in extra_parts)])
    if deficits:
        raise ValueError(
            "Insufficient unique coverage: " + json.dumps(deficits, sort_keys=True)
        )
    arrays = {
        key: np.stack([banks[bank][0][key][row] for bank, row, _ in selected])
        for key in array_schema
    }
    metadata = dict(reference)
    for field in ("seed", "rejections"):
        metadata.pop(field, None)
    metadata.update(
        sample_count=len(selected),
        training_ready=True,
        physics_validated=True,
        complete_requested_coverage=True,
        requested_samples_per_task=samples_per_task,
        requested_families=list(FAMILIES),
        entry_ids=arrays["entry_ids"].tolist(),
        settled_orientations=[
            banks[bank][1]["settled_orientations"][row] for bank, row, _ in selected
        ],
        physical_batches=sum(bank[1].get("physical_batches", 0) for bank in banks),
        discarded_duplicate_or_excluded_states=discarded,
        coverage={},
        active_part_coverage={},
        reset_generator_sha256=next(iter(generators)) if len(generators) == 1 else None,
        reset_generator_sha256s=sorted(generators),
        merge_sources=[
            {
                "path": str(path),
                "bank_sha256": _sha256(path / "bank.npz"),
                "metadata_sha256": _sha256(path / "bank.json"),
                "validation_log_sha256": bank[1]["validation_log_sha256"],
                "reset_generator_sha256s": bank[1].get("reset_generator_sha256s")
                or [bank[1]["reset_generator_sha256"]],
            }
            for path, bank in zip(inputs, banks, strict=True)
        ],
        excluded_entry_ids=sorted(
            {item for _, meta, _ in excluded for item in meta["entry_ids"]}
        ),
    )
    for task, name in enumerate(reference["task_names"]):
        metadata["coverage"][name] = {family: quota for family in FAMILIES}
        metadata["active_part_coverage"][name] = {
            family: dict(
                collections.Counter(
                    str(int(active))
                    for current_task, current_family, active in zip(
                        arrays["task_ids"],
                        arrays["family"],
                        arrays["active_part"],
                        strict=True,
                    )
                    if int(current_task) == task and str(current_family) == family
                )
            )
            for family in FAMILIES
        }
    validate_bank_entries(arrays, metadata)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "validation.jsonl").open("wb") as destination:
        for _, _, evidence in banks:
            with evidence.open("rb") as source:
                shutil.copyfileobj(source, destination)
    metadata["validation_log"] = "validation.jsonl"
    metadata["validation_log_sha256"] = _sha256(output / "validation.jsonl")
    np.savez_compressed(output / "bank.npz", **arrays)
    (output / "bank.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--exclude", type=pathlib.Path, action="append", default=[])
    parser.add_argument("--assets", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--samples-per-task", type=int, required=True)
    args = parser.parse_args()
    result = merge(
        args.input, args.out, args.assets, args.samples_per_task, exclude=args.exclude
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "sample_count",
                    "training_ready",
                    "coverage",
                    "active_part_coverage",
                    "physics_identity",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
