"""CPU-only merge checks using independently identified physical states."""

import hashlib
import json

import numpy as np
import pytest

from isaac.assembly_rl import reset_bank_builder as builder


def test_merge_deduplicates_excludes_and_rejects_incompatible_or_short_banks(tmp_path):
    from isaac.assembly_rl.merge_reset_banks import merge

    assets = tmp_path / "assets"
    assets.mkdir()
    names = ["one_movable", "two_movable", "three_movable"]
    hashes = {}
    for index, name in enumerate(names):
        path = assets / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "collision_geometry_ready": True,
                    "parts": [{"fixed": True}] + [{"fixed": False}] * (index + 1),
                }
            )
        )
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    def shard(name, offset, active_parts=None):
        path = tmp_path / name
        path.mkdir()
        rows = []
        for task in range(3):
            for family in builder.FAMILIES:
                for active in (active_parts or {}).get(
                    task, [1 + sample % (task + 1) for sample in range(3)]
                ):
                    poses = np.zeros((4, 7))
                    poses[:, 6] = 1
                    poses[active, 0] = offset + len(rows) * 0.01
                    row = {
                        "task_ids": task,
                        "part_poses": poses,
                        "goals": poses.copy(),
                        "joint_q": np.zeros(8),
                        "joint_qd": np.zeros(8),
                        "joint_targets": np.zeros(8),
                        "jaw_servo": np.zeros(2),
                        "arm_servo": np.zeros(6),
                        "arm_servo_velocity": np.zeros(6),
                        "part_velocities": np.zeros((4, 6)),
                        "gripper_open": 1.0,
                        "family": family,
                        "active_part": active,
                    }
                    row["entry_ids"] = builder.entry_content_id(
                        row, names[task], task + 2
                    )
                    rows.append(row)
        arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
        np.savez_compressed(path / "bank.npz", **arrays)
        evidence = b'{"batch": 0, "entries": []}\n'
        (path / "validation.jsonl").write_bytes(evidence)
        metadata = {
            "schema_version": 2,
            "content_id_schema": 2,
            "env_schema_version": 1,
            "physics_validated": True,
            "training_ready": True,
            "physics_identity": "same-model",
            "reset_generator_sha256": "same-generator",
            "grasp_challenge_protocol": "same-protocol",
            "task_names": names,
            "task_part_counts": [2, 3, 4],
            "source_hashes": hashes,
            "sample_count": len(rows),
            "entry_ids": arrays["entry_ids"].tolist(),
            "validation": {"stable_steps": 5},
            "validation_log": "validation.jsonl",
            "validation_log_sha256": hashlib.sha256(evidence).hexdigest(),
            "settled_orientations": [
                ["upright"] * (row["task_ids"] + 2) for row in rows
            ],
        }
        (path / "bank.json").write_text(json.dumps(metadata))
        return path

    first, second = shard("first", 0.1), shard("second", 1.0)
    output = tmp_path / "merged"
    result = merge([first, second, second], output, assets, 15, exclude=[first])
    assert result["training_ready"] and result["sample_count"] == 45
    assert set(result["entry_ids"]).isdisjoint(
        json.loads((first / "bank.json").read_text())["entry_ids"]
    )
    assert all(
        count == 3 for row in result["coverage"].values() for count in row.values()
    )
    assert all(
        set(parts) == {str(part) for part in range(1, index + 2)}
        for index, name in enumerate(names)
        for parts in result["active_part_coverage"][name].values()
    )
    assert (
        hashlib.sha256((output / "validation.jsonl").read_bytes()).hexdigest()
        == result["validation_log_sha256"]
    )
    with np.load(output / "bank.npz", allow_pickle=False) as arrays:
        builder.validate_bank_entries(dict(arrays), result)
    imbalanced = shard(
        "imbalanced",
        2.0,
        {0: [1] * 12, 1: [1] * 9 + [2] * 3, 2: [3] * 9 + [1] * 2 + [2] * 2},
    )
    for quota in (6, 7):
        balanced = merge(
            [imbalanced], tmp_path / f"balanced_{quota}", assets, 5 * quota
        )
        for families in balanced["active_part_coverage"].values():
            for counts in families.values():
                assert sum(counts.values()) == quota
                assert max(counts.values()) - min(counts.values()) <= 1
        assert len(set(balanced["entry_ids"])) == balanced["sample_count"]
    with pytest.raises(ValueError, match="available_by_active_part"):
        merge([imbalanced], tmp_path / "imbalanced_deficit", assets, 45)
    assert not (tmp_path / "imbalanced_deficit").exists()
    with pytest.raises(FileExistsError):
        merge([second], output, assets, 10)
    with pytest.raises(ValueError, match="coverage"):
        merge([second], tmp_path / "short", assets, 20)
    assert not (tmp_path / "short").exists()
    with pytest.raises(ValueError, match="quota"):
        merge([second], tmp_path / "small_quota", assets, 10)
    metadata = json.loads((second / "bank.json").read_text())
    metadata["physics_identity"] = "other-model"
    (second / "bank.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="physics_identity"):
        merge([first, second], tmp_path / "mixed", assets, 10)
    assert not (tmp_path / "mixed").exists()
    metadata["physics_identity"] = "same-model"
    metadata["reset_generator_sha256"] = "reviewed-proposal-change"
    (second / "bank.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="reset_generator_sha256"):
        merge([first, second], tmp_path / "unreviewed", assets, 15)
    reviewed = {"same-generator", "reviewed-proposal-change"}
    combined = merge(
        [first, second], tmp_path / "reviewed", assets, 15,
        compatible_generators=reviewed,
    )
    assert combined["reset_generator_sha256"] is None
    assert set(combined["reset_generator_sha256s"]) == reviewed
    assert combined["merge_sources"][1]["reset_generator_sha256s"] == [
        "reviewed-proposal-change"
    ]
    with pytest.raises(ValueError, match="Unreviewed"):
        merge(
            [first, second], tmp_path / "wrong_review", assets, 15,
            compatible_generators={"same-generator"},
        )
    metadata["validation"]["stable_steps"] = 1
    (second / "bank.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="validation"):
        merge(
            [first, second], tmp_path / "changed_validation", assets, 15,
            compatible_generators=reviewed,
        )
    with np.load(first / "bank.npz", allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays["active_part"][:] = 1
    np.savez_compressed(first / "bank.npz", **arrays)
    with pytest.raises(ValueError, match="missing_active_parts"):
        merge([first], tmp_path / "missing_part", assets, 15)
    arrays["joint_q"][0, 0] = 0.5
    np.savez_compressed(first / "bank.npz", **arrays)
    with pytest.raises(ValueError, match="content IDs"):
        merge([first], tmp_path / "tampered", assets, 10)
