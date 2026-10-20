"""Insertion checks must respect sequence without hiding final interference."""

import copy

import numpy as np
import trimesh

from isaac.assembly_rl import prepare_expansion_assets


def test_dependent_part_is_absent_only_during_prerequisite_insertion():
    parts = []
    for index, name in enumerate(("Base", "Lower", "Upper")):
        pose = np.eye(4)
        pose[2, 3] = index * 0.01
        parts.append(
            {
                "name": name,
                "fixed": index == 0,
                "assembled_transform": pose.tolist(),
                "disassembly_axis_local": [0, 0, 1],
                "requires_assembled": ["Lower"] if index == 2 else [],
            }
        )
    task = {"parts": parts}
    cube = trimesh.creation.box(extents=[0.004] * 3).vertices
    hulls = {part["name"]: [cube] for part in parts}
    assert prepare_expansion_assets.fit_task(task, hulls)["passed"]
    unordered = copy.deepcopy(task)
    unordered["parts"][2]["requires_assembled"] = []
    assert not prepare_expansion_assets.fit_task(unordered, hulls)["passed"]
    intersecting = copy.deepcopy(task)
    intersecting["parts"][2]["assembled_transform"][2][3] = 0.011
    assert not prepare_expansion_assets.fit_task(intersecting, hulls)["passed"]
