import hashlib
import pathlib
import subprocess
import sys

import pytest

from isaac.assembly_rl import ccd_workspace


def pinned_source() -> bytes:
    from mujoco_warp._src import collision_convex

    return pathlib.Path(collision_convex.__file__).read_bytes()


def test_transform_matches_audited_candidate_and_retains_license() -> None:
    original = pinned_source()
    transformed = ccd_workspace.transform_source(original)
    assert hashlib.sha256(transformed).hexdigest() == (
        "3b3237cdb5040fea088d00186b00455ad326358115f503b5be2dcb3f94ff1fc1"
    )
    assert transformed.split(b"from typing", 1)[0] == original.split(b"from typing", 1)[0]
    assert ccd_workspace.transform_source(original) == transformed


def test_unknown_runtime_source_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="pinned runtime"):
        ccd_workspace.transform_source(pinned_source() + b"\n")


def test_install_is_process_local_idempotent_and_rejects_conflicting_binding() -> None:
    code = """
import hashlib, pathlib, sys
from isaac.assembly_rl import ccd_workspace
from mujoco_warp._src import collision_convex, collision_driver
from mujoco_warp._src.types import GeomType
original = pathlib.Path(collision_convex.__file__).read_bytes()
assert ccd_workspace.MODULE_NAME not in sys.modules
assert collision_driver.convex_narrowphase is collision_convex.convex_narrowphase
args = (GeomType.MESH.value, GeomType.MESH.value, 50, 50, True, 0, 64, False)
original_kernel = collision_convex.ccd_kernel_builder(*args)
first = ccd_workspace.install()
module = sys.modules[ccd_workspace.MODULE_NAME]
assert first == ccd_workspace.fingerprint() == ccd_workspace.install()
assert first['contact_normal'] == 'epa_face_for_zero_witness_v1'
overlay_kernel = module.ccd_kernel_builder_thread_workspace_epa_v1(*args)
assert overlay_kernel is not original_kernel
assert module.ccd_kernel_builder_thread_workspace_epa_v1(*args) is overlay_kernel
assert collision_driver.convex_narrowphase is module.convex_narrowphase
assert module.__package__ == 'mujoco_warp._src'
assert hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest() == first['overlay_sha256']
assert pathlib.Path(collision_convex.__file__).read_bytes() == original
collision_driver.convex_narrowphase = collision_convex.convex_narrowphase
try:
    ccd_workspace.install()
except RuntimeError as error:
    assert 'binding' in str(error)
else:
    raise AssertionError('Conflicting collision binding was accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
