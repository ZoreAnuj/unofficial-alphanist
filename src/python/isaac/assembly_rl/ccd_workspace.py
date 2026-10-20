"""Process-local CCD workspace and contact normals for the audited runtime."""

import hashlib
import importlib.util
import pathlib
import sys
import tempfile

BASELINE_SHA256 = "9a030d1756b0c0cc6bc9df818011b0c44859bc45420cd37254df7087773bc85c"
OVERLAY_SHA256 = "3b3237cdb5040fea088d00186b00455ad326358115f503b5be2dcb3f94ff1fc1"
MODULE_NAME = "mujoco_warp._src.collision_convex_thread_workspace"
_source_directory = None


def fingerprint() -> dict[str, str]:
    return {
        "workspace": "grid_stride_thread_v1",
        "contact_normal": "epa_face_for_zero_witness_v1",
        "baseline_sha256": BASELINE_SHA256,
        "overlay_sha256": OVERLAY_SHA256,
        "module": MODULE_NAME,
    }


def transform_source(original: bytes) -> bytes:
    if hashlib.sha256(original).hexdigest() != BASELINE_SHA256:
        raise RuntimeError("CCD workspace requires the exact audited pinned runtime")
    source = original.decode("utf-8")
    source = source.replace(
        "    naccdmax_in: int,\n    # In:\n    epa_vert_in:",
        "    naccdmax_in: int,\n    scratch_slot: int,\n    # In:\n    epa_vert_in:",
    )
    source = source.replace(
        "      dist, ncollision, w1, w2, multiccd_idx = epa_phase(",
        "      # Each grid-stride thread completes this pair before reusing its row.\n"
        "      ccdid = scratch_slot\n"
        "      dist, ncollision, w1, w2, multiccd_idx = epa_phase(",
    )
    source = source.replace(
        "        naccdmax_in,\n        epa_vert_in,",
        "        naccdmax_in,\n        tid,\n        epa_vert_in,",
    )
    source = source.replace(
        "  # ccd collider count\n",
        """  # Temporary workspace belongs to concurrently executing threads, not all pairs.
  # Keep the total CCD overflow counter/capacity independent of scratch storage.
  ccd_launches = {}
  scratch_rows = 0
  for geom_pair in collision_table:
    g1, g2 = geom_pair[0].value, geom_pair[1].value
    count, geomgeomid = _pair_count(g1, g2)
    if not count:
      continue
    if g1 == GeomType.HFIELD or g2 == GeomType.HFIELD:
      # Heightfield kernels keep their original counter-indexed workspace.
      scratch_rows = d.naccdmax
      continue
    ccd_k = ccd_kernel_builder(
      g1, g2, m.opt.ccd_iterations, epa_iterations, use_multiccd,
      geomgeomid, m.block_dim.convex_ccd, bool(m.opt.warn_overflow),
    )
    ccd_grid = min(_ccd_grid_size(ccd_k, d.naconmax, d.ncollision.device), max(1, d.naccdmax))
    ccd_launches[geom_pair] = (ccd_k, ccd_grid)
    scratch_rows = max(scratch_rows, min(ccd_grid, d.naccdmax))

  # ccd collider count
""",
    )
    source = source.replace("shape=(d.naccdmax,", "shape=(scratch_rows,")
    source = source.replace(
        """      ccd_k = ccd_kernel_builder(
        g1,
        g2,
        m.opt.ccd_iterations,
        epa_iterations,
        use_multiccd,
        geomgeomid,
        m.block_dim.convex_ccd,
        bool(m.opt.warn_overflow),
      )
      ccd_grid = _ccd_grid_size(ccd_k, d.naconmax, d.ncollision.device)
""",
        "      ccd_k, ccd_grid = ccd_launches[geom_pair]\n",
    )
    source = source.replace(
        "    frame = make_frame(witness1[0] - witness2[0])",
        """    normal = witness1[0] - witness2[0]
    # EPA's projected Minkowski face equals geom1 witness minus geom2 witness.
    # Retain it when subtraction of nearly equal world-space points rounds to zero.
    if wp.length_sq(normal) == 0.0 and multiccd_idx >= 0:
      epa_normal = epa_pr_in[ccdid, multiccd_idx]
      if wp.length_sq(epa_normal) > 0.0 and wp.isfinite(epa_normal):
        normal = epa_normal
    frame = make_frame(normal)""",
    )
    # MJWarp's global factory cache keys use function names, not module names.
    source = source.replace(
        "ccd_kernel_builder(", "ccd_kernel_builder_thread_workspace_epa_v1("
    )
    transformed = source.encode("utf-8")
    if hashlib.sha256(transformed).hexdigest() != OVERLAY_SHA256:
        raise RuntimeError("CCD workspace transform differs from the audited candidate")
    return transformed


def install() -> dict[str, str]:
    """Install before environment construction; never edit installed runtime files."""
    from mujoco_warp._src import collision_convex, collision_driver

    global _source_directory
    source = transform_source(pathlib.Path(collision_convex.__file__).read_bytes())
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None:
        if (
            getattr(existing, "_assembly_workspace_fingerprint", None) != fingerprint()
            or collision_driver.convex_narrowphase is not existing.convex_narrowphase
        ):
            raise RuntimeError("CCD workspace module or collision binding conflicts")
        return fingerprint()
    if collision_driver.convex_narrowphase is not collision_convex.convex_narrowphase:
        raise RuntimeError("CCD workspace requires the original collision binding")
    # Warp inspects source files lazily; retain this process-owned file until exit.
    directory = tempfile.TemporaryDirectory(prefix="ur7-ccd-workspace-")
    path = pathlib.Path(directory.name) / "collision_convex_thread_workspace.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(MODULE_NAME, None)
        directory.cleanup()
        raise
    module._assembly_workspace_fingerprint = fingerprint()
    collision_driver.convex_narrowphase = module.convex_narrowphase
    _source_directory = directory
    return fingerprint()
