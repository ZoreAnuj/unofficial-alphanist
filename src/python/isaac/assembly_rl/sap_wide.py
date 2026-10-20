"""Process-local 64-bit sweep-and-prune work counters for the pinned runtime."""

import hashlib
import importlib.util
import pathlib
import sys
import tempfile

DRIVER_SHA256 = "6a50bbed06fc73ebf88921987f5ffdab5c5da307ff65930d7e9fa92baa8e5810"
CORE_SHA256 = "70efc41cee57876474cba322bf9a93c9119a7578c5de3828be5937c26df9dade"
OVERLAY_SHA256 = "a82cbb3915fe4a83bebc5cd8e967ce5b4f1f1d67ef83784663ba2bef5fc5c91c"
MODULE_NAME = "mujoco_warp._src.collision_driver_sap_wide"
_source_directory = None


def fingerprint() -> dict[str, str]:
    return {
        "driver_sha256": DRIVER_SHA256,
        "core_sha256": CORE_SHA256,
        "overlay_sha256": OVERLAY_SHA256,
    }


def transform_source(driver: bytes, core: bytes) -> bytes:
    if (
        hashlib.sha256(driver).hexdigest() != DRIVER_SHA256
        or hashlib.sha256(core).hexdigest() != CORE_SHA256
    ):
        raise RuntimeError("SAP64 requires the exact audited pinned runtime")
    source = driver.decode()
    core_source = core.decode()
    range_source = core_source[
        core_source.index("@wp.kernel\ndef sap_range(") : core_source.index(
            "\ndef create_collision_context("
        )
    ]
    range_source = range_source.replace(
        "range_out: wp.array2d[int]", "range_out: wp.array2d[wp.int64]"
    )
    range_source = range_source.replace(
        "= limit - sortedid", "= wp.int64(limit - sortedid)"
    )
    source = source.replace(
        "from mujoco_warp._src.collision_core import sap_range\n", ""
    )
    source = source.replace(
        "@cache_kernel\ndef _sap_project(",
        range_source + "\n@cache_kernel\ndef _sap_project(",
    )
    start = source.index("@cache_kernel\ndef _sap_broadphase(")
    end = source.index("@cache_kernel\ndef _segmented_sort(", start)
    kernel = source[start:end]
    kernel = kernel.replace(
        "cumulative_sum_in: wp.array[int]", "cumulative_sum_in: wp.array[wp.int64]"
    )
    kernel = kernel.replace(
        "worldgeomid = wp.tid()", "worldgeomid = wp.int64(wp.tid())"
    )
    kernel = kernel.replace(
        "j = i + worldgeomid + 1", "j = wp.int64(i) + worldgeomid + wp.int64(1)"
    )
    kernel = kernel.replace("j = j % ngeom", "localj = int(j % wp.int64(ngeom))")
    kernel = kernel.replace(
        "sort_index_in[worldid, j]", "sort_index_in[worldid, localj]"
    )
    kernel = kernel.replace(
        "worldgeomid += nsweep_in", "worldgeomid += wp.int64(nsweep_in)"
    )
    source = source[:start] + kernel + source[end:]
    # Pinned cache_kernel keys include function name, so avoid the original cache entry.
    source = source.replace("_sap_broadphase(", "_sap_broadphase_wide(")
    source = source.replace(
        "range_ = wp.empty((d.nworld, m.ngeom), dtype=int)",
        "range_ = wp.empty((d.nworld, m.ngeom), dtype=wp.int64)",
    )
    source = source.replace(
        "cumulative_sum = wp.empty((d.nworld, m.ngeom), dtype=int)",
        "cumulative_sum = wp.empty((d.nworld, m.ngeom), dtype=wp.int64)",
    )
    source = source.replace(
        "  nworldgeom = d.nworld * m.ngeom\n",
        "  nworldgeom = d.nworld * m.ngeom\n  if nworldgeom * 5 > 2147483647 or d.naconmax > 2147483647:\n    raise ValueError('SAP64 local geometry launch/contact dimensions exceed signed32 capacity')\n",
    )
    transformed = source.encode()
    if hashlib.sha256(transformed).hexdigest() != OVERLAY_SHA256:
        raise RuntimeError("SAP64 transform differs from the audited candidate")
    return transformed


def install() -> dict[str, str]:
    """Replace only SAP; preserve the caller's active CCD/narrowphase bindings."""
    from mujoco_warp._src import collision_core, collision_driver

    global _source_directory
    source = transform_source(
        pathlib.Path(collision_driver.__file__).read_bytes(),
        pathlib.Path(collision_core.__file__).read_bytes(),
    )
    identity = fingerprint()
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None:
        if (
            getattr(existing, "_assembly_fingerprint", None) != identity
            or collision_driver.sap_broadphase is not existing.sap_broadphase
        ):
            raise RuntimeError("SAP64 binding conflicts")
        return identity
    if collision_driver.sap_broadphase.__module__ != collision_driver.__name__:
        raise RuntimeError("SAP64 requires the original SAP binding")
    directory = tempfile.TemporaryDirectory(prefix="ur7-sap-wide-")
    path = pathlib.Path(directory.name) / "collision_driver_sap_wide.py"
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
    module._assembly_fingerprint = identity
    collision_driver.sap_broadphase = module.sap_broadphase
    _source_directory = directory
    return identity
