"""Exact, process-local host initialization for the pinned MuJoCo Warp runtime."""

import ast
import hashlib
import pathlib
from typing import Any

import numpy as np

SOURCE_SHA256 = "f9214e355f92a89583d85af04867403d1e8f81fcd45c907114d0baa279aa507d"
OVERLAY_SHA256 = "0b1b12136281a13b2675393855f0d4a9296589204ec9e16e0d6925258506e54c"
_installed = None


def fingerprint() -> dict[str, str]:
    return {
        "source_sha256": SOURCE_SHA256,
        "overlay_sha256": OVERLAY_SHA256,
        "implementation_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()
        ).hexdigest(),
        "initialization": "chunked_pair_histogram_body_nnz_v1",
    }


def _geom_pair_type_count(
    geom_types: np.ndarray, pairs: np.ndarray, type_count: int
) -> tuple[int, ...]:
    counts = np.zeros(type_count * (type_count + 1) // 2, dtype=np.int64)
    # Bound temporary storage independently of the hundreds of millions of pairs.
    for start in range(0, len(pairs), 1 << 20):
        chunk = pairs[start : start + (1 << 20)]
        first, second = geom_types[chunk[:, 0]], geom_types[chunk[:, 1]]
        low, high = np.minimum(first, second), np.maximum(first, second)
        indices = low * (2 * type_count - low - 1) // 2 + high
        counts += np.bincount(indices, minlength=len(counts))
    return tuple(counts)


def _body_contact_nnz(mjm: Any) -> int:
    from mujoco_warp._src import io

    contype = np.zeros(mjm.nbody, dtype=mjm.geom_contype.dtype)
    conaffinity = np.zeros(mjm.nbody, dtype=mjm.geom_conaffinity.dtype)
    # OR distributes over AND: a body pair passes iff any of its geom pairs passes.
    np.bitwise_or.at(contype, mjm.geom_bodyid, mjm.geom_contype)
    np.bitwise_or.at(conaffinity, mjm.geom_bodyid, mjm.geom_conaffinity)
    bodies = np.unique(mjm.geom_bodyid)
    result = 0
    for index, first in enumerate(bodies):
        for second in bodies[index + 1 :]:
            if mjm.body_weldid[first] == 0 and mjm.body_weldid[second] == 0:
                continue
            if (contype[first] & conaffinity[second]) or (
                contype[second] & conaffinity[first]
            ):
                result = max(result, io._body_pair_nnz(mjm, first, second))
    return result


def transform_source(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Initialization requires the exact audited runtime")
    text = source.decode()
    text = text.replace(
        """  m.geom_pair_type_count = tuple(
    np.bincount(
      [geom_trid_index(mjm.geom_type[geom1[i]], mjm.geom_type[geom2[i]]) for i in np.arange(len(geom1)) if nxn_include[i]],
      minlength=len(types.GeomType) * (len(types.GeomType) + 1) // 2,
    )
  )""",
        """  m.geom_pair_type_count = _assembly_geom_pair_type_count(
    mjm.geom_type, m.nxn_geom_pair_filtered, len(types.GeomType)
  )""",
    )
    text = text.replace(
        "    for idx in np.nonzero(nxn_include & (nxn_pairid_contact == -1))[0]:",
        "    for idx in (np.nonzero(nxn_include & (nxn_pairid_contact == -1))[0]\n"
        "                if np.any(mjm.geom_margin) else ()):",
    )
    start = text.index("  # filter geom-geom pairs (unique body pairs, filtered)")
    end = text.index("  if mjm.nflex == 0:", start)
    text = (
        text[:start]
        + "  max_contact_nnz = max(max_contact_nnz, _assembly_body_contact_nnz(mjm))\n\n"
        + text[end:]
    )
    transformed = text.encode()
    if hashlib.sha256(transformed).hexdigest() != OVERLAY_SHA256:
        raise RuntimeError("Initialization transform differs from audited source")
    return transformed


def install() -> dict[str, str]:
    """Replace host calculations only; preserve pair tables and physics kernels."""
    import mujoco_warp
    from mujoco_warp._src import io

    global _installed
    source = transform_source(pathlib.Path(io.__file__).read_bytes())
    if _installed is not None:
        if (
            io.put_model is not _installed[0]
            or mujoco_warp.put_model is not _installed[0]
            or io._calculate_max_contact_nnz is not _installed[1]
        ):
            raise RuntimeError("Initialization binding conflicts")
        return fingerprint()
    if (
        io.put_model.__module__ != io.__name__
        or mujoco_warp.put_model is not io.put_model
        or io._calculate_max_contact_nnz.__module__ != io.__name__
    ):
        raise RuntimeError("Initialization requires original host bindings")
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("put_model", "_calculate_max_contact_nnz")
    ]
    namespace = dict(vars(io))
    namespace.update(
        _assembly_geom_pair_type_count=_geom_pair_type_count,
        _assembly_body_contact_nnz=_body_contact_nnz,
    )
    exec(  # noqa: S102 - both original and transformed source hashes are pinned.
        compile(ast.Module(body=functions, type_ignores=[]), io.__file__, "exec"),
        namespace,
    )
    _installed = namespace["put_model"], namespace["_calculate_max_contact_nnz"]
    io.put_model, io._calculate_max_contact_nnz = _installed
    mujoco_warp.put_model = io.put_model
    return fingerprint()
