"""Skip the pinned runtime's discarded dense compact Jacobian allocation."""

import ast
import hashlib
import pathlib

SOURCE_SHA256 = "f9214e355f92a89583d85af04867403d1e8f81fcd45c907114d0baa279aa507d"
OVERLAY_SHA256 = "afc12f03dbccb4d0286b1393a3bdcb0c78c6a4d2ed38fcb06917108c056cf803"
_installed = None


def fingerprint() -> dict[str, str]:
    return {
        "source_sha256": SOURCE_SHA256,
        "overlay_sha256": OVERLAY_SHA256,
        "allocation": "skip_discarded_sparse_cJ_v1",
    }


def transform_source(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Compact allocation requires the exact audited runtime")
    anchor = (
        "  for f in dataclasses.fields(types.Data):\n"
        "    if f.name in d_kwargs:\n"
        "      continue\n"
        "    val = getattr(mjd, f.name, None)"
    )
    # The existing final compact allocator assigns an empty cJ for sparse models.
    transformed = (
        source.decode()
        .replace(anchor, '  if is_sparse(mjm):\n    d_kwargs["cJ"] = None\n' + anchor)
        .encode()
    )
    if hashlib.sha256(transformed).hexdigest() != OVERLAY_SHA256:
        raise RuntimeError("Compact allocation transform differs from audited source")
    return transformed


def install() -> dict[str, str]:
    """Replace host put_data only; preserve all runtime kernels and allocators."""
    import mujoco_warp
    from mujoco_warp._src import io

    global _installed
    source = transform_source(pathlib.Path(io.__file__).read_bytes())
    if _installed is not None:
        if io.put_data is not _installed or mujoco_warp.put_data is not _installed:
            raise RuntimeError("Compact allocation binding conflicts")
        return fingerprint()
    if io.put_data.__module__ != io.__name__ or mujoco_warp.put_data is not io.put_data:
        raise RuntimeError("Compact allocation requires the original put_data binding")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "put_data"
    )
    namespace = dict(vars(io))
    # Compile only the host function: no duplicate Warp kernels or cache entries.
    exec(  # noqa: S102 - both original and transformed source hashes are pinned above.
        compile(ast.Module(body=[function], type_ignores=[]), io.__file__, "exec"),
        namespace,
    )
    _installed = namespace["put_data"]
    io.put_data = _installed
    mujoco_warp.put_data = _installed
    return fingerprint()
