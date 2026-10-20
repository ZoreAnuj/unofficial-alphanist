"""Pinned, process-local float64 Hessian for sparse Newton assembly.

Precision boundaries follow MuJoCo Warp PR1801 (cfeaf4e4f608ea68859ac9e6ca9d216b6d193a48).
Only H, its products, factor and triangular solves change precision. No pivot
floor, regularization, contact modification, or force-law change is introduced.
"""

import ast
import hashlib
import importlib.util
import pathlib
import sys
import tempfile

SOLVER_SHA256 = "c7347ac24336f2c495f38286ed9fbc41d428f59ccef5065c5105009d3edc74d0"
LOCAL_SOLVER_SHA256 = "509b43da297dc6e49efeacdbb59df675213e84afbdb26770471a5ba1855a24b3"
BLOCK_SHA256 = "0773678b85d27692d2571ebecaca3ece6573fe30531660e8c993be285407af99"
SOLVER_OVERLAY_HASHES = {
    SOLVER_SHA256: "71abfacc2b26f7d538e47a59346ae780e1a624919fed694ea375df04102d352a",
    LOCAL_SOLVER_SHA256: "d16252a338241135d8f46eaaae27621250183c9f7bf3c25b6e3e25bf89276e5c",
}
BLOCK_OVERLAY_SHA256 = (
    "6f701997fa087b01c7808bef0e19c03c68271bc14596bd61f803c09ce73f185d"
)
SOLVER_MODULE = "mujoco_warp._src.solver_assembly_h64_v3"
BLOCK_MODULE = "mujoco_warp._src.block_cholesky_assembly_h64_v3"
_source_directory = None


def fingerprint(source_sha256: str | None = None) -> dict[str, str]:
    if source_sha256 is None:
        from mujoco_warp._src import solver

        source_sha256 = hashlib.sha256(
            pathlib.Path(solver.__file__).read_bytes()
        ).hexdigest()
    if source_sha256 not in SOLVER_OVERLAY_HASHES:
        raise RuntimeError("Hessian precision requires the exact audited runtime")
    return {
        "solver_sha256": source_sha256,
        "block_sha256": BLOCK_SHA256,
        "solver_overlay_sha256": SOLVER_OVERLAY_HASHES[source_sha256],
        "block_overlay_sha256": BLOCK_OVERLAY_SHA256,
        "precision": "sparse_hessian_float64_v3",
    }


def _edit(source, name, replacements):
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    lines = source.splitlines(keepends=True)
    body = "".join(lines[node.lineno - 1 : node.end_lineno])
    for old, new in replacements:
        if old not in body:
            raise RuntimeError(f"Missing audited Hessian expression in {name}: {old}")
        body = body.replace(old, new)
    return "".join(lines[: node.lineno - 1]) + body + "".join(lines[node.end_lineno :])


def transform_sources(solver_bytes: bytes, block_bytes: bytes) -> tuple[bytes, bytes]:
    if (
        hashlib.sha256(solver_bytes).hexdigest() not in SOLVER_OVERLAY_HASHES
        or hashlib.sha256(block_bytes).hexdigest() != BLOCK_SHA256
    ):
        raise RuntimeError("Hessian precision requires the exact audited runtime")
    block = block_bytes.decode()
    block = _edit(
        block,
        "solve_search_sums",
        [
            ("grad: float, solution: float", "grad: wp.float64, solution: wp.float64"),
            ("return wp.vec2(", "return wp.vec2d("),
        ],
    )
    block = _edit(
        block,
        "_create_newton_decrement_func",
        [
            ("solution_tile: wp.tile[float,", "solution_tile: wp.tile[wp.float64,"),
            (
                "grad_tile = wp.tile_load(b, shape=(vector_size_static, 1), offset=(0, 0), bounds_check=False)",
                "grad_tile = wp.tile_astype(wp.tile_load(b, shape=(vector_size_static, 1), offset=(0, 0), bounds_check=False), wp.float64)",
            ),
            (
                "wp.tile_map(wp.mul, active_solution, -1.0)",
                "wp.tile_astype(wp.tile_map(wp.mul, active_solution, wp.float64(-1.0)), float)",
            ),
            (
                "return wp.tile_reduce(wp.add, wp.tile_map(solve_search_sums, grad_tile, active_solution))[0]",
                "sums = wp.tile_reduce(wp.add, wp.tile_map(solve_search_sums, grad_tile, active_solution))[0]\n    return wp.vec2(float(sums[0]), float(sums[1]))",
            ),
        ],
    )
    block = _edit(
        block,
        "_create_blocked_cholesky_augmented_factorize_solve_func",
        [
            ("A: wp.array2d[float]", "A: wp.array2d[wp.float64]"),
            ("U_out: wp.array2d[float]", "U_out: wp.array2d[wp.float64]"),
            ('dtype=float, storage="shared"', 'dtype=wp.float64, storage="shared"'),
            (
                'input_rhs = wp.tile_load(b, shape=(block_size, 1), offset=(k, 0), storage="shared", bounds_check=False)',
                'input_rhs = wp.tile_zeros(shape=(block_size, 1), dtype=wp.float64, storage="shared")\n      wp.tile_assign(input_rhs, wp.tile_astype(wp.tile_load(b, shape=(block_size, 1), offset=(k, 0), bounds_check=False), wp.float64))',
            ),
            ("= 1.0e30", "= wp.float64(1.0e30)"),
            (
                "wp.tile_store(result_out, rhs_tile,",
                "wp.tile_store(result_out, wp.tile_astype(rhs_tile, float),",
            ),
        ],
    )
    block = _edit(
        block,
        "_create_blocked_cholesky_solve_func",
        [
            ("U: wp.array2d[float]", "U: wp.array2d[wp.float64]"),
            (
                'rhs_tile = wp.tile_load(b, shape=(matrix_size_static, 1), offset=(0, 0), storage="shared", bounds_check=False)',
                'rhs_tile = wp.tile_zeros(shape=(matrix_size_static, 1), dtype=wp.float64, storage="shared")\n    wp.tile_assign(rhs_tile, wp.tile_astype(wp.tile_load(b, shape=(matrix_size_static, 1), offset=(0, 0), bounds_check=False), wp.float64))',
            ),
            (
                "wp.tile_store(result_out, rhs_tile,",
                "wp.tile_store(result_out, wp.tile_astype(rhs_tile, float),",
            ),
        ],
    )
    source = solver_bytes.decode()
    source = _edit(
        source,
        "_create_solver_context",
        [
            (
                "h=wp.empty((nworld, nv_pad, nv_pad), dtype=float)",
                "h=wp.empty((nworld, nv_pad, nv_pad), dtype=wp.float64)",
            ),
            (
                "hfactor=wp.empty((nworld, nv_pad, nv_pad), dtype=float)",
                "hfactor=wp.empty((nworld, nv_pad, nv_pad), dtype=wp.float64)",
            ),
        ],
    )
    for name in ("_update_gradient_init_h_sparse", "_padding_h"):
        source = _edit(
            source,
            name,
            [
                ("ctx_h_out: wp.array3d[float]", "ctx_h_out: wp.array3d[wp.float64]"),
                (" = 1.0", " = wp.float64(1.0)"),
            ]
            if name == "_padding_h"
            else [
                ("ctx_h_out: wp.array3d[float]", "ctx_h_out: wp.array3d[wp.float64]"),
                (
                    "= wp.where(i == j, 1.0, 0.0)",
                    "= wp.float64(wp.where(i == j, 1.0, 0.0))",
                ),
                ("= M_in[worldid, elemid]", "= wp.float64(M_in[worldid, elemid])"),
                (" = 0.0", " = wp.float64(0.0)"),
            ],
        )
    # Retain the audited source's curvature formula (including the remote
    # normalization backport), promoting only its arithmetic precision.
    helper = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_elliptic_hessian_entry_from_projections"
    )
    helper_text = "\n".join(source.splitlines()[helper.lineno - 1 : helper.end_lineno])
    helper_text = (
        helper_text.replace(
            "_elliptic_hessian_entry_from_projections", "_elliptic_hessian_entry_h64_v3"
        )
        .replace(": float", ": wp.float64")
        .replace("-> float", "-> wp.float64")
    )
    source += "\n\n@wp.func\n" + helper_text + "\n"
    source += "\n_H64Vec16 = wp.types.vector(length=16, dtype=wp.float64)\n"
    source += """
@wp.func
def _safe_div_h64_v3(x: wp.float64, y: wp.float64) -> wp.float64:
  return x / wp.where(y != wp.float64(0.0), y, wp.float64(types.MJ_MINVAL))
"""
    source = _edit(
        source,
        "_JTDACJ_sparse",
        [
            ("types.vec16", "_H64Vec16"),
            ("-> float:", "-> wp.float64:"),
            ("float(0.0)", "wp.float64(0.0)"),
            ('dtype=float, storage="shared"', 'dtype=wp.float64, storage="shared"'),
            (
                "_elliptic_hessian_entry_from_projections(",
                "_elliptic_hessian_entry_h64_v3(",
            ),
            (
                "fri[0] * opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]",
                "wp.float64(fri[0]) * wp.float64(opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]])",
            ),
            (
                "math.safe_div(efc_D_in[worldid, efcid0],",
                "math.safe_div(wp.float64(efc_D_in[worldid, efcid0]),",
            ),
            (
                "ctx_Jaref_in[worldid, efcid0] * mu",
                "wp.float64(ctx_Jaref_in[worldid, efcid0]) * mu",
            ),
            ("scale = fri[dim - 1]", "scale = wp.float64(fri[dim - 1])"),
            (
                "ctx_Jaref_in[worldid, efcid] * scale",
                "wp.float64(ctx_Jaref_in[worldid, efcid]) * scale",
            ),
            ("types.MJ_MINVAL", "wp.float64(types.MJ_MINVAL)"),
        ]
        + (
            [("math.safe_div(1.0, t)", "math.safe_div(wp.float64(1.0), t)")]
            if hashlib.sha256(solver_bytes).hexdigest() == SOLVER_SHA256
            else []
        ),
    )
    source = _edit(
        source,
        "_JTDACJ_sparse",
        [
            ("math.safe_div(", "_safe_div_h64_v3("),
            ("1.0 + mu2", "wp.float64(1.0) + mu2"),
        ]
        + [
            (
                f"* efc_J_in[worldid, 0, rowadr[{idx}] + pos{pos}]",
                f"* wp.float64(efc_J_in[worldid, 0, rowadr[{idx}] + pos{pos}])",
            )
            for idx in ("0", "dim")
            for pos in (1, 2)
        ],
    )
    source = _edit(
        source,
        "_JTDACJ_sparse",
        [
            ("h_out: wp.array3d[float]", "h_out: wp.array3d[wp.float64]"),
            (
                "j_row = efc_J_in[worldid, 0, member_adr + block_row]",
                "j_row = wp.float64(efc_J_in[worldid, 0, member_adr + block_row])",
            ),
            (
                "j_col = efc_J_in[worldid, 0, member_adr + block_col]",
                "j_col = wp.float64(efc_J_in[worldid, 0, member_adr + block_col])",
            ),
            (
                "hval += j_row * efc_D_in[worldid, member_row] * j_col",
                "hval += j_row * wp.float64(efc_D_in[worldid, member_row]) * j_col",
            ),
        ],
    )
    source = _edit(
        source,
        "_update_gradient_h_incremental_sparse",
        [
            ("ctx_h_out: wp.array3d[float]", "ctx_h_out: wp.array3d[wp.float64]"),
            (
                "D = efc_D_in[worldid, efcid]",
                "D = wp.float64(efc_D_in[worldid, efcid])",
            ),
            ("sign = float(0.0)", "sign = wp.float64(0.0)"),
            (
                "Ji = efc_J_in[worldid, 0, rowadr + ii]",
                "Ji = wp.float64(efc_J_in[worldid, 0, rowadr + ii])",
            ),
            (
                "Jj = efc_J_in[worldid, 0, rowadr + jj]",
                "Jj = wp.float64(efc_J_in[worldid, 0, rowadr + jj])",
            ),
        ],
    )
    for name in (
        "_update_gradient_cholesky_blocked",
        "_update_gradient_cholesky_blocked_skip_unchanged",
    ):
        source = _edit(
            source,
            name,
            [
                ("ctx_h_in: wp.array3d[float]", "ctx_h_in: wp.array3d[wp.float64]"),
                (
                    "ctx_hfactor: wp.array3d[float]",
                    "ctx_hfactor: wp.array3d[wp.float64]",
                ),
            ],
        )
    source = _edit(
        source,
        "_update_gradient_cholesky",
        [
            ("h_in: wp.array3d[float]", "h_in: wp.array3d[wp.float64]"),
            (
                "input_tile = wp.tile_load(ctx_grad_in[worldid], shape=TILE_SIZE)",
                "input_tile = wp.tile_astype(wp.tile_load(ctx_grad_in[worldid], shape=TILE_SIZE), wp.float64)",
            ),
            (
                "ctx_search_dot_out[worldid] = sums[0]",
                "ctx_search_dot_out[worldid] = float(sums[0])",
            ),
            (
                "ctx_newton_decrement_out[worldid] = sums[1]",
                "ctx_newton_decrement_out[worldid] = float(sums[1])",
            ),
            (
                "wp.tile_map(wp.mul, output_tile, -1.0)",
                "wp.tile_astype(wp.tile_map(wp.mul, output_tile, wp.float64(-1.0)), float)",
            ),
        ],
    )
    # cache_kernel's global key includes the factory name, not its module/source.
    for name in (
        "_update_gradient_cholesky",
        "_update_gradient_init_h_sparse",
        "_JTDACJ_sparse",
        "_update_gradient_h_incremental_sparse",
        "_update_gradient_cholesky_blocked_skip_unchanged",
        "_update_gradient_cholesky_blocked",
    ):
        source = source.replace(name + "(", name + "_assembly_h64_v3(")
    for name in (
        "create_blocked_cholesky_augmented_factorize_solve_newton_func",
        "create_blocked_cholesky_solve_newton_func",
        "solve_search_sums",
    ):
        source = source.replace(
            "from mujoco_warp._src.block_cholesky import " + name,
            "from " + BLOCK_MODULE + " import " + name,
        )
    # All unused paths stay original and are rejected before allocating a context.
    guard = """  if (not m.is_sparse or m.opt.solver != types.SolverType.NEWTON
      or m.opt.cone not in (types.ConeType.PYRAMIDAL, types.ConeType.ELLIPTIC)
      or m.opt.integrator != types.IntegratorType.IMPLICITFAST
      or m.opt.enableflags & types.EnableBit.SLEEP):
    raise ValueError(
      "H64 requires sparse Newton, pyramidal/elliptic contacts, implicitfast, no sleep; "
      f"is_sparse={m.is_sparse}, solver={m.opt.solver}, cone={m.opt.cone}, "
      f"integrator={m.opt.integrator}, enableflags={m.opt.enableflags}, nv={m.nv}")
"""
    source = _edit(
        source,
        "solve",
        [
            (
                "  if m.opt.enableflags & types.EnableBit.SLEEP:",
                guard + "  if m.opt.enableflags & types.EnableBit.SLEEP:",
            )
        ],
    )
    transformed = source.encode(), block.encode()
    for actual, expected in zip(
        transformed,
        (
            SOLVER_OVERLAY_HASHES[hashlib.sha256(solver_bytes).hexdigest()],
            BLOCK_OVERLAY_SHA256,
        ),
        strict=True,
    ):
        if hashlib.sha256(actual).hexdigest() != expected:
            raise RuntimeError(
                "Hessian precision transform differs from audited candidate: "
                + hashlib.sha256(actual).hexdigest()
            )
    return transformed


def install() -> dict[str, str]:
    """Install before solver construction/graph capture; never edit runtime files."""
    from mujoco_warp._src import block_cholesky, solver

    global _source_directory
    sources = transform_sources(
        pathlib.Path(solver.__file__).read_bytes(),
        pathlib.Path(block_cholesky.__file__).read_bytes(),
    )
    identity = fingerprint()
    existing = sys.modules.get(SOLVER_MODULE)
    if existing is not None:
        if (
            getattr(existing, "_assembly_fingerprint", None) != identity
            or solver.solve is not existing.solve
        ):
            raise RuntimeError("Hessian precision binding conflicts")
        return identity
    if solver.solve.__module__ != solver.__name__ or BLOCK_MODULE in sys.modules:
        raise RuntimeError("Hessian precision requires original solver binding")
    directory = tempfile.TemporaryDirectory(prefix="ur7-hessian64-")
    try:
        for name, source in ((BLOCK_MODULE, sources[1]), (SOLVER_MODULE, sources[0])):
            path = pathlib.Path(directory.name) / (name.rsplit(".", 1)[-1] + ".py")
            path.write_bytes(source)
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            module._assembly_fingerprint = identity
        solver.solve = sys.modules[SOLVER_MODULE].solve
    except BaseException:
        sys.modules.pop(SOLVER_MODULE, None)
        sys.modules.pop(BLOCK_MODULE, None)
        directory.cleanup()
        raise
    _source_directory = directory
    return identity
