"""Render ONE photoreal still of a USD stage in Isaac Sim (RTX path tracing).

Import-light on purpose: only stdlib + numpy + PIL at module scope, and no
station imports, so it runs inside the isaac-sim container's python with just
``PYTHONPATH=/work/repo/src/python``. Everything that needs Kit (``carb``,
``omni``, ``pxr``) is imported inside :func:`main`, after the SimulationApp
exists — a boot stub can hand one in via ``app=``.
"""

import argparse
import pathlib
import sys
import time
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_CAMERA = "/Environment/cam_wide"
DOME_LIGHT_PATH = "/Environment/DomeLight"
WARMUP_UPDATES = 20
# /rtx/post/tonemap/op index for ACES filmic in Kit's tonemapper enum.
TONEMAP_ACES = 6


def parse_resolution(text: str) -> tuple[int, int]:
    """``"1920x1080"`` -> ``(1920, 1080)``."""
    try:
        width_text, height_text = text.lower().split("x")
        width, height = int(width_text), int(height_text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"resolution must be WxH, got {text!r}"
        ) from None
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"resolution must be positive, got {text!r}")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=pathlib.Path, required=True, help="USDA/USDC")
    ap.add_argument("--camera", default=DEFAULT_CAMERA, help="camera prim path")
    ap.add_argument("--out", type=pathlib.Path, help="output PNG")
    ap.add_argument("--resolution", type=parse_resolution, default="1920x1080")
    ap.add_argument("--spp", type=int, default=32)
    ap.add_argument("--total-spp", type=int, default=256)
    ap.add_argument("--subframes", type=int, default=16)
    ap.add_argument("--no-denoiser", action="store_true")
    ap.add_argument("--hdri", type=pathlib.Path, default=None)
    ap.add_argument("--hdri-intensity", type=float, default=900.0)
    ap.add_argument(
        "--stage-only",
        action="store_true",
        help="open the stage, print the prim tree + cameras, and exit",
    )
    return ap


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.stage_only and args.out is None:
        parser.error("--out is required unless --stage-only")
    return args


def image_stats(rgb: np.ndarray) -> str:
    """One-line exposure summary so a blank frame is diagnosable from the log."""
    gray = rgb.astype(np.float32).mean(axis=2)
    return (
        f"mean={gray.mean():.1f} std={gray.std():.1f} "
        f"black={(gray < 4).mean():.3f} white={(gray > 251).mean():.3f}"
    )


def _apply_render_settings(settings: Any, args: argparse.Namespace) -> None:
    # open_stage resets /rtx/*, so callers re-apply this after every open.
    settings.set_int("/rtx/pathtracing/spp", args.spp)
    settings.set_int("/rtx/pathtracing/totalSpp", args.total_spp)
    settings.set("/rtx/post/tonemap/op", TONEMAP_ACES)
    settings.set("/rtx/pathtracing/optixDenoiser/enabled", not args.no_denoiser)
    # Kit's "FFT Bloom" panel is the lensFlares group (no /rtx/post/bloom key
    # exists); histogram is auto-exposure. Both default off in the base
    # experience, pinned so a still and the episode lanes expose identically.
    settings.set_bool("/rtx/post/lensFlares/enabled", False)
    settings.set_bool("/rtx/post/histogram/enabled", False)


def _camera_prims(stage: Any) -> list[str]:
    from pxr import UsdGeom

    return [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]


def _print_stage_tree(stage: Any) -> None:
    for prim in stage.Traverse():
        depth = len(prim.GetPath().pathString.split("/")) - 2
        print(f"{'  ' * depth}{prim.GetName()} <{prim.GetTypeName()}>")
    print(f"cameras: {_camera_prims(stage)}", flush=True)


def _override_hdri(stage: Any, hdri: pathlib.Path, intensity: float) -> None:
    from pxr import Gf, UsdLux

    prim = stage.GetPrimAtPath(DOME_LIGHT_PATH)
    if not prim.IsValid():
        print(f"WARNING: {DOME_LIGHT_PATH} missing; --hdri ignored", flush=True)
        return
    dome = UsdLux.DomeLight(prim)
    dome.GetTextureFileAttr().Set(str(hdri))
    dome.GetTextureFormatAttr().Set(UsdLux.Tokens.latlong)
    dome.GetIntensityAttr().Set(intensity)
    dome.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))


def _capture(annotator: Any, width: int, height: int) -> np.ndarray:
    data = annotator.get_data()
    if isinstance(data, dict):
        data = data.get("data", next(iter(data.values())))
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr.reshape(height, width, -1)
    return arr[:, :, :3].astype(np.uint8)


def main(argv: list[str] | None = None, app: Any = None) -> int:
    args = parse_args(argv)
    width, height = args.resolution
    owns_app = app is None
    if owns_app:
        import isaacsim

        app = isaacsim.SimulationApp(
            {
                "headless": True,
                "width": width,
                "height": height,
                "renderer": "PathTracing",
            },
            experience="/isaac-sim/apps/isaacsim.exp.base.python.kit",
        )
    import carb.settings
    import omni.replicator.core as rep
    import omni.usd

    try:
        import omni.isaac.core.utils.stage as stage_utils
    except ImportError:  # Isaac 6 dropped the omni.isaac shim
        import isaacsim.core.utils.stage as stage_utils

    started = time.time()
    product = annotator = None
    try:
        if not stage_utils.open_stage(str(args.stage.resolve())):
            raise SystemExit(f"open_stage failed: {args.stage}")
        stage = omni.usd.get_context().get_stage()
        print(f"opened {args.stage} in {time.time() - started:.1f}s", flush=True)
        if args.stage_only:
            _print_stage_tree(stage)
            return 0
        if not stage.GetPrimAtPath(args.camera).IsValid():
            raise SystemExit(
                f"camera {args.camera} not found; cameras: {_camera_prims(stage)}"
            )
        settings = carb.settings.get_settings()
        _apply_render_settings(settings, args)
        if args.hdri is not None:
            _override_hdri(stage, args.hdri, args.hdri_intensity)

        product = rep.create.render_product(args.camera, resolution=(width, height))
        annotator = rep.AnnotatorRegistry.get_annotator("LdrColor")
        annotator.attach([product])
        for _ in range(WARMUP_UPDATES):
            app.update()
        _apply_render_settings(settings, args)
        rep.orchestrator.step(rt_subframes=args.subframes, pause_timeline=True)
        rgb = _capture(annotator, width, height)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(args.out)
        print(
            f"saved {args.out} {rgb.shape[1]}x{rgb.shape[0]} {image_stats(rgb)} "
            f"wall {time.time() - started:.1f}s",
            flush=True,
        )
        return 0
    finally:
        # Detach before destroy (a destroyed product invalidates attached
        # annotators) and release the app whether we return, SystemExit or
        # raise: a caller that hands in `app=` may render again in-process.
        if annotator is not None:
            annotator.detach()
        if product is not None:
            product.destroy()
        if owns_app:
            app.close()


if __name__ == "__main__":
    sys.exit(main())
