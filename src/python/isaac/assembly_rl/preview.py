"""Export a captured Newton state into the approved workshop; render with Isaac Sim.

No physics advances during rendering. USD carries the source snapshot SHA256 so
an image can be traced to its exact measured joints and object poses.
"""

import argparse
import hashlib
import json
import pathlib
from typing import Any

import numpy as np

PHYSICS_SOURCE = "Newton AssemblyEnvironment.snapshot"


def capture_snapshot(env: Any, output: pathlib.Path) -> pathlib.Path:
    """Capture the current environment state, including failures, without editing it."""
    values = {
        key: value.detach().cpu().numpy() for key, value in env.snapshot().items()
    }
    values.update(
        task_names=np.asarray(env.task_names),
        physics_source=np.asarray(PHYSICS_SOURCE),
        environment_config=np.asarray(json.dumps(env.config, sort_keys=True)),
    )
    output = pathlib.Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(handle, **values)
    return output


def read_snapshot(
    path: pathlib.Path, env_index: int
) -> tuple[str, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["physics_source"]) != PHYSICS_SOURCE:
            raise ValueError("Expected a captured Newton physics snapshot")
        names, ids = archive["task_names"], archive["task_ids"]
        if env_index < 0 or env_index >= len(ids):
            raise ValueError("Snapshot environment index out of range")
        task_index = int(ids[env_index])
        if not 0 <= task_index < len(names):
            raise ValueError("Snapshot task index out of range")
        joints, poses = archive["joint_q"][env_index], archive["part_poses"][env_index]
        if joints.shape != (8,) or poses.ndim != 2 or poses.shape[1] != 7:
            raise ValueError("Snapshot requires eight joints and part XYZ/XYZW poses")
        if not np.isfinite(joints).all() or not np.isfinite(poses).all():
            raise ValueError("Nonfinite snapshot")
        if not np.allclose(np.linalg.norm(poses[:, 3:], axis=-1), 1.0, atol=1e-5):
            raise ValueError("Snapshot part quaternion must be normalized")
        return str(names[task_index]), joints.copy(), poses.copy()


def export_snapshot(
    snapshot: pathlib.Path,
    scene: pathlib.Path,
    assets_path: pathlib.Path,
    output: pathlib.Path,
    *,
    env_index: int = 0,
    package: bool = False,
) -> pathlib.Path:
    from asset_pipeline.ur7e_cell import ur7e_model
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
    from scipy.spatial import transform

    task_id, joints, poses = read_snapshot(snapshot, env_index)
    definition = json.loads((pathlib.Path(assets_path) / f"{task_id}.json").read_text())
    if len(definition["parts"]) > len(poses):
        raise ValueError("Snapshot omits task parts")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output.resolve()))
    stage.GetRootLayer().subLayerPaths = [str(scene.resolve())]
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    def world_pose(path: str, matrix: np.ndarray) -> None:
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        parent = prim.GetParent()
        parent_pose = (
            np.eye(4)
            if parent.IsPseudoRoot()
            else np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(parent)).T
        )
        local = np.linalg.inv(parent_pose) @ matrix
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(Gf.Matrix4d(local.T))

    base = stage.GetPrimAtPath(ur7e_model.INSTANCES + "/ur7e_base")
    if not base:
        raise ValueError("Approved workshop must contain the UR7 base")
    mount = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(base)).T
    links = ur7e_model.link_poses(joints[:6])
    for member, link in ur7e_model.ARM_MEMBERS.items():
        world_pose(ur7e_model.INSTANCES + "/" + member, mount @ links[link])
    for member in ur7e_model.TOOL_MEMBERS:
        # Preserve measured right/left openings independently, including mimic error.
        opening = joints[6] if member == "hande_finger_right" else joints[7]
        world_pose(
            ur7e_model.INSTANCES + "/" + member,
            mount @ links["tool0"] @ ur7e_model.tool_pose(member, opening),
        )
    previous = stage.GetPrimAtPath("/Workshop/Assembly")
    if previous:
        previous.SetActive(False)
    for index, part in enumerate(definition["parts"]):
        if not Sdf.Path.IsValidIdentifier(part["name"]):
            raise ValueError("Part name must be a USD identifier")
        root = "/TrainingPreview/Parts/" + part["name"]
        matrix = np.eye(4)
        matrix[:3, :3] = transform.Rotation.from_quat(poses[index, 3:]).as_matrix()
        matrix[:3, 3] = poses[index, :3]
        world_pose(root, matrix)
        source = pathlib.Path(part["source"])
        if not source.is_file():
            raise FileNotFoundError(source)
        geometry = UsdGeom.Xform.Define(stage, root + "/Geometry").GetPrim()
        geometry.GetReferences().AddReference(str(source.resolve()))
    # Render the recorded state, including unsafe states, without a second simulator
    # moving objects to different positions or trying to resolve interpenetrations.
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
    camera = UsdGeom.Camera.Define(stage, "/TrainingPreview/Camera")
    eye, target = Gf.Vec3d(1.35, 1.80, 1.95), Gf.Vec3d(0.12, -0.02, 1.08)
    matrix = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
    camera.AddTransformOp().Set(matrix)
    camera.CreateFocalLengthAttr(48.0)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(27.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    stage.GetRootLayer().customLayerData = {
        "physics_source": PHYSICS_SOURCE,
        "task_id": task_id,
        "environment_index": env_index,
        "physics_snapshot_sha256": hashlib.sha256(
            pathlib.Path(snapshot).read_bytes()
        ).hexdigest(),
        "preview_only_no_physics_step": True,
    }
    stage.GetRootLayer().Save()
    unresolved = UsdUtils.ComputeAllDependencies(str(output))[2]
    if unresolved:
        raise ValueError(f"Unresolved preview assets: {unresolved}")
    if package:
        target_path = output.with_suffix(".usdz")
        if target_path.exists():
            raise FileExistsError(target_path)
        if not UsdUtils.CreateNewUsdzPackage(
            Sdf.AssetPath(str(output.resolve())), str(target_path.resolve())
        ):
            raise RuntimeError("Failed to package portable preview assets")
        return target_path
    return output


def render(
    scene: pathlib.Path, output: pathlib.Path, *, width: int = 1200, height: int = 900
) -> None:
    """Run under Isaac Sim python.sh on an idle RTX GPU; do not start physics."""
    from isaacsim import SimulationApp

    application = SimulationApp(
        {
            "headless": True,
            "renderer": "PathTracing",
            "width": width,
            "height": height,
            "fast_shutdown": True,
        }
    )
    exit_code = 0
    try:
        import omni.replicator.core as replicator
        import omni.usd
        from PIL import Image

        if not omni.usd.get_context().open_stage(str(scene.resolve())):
            raise RuntimeError("Isaac Sim could not open preview stage")
        for _ in range(30):
            application.update()
        product = replicator.create.render_product(
            "/TrainingPreview/Camera", (width, height)
        )
        rgb = replicator.annotators.get("rgb")
        rgb.attach(product)
        # Headless annotators may need several frames to initialize their buffers.
        pixels = np.empty((0,), dtype=np.uint8)
        for attempt in range(8):
            for _ in range(5):
                application.update()
            # NVIDIA's supported fixed-state capture path uses delta_time=0.
            replicator.orchestrator.step(
                rt_subframes=32, delta_time=0.0, pause_timeline=True
            )
            pixels = np.asarray(rgb.get_data())
            valid = pixels.shape[:2] == (height, width) and pixels.ndim == 3
            print(
                "PREVIEW_FRAME",
                attempt,
                pixels.shape,
                float(pixels[..., :3].std()) if valid else None,
                flush=True,
            )
            if valid and pixels[..., :3].std() >= 1:
                break
        else:
            raise RuntimeError(
                "Renderer returned missing or blank RGB pixels after eight captures"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels[..., :3].astype(np.uint8)).save(output)
        replicator.orchestrator.wait_until_complete()
        rgb.detach()
        product.destroy()
    except BaseException:
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        import inspect
        import os
        import sys

        if "exit_code" in inspect.signature(application.close).parameters:
            application.close(exit_code=exit_code)
        elif exit_code:
            # Isaac 5.1's fast shutdown always exits zero. Preserve failure before
            # entering that path; the process owns only this static render job.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
        else:
            application.close()
    if exit_code:
        raise RuntimeError("Isaac preview rendering failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    exporter = subparsers.add_parser("export")
    for key in ("snapshot", "scene", "assets", "out"):
        exporter.add_argument("--" + key, type=pathlib.Path, required=True)
    exporter.add_argument("--environment", type=int, default=0)
    exporter.add_argument("--package", action="store_true")
    renderer = subparsers.add_parser("render")
    renderer.add_argument("--scene", type=pathlib.Path, required=True)
    renderer.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        print(
            export_snapshot(
                args.snapshot,
                args.scene,
                args.assets,
                args.out,
                env_index=args.environment,
                package=args.package,
            )
        )
    else:
        render(args.scene, args.out)


if __name__ == "__main__":
    main()
