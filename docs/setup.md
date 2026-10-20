# Setup

## Learning core

Use Python 3.12 or newer and install the project as shown in the README. PyTorch, NumPy, and SciPy are sufficient for the policy, trainer, reset sampler, and CPU regression tests. Install a CUDA-enabled PyTorch build appropriate to your driver before using GPUs; the package does not select a CUDA wheel for you.

The original `isaac.assembly_rl` module path is preserved. The `isaac` namespace in this repository does not itself install NVIDIA Isaac Sim.

## Simulation runtime

The GPU training runtime inspected for this export had the following versions. They are a recorded environment, not a claim that every platform or package combination has been tested:

| Dependency | Inspected version |
| --- | --- |
| Newton | 1.6.0 |
| MuJoCo / MuJoCo-Warp | 3.12.0 / 3.12.0 |
| NVIDIA Warp | 1.17.0 |
| PyTorch | 2.9.0+cu128 |
| NumPy / SciPy | 1.26.4 / 1.15.3 |
| trimesh / Shapely | 5.1.0 / 2.2.0 |
| USD | 26.3 |

CPU regression checks used PyTorch 2.14.0+cpu, NumPy 2.5.3, and SciPy 1.18.1. CAD preparation additionally used CoACD 1.0.14; CoACD is not installed in the GPU training runtime.

Acquire [Newton](https://github.com/newton-physics/newton), [MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp), and [Warp](https://github.com/NVIDIA/warp) from their upstream distributions. `python -m pip install -e '.[test,geometry,simulation,render]'` installs the geometry/test extras; it does not install the full simulator. Isaac Sim is only needed for image rendering and has its own installation and license requirements.

**Exact runtime source matters.** `initialization.py`, `compact_allocation.py`, `hessian_precision.py`, `sap_wide.py`, and `ccd_workspace.py` apply process-local, hash-guarded transformations to the inspected MuJoCo-Warp runtime. Matching a version string alone is insufficient. A source-hash mismatch deliberately stops execution; do not disable that guard or label a changed runtime equivalent. Capacity qualification must be repeated after runtime or geometry changes.

## External assets

This source release does not include the experimental CAD caches, workshop USD, robot collision meshes, or validated reset banks. Obtain each asset under its own terms; [provenance](provenance.md) links the public sources. This is a concrete limitation of reproducing the complete ten-task experiment from a fresh clone.

The simulation expects:

- A UR7e URDF and native physical profile, with all referenced collision geometry resolvable. The packaged support code describes the required structure; robot mesh files remain external.
- A task catalog with per-task parts, source asset paths, fixed/moving roles, assembled transforms, and source hashes.
- A prepared directory containing one task JSON per task and numeric mesh caches referenced by those JSON files. `assets.py` creates coarse settling geometry; those outputs are explicitly **not** precision-training-ready. The task-specific `prepare_*` tools and collision-fit checks establish the stronger contract.
- A physically validated reset bank containing `bank.json` and `bank.npz`, with the current asset/controller/runtime identity. Proposal-only banks from `generate_bank.py` cannot certify training.
- A separate validated held-out bank for evaluation.

The UR7e URDF and numeric profile are included. Fetch the seven exact upstream collision meshes with `python scripts/fetch_robot_meshes.py`; the script verifies pinned SHA-256 values and refuses unexpected existing bytes. Meshes remain ignored by git. This source-checkout/editable-install layout is required by the relative native-profile paths.

Gantry source-CAD preparation is not exported because it depends on additional project geometry; the training environment supports a supplied prepared gantry task. A workshop USD must also be supplied for rendered robot previews.

Use repository-local `data/` for these inputs; it is ignored by git. The optional asset integration tests look under `data/training_assets`, `data/industrial_assembly_tasks/catalog.json`, and `data/workshop.usdc` and skip when absent. Run all simulator tests only after the runtime and assets are available. The preparation pipeline also requires source-native CAD and task metadata; downloading a mesh alone does not create a valid experiment.

## Check the installation

```bash
python -m isaac.assembly_rl.train --help
python -m isaac.assembly_rl.reset_bank_builder --help
python -m isaac.assembly_rl.merge_reset_banks --help
```

These commands parse arguments without advancing simulation. Use the CPU test command in the README before attempting the conditional workflow in [training](training.md).
