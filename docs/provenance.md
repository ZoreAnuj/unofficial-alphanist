# Provenance and acknowledgments

## Original paper and related methods

This repository implements an interpretation of [AlphaNIST: Emergent Assembly Skills from Binary Task Rewards](https://arxiv.org/abs/2610.13354), Octi Zhang et al. (2026). The method and paper belong to their authors. This repository is independent, and its UR7e task suite and implementation choices differ from the paper's benchmark. The complete original-paper citation is in [CITATION.bib](../CITATION.bib).

The following contributions are acknowledged separately from direct code reuse:

| Work | What this implementation references |
| --- | --- |
| [AlphaNIST](https://arxiv.org/abs/2610.13354) | Shared geometry-conditioned actor/critic, heterogeneous tasks, sparse completion reward, reset-bank sampling, distributed PPO |
| [OmniReset: Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning](https://omnireset.github.io/) — Patrick Yin et al., ICLR 2026 | Broad manipulation-relevant reset coverage; referenced configuration defaults |
| [A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control](https://sgs-rl.github.io/) — Octi Zhang et al., 2026 | Success-guided experience allocation |
| [SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning](https://openreview.net/forum?id=jXLiDKsuDo) — Hojoon Lee et al., ICLR 2025 | Running normalization and residual-network design/initialization |
| [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — John Schulman et al., 2017 | Clipped on-policy optimization |
| [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — John Schulman et al., 2015 | Advantage estimation with terminal/time-limit handling |
| [IndustReal](https://github.com/NVlabs/industrealkit) — NVIDIA and contributors | Gear source geometry, not policy or task-success results |

## Code and configuration provenance

The files are a curated export of the project's existing implementation. The learning algorithm is not copied from an official AlphaNIST release. It references the following pinned implementations; retaining their notices is attribution and traceability, not a claim that their licenses cover all project code.

| Upstream revision | Scope | Notice |
| --- | --- | --- |
| [UW-Lab/UWLab `825cfce`](https://github.com/UW-Lab/UWLab/tree/825cfce364f7e34b4b29aab73d37c5a1464f4375) | Referenced OmniReset configuration: PPO defaults, five-frame history, SGS history/epsilon, reset methodology | [BSD-3-Clause](../third_party/UWLab-BSD-3-Clause.txt) |
| [UW-Lab/rsl_rl `e7cd3c7`](https://github.com/UW-Lab/rsl_rl/tree/e7cd3c77bdb3c94753612f208c725e1add38a655) | Referenced adaptive Gaussian-KL learning-rate rule, clipped squared value loss, collection-time normalization behavior | [BSD-3-Clause](../third_party/RSL-RL-BSD-3-Clause.txt) |
| [SonyResearch/simba `7d0358b`](https://github.com/SonyResearch/simba/tree/7d0358b1f1fa3404fc5ea0ffedf9dc1424b49a45) | Referenced population moments, normalization epsilon/count, orthogonal projection and truncated-normal residual initialization | [Apache-2.0](../third_party/SimBa-Apache-2.0.txt) |
| [MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp) | Runtime dependency; hash-guarded source transformations in the five runtime-overlay modules. These contain matching/replacement source fragments and preserve installed upstream headers. Exact input/output hashes are in each module. | [Apache-2.0](../third_party/MuJoCo-Warp-Apache-2.0.txt) |
| [Universal Robots ROS2 Description `d4042c1`](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/tree/d4042c1f5123c86645cc5dbd650b4f461f697ee9) | UR7e chain, inertial/limit data and generated URDF; external collision meshes. Export adapts link frames and adds the experimental tool. | [BSD-3-Clause](../third_party/Universal-Robots-BSD-3-Clause.txt) |

Newton, MuJoCo, Warp, PyTorch, NumPy, SciPy, trimesh, Shapely, CoACD, USD, and Isaac Sim remain separately acquired dependencies under their own terms. No complete upstream runtime is vendored here.

Export changes are limited to packaging, portable fixture paths, task-count-independent fixture assertions, and removal of unrelated robot/station integrations. UR7e support retains its numerical kinematics and physical parameters. No private repository history is included.

## Explicit implementation choices

`policy.CONFIG_PROVENANCE` is the machine-readable source for the policy and optimizer attribution. Choices that should not be presented as confirmed author implementation include:

- Ordered flattened XYZ point aggregation, 1,024 points, and point/output-head initialization.
- A six-axis UR7e and Hand-E-style gripper, seven actions, and 140 proprioception values over five frames.
- The control-cost coefficient, action scaling, geometric pose metric, collision proxies, and physical reset acceptance thresholds.
- Storing collection-time normalized actor/critic features during PPO, and global cross-worker advantage normalization.
- An industrial task mix including gantry, motor mounting, shaft coupling, linear guide, enclosure, gears, threaded fasteners, rods, keyed pegs, and electrical connectors. This is not the paper's complete NIST task board.

No published success percentage, sample efficiency, or emergent behavior from the original work is claimed for this implementation.

## Rendered images

All five PNGs are existing Blender scene renders made for this project. Text/EXIF metadata has been removed for publication; compressed pixel-data chunks are unchanged. They show target configurations. No generated substitute images or paper figures are included. [Image provenance](images/provenance.json) records original source URLs, source hashes, image hashes, and transformations.

| Image | Geometry credit and terms |
| --- | --- |
| `02_motor_mount.png` | Charles Fox, motor and bracket, FreeCAD-library; CC BY 3.0 |
| `03_shaft_coupling.png` | Charles Fox (motor), WladIMirG (coupling), FreeCAD-library; CC BY 3.0 |
| `04_linear_guide.png` | FreeCAD-library contributors, MGN7 rail and carriage at the pinned source revision; CC BY 3.0 |
| `05_enclosure.png` | WladIMirG, enclosure and lid, FreeCAD-library; CC BY 3.0 |
| `06_gear_train.png` | NVIDIA IndustRealKit contributors, base and gears; NVIDIA research/evaluation license |

FreeCAD geometry comes from [FreeCAD-library `f5ca01c`](https://github.com/FreeCAD/FreeCAD-library/tree/f5ca01cdcaa2f69b9295a6382719c2d90628780c), with the original [asset license](../third_party/FreeCAD-assets-CC-BY-3.0.txt) and [CC BY 3.0 terms](https://creativecommons.org/licenses/by/3.0/). Changes: CAD conversion to USD, assembly placement, scene materials, and rendering. No endorsement by the model authors is implied.

Gear geometry comes from [IndustRealKit `0c91cb5`](https://github.com/NVlabs/industrealkit/tree/0c91cb5f7fb391e0cc8515fcdf669aab7e9ffe4b). The gear render is distributed under the included [NVIDIA license](../third_party/IndustReal-NVIDIA.txt), with its research/evaluation-only restriction. It is not covered by a permissive project-wide license.

## Excluded assets and licensing boundary

Raw CAD, workshop assets, bank archives, checkpoints, and infrastructure data are excluded. In particular, the inspected [UW-Lab asset dataset](https://huggingface.co/datasets/UW-Lab/uwlab-assets/tree/f9486a69b30a38355c1c2415fc15f6d6be8cb5b3) does not establish asset-specific redistribution rights for the NIST USD conversions used experimentally. A software repository's BSD license must not be assumed to cover third-party meshes. Those meshes and their renders are not published here.

Project-specific source has no blanket open-source grant in this initial export; see [LICENSE](../LICENSE). This preserves the distinction between permission to publish a curated implementation and ownership of every underlying contribution.
