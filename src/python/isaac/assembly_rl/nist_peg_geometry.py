"""Collision geometry for source NIST rods, keyed pegs and backed receivers."""

import itertools
import pathlib

import numpy as np

MOVABLES = {"RoundRod8x50", "RoundRod12x50", "KeyedPeg8x7", "KeyedPeg12x8"}
FIXTURES = {"RodReceiverFixture", "KeyedReceiverFixture"}
RECEIVERS = {
    "round_hole_8mm",
    "round_hole_12mm",
    "rectangular_hole_8mm",
    "rectangular_hole_12mm",
}


def mass_properties(hulls: list[np.ndarray], density: float) -> dict:
    """Integrate disjoint convex solids; material density is an explicit estimate."""
    from isaac.assembly_rl import collision_sections as cs

    if not np.isfinite(density) or density <= 0:
        raise ValueError("Expected positive finite nominal material density")
    properties = []
    for points in hulls:
        mesh = cs.convex_mesh(points)
        mesh.density = density
        properties.append(mesh.mass_properties)
    masses = np.asarray([p.mass for p in properties])
    centers = np.asarray([p.center_mass for p in properties])
    center = np.average(centers, weights=masses, axis=0)
    tensor = np.zeros((3, 3))
    for prop in properties:
        delta = prop.center_mass - center
        tensor += prop.inertia + prop.mass * (
            np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta)
        )
    if not np.isfinite(tensor).all() or np.any(np.linalg.eigvalsh(tensor) <= 0):
        raise ValueError("Invalid integrated inertia")
    return {
        "mass_kg": float(masses.sum()),
        "volume_m3": float(masses.sum() / density),
        "com_m": center.tolist(),
        "inertia_kg_m2": tensor.tolist(),
        "density_kg_m3": density,
        "provenance": "Disjoint source-profile/convex-envelope solid integrals with nominal homogeneous simulation material density; not measured mass or original CAD BRep mass.",
    }


def recipe(
    name: str, mesh, source_dir: pathlib.Path | None = None
) -> tuple[list[np.ndarray], dict]:
    """Preserve native receiver apertures; never widen a source mating surface."""
    import scipy.spatial
    import shapely
    import trimesh

    from isaac.assembly_rl import collision_sections as cs

    if name in MOVABLES:
        if not mesh.is_volume:
            raise ValueError(f"Expected closed source peg: {name}")
        hull = scipy.spatial.ConvexHull(mesh.vertices)
        points = np.asarray(mesh.vertices)[hull.vertices]
        proxy = cs.convex_mesh(points)
        samples = np.concatenate((proxy.vertices, proxy.triangles_center))
        # Millimetres avoid absolute triangle-degeneracy tolerances in proximity.
        reference_mm = mesh.copy()
        reference_mm.apply_scale(1000)
        _, distances, _ = trimesh.proximity.closest_point(reference_mm, samples * 1000)
        error = float(distances.max() / 1000)
        if error > 1e-6 or abs(hull.volume / mesh.volume - 1) > 1e-4:
            raise ValueError(f"Source peg is not convex within 1 micrometre: {name}")
        hulls = [points]
        report = {
            "method": "Convex envelope using unchanged source vertices; sub-micrometre source triangulation concavities bounded by sampled comparison",
            "source_volume_m3": float(mesh.volume),
            "maximum_sampled_surface_deviation_m": error,
            "surface_deviation_samples": len(samples),
        }
        density = 7800.0
    elif name in FIXTURES:
        if source_dir is None:
            raise ValueError(
                "Fixture recipe requires its actual component source directory"
            )
        profile = cs.section(mesh, 2, 0.0015)
        hulls, plate_report = cs.extrusion(profile, 2, 0, 0.003, error=1e-9)
        prefix = "round" if name == "RodReceiverFixture" else "rectangular"
        components = []
        for x, size in ((-0.04, 8), (0.04, 12)):
            component_name = f"{prefix}_hole_{size}mm"
            source = pathlib.Path(source_dir) / (component_name + ".obj")
            component = trimesh.load(source, force="mesh")
            pieces, detail = recipe(component_name, component)
            hulls.extend(points + np.array([x, 0, 0.003]) for points in pieces)
            components.append(
                {
                    "source_file": source.name,
                    "translation_m": [x, 0, 0.003],
                    "collision_report": detail,
                }
            )
        report = {
            "method": "Actual source receiver sections independently partitioned before fixture translation; source-backed 3 mm mount plate",
            "plate": plate_report,
            "components": components,
            "engineering_relief": None,
        }
        density = 2700.0
    elif name in RECEIVERS:
        levels = mesh.bounds[:, 2]
        hulls, slabs = [], []
        for low, high in itertools.pairwise(levels):
            middle = (low + high) / 2
            profile = cs.section(mesh, 2, middle)
            error = 0.0
            # End faces in the source receiver USD have small topological gaps;
            # verify the actual constant wall profile throughout each open slab.
            for z in np.linspace(low + 0.0001, high - 0.000192, 49):
                other = cs.section(mesh, 2, float(z))
                error = max(
                    error,
                    float(shapely.hausdorff_distance(profile.boundary, other.boundary)),
                )
            if error > 1e-7:
                raise ValueError(f"Receiver is not a constant source extrusion: {name}")
            pieces, detail = cs.extrusion(profile, 2, low, high, error=1e-9)
            hulls.extend(pieces)
            slabs.append(
                {
                    **detail,
                    "bottom_m": float(low),
                    "top_m": float(high),
                    "source_profile_comparison_samples": 49,
                    "maximum_sampled_profile_deviation_m": error,
                }
            )
        report = {
            "method": "Extrusion of actual constant source receiver sections with holes retained; supplied mount plate provides the bottom stop",
            "slabs": slabs,
            "engineering_relief": None,
        }
        density = 2700.0
    else:
        raise ValueError(f"Unknown NIST peg geometry: {name}")
    points = np.concatenate(hulls)
    if np.any(points.min(0) < mesh.bounds[0] - 1e-7) or np.any(
        points.max(0) > mesh.bounds[1] + 1e-7
    ):
        raise ValueError(f"Collision bounds exceed source: {name}")
    report.update(
        source_bounds_m=mesh.bounds.tolist(),
        collision_bounds_m=[points.min(0).tolist(), points.max(0).tolist()],
        convex_pieces=len(hulls),
        mass_properties=mass_properties(hulls, density),
        scope="Source-section and surface samples are not certified global error bounds; swept fit and dynamic contact validation remain required",
    )
    return hulls, report
