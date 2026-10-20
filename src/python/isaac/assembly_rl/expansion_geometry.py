"""Source-derived collision partitions and CAD mass for the four added assemblies.

Section recipes are sampled approximations, not physical readiness certificates.
All coordinates are source-local metres; the original CAD is in millimetres.
"""

import hashlib
import pathlib

import numpy as np

SOURCES = {
    "SK8Support": ("sk8_support.step", 2700.0),
    "FlangedBearing8": ("flanged_bearing8_gobilda.step", 7800.0),
    "CornerCube": ("makerbeam_corner.step", 2700.0),
    "FrameMemberX": ("makerbeam40.step", 2700.0),
    "FrameMemberY": ("makerbeam40.step", 2700.0),
    "DrivenShaft8x50": ("shaft8x50_gobilda.step", 7800.0),
    "StainlessShaft8x50": ("shaft8x50_gobilda.step", 7800.0),
}


def mass_properties(name: str, source_dir: pathlib.Path) -> dict:
    """Integrate valid source BRep, independently of tessellation or bore relief."""
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.STEPControl import STEPControl_Reader

    filename, density = SOURCES[name]
    source = pathlib.Path(source_dir) / "expansion_candidates" / filename
    reader = STEPControl_Reader()
    if reader.ReadFile(str(source)) != 1 or reader.TransferRoots() < 1:
        raise ValueError(f"Cannot read source CAD: {source}")
    shape = reader.OneShape()
    if shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid():
        raise ValueError(f"Invalid source BRep: {source}")
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    center, inertia = properties.CentreOfMass(), properties.MatrixOfInertia()
    tensor = np.array(
        [[inertia.Value(row, col) for col in range(1, 4)] for row in range(1, 4)]
    ) * (density * 1e-15)
    if properties.Mass() <= 0 or not np.all(np.linalg.eigvalsh(tensor) > 0):
        raise ValueError(f"Non-positive source mass or inertia: {source}")
    return {
        "mass_kg": properties.Mass() * density * 1e-9,
        "volume_m3": properties.Mass() * 1e-9,
        "com_m": [center.X() / 1000, center.Y() / 1000, center.Z() / 1000],
        "inertia_kg_m2": tensor.tolist(),
        "density_kg_m3": density,
        "source_filename": str(source.relative_to(source_dir)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "provenance": "Valid source OpenCascade BRep volume, COM and inertia integrals; nominal homogeneous density, not measured mass. Collision relief does not alter inertia.",
    }


def recipe(name: str, mesh) -> tuple[list[np.ndarray], dict]:
    """Retain source apertures, including explicit collision-only shaft clearance."""
    import scipy.spatial
    import shapely

    from isaac.assembly_rl import collision_sections as cs

    if name not in SOURCES:
        raise ValueError(f"No expansion geometry recipe for {name}")
    relief = None
    if name in ("DrivenShaft8x50", "StainlessShaft8x50"):
        source_hull = scipy.spatial.ConvexHull(mesh.vertices)
        if not mesh.is_volume or not np.isclose(
            source_hull.volume, mesh.volume, rtol=1e-8, atol=1e-16
        ):
            raise ValueError("Expected the unchanged convex source shaft")
        hulls = [np.asarray(mesh.vertices)[source_hull.vertices]]
        report = {"method": "Unchanged convex source shaft, including end chamfers"}
    elif name == "FlangedBearing8":
        hulls, report = _bearing(mesh)
    elif name in ("FrameMemberX", "FrameMemberY"):
        hulls, report = cs.layered(mesh, axis=2, error=1e-5)
    else:
        if name == "SK8Support":
            axis = 2
            levels = [0, 0.0002, *np.arange(0.003, 0.01101, 0.001), 0.0138, 0.014]
            center = np.array([0.009, 0.014])
        else:
            axis = 2
            levels = np.linspace(-0.005, 0.005, 41)
        if name == "SK8Support":
            sides = 32
            angles = np.arange(sides) * 2 * np.pi / sides
            radius = 0.00405 / np.cos(np.pi / sides)
            cutter = shapely.Polygon(
                center + radius * np.c_[np.cos(angles), np.sin(angles)]
            )

            def relief(polygon, low, high):
                return polygon.difference(cutter)

        hulls, report = cs.common_sections(
            mesh, axis, levels, error=2.5e-5, sample_step=0.0001, relief=relief
        )
        report["maximum_slab_span_m"] = float(np.diff(levels).max())
        if relief is not None:
            report["engineering_relief"] = {
                "description": "Source 8 mm shaft bore enlarged by a circumscribed 32-sided aperture; source shaft and physical inertia unchanged",
                "nominal_radial_clearance_m": 0.00005,
                "maximum_aperture_radius_m": float(radius),
                "axis": axis,
                "center_in_section_m": center.tolist(),
            }
    if name in ("SK8Support", "FlangedBearing8"):
        axes = [0, 1] if name == "SK8Support" else [0, 2]
        center = [0.009, 0.014] if name == "SK8Support" else [0, 0]
        clearance = (
            min(
                shapely.MultiPoint(p[:, axes]).convex_hull.distance(
                    shapely.Point(center)
                )
                for p in hulls
            )
            - 0.004
        )
        if clearance < 0.00005 - 1e-9:
            raise ValueError(
                f"Collision simplification consumed bore clearance: {name}"
            )
        report["measured_minimum_bore_radial_clearance_m"] = float(clearance)
    points = np.concatenate(hulls)
    if np.any(points.min(0) < mesh.bounds[0] - 1e-7) or np.any(
        points.max(0) > mesh.bounds[1] + 1e-7
    ):
        raise ValueError(f"Collision partition exceeds source bounds: {name}")
    report.update(
        source_bounds_m=mesh.bounds.tolist(),
        collision_bounds_m=[points.min(0).tolist(), points.max(0).tolist()],
        convex_pieces=len(hulls),
        collision_volume_m3=sum(scipy.spatial.ConvexHull(p).volume for p in hulls),
        scope="Source-section sampling is not a certified surface bound; swept-fit and dynamic contact validation remain required",
    )
    return hulls, report


def _bearing(mesh):
    """Partition the source axial race profile, then sweep each convex piece."""
    import scipy.spatial
    import shapely
    import trimesh

    from isaac.assembly_rl import collision_sections as cs

    profile = cs.section(mesh, 2, 0).intersection(shapely.box(0, -1, 1, 1))
    symmetry_error = 0.0
    for angle in (0.37, 0.83, 1.29):
        rotated = mesh.copy()
        rotated.apply_transform(
            trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        )
        check = cs.section(rotated, 2, 0).intersection(shapely.box(0, -1, 1, 1))
        symmetry_error = max(
            symmetry_error, shapely.hausdorff_distance(profile.boundary, check.boundary)
        )
    if symmetry_error > 2.5e-5:
        raise ValueError(
            "Bearing source is not rotationally symmetric within 25 micrometres"
        )
    sides = 48
    bore_radius = 0.00405 / np.cos(np.pi / sides)
    profile = profile.intersection(shapely.box(bore_radius, -1, 1, 1))
    pieces, report = cs.extrusion(profile, 2, 0, 0.001, error=1e-5)
    hulls = []
    for piece in pieces:
        radial_axial = np.unique(piece[:, :2], axis=0)
        radius, axial = radial_axial.T
        for index in range(sides):
            angles = np.array([index, index + 1]) * 2 * np.pi / sides
            points = np.concatenate(
                [np.c_[radius * np.cos(a), axial, radius * np.sin(a)] for a in angles]
            )
            hull = scipy.spatial.ConvexHull(points)
            hulls.append(points[hull.vertices])
    report.update(
        method="Source axial race profile convex partition swept through 48 angular sectors; actual grooves and apertures retained",
        angular_sectors=sides,
        source_profile_symmetry_sample_error_m=float(symmetry_error),
        maximum_angular_chord_error_m=float(
            mesh.extents[0] / 2 * (1 - np.cos(np.pi / sides))
        ),
        engineering_relief={
            "description": "Collision-only bearing bore relief; source shaft and CAD mass unchanged",
            "nominal_radial_clearance_m": 0.00005,
            "maximum_aperture_radius_m": float(bore_radius),
            "axis": 1,
            "center_in_section_m": [0.0, 0.0],
        },
    )
    return hulls, report
