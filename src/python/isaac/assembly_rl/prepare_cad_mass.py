"""Source CAD mass integrals with explicit nominal homogeneous densities.

Run with an OCP-enabled Python before prepare_source_assets; the JSON carries
source hashes so physics preparation rejects mass data from different CAD.
"""

import argparse
import hashlib
import json
import pathlib
import tempfile
import xml.etree.ElementTree as ET
import zipfile


def prepare(source_dir: pathlib.Path, output: pathlib.Path) -> None:
    from OCP.BRep import BRep_Builder
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepTools import BRepTools
    from OCP.GProp import GProp_GProps
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopoDS import TopoDS_Shape

    result = {}
    for name in (
        "motor",
        "motor_bracket",
        "coupling",
        "rail_mgn7",
        "carriage_detailed",
        "enclosure",
        "lid",
    ):
        if name in ("enclosure", "lid"):
            source = source_dir / "enclosure_assembly.FCStd"
            with zipfile.ZipFile(source) as archive:
                root = ET.fromstring(archive.read("Document.xml"))
                target = "Cut008001" if name == "enclosure" else "Common"
                obj = next(
                    o
                    for o in root.findall("./ObjectData/Object")
                    if o.attrib["name"] == target
                )
                filename = obj.find("./Properties/Property[@name='Shape']/Part").attrib[
                    "file"
                ]
                with tempfile.TemporaryDirectory() as tmp:
                    path = pathlib.Path(tmp) / "source.brp"
                    path.write_bytes(archive.read(filename))
                    shape = TopoDS_Shape()
                    BRepTools.Read_s(shape, str(path), BRep_Builder())
        else:
            source = source_dir / (name + ".step")
            reader = STEPControl_Reader()
            reader.ReadFile(str(source))
            reader.TransferRoots()
            shape = reader.OneShape()
        if shape.IsNull() or not BRepCheck_Analyzer(shape).IsValid():
            raise ValueError(f"Invalid source CAD: {source}")
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)
        if props.Mass() <= 0:
            raise ValueError(f"Non-positive source volume: {source}")
        center, inertia = props.CentreOfMass(), props.MatrixOfInertia()
        density = 1200 if name in ("enclosure", "lid") else 2700
        result[name] = {
            "mass_kg": props.Mass() * 1e-9 * density,
            "volume_m3": props.Mass() * 1e-9,
            "com_m": [center.X() / 1000, center.Y() / 1000, center.Z() / 1000],
            "inertia_kg_m2": [
                [inertia.Value(r, c) * 1e-15 * density for c in range(1, 4)]
                for r in range(1, 4)
            ],
            "density_kg_m3": density,
            "source_filename": source.name,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "provenance": "OpenCascade source BRep volume, center of mass and inertia integrals; nominal homogeneous density, not measured material or mass. Collision-only relief does not alter nominal physical inertia.",
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    prepare(args.sources, args.out)
