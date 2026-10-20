"""Fetch the seven upstream collision meshes used by the packaged UR7e URDF."""

import hashlib
import json
import pathlib
import urllib.request


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "scripts/robot_meshes.json").read_text())
    target = root / "src/python/asset_pipeline/ur7e_cell/official/collision"
    target.mkdir(parents=True, exist_ok=True)
    prefix = (
        "https://raw.githubusercontent.com/UniversalRobots/"
        "Universal_Robots_ROS2_Description/"
        f"{manifest['revision']}/{manifest['directory']}/"
    )
    for name, expected in manifest["sha256"].items():
        output = target / name
        if output.exists():
            content = output.read_bytes()
        else:
            with urllib.request.urlopen(prefix + name, timeout=60) as response:
                content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Unexpected mesh bytes: {name}; existing files are not overwritten")
        if not output.exists():
            with output.open("xb") as handle:
                handle.write(content)
        print(f"Verified {name}")


if __name__ == "__main__":
    main()
